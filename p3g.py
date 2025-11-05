# pip install --upgrade gradio llama-cpp-python
import base64, mimetypes, os, traceback
import gradio as gr
from llama_cpp import Llama

MODEL_PATH  = "/Users/x/Downloads/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "/Users/x/Downloads/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    mmproj=MMPROJ_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    chat_format="qwen",   # or remove if your build prefers raw messages
)

def _img_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime: mime = "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def generate_response_stream(prompt, image_path):
    try:
        # Build messages
        if image_path and os.path.exists(image_path):
            data_uri = _img_to_data_uri(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe the image."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }]
        else:
            messages = [{"role": "user", "content": prompt}]

        acc = ""
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=True,                 # <— streaming from llama-cpp-python
        ):
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content") or delta.get("text") or ""
            if piece:
                acc += piece
                yield acc               # <— yield incremental text to Gradio

    except Exception as e:
        yield f"Exception: {e}\n\n{traceback.format_exc()}"

with gr.Blocks(title="Qwen2-VL · Streaming") as demo:
    gr.Markdown("### Qwen2-VL (GGUF) · Multimodal · Streaming")
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", lines=5)
        image_in  = gr.Image(label="Image (optional)", type="filepath")
    out = gr.Textbox(label="Response", lines=14)
    btn = gr.Button("Generate")
    btn.click(generate_response_stream, inputs=[prompt_in, image_in], outputs=out)

    # Enable Gradio streaming and concurrency control
    demo.queue(max_size=32).launch(server_port=7860, server_name="0.0.0.0")
