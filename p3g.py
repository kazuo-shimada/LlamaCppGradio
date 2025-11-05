import base64, mimetypes, os, traceback
import gradio as gr
from llama_cpp import Llama

MODEL_PATH  = "/Users/x/Downloads/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "/Users/x/Downloads/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"

# Try Qwen handler. If not available, run without a handler.
def _load_llm():
    try:
        return Llama(
            model_path=MODEL_PATH,
            mmproj=MMPROJ_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,          # Metal on Apple Silicon
            chat_format="qwen",       # <- works on builds that lack 'qwen2_vl'
        )
    except Exception:
        return Llama(
            model_path=MODEL_PATH,
            mmproj=MMPROJ_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            # no chat_format; we’ll send ChatML manually if needed
        )

llm = _load_llm()

def _img_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

def _infer(messages, max_new_tokens=512, temperature=0.2):
    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    content = out["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "\n".join(
            p.get("text","") if isinstance(p,dict) and p.get("type")=="text" else str(p)
            for p in content
        ).strip()
    return str(content)

def generate_response(prompt, image_path):
    try:
        if image_path and os.path.exists(image_path):
            data_uri = _img_to_data_uri(image_path)
            # Qwen-style multimodal content
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe the image."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
        return _infer(messages)
    except Exception as e:
        return f"Exception: {e}\n\n{traceback.format_exc()}"

with gr.Blocks(title="Qwen2-VL via llama.cpp") as demo:
    gr.Markdown("## Qwen2-VL (GGUF) · Multimodal")
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", lines=5)
        image_in  = gr.Image(label="Image (optional)", type="filepath")
    out = gr.Textbox(label="Response", lines=14)
    btn = gr.Button("Generate")
    btn.click(generate_response, inputs=[prompt_in, image_in], outputs=out)

demo.launch(server_port=7860, server_name="0.0.0.0")
