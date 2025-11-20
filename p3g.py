# pip install --upgrade "llama-cpp-python[server]" gradio
import os, base64, mimetypes, traceback, threading
import gradio as gr
from typing import Optional

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# ----------------------------
# Globals + simple load lock
# ----------------------------
LLM_LOCK = threading.Lock()
LLM: Optional[Llama] = None
CURR_DESC = ""

# ----------------------------
# Utilities
# ----------------------------
def _img_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def _build_messages(prompt: str, image_path: Optional[str], put_image_first=True):
    if image_path and os.path.exists(image_path):
        data_uri = _img_to_data_uri(image_path)
        content = (
            [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt or "Describe the image."},
            ]
            if put_image_first
            else [
                {"type": "text", "text": prompt or "Describe the image."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        )
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": prompt or "Say hello."}]

# -------------------------------------------------------------------
# Auto-handler with NEW behavior:
# ANYTHING unknown → fallback to LLaVA15 handler instead of None
# -------------------------------------------------------------------
def _infer_handler_from_name(model_path: str) -> str:
    name = os.path.basename(model_path).lower()

    # Qwen2.5-VL explicitly supported
    if (
        "qwen2.5-vl" in name
        or "qwen2_5-vl" in name
        or "qwen25-vl" in name
        or "qwen2.5_vl" in name
    ):
        return "Qwen2.5-VL (chat_format)"

    # Llava variants → LLaVA15
    if "llava" in name:
        return "LLaVA15 (chat_handler)"

    # NEW REQUESTED BEHAVIOR:
    # Qwen2-VL is UNKNOWN but now forced to fallback
    if "qwen2-vl" in name or "qwen-vl" in name or "qwen2_vl" in name:
        return "LLaVA15 (chat_handler)"

    # EVERYTHING ELSE → fallback to LLaVA15
    return "LLaVA15 (chat_handler)"

# ----------------------------
# Model loader
# ----------------------------
def load_model(
    model_file: str,
    mmproj_file: Optional[str],
    handler_choice: str,
    n_ctx: int,
    n_gpu_layers: int,
    verbose: bool
):
    global LLM, CURR_DESC
    with LLM_LOCK:
        LLM = None

        kwargs = dict(
            model_path=model_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

        desc_lines = [f"Model: {model_file}"]

        if handler_choice == "Auto from filename":
            handler_choice = _infer_handler_from_name(model_file)

        if handler_choice == "Qwen2.5-VL (chat_format)":
            if not mmproj_file or not os.path.exists(mmproj_file):
                return None, "Qwen2.5-VL needs an mmproj file. Provide one.", ""
            kwargs.update(
                dict(
                    chat_format="qwen2.5-vl",
                    mmproj=mmproj_file,
                )
            )
            desc_lines.append("Handler: chat_format='qwen2.5-vl'")
            desc_lines.append(f"mmproj: {mmproj_file}")

        elif handler_choice == "LLaVA15 (chat_handler)":
            if not mmproj_file or not os.path.exists(mmproj_file):
                return None, "LLaVA15 handler needs an mmproj file. Provide one.", ""
            chat_handler = Llava15ChatHandler(clip_model_path=mmproj_file)
            kwargs.update(dict(chat_handler=chat_handler))
            desc_lines.append("Handler: Llava15ChatHandler")
            desc_lines.append(f"mmproj: {mmproj_file}")

        elif handler_choice == "Raw text (no images)":
            desc_lines.append("Handler: raw text only. Images ignored.")

        else:
            return None, f"Unknown handler: {handler_choice}", ""

        try:
            llm = Llama(**kwargs)
        except Exception as e:
            return None, f"Load error: {e}", ""

        LLM = llm
        CURR_DESC = "\n".join(
            desc_lines + [f"n_ctx={n_ctx}", f"n_gpu_layers={n_gpu_layers}"]
        )
        return True, "Loaded.", CURR_DESC

# ----------------------------
# Inference
# ----------------------------
def generate_response_stream(prompt, image_path):
    global LLM
    if LLM is None:
        yield "Model is not loaded. Load a model first."
        return

    try:
        messages = _build_messages(prompt, image_path, put_image_first=True)
        acc = ""
        for chunk in LLM.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=True,
        ):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            piece = (
                delta.get("content")
                or delta.get("text")
                or choice.get("text")
                or ""
            )
            if piece:
                acc += piece
                yield acc
    except Exception as e:
        yield f"Exception: {e}\n\n{traceback.format_exc()}"

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="Local VL · GGUF+mmproj · Streaming") as demo:
    gr.Markdown("### Local Multimodal LLM · Drag-and-drop model and mmproj")

    with gr.Row():
        model_file = gr.File(
            label="Model GGUF",
            file_types=[".gguf"],
            type="filepath"
        )
        mmproj_file = gr.File(
            label="mmproj GGUF (optional for text-only, required for images)",
            file_types=[".gguf"],
            type="filepath"
        )

    handler_choice = gr.Dropdown(
        choices=[
            "Auto from filename",
            "Qwen2.5-VL (chat_format)",
            "LLaVA15 (chat_handler)",
            "Raw text (no images)",
        ],
        value="Auto from filename",
        label="Vision handler"
    )

    with gr.Row():
        n_ctx = gr.Slider(2048, 16384, value=8192, step=1024, label="n_ctx")
        n_gpu_layers = gr.Number(value=-1, precision=0, label="n_gpu_layers (-1 = all)")
        verbose = gr.Checkbox(value=True, label="verbose")

    load_btn = gr.Button("Load / Reload model", variant="primary")
    load_status = gr.Markdown()
    model_desc = gr.Textbox(label="Current model config", lines=6)

    gr.Markdown("---")
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", lines=5)
        image_in = gr.Image(label="Image (optional)", type="filepath")
    out = gr.Textbox(label="Response", lines=14)
    gen_btn = gr.Button("Generate")

    def _on_load(model_path, mmproj_path, handler, ctx, ngpu, verb):
        if model_path is None:
            return "Provide a model GGUF.", ""
        ok, msg, desc = load_model(
            model_file=model_path,
            mmproj_file=mmproj_path,
            handler_choice=handler,
            n_ctx=int(ctx),
            n_gpu_layers=int(ngpu),
            verbose=bool(verb),
        )
        if ok is None:
            return msg, ""
        return msg, desc

    load_btn.click(
        _on_load,
        inputs=[model_file, mmproj_file, handler_choice, n_ctx, n_gpu_layers, verbose],
        outputs=[load_status, model_desc],
    )

    gen_btn.click(generate_response_stream, inputs=[prompt_in, image_in], outputs=out)

    demo.queue(max_size=32).launch(server_port=7860, server_name="0.0.0.0")
