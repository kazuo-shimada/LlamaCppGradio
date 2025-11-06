import os, base64, mimetypes, traceback
import gradio as gr
from llama_cpp import Llama

# ---------- Helpers ----------
def _img_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime: mime = "image/png"
    with open(path, "rb") as f:
        return f"data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode())

def _load_llm(model_path: str, mmproj_path: str | None, n_ctx: int, n_gpu_layers: int, chat_format: str | None):
    kwargs = dict(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )
    if mmproj_path:
        kwargs["mmproj"] = mmproj_path

    # Chat format handling
    if chat_format == "qwen":
        kwargs["chat_format"] = "qwen"
    # "none" => no chat_format; "auto" => try qwen then fallback
    elif chat_format == "auto":
        try:
            return Llama(**kwargs, chat_format="qwen")
        except Exception:
            pass  # fall through to no handler

    return Llama(**kwargs)

def _model_summary(llm: Llama, model_path: str, mmproj_path: str | None) -> str:
    parts = [f"**Model loaded**: `{os.path.basename(model_path)}`"]
    if mmproj_path:
        parts.append(f"**Vision projector**: `{os.path.basename(mmproj_path)}`")
    parts.append(f"**Context**: {llm.n_ctx_train} tokens  |  **GPU layers**: set")
    return "\n\n".join(parts)

# ---------- Inference (streaming) ----------
def generate_response_stream(prompt, image_path, llm_state, loaded_ok):
    try:
        if not loaded_ok or llm_state is None:
            yield "Model not loaded. Use the Load Model button above. Provide a valid model path and try again."
            return

        # Build messages for text or multimodal
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
        for chunk in llm_state.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content") or delta.get("text") or ""
            if piece:
                acc += piece
                yield acc

    except Exception as e:
        yield f"Exception: {e}\n\n{traceback.format_exc()}"

# ---------- Load / Unload actions ----------
def do_load(model_path, mmproj_path, n_ctx, n_gpu_layers, chat_format):
    # Validate inputs
    errs = []
    if not model_path:
        errs.append("Provide a path to a GGUF model file.")
    elif not os.path.exists(model_path):
        errs.append(f"Model file not found: `{model_path}`")
    elif not model_path.endswith(".gguf"):
        errs.append("Model must be a `.gguf` file.")

    if mmproj_path:
        if not os.path.exists(mmproj_path):
            errs.append(f"Projector file not found: `{mmproj_path}`")
        elif not mmproj_path.endswith(".gguf"):
            errs.append("Projector must be a `.gguf` file for vision.")

    if errs:
        msg = "### Setup incomplete\n" + "\n".join(f"- {e}" for e in errs) + \
              "\n\nFix these issues, then click **Load Model** again."
        return (gr.Markdown.update(value=msg),
                gr.Button.update(interactive=False),
                None, False)

    # Try instantiate
    try:
        llm = _load_llm(model_path, mmproj_path or None, int(n_ctx), int(n_gpu_layers), chat_format)
        msg = "### Ready\n" + _model_summary(llm, model_path, mmproj_path or None) + \
              "\n\nTips:\n- If you attach an image without a projector, responses will be text-only.\n- If tokens stall, reduce max tokens or raise context."
        return (gr.Markdown.update(value=msg),
                gr.Button.update(interactive=True),
                llm, True)
    except Exception as e:
        msg = "### Load failed\n" \
              f"- Exception: `{type(e).__name__}`\n" \
              f"- Details:\n```\n{e}\n```\n" \
              "Check that model and projector match the same family and that your llama-cpp-python build supports them."
        return (gr.Markdown.update(value=msg),
                gr.Button.update(interactive=False),
                None, False)

def do_unload():
    return (gr.Markdown.update(value="### Unloaded\nLoad a model to begin."),
            gr.Button.update(interactive=False),
            None, False)

# ---------- UI ----------
with gr.Blocks(title="Local LLM · Load-Then-Chat") as demo:
    gr.Markdown("## Local LLM with optional Vision\nSet paths, load the model, then chat. App stays disabled until a model is loaded.")

    with gr.Group():
        with gr.Row():
            model_path = gr.Textbox(label="Model GGUF path", placeholder="/Users/you/Downloads/Qwen2-VL-7B-Instruct-Q4_K_M.gguf")
            mmproj_path = gr.Textbox(label="Projector GGUF path (optional for vision)", placeholder="/Users/you/Downloads/mmproj-Qwen2-VL-7B-Instruct-f16.gguf")
        with gr.Row():
            chat_format = gr.Dropdown(choices=["auto", "qwen", "none"], value="qwen", label="Chat format")
            n_ctx = gr.Slider(2048, 8192, value=4096, step=512, label="Context tokens (n_ctx)")
            n_gpu_layers = gr.Slider(-1, 64, value=-1, step=1, label="GPU layers (-1 = auto/all)")

        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload")

        status = gr.Markdown("### Unloaded\nProvide paths and click **Load Model**.")
        llm_state = gr.State(None)
        loaded_ok = gr.State(False)

    gr.Markdown("---")
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", lines=5, placeholder="Ask something…")
        image_in  = gr.Image(label="Image (optional)", type="filepath")
    out = gr.Textbox(label="Response", lines=14)
    gen_btn = gr.Button("Generate", interactive=False)

    # Wiring
    load_btn.click(
        do_load,
        inputs=[model_path, mmproj_path, n_ctx, n_gpu_layers, chat_format],
        outputs=[status, gen_btn, llm_state, loaded_ok],
    )
    unload_btn.click(
        do_unload,
        inputs=None,
        outputs=[status, gen_btn, llm_state, loaded_ok],
    )
    gen_btn.click(
        generate_response_stream,
        inputs=[prompt_in, image_in, llm_state, loaded_ok],
        outputs=out,
    )

    demo.queue(max_size=32).launch(server_port=7860, server_name="0.0.0.0")
