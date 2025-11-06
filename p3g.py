#!/usr/bin/env python3
import os, base64, mimetypes, traceback
import gradio as gr
from llama_cpp import Llama

# ----------------------------
# Helpers
# ----------------------------
def _img_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        return f"data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode())

def _humanize_err(e: Exception) -> str:
    msg = str(e) or e.__class__.__name__
    hints = []
    s = msg.lower()
    if "unknown model architecture" in s:
        hints.append("Model file is not supported by your llama.cpp build.")
    if "no such file" in s or "not found" in s:
        hints.append("Check the file path or upload again.")
    if "metal" in s and "n_gpu_layers" in s:
        hints.append("Try reducing n_gpu_layers or use CPU by setting it to 0.")
    if "mmproj" in s and "clip" in s:
        hints.append("mmproj must match the model family. Use the correct mmproj for your model.")
    if "permission" in s:
        hints.append("You may not have read permission on that file.")
    extra = ("\n\nHints:\n- " + "\n- ".join(hints)) if hints else ""
    return f"{msg}{extra}"

# ----------------------------
# Model lifecycle
# ----------------------------
def load_model(model_path, mmproj_path, n_ctx, n_gpu_layers, chat_format, state):
    try:
        # Validate inputs
        if not model_path:
            return gr.update(value="Load failed: No model selected."), state, gr.update(interactive=False)
        if isinstance(model_path, dict) and "temp_path" in model_path:
            model_path = model_path["temp_path"]  # gr.File returns dict
        if isinstance(mmproj_path, dict) and "temp_path" in mmproj_path:
            mmproj_path = mmproj_path["temp_path"]

        if not os.path.exists(model_path):
            return gr.update(value=f"Load failed: Model path not found:\n{model_path}"), state, gr.update(interactive=False)
        if mmproj_path and not os.path.exists(mmproj_path):
            return gr.update(value=f"Load failed: mmproj path not found:\n{mmproj_path}"), state, gr.update(interactive=False)

        # Close prior model if any
        if state and state.get("llm"):
            try:
                del state["llm"]
            except Exception:
                pass
            state = {"llm": None, "loaded": False}

        kwargs = dict(
            model_path=model_path,
            n_ctx=int(n_ctx) if n_ctx else 4096,
            n_gpu_layers=int(n_gpu_layers) if n_gpu_layers is not None else -1,
        )
        if chat_format and chat_format != "auto":
            kwargs["chat_format"] = chat_format
        if mmproj_path:
            kwargs["mmproj"] = mmproj_path

        llm = Llama(**kwargs)
        state = {"llm": llm, "loaded": True, "chat_format": chat_format or "auto"}
        status = (
            "Model loaded.\n"
            f"- model: {model_path}\n"
            f"- mmproj: {mmproj_path or '(none)'}\n"
            f"- n_ctx: {kwargs['n_ctx']}\n"
            f"- n_gpu_layers: {kwargs['n_gpu_layers']}\n"
            f"- chat_format: {state['chat_format']}\n\n"
            "You can generate now."
        )
        return gr.update(value=status), state, gr.update(interactive=True)

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        msg = _humanize_err(e)
        return gr.update(value=f"Load failed:\n{msg}\n\n{tb}"), state, gr.update(interactive=False)

def unload_model(state):
    if state and state.get("llm"):
        try:
            del state["llm"]
        except Exception:
            pass
    return {"llm": None, "loaded": False, "chat_format": "auto"}, gr.update(value="Unloaded. Load a model to continue."), gr.update(interactive=False)

# ----------------------------
# Generation (streaming)
# ----------------------------
def generate_stream(prompt, image_file, state, max_tokens, temperature):
    try:
        if not state or not state.get("loaded") or not state.get("llm"):
            yield "No model loaded. Use the Load Model button first."
            return

        llm: Llama = state["llm"]
        vision = False
        messages = []

        if image_file:
            # gr.Image returns a dict with 'path' in some configs; handle both
            img_path = image_file if isinstance(image_file, str) else image_file.get("path")
            if img_path and os.path.exists(img_path):
                data_uri = _img_to_data_uri(img_path)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Describe the image."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }]
                vision = True

        if not messages:
            # Text only
            messages = [{"role": "user", "content": prompt or "Say hello."}]

        acc = ""
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens) if max_tokens else 512,
            temperature=float(temperature) if temperature is not None else 0.2,
            stream=True,
        ):
            # llama-cpp-python streams OpenAI-style deltas
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content") or delta.get("text") or ""
            if piece:
                acc += piece
                yield acc
        # Ensure final value returned
        if not acc:
            yield "No output produced."
    except Exception as e:
        yield f"Exception: {e}\n\n{traceback.format_exc()}"

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="LLM · GGUF · Streaming") as demo:
    gr.Markdown("### Local LLM (llama.cpp · GGUF) — Load models in the UI and stream outputs")

    state = gr.State({"llm": None, "loaded": False, "chat_format": "auto"})

    with gr.Tab("1) Model"):
        gr.Markdown("Select your model files. Click **Load Model**. Generation stays disabled until load succeeds.")
        with gr.Row():
            model_in = gr.File(label="Model .gguf", file_count="single", file_types=[".gguf"])
            mmproj_in = gr.File(label="mmproj .gguf (optional for vision models)", file_count="single", file_types=[".gguf"])
        with gr.Row():
            n_ctx_in = gr.Number(value=4096, precision=0, label="n_ctx")
            n_gpu_layers_in = gr.Number(value=-1, precision=0, label="n_gpu_layers (-1 = all GPU, 0 = CPU)")
            chat_format_in = gr.Dropdown(
                choices=["auto", "qwen", "qwen2", "llama-3", "chatml", "mistral"],
                value="auto",
                label="chat_format"
            )
        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload")
        status_out = gr.Textbox(label="Status", lines=8, interactive=False)

    with gr.Tab("2) Generate"):
        with gr.Row():
            prompt_in = gr.Textbox(label="Prompt", lines=6, placeholder="Ask something…")
            image_in = gr.Image(label="Image (optional)", type="filepath")
        with gr.Row():
            max_tokens_in = gr.Slider(minimum=16, maximum=4096, step=16, value=512, label="max_tokens")
            temp_in = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.2, label="temperature")
        generate_btn = gr.Button("Generate", variant="primary", interactive=False)
        out_box = gr.Textbox(label="Response (streaming)", lines=16)

    # Wiring
    load_btn.click(
        fn=load_model,
        inputs=[model_in, mmproj_in, n_ctx_in, n_gpu_layers_in, chat_format_in, state],
        outputs=[status_out, state, generate_btn],
        queue=True
    )

    unload_btn.click(
        fn=unload_model,
        inputs=[state],
        outputs=[state, status_out, generate_btn],
        queue=False
    )

    generate_btn.click(
        fn=generate_stream,
        inputs=[prompt_in, image_in, state, max_tokens_in, temp_in],
        outputs=out_box,
        queue=True
    )

    demo.queue(max_size=64).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
