# pip install --upgrade "llama-cpp-python[server]" gradio
import os, base64, mimetypes, traceback, threading
import gradio as gr
from typing import Optional, Dict, Any

from llama_cpp import Llama
import llama_cpp.llama_chat_format as lcf  # used only to dynamically list text chat_formats

# ============================
# Globals
# ============================
LLM_LOCK = threading.Lock()
LLM: Optional[Llama] = None
CURR_DESC = ""          # human-readable current config
IS_TEXT_ONLY = True     # runtime flag so we can ignore images when needed

# ============================
# Dynamic discovery
# ============================
def _discover_chat_formats() -> list[str]:
    """
    Ask llama-cpp-python for the chat formats it currently registers.
    We don't hardcode; this updates automatically as your package updates.
    """
    names: set[str] = set()

    # Known registries across versions
    for attr in ("CHAT_FORMATS", "CHAT_FORMAT_REGISTRY", "_CHAT_FORMATS", "_CHAT_FORMAT_REGISTRY"):
        if hasattr(lcf, attr):
            reg = getattr(lcf, attr)
            if isinstance(reg, dict):
                names.update(map(str, reg.keys()))

    # Optional helper functions in some versions
    for fn in ("chat_format_names", "list_chat_format_names", "get_chat_format_names"):
        if hasattr(lcf, fn):
            try:
                maybe = getattr(lcf, fn)()
                if isinstance(maybe, (list, tuple)):
                    names.update(map(str, maybe))
            except Exception:
                pass

    # Fallback list if discovery fails
    if not names:
        names = {
            "llama-2", "llama-3", "alpaca", "qwen", "vicuna", "oasst_llama",
            "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy",
            "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion",
            "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma",
            "functionary", "functionary-v2", "functionary-v1", "chatml-function-calling",
        }
    return sorted(names)

CHAT_FORMATS = _discover_chat_formats()

# ============================
# Utilities
# ============================
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

# ============================
# Loader (AUTO like llama-cpp-python)
# ============================
def load_model(
    model_file: str,
    mmproj_file: Optional[str],
    load_mode: str,          # "Auto (recommended)" | "Force text-only" | "Advanced (manual chat_format override)"
    chat_format_choice: str, # used only for "Advanced (manual ...)"
    n_ctx: int,
    n_gpu_layers: int,
    verbose: bool
):
    """
    Auto-behavior mirrors llama-cpp-python:
      - We don't guess from filenames.
      - We do NOT set chat_format or chat_handler in Auto mode.
      - If mmproj is provided, we pass it as `mmproj=...` so llama-cpp-python can
        automatically instantiate the appropriate vision handler for that model.
      - If no mmproj is provided, the model runs text-only.
      - If Auto fails to initialize, we fall back to text-only automatically.
    """
    global LLM, CURR_DESC, IS_TEXT_ONLY
    with LLM_LOCK:
        LLM = None
        IS_TEXT_ONLY = True

        kwargs: Dict[str, Any] = dict(
            model_path=model_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        desc_lines = [f"Model: {model_file}"]

        # -----------------------
        # Mode selection
        # -----------------------
        if load_mode == "Auto (recommended)":
            # Do not set chat_format or chat_handler here.
            # Pass mmproj only if provided, to allow llama-cpp-python to auto-wire vision.
            if mmproj_file and os.path.exists(mmproj_file):
                kwargs["mmproj"] = mmproj_file
                desc_lines.append(f"Auto: provided mmproj -> {mmproj_file}")
            else:
                desc_lines.append("Auto: no mmproj -> text-only")

            # Try create Llama. If this fails (e.g., incompatible mmproj), fall back to text-only retry.
            try:
                llm = Llama(**kwargs)
                LLM = llm
                IS_TEXT_ONLY = "mmproj" not in kwargs  # if we didn't pass mmproj, it's text-only
                handler_line = "Handler: auto-vision (mmproj)" if not IS_TEXT_ONLY else "Handler: auto text-only"
                desc_lines.append(handler_line)

            except Exception as e:
                # Fallback: strip mmproj and try text-only
                err = f"Auto load failed with mmproj (if any). Falling back to text-only. Reason: {e}"
                desc_lines.append(f"Notice: {err}")
                kwargs.pop("mmproj", None)
                try:
                    llm = Llama(**kwargs)
                    LLM = llm
                    IS_TEXT_ONLY = True
                    desc_lines.append("Handler: auto text-only (fallback)")
                except Exception as e2:
                    return None, f"Load error (text-only fallback also failed): {e2}", ""

        elif load_mode == "Force text-only":
            # Explicitly force text-only. Optionally allow a user-provided chat_format override.
            desc_lines.append("Mode: Force text-only")
            cf = (chat_format_choice or "").strip()
            if cf:
                kwargs["chat_format"] = cf
                desc_lines.append(f"chat_format: {cf}")
            try:
                llm = Llama(**kwargs)
                LLM = llm
                IS_TEXT_ONLY = True
                desc_lines.append("Handler: text-only")
            except Exception as e:
                return None, f"Load error (text-only): {e}", ""

        else:  # "Advanced (manual chat_format override)"
            # This mode lets power users set a chat_format for text-only use.
            desc_lines.append("Mode: Advanced (manual chat_format override, text-only)")
            cf = (chat_format_choice or "").strip()
            if cf:
                kwargs["chat_format"] = cf
                desc_lines.append(f"chat_format: {cf}")
            try:
                llm = Llama(**kwargs)
                LLM = llm
                IS_TEXT_ONLY = True
                desc_lines.append("Handler: text-only (manual chat_format)")
            except Exception as e:
                return None, f"Load error (advanced/manual): {e}", ""

        CURR_DESC = "\n".join(desc_lines + [f"n_ctx={n_ctx}", f"n_gpu_layers={n_gpu_layers}"])
        return True, "Loaded.", CURR_DESC

# ============================
# Inference
# ============================
def generate_response_stream(prompt, image_path):
    global LLM, IS_TEXT_ONLY
    if LLM is None:
        yield "Model is not loaded. Load a model first."
        return
    try:
        # In text-only mode, ignore any provided image.
        effective_image = None if IS_TEXT_ONLY else image_path
        messages = _build_messages(prompt, effective_image, put_image_first=True)

        acc = ""
        for chunk in LLM.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=True,
        ):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            piece = delta.get("content") or delta.get("text") or choice.get("text") or ""
            if piece:
                acc += piece
                yield acc
    except Exception as e:
        yield f"Exception: {e}\n\n{traceback.format_exc()}"

# ============================
# Gradio UI
# ============================
with gr.Blocks(title="Local LLM · Auto chat_format/vision · GGUF + optional mmproj") as demo:
    gr.Markdown("### Local LLM with AUTO chat_format / AUTO vision handling (llama-cpp-python)")

    with gr.Row():
        model_file = gr.File(label="Model GGUF", file_types=[".gguf"], type="filepath")
        mmproj_file = gr.File(
            label="mmproj GGUF (optional; provide to enable auto vision if compatible)",
            file_types=[".gguf"],
            type="filepath"
        )

    load_mode = gr.Dropdown(
        choices=[
            "Auto (recommended)",
            "Force text-only",
            "Advanced (manual chat_format override)",
        ],
        value="Auto (recommended)",
        label="Load mode"
    )

    chat_format_choice = gr.Dropdown(
        choices=[""] + CHAT_FORMATS,   # "" → let llama-cpp-python auto-pick from tokenizer.chat_template
        value="",
        label="Chat format (used only for 'Force text-only' or 'Advanced' modes)"
    )

    with gr.Row():
        n_ctx = gr.Slider(2048, 16384, value=8192, step=1024, label="n_ctx")
        n_gpu_layers = gr.Number(value=-1, precision=0, label="n_gpu_layers (-1 = all)")
        verbose = gr.Checkbox(value=True, label="verbose")

    load_btn = gr.Button("Load / Reload model", variant="primary")
    load_status = gr.Markdown()
    model_desc = gr.Textbox(label="Current model config", lines=12)

    gr.Markdown("---")
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", lines=5)
        image_in  = gr.Image(label="Image (optional; ignored if text-only)", type="filepath")
    out = gr.Textbox(label="Response (streaming)", lines=14)
    gen_btn = gr.Button("Generate")

    def _on_load(model_path, mmproj_path, mode, cf_choice, ctx, ngpu, verb):
        if model_path is None:
            return "Provide a model GGUF.", ""
        ok, msg, desc = load_model(
            model_file=model_path,
            mmproj_file=mmproj_path,
            load_mode=mode,
            chat_format_choice=cf_choice,
            n_ctx=int(ctx),
            n_gpu_layers=int(ngpu),
            verbose=bool(verb),
        )
        if ok is None:
            return msg, ""
        return msg, desc

    load_btn.click(
        _on_load,
        inputs=[model_file, mmproj_file, load_mode, chat_format_choice, n_ctx, n_gpu_layers, verbose],
        outputs=[load_status, model_desc],
    )

    gen_btn.click(generate_response_stream, inputs=[prompt_in, image_in], outputs=out)

    demo.queue(max_size=32).launch(server_port=7860, server_name="0.0.0.0")
