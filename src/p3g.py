# pip install --upgrade "llama-cpp-python[server]" gradio
import os, base64, mimetypes, traceback, threading
import gradio as gr
from typing import Optional, Dict, Any, Tuple

from llama_cpp import Llama
import llama_cpp.llama_chat_format as lcf  # dynamic chat-format discovery

"""Core UI and model logic for p3g. Importable without side effects. Use a separate runner to launch the app."""

# ----------------------------
# Try to import all vision chat handlers that may exist in your installed version.
# We build a dynamic registry so the UI only shows handlers that are actually available.
# ----------------------------
_HANDLER_IMPORTS = {
    "LLaVA 1.5 (Llava15ChatHandler)": ("llama_cpp.llama_chat_format", "Llava15ChatHandler"),
    "LLaVA 1.6 (Llava16ChatHandler)": ("llama_cpp.llama_chat_format", "Llava16ChatHandler"),
    "Moondream2 (MoondreamChatHandler)": ("llama_cpp.llama_chat_format", "MoondreamChatHandler"),
    "NanoLLaVA (NanollavaChatHandler)": ("llama_cpp.llama_chat_format", "NanollavaChatHandler"),
    "Llama-3 Vision Alpha (Llama3VisionAlphaChatHandler)": ("llama_cpp.llama_chat_format", "Llama3VisionAlphaChatHandler"),
    "MiniCPM-V 2.6 (MiniCPMv26ChatHandler)": ("llama_cpp.llama_chat_format", "MiniCPMv26ChatHandler"),
    "Qwen2.5-VL (Qwen25VLChatHandler)": ("llama_cpp.llama_chat_format", "Qwen25VLChatHandler"),
}

AVAILABLE_HANDLERS: Dict[str, Any] = {}
for label, (mod, cls_name) in _HANDLER_IMPORTS.items():
    try:
        module = __import__(mod, fromlist=[cls_name])
        AVAILABLE_HANDLERS[label] = getattr(module, cls_name)
    except Exception:
        pass  # Not available in this installed version

# ----------------------------
# Globals + simple load lock
# ----------------------------
LLM_LOCK = threading.Lock()
LLM: Optional[Llama] = None
CURR_DESC = ""

# ----------------------------
# Discover supported chat formats dynamically from llama_cpp.llama_chat_format.
# Falls back to a conservative static list if discovery fails.
# ----------------------------
def _discover_chat_formats() -> list[str]:
    names: set[str] = set()

    # Known registries across versions
    for attr in ("CHAT_FORMATS", "CHAT_FORMAT_REGISTRY", "_CHAT_FORMATS", "_CHAT_FORMAT_REGISTRY"):
        if hasattr(lcf, attr):
            reg = getattr(lcf, attr)
            if isinstance(reg, dict):
                names.update(map(str, reg.keys()))

    # Helper functions, if present
    for fn in ("chat_format_names", "list_chat_format_names", "get_chat_format_names"):
        if hasattr(lcf, fn):
            try:
                maybe = getattr(lcf, fn)()
                if isinstance(maybe, (list, tuple)):
                    names.update(map(str, maybe))
            except Exception:
                pass

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
    # If user passed an image but the loaded model is text-only, we will ignore it upstream.
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

# ----------------------------
# Filename-based auto-detect → pick the most likely vision handler.
# Fallback policy: if no match, choose LLaVA 1.5 (if available), else first available.
# ----------------------------
_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("LLaVA 1.5 (Llava15ChatHandler)", ("llava-1.5", "llava15", "llava_v1.5")),
    ("LLaVA 1.6 (Llava16ChatHandler)", ("llava-1.6", "llava16", "llava_v1.6")),
    ("Moondream2 (MoondreamChatHandler)", ("moondream2", "moondream")),
    ("NanoLLaVA (NanollavaChatHandler)", ("nanollava", "nano-llava")),
    ("Llama-3 Vision Alpha (Llama3VisionAlphaChatHandler)", ("llama-3-vision", "llama3-vision", "llama-3-vision-alpha")),
    ("MiniCPM-V 2.6 (MiniCPMv26ChatHandler)", ("minicpm-v-2.6", "minicpmv26", "minicpm-v2.6")),
    ("Qwen2.5-VL (Qwen25VLChatHandler)", ("qwen2.5-vl", "qwen2_5-vl", "qwen25-vl", "qwen-2.5-vl")),
)

def _infer_handler_from_name(model_path: str) -> Optional[str]:
    name = os.path.basename(model_path).lower()
    for label, keys in _PATTERNS:
        if label in AVAILABLE_HANDLERS and any(k in name for k in keys):
            return label
    if "LLaVA 1.5 (Llava15ChatHandler)" in AVAILABLE_HANDLERS:
        return "LLaVA 1.5 (Llava15ChatHandler)"
    return next(iter(AVAILABLE_HANDLERS.keys()), None)

# ----------------------------
# Model loader (now with automatic TEXT-ONLY FALLBACK if mmproj is missing)
# ----------------------------
def load_model(
    model_file: str,
    mmproj_file: Optional[str],
    handler_choice: str,         # "Auto from filename" | any AVAILABLE_HANDLERS key | "Raw text (no images)"
    chat_format_choice: str,     # "" for auto (use gguf metadata if present) or any discovered chat_format (text-only path)
    n_ctx: int,
    n_gpu_layers: int,
    verbose: bool
):
    global LLM, CURR_DESC
    with LLM_LOCK:
        LLM = None

        kwargs: Dict[str, Any] = dict(
            model_path=model_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        desc_lines = [f"Model: {model_file}"]

        # Resolve vision handler selection (auto)
        if handler_choice == "Auto from filename":
            inferred = _infer_handler_from_name(model_file)
            handler_choice = inferred if inferred is not None else "Raw text (no images)"

        # If a vision handler was selected but mmproj is missing → FALL BACK to text-only
        fallback_to_text = handler_choice in AVAILABLE_HANDLERS and not (mmproj_file and os.path.exists(mmproj_file))
        if fallback_to_text:
            missing = "No mmproj provided; falling back to text-only mode."
            handler_choice = "Raw text (no images)"
            desc_lines.append(f"Notice: {missing}")

        # Vision path using a chat_handler
        if handler_choice in AVAILABLE_HANDLERS:
            handler_cls = AVAILABLE_HANDLERS[handler_choice]
            try:
                chat_handler = handler_cls(clip_model_path=mmproj_file)  # typical kw
            except TypeError:
                try:
                    chat_handler = handler_cls(mmproj=mmproj_file)
                except TypeError:
                    chat_handler = handler_cls()
            kwargs["chat_handler"] = chat_handler
            desc_lines.append(f"Handler: {handler_choice}")
            desc_lines.append(f"mmproj: {mmproj_file}")

        # Text-only path; optionally apply a chat_format from dropdown
        elif handler_choice == "Raw text (no images)":
            desc_lines.append("Handler: raw text only. Images ignored.")
            cf = (chat_format_choice or "").strip()
            if cf:
                if cf in CHAT_FORMATS:
                    kwargs["chat_format"] = cf
                    desc_lines.append(f"chat_format: {cf}")
                else:
                    return None, f"Unsupported chat_format selected: {cf}", ""

        else:
            return None, f"Unknown handler: {handler_choice}", ""

        try:
            llm = Llama(**kwargs)
        except Exception as e:
            return None, f"Load error: {e}", ""

        LLM = llm
        CURR_DESC = "\n".join(desc_lines + [f"n_ctx={n_ctx}", f"n_gpu_layers={n_gpu_layers}"])
        return True, "Loaded.", CURR_DESC

# ----------------------------
# Inference
# ----------------------------
def generate_response_stream(prompt, image_path):
    if LLM is None:
        yield "Model is not loaded. Load a model first."
        return
    try:
        # If current config is text-only, ignore any passed image by forcing text messages
        is_text_only = "Handler: raw text only." in CURR_DESC
        messages = _build_messages(prompt, None if is_text_only else image_path, put_image_first=True)
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

# ----------------------------
# Gradio UI
# ----------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local VL · GGUF+mmproj · Streaming") as demo:
        gr.Markdown("### Local Multimodal LLM · Drag-and-drop model and mmproj")

        with gr.Row():
            model_file = gr.File(label="Model GGUF", file_types=[".gguf"], type="filepath")
            mmproj_file = gr.File(label="mmproj GGUF (optional; required only for vision handlers)", file_types=[".gguf"], type="filepath")

        # Build handler dropdown from available handlers dynamically
        handler_choices = ["Auto from filename"] + list(AVAILABLE_HANDLERS.keys()) + ["Raw text (no images)"]
        default_handler = "Auto from filename" if AVAILABLE_HANDLERS else "Raw text (no images)"
        handler_choice = gr.Dropdown(choices=handler_choices, value=default_handler, label="Vision handler")

        # Text chat_format dropdown (used only when handler = Raw text)
        chat_format_choice = gr.Dropdown(
            choices=[""] + CHAT_FORMATS,  # "" = Auto (use gguf tokenizer.chat_template / default)
            value="",
            label="Chat format (text-only; ignored if using a vision handler)"
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
            image_in  = gr.Image(label="Image (optional; ignored if text-only fallback used)", type="filepath")
        out = gr.Textbox(label="Response (streaming)", lines=14)
        gen_btn = gr.Button("Generate")

        def _on_load(model_path, mmproj_path, handler, cf_choice, ctx, ngpu, verb):
            if model_path is None:
                return "Provide a model GGUF.", ""
            ok, msg, desc = load_model(
                model_file=model_path,
                mmproj_file=mmproj_path,
                handler_choice=handler,
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
            inputs=[model_file, mmproj_file, handler_choice, chat_format_choice, n_ctx, n_gpu_layers, verbose],
            outputs=[load_status, model_desc],
        )

        gen_btn.click(generate_response_stream, inputs=[prompt_in, image_in], outputs=out)

    return demo

__all__ = [
    "_build_messages",
    "load_model",
    "generate_response_stream",
    "_infer_handler_from_name",
    "AVAILABLE_HANDLERS",
    "CHAT_FORMATS",
    "LLM",
    "CURR_DESC",
    "LLM_LOCK",
    "build_ui",
]
