# tests/test_model_loading.py
import os
import tempfile
import importlib

def reload_app():
    # Ensure fresh globals (AVAILABLE_HANDLERS, CURR_DESC, etc.) each test
    if "app" in globals():
        import sys
        sys.modules.pop("app", None)
    import app
    importlib.reload(app)
    return app

def test_fallback_to_text_when_mmproj_missing(tmp_path):
    app = reload_app()

    # Choose a vision handler label that is guaranteed to exist from your registry
    # e.g., Qwen2.5
    label = "Qwen2.5-VL (Qwen25VLChatHandler)"
    assert label in app.AVAILABLE_HANDLERS  # sanity

    # No mmproj -> should fall back to text-only
    ok, msg, desc = app.load_model(
        model_file=str(tmp_path / "model.gguf"),
        mmproj_file=None,
        handler_choice=label,          # explicitly pick a vision handler
        chat_format_choice="",         # not used in text-only unless set
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False,
    )

    assert ok is True
    assert "Handler: raw text only. Images ignored." in desc
    # Ensure no chat_handler was passed to the fake Llama
    assert "chat_handler" not in app.LLM.kwargs

def test_uses_vision_handler_when_mmproj_present(tmp_path):
    app = reload_app()

    # Create a real temp file for mmproj so os.path.exists(mmproj) is True
    mmproj = tmp_path / "vision.mmproj.gguf"
    mmproj.write_bytes(b"fake-mmproj")

    label = "LLaVA 1.5 (Llava15ChatHandler)"
    assert label in app.AVAILABLE_HANDLERS

    ok, msg, desc = app.load_model(
        model_file=str(tmp_path / "model.gguf"),
        mmproj_file=str(mmproj),
        handler_choice=label,
        chat_format_choice="",
        n_ctx=4096,
        n_gpu_layers=0,
        verbose=True,
    )

    assert ok is True
    assert f"Handler: {label}" in desc
    assert f"mmproj: {mmproj}" in desc
    # Now a chat_handler must be present
    assert "chat_handler" in app.LLM.kwargs
    # And the handler should have been constructed with our mmproj path
    assert getattr(app.LLM.kwargs["chat_handler"], "clip_model_path") == str(mmproj)

def test_auto_infer_handler_from_filename(tmp_path):
    app = reload_app()

    mmproj = tmp_path / "qwen.mmproj.gguf"
    mmproj.write_bytes(b"fake-mmproj")

    # Model filename contains qwen2.5-vl so Auto should pick the Qwen handler
    model_path = tmp_path / "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"
    model_path.write_bytes(b"fake-model")

    ok, msg, desc = app.load_model(
        model_file=str(model_path),
        mmproj_file=str(mmproj),
        handler_choice="Auto from filename",
        chat_format_choice="",
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )

    assert ok is True
    # Either Qwen handler or sensible fallback; prefer the Qwen label
    picked = [lbl for lbl in app.AVAILABLE_HANDLERS.keys() if lbl in desc]
    assert picked, "Auto should select a vision handler when mmproj exists"
    assert "mmproj:" in desc
