import importlib

def reload_app(modname="p3g"):
    import sys
    if modname in sys.modules:
        sys.modules.pop(modname, None)
    mod = importlib.import_module(modname)
    return importlib.reload(mod)

def test_fallback_to_text_when_mmproj_missing(tmp_path):
    app = reload_app()

    label = "Qwen2.5-VL (Qwen25VLChatHandler)"
    assert label in app.AVAILABLE_HANDLERS

    ok, msg, desc = app.load_model(
        model_file=str(tmp_path / "model.gguf"),
        mmproj_file=None,
        handler_choice=label,
        chat_format_choice="",
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False,
    )

    assert ok is True
    assert "Handler: raw text only. Images ignored." in desc
    assert "chat_handler" not in app.LLM.kwargs

def test_uses_vision_handler_when_mmproj_present(tmp_path):
    app = reload_app()

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
    assert "chat_handler" in app.LLM.kwargs
    assert getattr(app.LLM.kwargs["chat_handler"], "clip_model_path") == str(mmproj)

def test_auto_infer_handler_from_filename(tmp_path):
    app = reload_app()

    mmproj = tmp_path / "qwen.mmproj.gguf"
    mmproj.write_bytes(b"fake-mmproj")

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
    assert "mmproj:" in desc
    picked = [lbl for lbl in app.AVAILABLE_HANDLERS.keys() if lbl in desc]
    assert picked, "Auto should select a vision handler when mmproj exists"
