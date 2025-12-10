import types
import sys

# ----- Fake llama_cpp.llama_chat_format -----
lcf = types.ModuleType("llama_cpp.llama_chat_format")

class _DummyHandler:
    def __init__(self, clip_model_path=None, mmproj=None, **kwargs):
        self.clip_model_path = clip_model_path or mmproj

Llava15ChatHandler = _DummyHandler
Llava16ChatHandler = _DummyHandler
MoondreamChatHandler = _DummyHandler
NanollavaChatHandler = _DummyHandler
Llama3VisionAlphaChatHandler = _DummyHandler
MiniCPMv26ChatHandler = _DummyHandler
Qwen25VLChatHandler = _DummyHandler

lcf.CHAT_FORMATS = {"llama-2": object(), "chatml": object()}
lcf.Llava15ChatHandler = Llava15ChatHandler
lcf.Llava16ChatHandler = Llava16ChatHandler
lcf.MoondreamChatHandler = MoondreamChatHandler
lcf.NanollavaChatHandler = NanollavaChatHandler
lcf.Llama3VisionAlphaChatHandler = Llama3VisionAlphaChatHandler
lcf.MiniCPMv26ChatHandler = MiniCPMv26ChatHandler
lcf.Qwen25VLChatHandler = Qwen25VLChatHandler

# ----- Fake llama_cpp root with Llama -----
llama_cpp = types.ModuleType("llama_cpp")

class Llama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

llama_cpp.Llama = Llama
llama_cpp.llama_chat_format = lcf

sys.modules["llama_cpp"] = llama_cpp
sys.modules["llama_cpp.llama_chat_format"] = lcf

print("Fake llama_cpp module has been set up for testing.")
