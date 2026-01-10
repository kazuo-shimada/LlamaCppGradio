import src.p3g

def test_build_messages_image_and_text(tmp_path):

    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    msgs = src.p3g._build_messages("hello", str(img), put_image_first=True)
    assert isinstance(msgs, list) and msgs and isinstance(msgs[0], dict)
    content = msgs[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "hello"

def test_build_messages_text_only():
    msgs = src.p3g._build_messages("hi", None, put_image_first=True)
    assert msgs == [{"role": "user", "content": "hi"}]
