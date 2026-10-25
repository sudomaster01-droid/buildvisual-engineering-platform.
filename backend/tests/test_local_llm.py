from __future__ import annotations


def test_local_vision_adapter_uses_configured_endpoint_without_hosted_key(monkeypatch):
    from app.config import get_settings
    from app.services.interpret import llm

    settings = get_settings()
    monkeypatch.setattr(settings, "llm_provider", "local_openai")
    monkeypatch.setattr(settings, "llm_model", "Qwen/Qwen3-VL-2B-Instruct")
    monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:8001/v1")
    monkeypatch.setattr(settings, "llm_api_key", "")
    assert llm.llm_available()

    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"rooms": []}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, *, headers, json):
            captured.update(url=url, headers=headers, payload=json)
            return Response()

    monkeypatch.setattr(llm.httpx, "Client", Client)
    client = llm.build_llm_client()
    result = client.generate_json(
        system="Return JSON.",
        user="Inspect this drawing.",
        images=[(b"image-bytes", "image/png")],
    )

    assert result == {"rooms": []}
    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert captured["payload"]["messages"][1]["content"][1]["image_url"]["url"].endswith(
        "aW1hZ2UtYnl0ZXM="
    )
    assert client.last_usage.provider == "local_openai"
