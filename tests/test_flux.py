import json
from unittest.mock import Mock, patch

from flux import Flux


def _make_response(data):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = data
    return mock_resp


def test_generate_prompts_success():
    result_payload = {
        "flux_prompt": "flux",
        "wan_prompt": "wan",
        "negative_prompt": "neg",
        "explanation": "ok",
    }
    api_response = {"choices": [{"message": {"content": json.dumps(result_payload)}}]}
    with patch("flux.requests.post", return_value=_make_response(api_response)):
        out = Flux().generate_prompts("a dog", "key", "run")
    assert out["result"] == ("flux", "wan", "neg")
    assert out["ui"]["text"] == ["ok"]


def test_generate_prompts_missing_api_key():
    out = Flux().generate_prompts("a dog", "", "run")
    assert out["ui"]["text"] == ["No API key provided."]
    assert out["result"] == (
        "a dog",
        "",
        "blurry, low_detail, bad_anatomy",
    )


def test_generate_prompts_api_failure():
    with patch("flux.requests.post", side_effect=Exception("fail")):
        out = Flux().generate_prompts("a dog", "key", "run")
    assert out["result"] == (
        "a dog",
        "",
        "blurry, low_detail, bad_anatomy",
    )
    assert out["ui"]["text"][0].startswith("Error calling Grok API:")
