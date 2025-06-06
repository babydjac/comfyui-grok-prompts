import json
from unittest.mock import Mock, patch

from ponyxl import PonyXL


def _make_response(data):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = data
    return mock_resp


def test_generate_prompts_success():
    result_payload = {
        "ponyxl_prompt": "pony",
        "wan_prompt": "wan",
        "negative_prompt": "neg",
        "explanation": "ok",
    }
    api_response = {"choices": [{"message": {"content": json.dumps(result_payload)}}]}
    with patch("ponyxl.requests.post", return_value=_make_response(api_response)):
        out = PonyXL().generate_prompts("a cat", "key", "jump")
    assert out["result"] == ("pony", "wan", "neg")
    assert out["ui"]["text"] == ["ok"]


def test_generate_prompts_missing_api_key():
    out = PonyXL().generate_prompts("a cat", "", "jump")
    assert out["ui"]["text"] == ["No API key provided."]
    assert out["result"] == (
        "a cat",
        "",
        "blurry, low_quality, bad_anatomy, oversaturated",
    )


def test_generate_prompts_api_failure():
    with patch("ponyxl.requests.post", side_effect=Exception("fail")):
        out = PonyXL().generate_prompts("a cat", "key", "jump")
    assert out["result"] == (
        "a cat",
        "",
        "blurry, low_quality, bad_anatomy, oversaturated",
    )
    assert out["ui"]["text"][0].startswith("Error calling Grok API:")
