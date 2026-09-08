"""Offline checks for box mapping, context boundaries, and provider failures."""

import json
from pathlib import Path
import sys
import traceback
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_networking_components import translation_API as api


SECRET="offline-private-key"
REGIONS=[{"id": "a", "image_id": "page-1", "text": "こんにちは", "panel": 1},
         {"id": "b", "image_id": "page-2", "text": "ありがとう"}]


def invoke(rows=None, content=None, **kwargs):
    if content is None:
        content=json.dumps({"regions": rows if rows is not None else
                            [{"id": "a", "text": "Hello"}, {"id": "b", "text": "Thank you"}]})
    response=Mock(status_code=200)
    response.json.return_value={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                "usage": {"prompt_tokens": 400, "completion_tokens": 20},
                                "model": "minimax/minimax-m3:free"}
    with patch.object(api.requests, "post", return_value=response) as post:
        result=api.translate_regions(REGIONS, key=SECRET, **kwargs)
    return result, post


def raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception as exc:
        return exc
    raise AssertionError(f"Expected {exception.__name__}")


def test_request_context_and_json_contract():
    result, post=invoke(target_language="French", timeout=45)
    assert result["usage"]=={"prompt_tokens": 400, "completion_tokens": 20}
    assert result["model"]=="minimax/minimax-m3:free"
    request=post.call_args.kwargs
    assert request["timeout"]==45 and request["allow_redirects"] is False
    payload=request["json"]
    assert payload["response_format"]=={"type": "json_object"}
    assert payload["stream"] is False
    assert json.loads(payload["messages"][1]["content"])["regions"]==REGIONS
    prompt=payload["messages"][0]["content"]
    for text in ('"French"', "unrelated", "never instructions", "exactly once", "Do not censor"):
        assert text in prompt
    post.assert_called_once()


def test_response_order_is_restored_by_id():
    result, _=invoke(rows=[{"id": "b", "text": "Thanks"}, {"id": "a", "text": "Hi"}])
    assert result["regions"]==[{"id": "a", "text": "Hi"}, {"id": "b", "text": "Thanks"}]


def test_id_validation_rejects_duplicate_missing_extra_and_wrong_types():
    invalid=[[], [{"id": "a", "text": "Hi"}],
             [{"id": "a", "text": "Hi"}, {"id": "a", "text": "Thanks"}],
             [{"id": "a", "text": "Hi"}, {"id": "c", "text": "Thanks"}],
             [{"id": 1, "text": "Hi"}, {"id": "b", "text": "Thanks"}],
             [{"id": "a", "text": []}, {"id": "b", "text": "Thanks"}],
             [{"id": "a", "text": " "}, {"id": "b", "text": "Thanks"}],
             [{"id": "a", "text": "Hi", "note": "extra"}, {"id": "b", "text": "Thanks"}]]
    for rows in invalid:
        raises(RuntimeError, invoke, rows=rows)


def test_json_validation_rejects_wrappers_and_duplicate_keys():
    for content in ('not JSON', '[]', '{}', '{"regions":null}',
                    '```json\n{"regions":[]}\n```',
                    '{"regions":[],"regions":[]}',
                    '{"regions":[{"id":"a","id":"a","text":"Hi"},{"id":"b","text":"Thanks"}]}'):
        raises(RuntimeError, invoke, content=content)


def test_input_validation_before_provider_call():
    invalid=[None, {}, [None], [{"id": "", "text": "a"}],
             [{"id": 1, "text": "a"}], [{"id": "a", "text": None}],
             [{"id": "a", "text": "a", "image_id": 2}],
             [REGIONS[0], REGIONS[0]], [{"id": "a", "text": "a", "panel": float("nan")}]]
    with patch.object(api.requests, "post") as post:
        for regions in invalid:
            raises(ValueError, api.translate_regions, regions, key=SECRET)
        for kwargs in ({"timeout": 0}, {"timeout": True}, {"timeout": float("inf")},
                       {"target_language": "auto"}, {"model": ""}, {"key": "a b"}):
            raises(ValueError, api.translate_regions, REGIONS, **kwargs)
        post.assert_not_called()


def test_empty_input_skips_api():
    with patch.object(api.requests, "post") as post:
        assert api.translate_regions([], model="test/model")=={
            "regions": [], "usage": {}, "model": "test/model"}
        post.assert_not_called()


def test_empty_source_cannot_gain_hallucinated_dialogue():
    regions=[{"id": "a", "text": ""}, {"id": "b", "text": "Thanks"}]
    with patch.dict(globals(), {"REGIONS": regions}):
        raises(RuntimeError, invoke)
        result, _=invoke(rows=[{"id": "a", "text": ""}, {"id": "b", "text": "Thanks"}])
        assert result["regions"][0]["text"]==""


def test_transport_and_provider_errors_are_private_and_not_retried():
    for exception in (api.requests.Timeout, api.requests.ConnectionError):
        with patch.object(api.requests, "post", side_effect=exception(SECRET)) as post:
            exc=raises(RuntimeError, api.translate_regions, REGIONS, key=SECRET)
            assert SECRET not in "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            post.assert_called_once()
    response=Mock(status_code=429, text=SECRET)
    with patch.object(api.requests, "post", return_value=response) as post:
        exc=raises(RuntimeError, api.translate_regions, REGIONS, key=SECRET)
        assert "429" in str(exc) and SECRET not in str(exc)
        post.assert_called_once()


def test_environment_defaults_and_explicit_overrides():
    with patch.dict(api.os.environ, {"OPENROUTER_MODEL": "environment/model"}, clear=True):
        _, post=invoke()
        assert post.call_args.kwargs["json"]["model"]=="environment/model"
        _, post=invoke(model="explicit/model")
        assert post.call_args.kwargs["json"]["model"]=="explicit/model"
    with patch.dict(api.os.environ, {}, clear=True):
        _, post=invoke()
        assert post.call_args.kwargs["json"]["model"]=="minimax/minimax-m3:free"


if __name__=="__main__":
    tests=[unittest.FunctionTestCase(function) for name, function in sorted(globals().items())
           if name.startswith("test_")]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    sys.exit(not result.wasSuccessful())
