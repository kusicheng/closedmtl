"""Offline contract checks; run this file directly with the project Python."""

import contextlib
import io
from pathlib import Path
import sys
import traceback
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_networking_components import translation_API as api
import main as cli


SECRET="test-private-key-do-not-print"


def reply(content="Translated", finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason,
                         "message": {"content": content}}]}


def invoke(payload=None, status=200, **kwargs):
    response=Mock(status_code=status)
    response.json.return_value=reply() if payload is None else payload
    with patch.object(api.requests, "post", return_value=response) as post:
        result=api.translation_request("Source", key=SECRET, **kwargs)
    return result, post


def raises(exception, callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except exception as exc:
        return exc
    raise AssertionError(f"Expected {exception.__name__}")


def test_unicode_and_request_contract():
    source="  你好！\n世界 🌍  "
    translated="  Bonjour !\nLe monde 🌍  "
    response=Mock(status_code=200)
    response.json.return_value=reply(translated)
    with patch.object(api.requests, "post", return_value=response) as post:
        actual=api.translation_request(source, target_language="French", key=SECRET,
                                         model="test/model", timeout=12.5)
    assert actual==translated
    post.assert_called_once()
    assert post.call_args.args==(api.API_URL,)
    request=post.call_args.kwargs
    assert request["headers"]["Authorization"]==f"Bearer {SECRET}"
    assert request["timeout"]==12.5 and request["allow_redirects"] is False
    assert request["json"]["stream"] is False
    assert request["json"]["messages"][1]=={"role": "user", "content": source}
    system=request["json"]["messages"][0]["content"]
    assert "Detect the source language automatically" in system
    assert '"French"' in system


def test_key_and_model_precedence():
    response=Mock(status_code=200)
    response.json.return_value=reply()
    with patch.dict(api.os.environ, {"OPENROUTER_API_KEY": "environment-key",
                                    "OPENROUTER_MODEL": "environment/model"}, clear=True):
        with patch.object(api.requests, "post", return_value=response) as post:
            api.translation_request("text")
            assert post.call_args.kwargs["headers"]["Authorization"]=="Bearer environment-key"
            assert post.call_args.kwargs["json"]["model"]=="environment/model"
            api.translation_request("text", key=SECRET, model="explicit/model")
            assert post.call_args.kwargs["headers"]["Authorization"]==f"Bearer {SECRET}"
            assert post.call_args.kwargs["json"]["model"]=="explicit/model"
            raises(ValueError, api.translation_request, "text", key="")
    with patch.dict(api.os.environ, {}, clear=True):
        raises(ValueError, api.translation_request, "text")
        _, post=invoke()
        assert post.call_args.kwargs["json"]["model"]==api.DEFAULT_MODEL


def test_input_validation_makes_no_requests():
    invalid=([{"target_language": value} for value in
                ("", " ", None, 1, "auto", "DETECT", " detect language ", "selected language")]
               + [{"timeout": value} for value in
                  (0, -1, True, None, "60", float("inf"), float("nan"))]
               + [{"source_language": ""}, {"type": "unknown"}, {"model": ""}])
    with patch.object(api.requests, "post") as post:
        for kwargs in invalid:
            raises(ValueError, api.translation_request, "text", key=SECRET, **kwargs)
        for value in (None, "", " ", 17):
            raises(ValueError, api.translation_request, value, key=SECRET)
        for key in ("", " ", "key\nvalue", "key value", 17):
            raises(ValueError, api.translation_request, "text", key=key)
        post.assert_not_called()


def test_explicit_source_and_auto_aliases():
    for source in ("auto", "DETECT", " detect language "):
        _, post=invoke(source_language=source)
        assert "Detect the source language automatically" in post.call_args.kwargs["json"]["messages"][0]["content"]
    _, post=invoke(source_language="Japanese", target_language="en")
    assert 'The source language is "Japanese"' in post.call_args.kwargs["json"]["messages"][0]["content"]


def test_transport_errors_do_not_leak_secrets_or_retry():
    for exception in (api.requests.Timeout, api.requests.ConnectionError,
                      api.requests.RequestException):
        output=io.StringIO()
        with patch.object(api.requests, "post", side_effect=exception(SECRET)) as post:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exc=raises(RuntimeError, api.translation_request, "text", key=SECRET)
                rendered="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        assert SECRET not in rendered + output.getvalue()
        post.assert_called_once()


def test_http_failures_are_actionable_and_private():
    for status, hint in ((401, "API key"), (402, "credit balance"),
                         (403, "denied"), (429, "Rate limit"), (500, "Try again"),
                         (503, "Try again"), (302, "Try again")):
        response=Mock(status_code=status, text=SECRET)
        response.json.return_value={"error": {"message": SECRET}}
        with patch.object(api.requests, "post", return_value=response) as post:
            exc=raises(RuntimeError, api.translation_request, "text", key=SECRET)
        assert str(status) in str(exc) and hint in str(exc)
        assert SECRET not in str(exc)
        response.json.assert_not_called()
        post.assert_called_once()


def test_malformed_empty_blocked_and_truncated_replies():
    payloads=[[], "bad", {}, {"choices": []}, {"choices": [None]},
                {"choices": [{}]}, {"choices": [{"finish_reason": "stop", "message": None}]},
                {"error": {"message": SECRET}}, reply(""), reply(" \n"), reply(None),
                reply([]), reply("partial", "length"), reply("blocked", "content_filter"),
                reply("text", None)]
    for payload in payloads:
        exc=raises(RuntimeError, invoke, payload=payload)
        assert "invalid, blocked, or incomplete" in str(exc)
        assert SECRET not in str(exc)
    response=Mock(status_code=200)
    response.json.side_effect=ValueError(SECRET)
    with patch.object(api.requests, "post", return_value=response):
        exc=raises(RuntimeError, api.translation_request, "text", key=SECRET)
    assert SECRET not in "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def test_cli_success():
    output=io.StringIO()
    with patch.object(sys, "argv", ["main.py", "--text", "Hello", "--target", "French",
                                    "--source", "English", "--model", "test/model"]):
        with patch.object(cli, "translation_request", return_value="Bonjour") as translate:
            with contextlib.redirect_stdout(output):
                status=cli.main()
    assert status==0 and output.getvalue()=="Bonjour\n"
    translate.assert_called_once_with("Hello", target_language="French",
                                       source_language="English", model="test/model")


def test_cli_stdin_input():
    source="First line\nSecond line\n"
    output=io.StringIO()
    with patch.object(sys, "argv", ["main.py", "--target", "en"]):
        with patch.object(sys, "stdin", io.StringIO(source)):
            with patch.object(cli, "translation_request", return_value=source) as translate:
                with contextlib.redirect_stdout(output):
                    status=cli.main()
    assert status==0 and output.getvalue()==source+"\n"
    translate.assert_called_once_with(source, target_language="en",
                                       source_language="detect language", model=None)


def test_cli_missing_key_failure():
    output=io.StringIO()
    errors=io.StringIO()
    with patch.dict(api.os.environ, {}, clear=True):
        with patch.object(sys, "argv", ["main.py", "--text", "Hello", "--target", "en"]):
            with patch.object(api.requests, "post") as post:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    status=cli.main()
    assert status==1 and output.getvalue()==""
    assert "API key must be a nonempty string" in errors.getvalue()
    assert "Traceback" not in errors.getvalue()
    post.assert_not_called()


if __name__=="__main__":
    tests=[unittest.FunctionTestCase(function) for name, function in sorted(globals().items())
             if name.startswith("test_")]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    sys.exit(not result.wasSuccessful())
