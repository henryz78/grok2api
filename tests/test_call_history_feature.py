import json
import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_common_stubs() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if "orjson" not in sys.modules:
        orjson = types.ModuleType("orjson")
        orjson.dumps = lambda obj: json.dumps(obj, separators=(",", ":")).encode("utf-8")
        orjson.loads = lambda data: json.loads(
            data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        )
        orjson.JSONDecodeError = json.JSONDecodeError
        sys.modules["orjson"] = orjson

    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _Logger:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        loguru.logger = _Logger()
        sys.modules["loguru"] = loguru

    if "tiktoken" not in sys.modules:
        tiktoken = types.ModuleType("tiktoken")

        class _Encoding:
            def encode(self, text, *args, **kwargs):
                return list((text or "").encode("utf-8"))

        tiktoken.Encoding = _Encoding
        tiktoken.get_encoding = lambda _name: _Encoding()
        tiktoken.encoding_for_model = lambda _name: _Encoding()
        sys.modules["tiktoken"] = tiktoken

    if "fastapi" not in sys.modules:
        fastapi = types.ModuleType("fastapi")

        class _Request:
            pass

        fastapi.Request = _Request
        sys.modules["fastapi"] = fastapi


def _load_openai_call_history_module():
    _install_common_stubs()
    module_name = "_grok2api_call_history_under_test"
    module_path = REPO_ROOT / "app" / "products" / "openai" / "call_history.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CallHistoryStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        from app.platform.storage.call_history import CallHistoryStore

        cls.CallHistoryStore = CallHistoryStore

    def test_store_can_append_list_and_get_sensitive_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "call_history.db"
            store = self.CallHistoryStore(db_path=db_path)
            entry = store.build_entry(
                created_at_ms=1000,
                finished_at_ms=2345,
                route="/v1/chat/completions",
                model="c/grok-4.3",
                stream=True,
                success=True,
                prompt_tokens=14,
                completion_tokens=22,
                reasoning_tokens=3,
                total_tokens=39,
                client_ip="203.0.113.9",
                request_body={"model": "c/grok-4.3", "messages": [{"role": "user", "content": "hi"}]},
                response_body={"choices": [{"message": {"content": "hello"}}]},
                meta={"route_kind": "chat"},
            )

            self.assertTrue(entry["id"].startswith("call_"))
            self.assertEqual(entry["duration_ms"], 1345)
            self.assertIn("messages", entry["request_body"])

            import asyncio

            asyncio.run(store.initialize())
            asyncio.run(store.record(entry))
            page = asyncio.run(store.list_calls(page=1, page_size=10))
            self.assertEqual(page.total, 1)
            self.assertEqual(page.items[0].model, "c/grok-4.3")
            self.assertEqual(page.items[0].client_ip, "203.0.113.9")
            self.assertEqual(page.items[0].token_summary, "1.35s · 流 · 29 t/s · 14 / 22")

            got = asyncio.run(store.get(entry["id"]))
            self.assertIsNotNone(got)
            self.assertEqual(got.request_preview, '{"model":"c/grok-4.3","messages":[{"role":"user","content":"hi"}]}')
            self.assertIn('"content":"hello"', got.response_body)

    def test_store_filters_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "call_history.db"
            store = self.CallHistoryStore(db_path=db_path)

            import asyncio

            asyncio.run(store.initialize())
            asyncio.run(
                store.record(
                    store.build_entry(
                        created_at_ms=10,
                        finished_at_ms=20,
                        route="/v1/chat/completions",
                        model="c/grok-4.3",
                        stream=False,
                        success=True,
                        request_body="hello",
                    )
                )
            )
            asyncio.run(
                store.record(
                    store.build_entry(
                        created_at_ms=30,
                        finished_at_ms=50,
                        route="/v1/responses",
                        model="c/grok-4.20-multi-agent",
                        stream=True,
                        success=False,
                        status_code=400,
                        error_message="betaaccess required",
                        request_body="search",
                    )
                )
            )
            failed = asyncio.run(store.list_calls(status="failed"))
            self.assertEqual(failed.total, 1)
            self.assertEqual(failed.items[0].route, "/v1/responses")
            streamed = asyncio.run(store.list_calls(stream=True))
            self.assertEqual(streamed.total, 1)
            searched = asyncio.run(store.list_calls(q="betaaccess"))
            self.assertEqual(searched.total, 1)

    def test_summary_redacts_bodies_by_default(self):
        from app.platform.storage.call_history import summarize_call_history

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "call_history.db"
            store = self.CallHistoryStore(db_path=db_path)
            entry = store.build_entry(
                created_at_ms=100,
                finished_at_ms=200,
                route="/v1/chat/completions",
                model="c/grok-4.3",
                stream=False,
                success=True,
                client_ip="203.0.113.9",
                request_body="secret prompt",
                response_body="secret response",
            )

            import asyncio

            asyncio.run(store.initialize())
            asyncio.run(store.record(entry))
            record = asyncio.run(store.get(entry["id"]))
            redacted = summarize_call_history(record, include_sensitive=False)
            exposed = summarize_call_history(record, include_sensitive=True)

            self.assertEqual(redacted["client_ip"], "")
            self.assertNotIn("request_body", redacted)
            self.assertNotIn("response_body", redacted)
            self.assertEqual(exposed["client_ip"], "203.0.113.9")
            self.assertEqual(exposed["request_body"], "secret prompt")

    def test_stream_history_estimates_usage_and_stores_plain_response_text(self):
        history = _load_openai_call_history_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "call_history.db"
            store = self.CallHistoryStore(db_path=db_path)
            original_store = history.call_history_store
            history.call_history_store = store

            async def _stream():
                yield 'data: {"choices":[{"delta":{"reasoning_content":"Thinking about your request\\n"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}\n\n'
                yield "data: [DONE]\n\n"

            async def _collect():
                chunks = []
                async for chunk in history.recording_sse(
                    _stream(),
                    started_at_ms=1000,
                    route="/v1/chat/completions",
                    model="grok-4.20-fast",
                    request_body={
                        "model": "grok-4.20-fast",
                        "messages": [{"role": "user", "content": "你好"}],
                        "tools": [{"type": "function", "function": {"name": "tool"}}],
                    },
                    client_ip="203.0.113.9",
                ):
                    chunks.append(chunk)
                return chunks

            import asyncio

            try:
                asyncio.run(store.initialize())
                chunks = asyncio.run(_collect())
                page = asyncio.run(store.list_calls(page=1, page_size=10))
            finally:
                history.call_history_store = original_store

            self.assertEqual(len(chunks), 5)
            self.assertEqual(page.total, 1)
            record = page.items[0]
            self.assertIsNotNone(record.prompt_tokens)
            self.assertIsNotNone(record.completion_tokens)
            self.assertIsNotNone(record.total_tokens)
            self.assertGreater(record.total_tokens or 0, 0)
            self.assertEqual(record.response_body, "Thinking about your request\n你好")
            self.assertNotIn("data:", record.response_body)


class CallHistorySurfaceTests(unittest.TestCase):
    def test_static_surface_contains_calls_page_and_sensitive_toggle(self):
        header = (REPO_ROOT / "app" / "statics" / "admin" / "header.html").read_text(encoding="utf-8")
        config = (REPO_ROOT / "app" / "statics" / "admin" / "config.html").read_text(encoding="utf-8")
        calls = (REPO_ROOT / "app" / "statics" / "admin" / "calls.html").read_text(encoding="utf-8")
        admin_header_js = (REPO_ROOT / "app" / "statics" / "js" / "admin-header.js").read_text(encoding="utf-8")

        self.assertIn("/admin/calls", header)
        self.assertIn("call_history_show_sensitive", config)
        self.assertIn("调用历史", calls)
        self.assertIn("openCallModal", calls)
        fallback_nav = admin_header_js.split("<nav class=\"admin-nav\">", 1)[1]
        self.assertLess(fallback_nav.index("/admin/calls"), fallback_nav.index("/admin/logs"))
        self.assertIn("request_text", calls)
        self.assertIn("response_text", calls)

    def test_admin_detail_text_extracts_plain_request_and_response(self):
        history = _load_openai_call_history_module()

        request_body = json.dumps(
            {
                "model": "grok-4.20-fast",
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "你好"},
                ],
                "tools": [{"type": "function", "function": {"name": "eval_javascript"}}],
            },
            ensure_ascii=False,
        )
        response_body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"Thinking about your request\\n"}}]}',
                'data: {"choices":[{"delta":{"content":"你"}}]}',
                'data: {"choices":[{"delta":{"content":"好"}}]}',
                "data: [DONE]",
            ]
        )

        request_text = history.display_text_from_request(request_body)
        response_text = history.display_text_from_response(response_body)

        self.assertIn("system: policy", request_text)
        self.assertIn("user: 你好", request_text)
        self.assertNotIn("eval_javascript", request_text)
        self.assertEqual(response_text, "Thinking about your request\n你好")

    def test_usage_merge_fills_missing_stream_totals(self):
        history = _load_openai_call_history_module()

        usage = history.merge_usage_for_history(
            {"prompt_tokens": None, "completion_tokens": 12, "reasoning_tokens": None, "total_tokens": None},
            {"messages": [{"role": "user", "content": "你好"}]},
            "你好",
        )

        self.assertIsNotNone(usage["prompt_tokens"])
        self.assertEqual(usage["completion_tokens"], 12)
        self.assertEqual(usage["total_tokens"], (usage["prompt_tokens"] or 0) + 12)

    def test_storage_paths_export(self):
        from app.platform.storage import call_history_db_path

        self.assertTrue(str(call_history_db_path()).endswith(os.path.join("data", "call_history.db")))

    def test_openai_router_contains_call_history_instrumentation(self):
        router_src = (REPO_ROOT / "app" / "products" / "openai" / "router.py").read_text(encoding="utf-8")

        self.assertIn("recording_sse", router_src)
        self.assertIn("record_call_history", router_src)
        self.assertIn("chat_completions_endpoint(req: ChatCompletionRequest, request: Request)", router_src)
        self.assertIn("responses_endpoint(req: ResponsesCreateRequest, request: Request)", router_src)


if __name__ == "__main__":
    unittest.main()
