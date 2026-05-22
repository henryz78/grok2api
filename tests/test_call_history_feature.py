import json
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
            def encode(self, text):
                return list((text or "").encode("utf-8"))

        tiktoken.Encoding = _Encoding
        tiktoken.get_encoding = lambda _name: _Encoding()
        tiktoken.encoding_for_model = lambda _name: _Encoding()
        sys.modules["tiktoken"] = tiktoken


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


class CallHistorySurfaceTests(unittest.TestCase):
    def test_static_surface_contains_calls_page_and_sensitive_toggle(self):
        header = (REPO_ROOT / "app" / "statics" / "admin" / "header.html").read_text(encoding="utf-8")
        config = (REPO_ROOT / "app" / "statics" / "admin" / "config.html").read_text(encoding="utf-8")
        calls = (REPO_ROOT / "app" / "statics" / "admin" / "calls.html").read_text(encoding="utf-8")

        self.assertIn("/admin/calls", header)
        self.assertIn("call_history_show_sensitive", config)
        self.assertIn("调用历史", calls)
        self.assertIn("openCallModal", calls)

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
