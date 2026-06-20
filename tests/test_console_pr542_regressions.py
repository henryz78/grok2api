import asyncio
import importlib
import importlib.util
import json
import sys
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


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def _ensure_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_anthropic_messages_module():
    _install_common_stubs()
    _purge_modules(("app.products.anthropic",))

    _ensure_package("app", REPO_ROOT / "app")
    _ensure_package("app.products", REPO_ROOT / "app" / "products")
    _ensure_package("app.products.anthropic", REPO_ROOT / "app" / "products" / "anthropic")

    stub_chat = types.ModuleType("app.products.openai.chat")
    stub_chat._stream_chat = None
    stub_chat._extract_message = lambda messages: ("", [])
    stub_chat._resolve_image = None
    stub_chat._quota_sync = None
    stub_chat._fail_sync = None
    stub_chat._parse_retry_codes = lambda *_args, **_kwargs: frozenset()
    stub_chat._feedback_kind = lambda exc: exc
    stub_chat._log_task_exception = lambda *_args, **_kwargs: None
    stub_chat._configured_retry_codes = lambda *_args, **_kwargs: frozenset()
    stub_chat._should_retry_upstream = lambda *_args, **_kwargs: False
    stub_chat._console_completions = None
    sys.modules["app.products.openai.chat"] = stub_chat

    stub_sieve = types.ModuleType("app.products.openai._tool_sieve")
    stub_sieve.ToolSieve = object
    sys.modules["app.products.openai._tool_sieve"] = stub_sieve

    spec = importlib.util.spec_from_file_location(
        "app.products.anthropic.messages",
        REPO_ROOT / "app" / "products" / "anthropic" / "messages.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConsoleProtocolRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        cls.xai_console = importlib.import_module("app.dataplane.reverse.protocol.xai_console")
        cls.ValidationError = importlib.import_module("app.platform.errors").ValidationError
        cls.UpstreamError = importlib.import_module("app.platform.errors").UpstreamError

    def test_build_console_input_rejects_unsupported_blocks(self):
        for block in (
            {"type": "file", "file": {"data": "data:application/pdf;base64,AAAA"}},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ):
            with self.subTest(block_type=block["type"]):
                with self.assertRaises(self.ValidationError):
                    self.xai_console.build_console_input([{"role": "user", "content": [block]}])

    def test_extract_console_stream_error_detects_failed_events(self):
        exc = self.xai_console.extract_console_sse_error(
            "response.failed",
            json.dumps({"type": "response.failed", "error": {"message": "credits exhausted"}}),
        )
        self.assertIsNotNone(exc)
        self.assertIsInstance(exc, self.UpstreamError)
        self.assertEqual(exc.status, 502)
        self.assertIn("credits exhausted", exc.message)

    def test_multi_agent_filters_client_function_tools(self):
        tools = self.xai_console.convert_openai_tools_to_console(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                },
                {"type": "web_search"},
            ]
        )

        filtered = self.xai_console.filter_console_tools_for_model(
            "grok-4.20-multi-agent-0309",
            tools,
        )

        self.assertEqual(filtered, [{"type": "web_search"}])

    def test_multi_agent_can_keep_client_function_tools_when_enabled(self):
        tools = self.xai_console.convert_openai_tools_to_console(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                },
                {"type": "web_search"},
            ]
        )

        filtered = self.xai_console.filter_console_tools_for_model(
            "grok-4.20-multi-agent-0309",
            tools,
            allow_multi_agent_client_tools=True,
        )

        self.assertEqual([tool["type"] for tool in filtered], ["function", "web_search"])

    def test_non_multi_agent_keeps_client_function_tools(self):
        tools = self.xai_console.convert_openai_tools_to_console(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                },
                {"type": "web_search"},
            ]
        )

        filtered = self.xai_console.filter_console_tools_for_model("grok-4.3", tools)

        self.assertEqual([tool["type"] for tool in filtered], ["function", "web_search"])

    def test_normalize_console_usage_does_not_double_count_reasoning(self):
        usage = self.xai_console.normalize_console_usage(
            {
                "prompt_tokens": 40,
                "completion_tokens": 120,
                "reasoning_tokens": 10,
            },
            prompt_tokens_fallback=5,
            completion_tokens_fallback=6,
            reasoning_tokens_fallback=7,
        )
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 40,
                "completion_tokens": 120,
                "reasoning_tokens": 10,
            },
        )

    def test_format_console_reasoning_synthesizes_tool_trace_when_summary_missing(self):
        response = {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "type": "search",
                        "query": "Donald Trump recent activities May 2026",
                        "sources": [{"url": "https://example.com", "title": "Example"}],
                    },
                },
                {
                    "type": "x_search_call",
                    "action": {
                        "type": "search",
                        "query": "Trump OR 特朗普 since:2026-05-01",
                    },
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_account",
                    "arguments": "{\"query\":\"account status\"}",
                },
            ]
        }

        self.assertEqual(
            self.xai_console.format_console_reasoning(response),
            "Thinking about your request\n"
            "🔍 web_search: Donald Trump recent activities May 2026\n"
            "🔍 x_search: Trump OR 特朗普 since:2026-05-01\n"
            "🔧 lookup_account: account status\n",
        )

    def test_console_stream_adapter_emits_synthetic_search_reasoning(self):
        adapter = self.xai_console.ConsoleStreamAdapter()
        adapter.feed_event("response.output_item.done")

        event = adapter.feed_data(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "web_search_call",
                        "action": {
                            "type": "search",
                            "query": "特朗普 最近行程 2026",
                        },
                    },
                }
            )
        )

        self.assertEqual(event["kind"], "thinking")
        self.assertEqual(
            event["content"],
            "Thinking about your request\n🔍 web_search: 特朗普 最近行程 2026\n",
        )

    def test_internal_function_tool_names_are_not_client_tools(self):
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "open_page"}},
            {"type": "function", "function": {"name": "search"}},
            {"type": "web_search", "filters": {"allowed_domains": ["x.ai"]}},
        ]

        self.assertEqual(self.xai_console.client_function_tool_names(tools), {"search"})

        converted = self.xai_console.convert_openai_tools_to_console(tools)
        converted_function_names = {
            tool.get("name")
            for tool in converted
            if isinstance(tool, dict) and tool.get("type") == "function"
        }

        self.assertEqual(converted_function_names, {"search"})
        self.assertIn({"type": "web_search", "filters": {"allowed_domains": ["x.ai"]}}, converted)
        self.assertEqual(
            self.xai_console.convert_openai_tool_choice(
                {"type": "function", "function": {"name": "web_search"}}
            ),
            "auto",
        )

    def test_console_stream_adapter_filters_internal_function_events(self):
        adapter = self.xai_console.ConsoleStreamAdapter(function_tool_names={"lookup_order"})
        adapter.feed_event("response.output_item.added")

        started = adapter.feed_data(
            json.dumps(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "id": "fc_builtin",
                        "type": "function_call",
                        "call_id": "call_builtin",
                        "name": "web_search",
                    },
                }
            )
        )
        adapter.feed_event("response.function_call_arguments.done")
        finished = adapter.feed_data(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_builtin",
                    "arguments": "{\"query\":\"xAI latest\"}",
                }
            )
        )

        self.assertEqual(started["kind"], "skip")
        self.assertEqual(finished["kind"], "thinking")
        self.assertIn("web_search: xAI latest", finished["content"])
        self.assertEqual(adapter.tool_calls, [])

    def test_console_stream_adapter_allows_declared_client_function_events(self):
        adapter = self.xai_console.ConsoleStreamAdapter(function_tool_names={"lookup_order"})
        adapter.feed_event("response.output_item.added")

        started = adapter.feed_data(
            json.dumps(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "id": "fc_client",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_order",
                    },
                }
            )
        )
        adapter.feed_event("response.function_call_arguments.done")
        finished = adapter.feed_data(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_client",
                    "arguments": "{\"order_id\":\"A123\"}",
                }
            )
        )

        self.assertEqual(started["kind"], "tool_call_start")
        self.assertEqual(started["name"], "lookup_order")
        self.assertEqual(finished["kind"], "tool_call_done")
        self.assertEqual(adapter.tool_calls[0]["function"]["arguments"], "{\"order_id\":\"A123\"}")

    def test_extract_console_tool_calls_filters_to_client_function_names(self):
        response = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_builtin",
                    "name": "web_search_with_snippets",
                    "arguments": "{\"query\":\"latest\"}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_order",
                    "arguments": "{\"order_id\":\"A123\"}",
                },
            ]
        }

        self.assertEqual(
            self.xai_console.extract_console_tool_calls(
                response,
                function_tool_names={"lookup_order"},
            ),
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_order",
                        "arguments": "{\"order_id\":\"A123\"}",
                    },
                }
            ],
        )

    def test_filter_console_response_output_removes_internal_function_calls(self):
        response = {
            "id": "resp_1",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_builtin",
                    "name": "open_page",
                    "arguments": "{\"url\":\"https://x.ai\"}",
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_order",
                    "arguments": "{\"order_id\":\"A123\"}",
                },
            ],
        }

        filtered = self.xai_console.filter_console_response_tool_calls(
            response,
            function_tool_names={"lookup_order"},
        )

        self.assertEqual(
            [item.get("name") for item in filtered["output"] if item.get("type") == "function_call"],
            ["lookup_order"],
        )


class AuthNsfwSequenceRegressionTests(unittest.TestCase):
    def _load_auth_module(self):
        _install_common_stubs()
        prefixes = (
            "app.dataplane.reverse.protocol.xai_auth",
            "app.dataplane.proxy.adapters",
            "app.dataplane.reverse.transport",
        )
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        }

        def _restore_modules():
            for name in list(sys.modules):
                if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                    del sys.modules[name]
            sys.modules.update(saved_modules)

        self.addCleanup(_restore_modules)
        _purge_modules(prefixes)

        adapters_pkg = types.ModuleType("app.dataplane.proxy.adapters")
        adapters_pkg.__path__ = []
        sys.modules["app.dataplane.proxy.adapters"] = adapters_pkg

        session_mod = types.ModuleType("app.dataplane.proxy.adapters.session")

        class _Session:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        session_mod.ResettableSession = _Session
        session_mod.build_session_kwargs = lambda **kwargs: dict(kwargs)
        sys.modules["app.dataplane.proxy.adapters.session"] = session_mod

        transport_pkg = types.ModuleType("app.dataplane.reverse.transport")
        transport_pkg.__path__ = []
        sys.modules["app.dataplane.reverse.transport"] = transport_pkg

        grpc_web_mod = types.ModuleType("app.dataplane.reverse.transport.grpc_web")
        grpc_web_mod.post_grpc_web = None
        sys.modules["app.dataplane.reverse.transport.grpc_web"] = grpc_web_mod

        http_mod = types.ModuleType("app.dataplane.reverse.transport.http")
        http_mod.post_json = None
        sys.modules["app.dataplane.reverse.transport.http"] = http_mod

        return importlib.import_module("app.dataplane.reverse.protocol.xai_auth")

    def test_nsfw_sequence_skips_locked_birth_date_429(self):
        module = self._load_auth_module()
        calls = []

        class _Proxy:
            async def acquire(self, **kwargs):
                calls.append(("acquire", kwargs.get("clearance_origin")))
                return object()

            async def feedback(self, lease, result):
                calls.append(("feedback", result.kind))

        async def _accept_tos(token):
            calls.append(("accept_tos", token))

        async def _set_birth_date(token, **kwargs):
            calls.append(("set_birth_date", token))
            raise module.UpstreamError(
                "birth date locked",
                status=429,
                body="{\"error\":\"birth-date-change-limit-reached\"}",
            )

        async def _grpc_call(*args, **kwargs):
            calls.append(("grpc", kwargs.get("label")))

        async def _get_proxy_runtime():
            return _Proxy()

        module.accept_tos = _accept_tos
        module.set_birth_date = _set_birth_date
        module._grpc_call = _grpc_call
        module.get_proxy_runtime = _get_proxy_runtime

        asyncio.run(module.nsfw_sequence("token-test"))

        self.assertIn(("grpc", "enable_nsfw"), calls)

    def test_nsfw_sequence_keeps_other_birth_date_429_failures(self):
        module = self._load_auth_module()
        calls = []

        class _Proxy:
            async def acquire(self, **kwargs):
                return object()

            async def feedback(self, lease, result):
                calls.append(("feedback", result.kind))

        async def _accept_tos(token):
            calls.append(("accept_tos", token))

        async def _set_birth_date(token, **kwargs):
            calls.append(("set_birth_date", token))
            raise module.UpstreamError("real rate limit", status=429, body="{\"error\":\"rate_limit\"}")

        async def _grpc_call(*args, **kwargs):
            calls.append(("grpc", kwargs.get("label")))

        async def _get_proxy_runtime():
            return _Proxy()

        module.accept_tos = _accept_tos
        module.set_birth_date = _set_birth_date
        module._grpc_call = _grpc_call
        module.get_proxy_runtime = _get_proxy_runtime

        with self.assertRaises(module.UpstreamError):
            asyncio.run(module.nsfw_sequence("token-test"))

        self.assertNotIn(("grpc", "enable_nsfw"), calls)


class ConsoleReasoningDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        cls.registry = importlib.import_module("app.control.model.registry")

    def test_hybrid_console_models_default_to_high_reasoning_effort(self):
        self.assertEqual(
            self.registry.resolve("c/grok-4.3").default_reasoning_effort,
            "high",
        )
        self.assertEqual(
            self.registry.resolve("c/grok-4.20-reasoning").default_reasoning_effort,
            "",
        )
        self.assertEqual(
            self.registry.resolve("c/grok-4.20-non-reasoning").default_reasoning_effort,
            "",
        )
        self.assertEqual(
            self.registry.resolve("c/grok-4.20-multi-agent").default_reasoning_effort,
            "",
        )

    def test_console_effort_variant_models_resolve_to_upstream_console_models(self):
        cases = {
            "c/grok-4.3-low": ("grok-4.3", "low"),
            "c/grok-4.3-medium": ("grok-4.3", "medium"),
            "c/grok-4.3-high": ("grok-4.3", "high"),
            "c/grok-4.20-multi-agent-low": ("grok-4.20-multi-agent-0309", "low"),
            "c/grok-4.20-multi-agent-medium": ("grok-4.20-multi-agent-0309", "medium"),
            "c/grok-4.20-multi-agent-high": ("grok-4.20-multi-agent-0309", "high"),
            "c/grok-4.20-multi-agent-xhigh": ("grok-4.20-multi-agent-0309", "xhigh"),
        }

        for model_name, (console_model, effort) in cases.items():
            with self.subTest(model_name=model_name):
                spec = self.registry.resolve(model_name)
                self.assertEqual(spec.console_model, console_model)
                self.assertEqual(spec.default_reasoning_effort, effort)

    def test_grok_build_console_model_uses_clean_c_prefixed_name(self):
        spec = self.registry.resolve("c/grok-build")

        self.assertTrue(spec.is_console())
        self.assertEqual(spec.console_model, "grok-build-0.1")
        self.assertEqual(spec.public_name, "c/Grok Build")
        self.assertEqual(
            self.registry.resolve("grok-build-console").model_name,
            "c/grok-build",
        )

    def test_console_aliases_resolve_to_canonical_c_prefixed_models(self):
        self.assertEqual(
            self.registry.resolve("grok-4.3").model_name,
            "c/grok-4.3",
        )
        self.assertEqual(
            self.registry.resolve("grok-4.20-reasoning").model_name,
            "c/grok-4.20-reasoning",
        )
        self.assertEqual(
            self.registry.resolve("grok-4.20-non-reasoning").model_name,
            "c/grok-4.20-non-reasoning",
        )
        self.assertEqual(
            self.registry.resolve("grok-4.20-multi-agent").model_name,
            "c/grok-4.20-multi-agent",
        )
        with self.assertRaises(ValueError):
            self.registry.resolve("grok-4.20")


class ConsoleQuotaIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        cls.registry = importlib.import_module("app.control.model.registry")
        cls.model_enums = importlib.import_module("app.control.model.enums")
        cls.quota_defaults = importlib.import_module("app.control.account.quota_defaults")
        cls.account_models = importlib.import_module("app.control.account.models")
        cls.account_sync = importlib.import_module("app.dataplane.account.sync")

    def test_c_prefixed_console_models_use_console_mode(self):
        console_mode = self.model_enums.ModeId.CONSOLE
        for model_name in (
            "c/grok-4.3",
            "c/grok-4.20-reasoning",
            "c/grok-4.20-non-reasoning",
            "c/grok-4.20-multi-agent",
            "c/grok-build",
        ):
            with self.subTest(model_name=model_name):
                spec = self.registry.resolve(model_name)
                self.assertTrue(spec.is_console())
                self.assertEqual(spec.mode_id, console_mode)

    def test_basic_pool_has_independent_console_quota_window(self):
        console_mode = int(self.model_enums.ModeId.CONSOLE)
        quota_set = self.quota_defaults.default_quota_set("basic")

        self.assertTrue(self.quota_defaults.supports_mode("basic", console_mode))
        self.assertIn(console_mode, self.quota_defaults.supported_mode_ids("basic"))
        self.assertIsNotNone(quota_set.console)
        self.assertEqual(quota_set.fast.total, 30)
        self.assertEqual(quota_set.fast.window_seconds, 86_400)
        self.assertEqual(quota_set.console.total, 20)
        self.assertEqual(quota_set.console.window_seconds, 3_600)

    def test_runtime_sync_indexes_console_quota_separately_from_fast(self):
        console_mode = int(self.model_enums.ModeId.CONSOLE)
        quota_set = self.quota_defaults.default_quota_set("basic")
        quota_set.fast.remaining = 0
        quota_set.console.remaining = 7
        record = self.account_models.AccountRecord(
            token="token-for-console-quota-test",
            pool="basic",
            quota=quota_set.to_dict(),
        )

        args = self.account_sync._record_to_slot_args(record)

        self.assertEqual(args["quota_fast"], 0)
        self.assertEqual(args["quota_console"], 7)
        self.assertEqual(args["window_fast"], 86_400)
        self.assertEqual(args["window_console"], 3_600)
        self.assertEqual(console_mode, 5)

    def test_console_mode_quota_refresh_does_not_hit_grok_rate_limits(self):
        console_mode = int(self.model_enums.ModeId.CONSOLE)
        xai_usage = importlib.import_module("app.dataplane.reverse.protocol.xai_usage")
        calls = []
        original_do_fetch = xai_usage._do_fetch

        async def _fake_do_fetch(_token, mode_name):
            calls.append(mode_name)
            return {
                "remainingQueries": 99,
                "totalQueries": 99,
                "windowSizeSeconds": 7200,
            }

        async def _run():
            xai_usage._do_fetch = _fake_do_fetch
            try:
                return await xai_usage.fetch_mode_quota("token", console_mode)
            finally:
                xai_usage._do_fetch = original_do_fetch

        self.assertIsNone(asyncio.run(_run()))
        self.assertEqual(calls, [])


class AnthropicConsoleBridgeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.messages = _load_anthropic_messages_module()

    def test_chat_completion_to_anthropic_preserves_annotations_and_search_sources(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://example.com",
                                    "title": "Example",
                                    "start_index": 0,
                                    "end_index": 6,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 22},
            "search_sources": [{"url": "https://example.com", "title": "Example"}],
        }

        result = self.messages._chat_completion_to_anthropic(response, "msg_test", "grok-4.3")

        self.assertEqual(
            result["content"][0]["annotations"],
            [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.com",
                        "title": "Example",
                        "start_index": 0,
                        "end_index": 6,
                    },
                }
            ],
        )
        self.assertEqual(
            result["search_sources"],
            [{"url": "https://example.com", "title": "Example"}],
        )

    def test_chat_stream_to_anthropic_sse_preserves_final_metadata(self):
        async def _chat_stream():
            first = {
                "choices": [{"delta": {"content": "Hello"}}],
            }
            final = {
                "choices": [
                    {
                        "delta": {
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://example.com",
                                        "title": "Example",
                                        "start_index": 0,
                                        "end_index": 5,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 12},
                "search_sources": [{"url": "https://example.com", "title": "Example"}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        async def _collect():
            chunks = []
            async for chunk in self.messages._chat_stream_to_anthropic_sse(
                _chat_stream(), "msg_test", "grok-4.3"
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        message_delta = next(chunk for chunk in chunks if "event: message_delta" in chunk)
        payload = json.loads(message_delta.split("data: ", 1)[1].strip())

        self.assertEqual(
            payload["delta"]["annotations"],
            [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.com",
                        "title": "Example",
                        "start_index": 0,
                        "end_index": 5,
                    },
                }
            ],
        )
        self.assertEqual(
            payload["delta"]["search_sources"],
            [{"url": "https://example.com", "title": "Example"}],
        )


if __name__ == "__main__":
    unittest.main()
