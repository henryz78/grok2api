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

    if "curl_cffi" not in sys.modules:
        curl_cffi = types.ModuleType("curl_cffi")
        curl_const = types.ModuleType("curl_cffi.const")

        class _CurlOpt:
            PROXY_SSL_VERIFYPEER = object()
            PROXY_SSL_VERIFYHOST = object()

        curl_const.CurlOpt = _CurlOpt
        sys.modules["curl_cffi"] = curl_cffi
        sys.modules["curl_cffi.const"] = curl_const

    fastapi = sys.modules.get("fastapi") or types.ModuleType("fastapi")

    class _APIRouter:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def put(self, *args, **kwargs):
            return lambda fn: fn

        def delete(self, *args, **kwargs):
            return lambda fn: fn

    if not hasattr(fastapi, "APIRouter"):
        fastapi.APIRouter = _APIRouter
    if not hasattr(fastapi, "Body"):
        fastapi.Body = lambda default=None, *args, **kwargs: default
    if not hasattr(fastapi, "Depends"):
        fastapi.Depends = lambda dependency=None, *args, **kwargs: dependency
    if not hasattr(fastapi, "Query"):
        fastapi.Query = lambda default=None, *args, **kwargs: default
    sys.modules["fastapi"] = fastapi

    responses = sys.modules.get("fastapi.responses") or types.ModuleType("fastapi.responses")

    class _Response:
        def __init__(self, content=b"", media_type=None, status_code=200):
            self.body = content if isinstance(content, bytes) else str(content).encode("utf-8")
            self.media_type = media_type
            self.status_code = status_code

    if not hasattr(responses, "Response"):
        responses.Response = _Response
    if not hasattr(responses, "StreamingResponse"):
        responses.StreamingResponse = _Response
    sys.modules["fastapi.responses"] = responses


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def _ensure_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_openai_chat_module():
    _install_common_stubs()
    _purge_modules(("app.products.openai",))

    _ensure_package("app", REPO_ROOT / "app")
    _ensure_package("app.products", REPO_ROOT / "app" / "products")
    _ensure_package("app.products.openai", REPO_ROOT / "app" / "products" / "openai")

    spec = importlib.util.spec_from_file_location(
        "app.products.openai.chat",
        REPO_ROOT / "app" / "products" / "openai" / "chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_admin_tokens_module():
    _install_common_stubs()
    _purge_modules(("app.products.web",))

    _ensure_package("app", REPO_ROOT / "app")
    _ensure_package("app.products", REPO_ROOT / "app" / "products")
    _ensure_package("app.products.web", REPO_ROOT / "app" / "products" / "web")

    admin_pkg = types.ModuleType("app.products.web.admin")
    admin_pkg.__path__ = [str(REPO_ROOT / "app" / "products" / "web" / "admin")]
    admin_pkg.get_refresh_svc = lambda *args, **kwargs: None
    admin_pkg.get_repo = lambda *args, **kwargs: None
    sys.modules["app.products.web.admin"] = admin_pkg

    spec = importlib.util.spec_from_file_location(
        "app.products.web.admin.tokens",
        REPO_ROOT / "app" / "products" / "web" / "admin" / "tokens.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OfficialModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        cls.registry = importlib.import_module("app.control.model.registry")
        cls.model_enums = importlib.import_module("app.control.model.enums")

    def test_grok_4_3_fast_is_official_web_fast_model_not_console(self):
        spec = self.registry.resolve("grok-4.3-fast")

        self.assertEqual(spec.mode_id, self.model_enums.ModeId.FAST)
        self.assertEqual(spec.pool_candidates(), (2, 1, 0))
        self.assertTrue(spec.prefer_best)
        self.assertFalse(spec.is_console())
        self.assertEqual(spec.console_model, "")


class AccountRefreshBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        cls.refresh_mod = importlib.import_module("app.control.account.refresh")
        cls.models = importlib.import_module("app.control.account.models")
        cls.enums = importlib.import_module("app.control.account.enums")
        cls.quota_defaults = importlib.import_module("app.control.account.quota_defaults")

    def _window(self, *, remaining: int, total: int, mode_id: int = 4):
        return self.models.QuotaWindow(
            remaining=remaining,
            total=total,
            window_seconds=7200,
            reset_at=123456789,
            synced_at=123450000,
            source=self.enums.QuotaSource.REAL,
        )

    def test_bootstrap_fetches_entitlement_probe_modes_for_basic_records(self):
        xai_usage = importlib.import_module("app.dataplane.reverse.protocol.xai_usage")
        calls = []
        original = xai_usage.fetch_all_quotas

        async def _fake_fetch(token, mode_ids):
            calls.append((token, mode_ids))
            return {}

        async def _run():
            xai_usage.fetch_all_quotas = _fake_fetch
            try:
                svc = self.refresh_mod.AccountRefreshService(repository=object())
                await svc._fetch_all_quotas("tok-bootstrap", "basic", bootstrap=True)
            finally:
                xai_usage.fetch_all_quotas = original

        asyncio.run(_run())

        self.assertEqual(calls, [("tok-bootstrap", (0, 2, 3, 4, 1, 5))])

    def test_refresh_one_bootstrap_uses_live_grok_4_3_window_to_patch_heavy_pool(self):
        patches = []
        quota = self.quota_defaults.default_quota_set("basic").to_dict()
        record = self.models.AccountRecord(token="tok-heavy", pool="basic", quota=quota)

        class _Repo:
            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        class _Service(self.refresh_mod.AccountRefreshService):
            async def _fetch_all_quotas(self, token, pool, *, bootstrap=False):
                self.fetch_args = (token, pool, bootstrap)
                return {4: self.test_case._window(remaining=149, total=150)}

        async def _run():
            svc = _Service(_Repo())
            svc.test_case = self
            result = await svc._refresh_one(record, bootstrap=True)
            return svc, result

        svc, result = asyncio.run(_run())

        self.assertEqual(svc.fetch_args, ("tok-heavy", "basic", True))
        self.assertEqual(result.refreshed, 1)
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].pool, "heavy")
        self.assertIsNotNone(patches[0].quota_grok_4_3)

    def test_reset_expired_console_windows_restores_local_console_quota(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 0
        quota_set.console.reset_at = 1
        record = self.models.AccountRecord(
            token="tok-console-expired",
            pool="basic",
            quota=quota_set.to_dict(),
        )

        class _Snapshot:
            items = [record]

        class _Repo:
            async def runtime_snapshot(self):
                return _Snapshot()

            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            return await svc.reset_expired_console_windows()

        count = asyncio.run(_run())

        self.assertEqual(count, 1)
        restored = self.models.QuotaWindow.from_dict(patches[0].quota_console)
        self.assertEqual(restored.remaining, 30)
        self.assertEqual(restored.total, 30)
        self.assertEqual(restored.window_seconds, 1_800)

    def test_refresh_call_async_console_deducts_locally_without_upstream_fetch(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 30
        quota_set.console.reset_at = None
        record = self.models.AccountRecord(
            token="tok-console-local",
            pool="basic",
            quota=quota_set.to_dict(),
        )

        class _Repo:
            async def get_accounts(self, tokens):
                return [record]

            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        class _Service(self.refresh_mod.AccountRefreshService):
            async def _fetch_mode_quota(self, token, pool, mode_id):
                raise AssertionError("console quota must stay local")

        async def _run():
            svc = _Service(_Repo())
            await svc.refresh_call_async("tok-console-local", 5)

        asyncio.run(_run())

        self.assertEqual(len(patches), 1)
        patched = patches[0]
        window = self.models.QuotaWindow.from_dict(patched.quota_console)
        self.assertEqual(window.remaining, 29)
        self.assertIsNone(window.reset_at)
        self.assertEqual(patched.usage_use_delta, 1)
        self.assertIsNone(patched.usage_sync_delta)

    def test_console_local_deduct_starts_reset_timer_at_rotation_threshold(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 21
        quota_set.console.reset_at = None
        record = self.models.AccountRecord(
            token="tok-console-threshold",
            pool="basic",
            quota=quota_set.to_dict(),
        )

        class _Repo:
            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            await svc._apply_single_mode(record, 5, None, is_use=True, use_at_ms=1234)

        asyncio.run(_run())

        window = self.models.QuotaWindow.from_dict(patches[0].quota_console)
        self.assertEqual(window.remaining, 20)
        self.assertIsNotNone(window.reset_at)
        self.assertEqual(window.window_seconds, 1_800)

    def test_console_local_deduct_resets_expired_window_before_counting_call(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 0
        quota_set.console.reset_at = 1
        record = self.models.AccountRecord(
            token="tok-console-expired-use",
            pool="basic",
            quota=quota_set.to_dict(),
        )

        class _Repo:
            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            await svc._apply_single_mode(record, 5, None, is_use=True, use_at_ms=1234)

        asyncio.run(_run())

        window = self.models.QuotaWindow.from_dict(patches[0].quota_console)
        self.assertEqual(window.remaining, 29)
        self.assertEqual(window.total, 30)
        self.assertIsNotNone(window.reset_at)
        self.assertEqual(window.source, self.enums.QuotaSource.DEFAULT)


class ConsoleQuotaSyncTests(unittest.TestCase):
    def test_console_quota_sync_runs_in_random_strategy(self):
        chat = _load_openai_chat_module()
        calls = []

        class _RefreshService:
            async def refresh_call_async(self, token, mode_id):
                calls.append((token, mode_id))

        async def _run():
            original_strategy = chat.current_strategy
            original_service = chat.get_refresh_service
            try:
                chat.current_strategy = lambda: "random"
                chat.get_refresh_service = lambda: _RefreshService()
                await chat._quota_sync("tok-console", 5)
            finally:
                chat.current_strategy = original_strategy
                chat.get_refresh_service = original_service

        asyncio.run(_run())

        self.assertEqual(calls, [("tok-console", 5)])


class StableJiujiuUpdateTests(unittest.TestCase):
    def test_run_batch_uses_worker_pool_and_preserves_order(self):
        runtime_batch = importlib.import_module("app.platform.runtime.batch")
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def _handler(item):
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0)
            async with lock:
                active -= 1
            return item * 2

        async def _run():
            return await runtime_batch.run_batch(range(12), _handler, concurrency=3)

        result = asyncio.run(_run())

        self.assertEqual(result, [i * 2 for i in range(12)])
        self.assertLessEqual(max_active, 3)

        source = (REPO_ROOT / "app" / "platform" / "runtime" / "batch.py").read_text(encoding="utf-8")
        self.assertIn("asyncio.create_task(_worker())", source)
        self.assertNotIn("asyncio.gather(*[_guarded", source)

    def test_admin_batch_async_path_uses_shared_batch_runner(self):
        source = (REPO_ROOT / "app" / "products" / "web" / "admin" / "batch.py").read_text(encoding="utf-8")

        self.assertIn("_MAX_BATCH_CONCURRENCY = 50", source)
        self.assertIn("await run_batch(tokens, _one, concurrency=concurrency)", source)
        self.assertNotIn("await asyncio.gather(*[_one(t) for t in tokens])", source)

    def test_redis_get_accounts_uses_pipeline(self):
        source = (REPO_ROOT / "app" / "control" / "account" / "backends" / "redis.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("async with self._r.pipeline() as pipe", source)
        self.assertIn("hashes = await pipe.execute()", source)
        self.assertIn("for token, h in zip(tokens, hashes, strict=True):", source)

    def test_config_number_input_honors_min_max(self):
        source = (REPO_ROOT / "app" / "statics" / "admin" / "config.html").read_text(encoding="utf-8")

        self.assertIn("if (field.min != null) attrs.min = field.min;", source)
        self.assertIn("if (field.max != null) attrs.max = field.max;", source)
        self.assertIn("if (field.min != null && n < field.min) n = field.min;", source)
        self.assertIn("if (field.max != null && n > field.max) n = field.max;", source)
        self.assertIn("key: 'refresh_concurrency', label: '刷新 Usage 并发数'", source)
        self.assertIn("type: 'number', min: 1, max: 50", source)

    def test_streaming_routes_emit_sse_heartbeats(self):
        chat = (REPO_ROOT / "app" / "products" / "openai" / "chat.py").read_text(encoding="utf-8")
        responses = (REPO_ROOT / "app" / "products" / "openai" / "responses.py").read_text(
            encoding="utf-8"
        )
        messages = (REPO_ROOT / "app" / "products" / "anthropic" / "messages.py").read_text(
            encoding="utf-8"
        )

        self.assertGreaterEqual(chat.count('yield ": heartbeat\\n\\n"'), 2)
        self.assertGreaterEqual(responses.count('yield ": heartbeat\\n\\n"'), 2)
        self.assertGreaterEqual(messages.count('yield ": heartbeat\\n\\n"'), 2)

    def test_admin_batch_all_manageable_must_be_explicit(self):
        source = (REPO_ROOT / "app" / "products" / "web" / "admin" / "batch.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("all_manageable: bool = Query(False)", source)
        self.assertIn('raise ValidationError("No tokens provided", param="tokens")', source)
        self.assertIn('raise ValidationError("tokens must be empty when all_manageable=true"', source)
        self.assertIn("await _filter_manageable_tokens(repo, tokens)", source)

    def test_delete_invalid_endpoint_only_deletes_expired_accounts(self):
        tokens_mod = _load_admin_tokens_module()
        enums = importlib.import_module("app.control.account.enums")

        class _Record:
            def __init__(self, token, status):
                self.token = token
                self.status = status

        class _Repo:
            def __init__(self):
                self.deleted = []

            async def list_accounts(self, _query):
                return types.SimpleNamespace(
                    items=[
                        _Record("tok-active", enums.AccountStatus.ACTIVE),
                        _Record("tok-cooling", enums.AccountStatus.COOLING),
                        _Record("tok-disabled", enums.AccountStatus.DISABLED),
                        _Record("tok-expired", enums.AccountStatus.EXPIRED),
                    ],
                    total_pages=1,
                )

            async def delete_accounts(self, tokens):
                self.deleted = list(tokens)

        repo = _Repo()
        response = asyncio.run(tokens_mod.delete_invalid_tokens(repo))
        body = json.loads(response.body.decode("utf-8"))

        self.assertEqual(repo.deleted, ["tok-expired"])
        self.assertEqual(body, {"deleted": 1})

    def test_account_admin_safe_bulk_controls_are_present(self):
        account = (REPO_ROOT / "app" / "statics" / "admin" / "account.html").read_text(encoding="utf-8")
        config = (REPO_ROOT / "app" / "statics" / "admin" / "config.html").read_text(encoding="utf-8")
        defaults = (REPO_ROOT / "config.defaults.toml").read_text(encoding="utf-8")

        self.assertIn("id=\"import-auto-nsfw\"", account)
        self.assertIn("id=\"import-file-auto-nsfw\"", account)
        self.assertIn("/batch/refresh?all_manageable=true", account)
        self.assertIn("/batch/nsfw?all_manageable=true", account)
        self.assertIn("window.prompt", account)
        self.assertIn("DELETE", account)
        self.assertIn("isDeletableInvalidStatus(status)", account)
        self.assertIn("const ACCOUNT_AUTO_RELOAD_MS = 30000;", account)
        self.assertIn("startAccountAutoReload();", account)
        self.assertIn("load({ silent: true })", account)
        self.assertIn("if (_batchEs) return false;", account)
        self.assertIn("auto_nsfw_on_import", config)
        self.assertIn("auto_nsfw_on_import = false", defaults)
        self.assertIn("console_multi_agent_native_tools", config)
        self.assertIn("console_multi_agent_native_tools = false", defaults)

    def test_auto_pool_import_refreshes_in_background(self):
        source = (REPO_ROOT / "app" / "products" / "web" / "admin" / "tokens.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("sync_auto_detect", source)
        self.assertNotIn("admin auto-detect quota sync completed", source)
        self.assertIn("_fire_and_forget(_refresh_then_auto_nsfw(", source)

    def test_security_dependency_floors_are_updated(self):
        source = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"cryptography>=48.0.1"', source)
        self.assertIn('"python-multipart>=0.0.31"', source)
        self.assertIn('"starlette>=1.1.0"', source)

    def test_console_multi_agent_native_tools_config_is_wired(self):
        chat = (REPO_ROOT / "app" / "products" / "openai" / "chat.py").read_text(encoding="utf-8")
        responses = (REPO_ROOT / "app" / "products" / "openai" / "responses.py").read_text(
            encoding="utf-8"
        )

        for source in (chat, responses):
            self.assertIn('get_bool("features.console_multi_agent_native_tools", False)', source)
            self.assertIn("allow_multi_agent_client_tools=allow_multi_agent_tools", source)
