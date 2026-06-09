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
        self.assertEqual(restored.window_seconds, 900)

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

    def test_console_local_deduct_starts_reset_timer_at_half_quota(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 16
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
        self.assertEqual(window.remaining, 15)
        self.assertIsNotNone(window.reset_at)
        self.assertEqual(window.window_seconds, 900)

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
