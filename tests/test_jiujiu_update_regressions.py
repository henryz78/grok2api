import asyncio
import importlib
import importlib.util
import json
import tempfile
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
        class _Repo:
            async def reset_expired_console_windows(self):
                return 3

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            return await svc.reset_expired_console_windows()

        count = asyncio.run(_run())

        self.assertEqual(count, 3)

    def test_refresh_call_async_console_deducts_locally_without_upstream_fetch(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 20
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
        self.assertEqual(window.remaining, 19)
        self.assertIsNone(window.reset_at)
        self.assertEqual(patched.usage_use_delta, 1)
        self.assertIsNone(patched.usage_sync_delta)

    def test_console_local_deduct_starts_reset_timer_at_rotation_threshold(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 13
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
        self.assertEqual(window.remaining, 12)
        self.assertIsNotNone(window.reset_at)
        self.assertEqual(window.window_seconds, 3_600)

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
        self.assertEqual(window.remaining, 19)
        self.assertEqual(window.total, 20)
        self.assertIsNone(window.reset_at)
        self.assertEqual(window.source, self.enums.QuotaSource.DEFAULT)

    def test_record_failure_async_console_429_decrements_without_zeroing(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 20
        quota_set.console.reset_at = None
        record = self.models.AccountRecord(
            token="tok-console-429",
            pool="basic",
            quota=quota_set.to_dict(),
        )
        upstream_error = importlib.import_module("app.platform.errors").UpstreamError(
            "rate limited",
            status=429,
        )

        class _Repo:
            async def get_accounts(self, tokens):
                return [record]

            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            await svc.record_failure_async("tok-console-429", 5, upstream_error)

        asyncio.run(_run())

        self.assertEqual(len(patches), 1)
        patch = patches[0]
        window = self.models.QuotaWindow.from_dict(patch.quota_console)
        self.assertEqual(window.remaining, 10)
        self.assertIsNotNone(window.reset_at)
        self.assertEqual(patch.usage_fail_delta, 1)
        self.assertEqual(patch.last_fail_reason, "rate_limited")
        self.assertEqual(patch.ext_merge["console_429_count"], 1)
        self.assertIsInstance(patch.ext_merge["console_429_last_at"], int)

    def test_record_failure_async_console_429_uses_independent_counter(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 20
        record = self.models.AccountRecord(
            token="tok-console-counter",
            pool="basic",
            quota=quota_set.to_dict(),
            usage_fail_count=99,
            ext={"console_429_count": 1},
        )
        upstream_error = importlib.import_module("app.platform.errors").UpstreamError(
            "rate limited",
            status=429,
        )

        class _Repo:
            async def get_accounts(self, tokens):
                return [record]

            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            await svc.record_failure_async("tok-console-counter", 5, upstream_error)

        asyncio.run(_run())

        self.assertEqual(len(patches), 1)
        patch = patches[0]
        window = self.models.QuotaWindow.from_dict(patch.quota_console)
        self.assertEqual(window.remaining, 10)
        self.assertIsNone(patch.status)
        self.assertEqual(patch.ext_merge["console_429_count"], 2)
        self.assertIsInstance(patch.ext_merge["console_429_last_at"], int)

    def test_record_failure_async_console_429_sliding_window_resets_old_counter(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 20
        record = self.models.AccountRecord(
            token="tok-console-sliding",
            pool="basic",
            quota=quota_set.to_dict(),
            ext={
                "console_429_count": 2,
                "console_429_last_at": 1_000,
            },
        )
        upstream_error = importlib.import_module("app.platform.errors").UpstreamError(
            "rate limited",
            status=429,
        )

        class _Repo:
            async def get_accounts(self, tokens):
                return [record]

            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            original_now_ms = self.refresh_mod.now_ms
            self.refresh_mod.now_ms = lambda: 1_000 + 13 * 3600 * 1000
            try:
                svc = self.refresh_mod.AccountRefreshService(_Repo())
                await svc.record_failure_async("tok-console-sliding", 5, upstream_error)
            finally:
                self.refresh_mod.now_ms = original_now_ms

        asyncio.run(_run())

        self.assertEqual(len(patches), 1)
        patch = patches[0]
        self.assertEqual(patch.ext_merge["console_429_count"], 1)
        self.assertIsNone(patch.status)

    def test_record_failure_async_console_429_expires_after_third_console_429(self):
        patches = []
        quota_set = self.quota_defaults.default_quota_set("basic")
        assert quota_set.console is not None
        quota_set.console.remaining = 5
        record = self.models.AccountRecord(
            token="tok-console-expire",
            pool="basic",
            quota=quota_set.to_dict(),
            ext={"console_429_count": 2},
        )
        upstream_error = importlib.import_module("app.platform.errors").UpstreamError(
            "rate limited",
            status=429,
        )

        class _Repo:
            async def get_accounts(self, tokens):
                return [record]

            async def patch_accounts(self, account_patches):
                patches.extend(account_patches)

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            await svc.record_failure_async("tok-console-expire", 5, upstream_error)

        asyncio.run(_run())

        self.assertEqual(len(patches), 1)
        patch = patches[0]
        window = self.models.QuotaWindow.from_dict(patch.quota_console)
        self.assertEqual(window.remaining, 0)
        self.assertEqual(patch.status, self.enums.AccountStatus.EXPIRED)
        self.assertEqual(patch.state_reason, "console_429_threshold_exceeded")
        self.assertEqual(patch.ext_merge["console_429_count"], 3)
        self.assertIsInstance(patch.ext_merge["console_429_last_at"], int)
        self.assertEqual(patch.ext_merge["expired_reason"], "console_429_threshold_exceeded")

    def test_recover_console_expired_accounts_delegates_to_repository(self):
        class _Repo:
            async def recover_console_expired_accounts(self):
                return 2

        async def _run():
            svc = self.refresh_mod.AccountRefreshService(_Repo())
            return await svc.recover_console_expired_accounts()

        self.assertEqual(asyncio.run(_run()), 2)

    def test_local_repository_bulk_resets_expired_console_windows(self):
        _install_common_stubs()
        local_mod = importlib.import_module("app.control.account.backends.local")
        commands = importlib.import_module("app.control.account.commands")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = local_mod.LocalAccountRepository(Path(tmpdir) / "accounts.db")
            asyncio.run(repo.initialize())
            asyncio.run(
                repo.upsert_accounts(
                    [
                        commands.AccountUpsert(token="tok-zero"),
                        commands.AccountUpsert(token="tok-stale"),
                        commands.AccountUpsert(token="tok-good"),
                    ]
                )
            )

            quota_zero = self.quota_defaults.default_quota_set("basic")
            assert quota_zero.console is not None
            quota_zero.console.remaining = 0
            quota_zero.console.reset_at = None

            quota_stale = self.quota_defaults.default_quota_set("basic")
            assert quota_stale.console is not None
            quota_stale.console.remaining = 7
            quota_stale.console.reset_at = 1

            quota_good = self.quota_defaults.default_quota_set("basic")
            assert quota_good.console is not None
            quota_good.console.remaining = 7
            quota_good.console.reset_at = 9_999_999_999_999

            patch_cls = commands.AccountPatch
            asyncio.run(
                repo.patch_accounts(
                    [
                        patch_cls(token="tok-zero", quota_console=quota_zero.console.to_dict()),
                        patch_cls(token="tok-stale", quota_console=quota_stale.console.to_dict()),
                        patch_cls(token="tok-good", quota_console=quota_good.console.to_dict()),
                    ]
                )
            )

            count = asyncio.run(repo.reset_expired_console_windows())
            records = {
                record.token: record
                for record in asyncio.run(repo.get_accounts(["tok-zero", "tok-stale", "tok-good"]))
            }

        self.assertEqual(count, 2)
        self.assertEqual(records["tok-zero"].quota_set().console.remaining, 20)
        self.assertEqual(records["tok-stale"].quota_set().console.remaining, 20)
        self.assertEqual(records["tok-good"].quota_set().console.remaining, 7)

    def test_local_repository_clear_failures_clears_console_429_counter(self):
        _install_common_stubs()
        local_mod = importlib.import_module("app.control.account.backends.local")
        commands = importlib.import_module("app.control.account.commands")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = local_mod.LocalAccountRepository(Path(tmpdir) / "accounts.db")
            asyncio.run(repo.initialize())
            asyncio.run(repo.upsert_accounts([commands.AccountUpsert(token="tok-clear")]))
            asyncio.run(
                repo.patch_accounts(
                    [
                        commands.AccountPatch(
                            token="tok-clear",
                            status=self.enums.AccountStatus.EXPIRED,
                            ext_merge={
                                "console_429_count": 3,
                                "console_429_last_at": 123,
                                "expired_reason": "console_429_threshold_exceeded",
                            },
                        )
                    ]
                )
            )
            asyncio.run(
                repo.patch_accounts(
                    [commands.AccountPatch(token="tok-clear", clear_failures=True)]
                )
            )
            record = asyncio.run(repo.get_accounts(["tok-clear"]))[0]

        self.assertEqual(record.status, self.enums.AccountStatus.ACTIVE)
        self.assertNotIn("console_429_count", record.ext)
        self.assertNotIn("console_429_last_at", record.ext)
        self.assertNotIn("expired_reason", record.ext)

    def test_local_repository_recovers_console_429_expired_accounts(self):
        _install_common_stubs()
        local_mod = importlib.import_module("app.control.account.backends.local")
        commands = importlib.import_module("app.control.account.commands")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = local_mod.LocalAccountRepository(Path(tmpdir) / "accounts.db")
            asyncio.run(repo.initialize())
            asyncio.run(
                repo.upsert_accounts(
                    [
                        commands.AccountUpsert(token="tok-recover"),
                        commands.AccountUpsert(token="tok-low-use"),
                    ]
                )
            )
            old_expired_at = self.refresh_mod.now_ms() - 2 * 3600 * 1000
            asyncio.run(
                repo.patch_accounts(
                    [
                        commands.AccountPatch(
                            token="tok-recover",
                            status=self.enums.AccountStatus.EXPIRED,
                            state_reason="console_429_threshold_exceeded",
                            usage_use_delta=6,
                            ext_merge={
                                "console_429_count": 3,
                                "console_429_last_at": old_expired_at,
                                "expired_at": old_expired_at,
                                "expired_reason": "console_429_threshold_exceeded",
                            },
                        ),
                        commands.AccountPatch(
                            token="tok-low-use",
                            status=self.enums.AccountStatus.EXPIRED,
                            state_reason="console_429_threshold_exceeded",
                            usage_use_delta=5,
                            ext_merge={
                                "console_429_count": 3,
                                "expired_at": old_expired_at,
                                "expired_reason": "console_429_threshold_exceeded",
                            },
                        ),
                    ]
                )
            )

            count = asyncio.run(repo.recover_console_expired_accounts())
            records = {
                record.token: record
                for record in asyncio.run(repo.get_accounts(["tok-recover", "tok-low-use"]))
            }

        self.assertEqual(count, 1)
        self.assertEqual(records["tok-recover"].status, self.enums.AccountStatus.ACTIVE)
        self.assertIsNone(records["tok-recover"].state_reason)
        self.assertNotIn("console_429_count", records["tok-recover"].ext)
        self.assertNotIn("console_429_last_at", records["tok-recover"].ext)
        self.assertEqual(records["tok-low-use"].status, self.enums.AccountStatus.EXPIRED)


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
    def test_sqlite_wal_fallback_keeps_connection_usable(self):
        _install_common_stubs()
        local_mod = importlib.import_module("app.control.account.backends.local")

        class _FakeConnection:
            def __init__(self):
                self.row_factory = None
                self.executed: list[str] = []

            def execute(self, sql, *args, **kwargs):
                self.executed.append(str(sql))
                if str(sql) == "PRAGMA journal_mode=WAL":
                    raise local_mod.sqlite3.OperationalError("wal unsupported")
                return self

        fake = _FakeConnection()
        original_connect = local_mod.sqlite3.connect
        local_mod.sqlite3.connect = lambda *args, **kwargs: fake
        try:
            repo = local_mod.LocalAccountRepository(Path("dummy.db"))
            self.assertIs(repo._connect(), fake)
        finally:
            local_mod.sqlite3.connect = original_connect

        self.assertIn("PRAGMA journal_mode=WAL", fake.executed)
        self.assertIn("PRAGMA busy_timeout=5000", fake.executed)

    def test_console_expired_recovery_loop_is_wired(self):
        source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

        self.assertIn("console_recovery_interval = 600", source)
        self.assertIn("await refresh_svc.recover_console_expired_accounts()", source)
        self.assertIn('name="console-expired-recovery"', source)
        self.assertIn("console_recovery_task.cancel()", source)

    def test_deleted_account_cleanup_is_wired_with_safe_defaults(self):
        defaults = (REPO_ROOT / "config.defaults.toml").read_text(encoding="utf-8")
        config = (REPO_ROOT / "app" / "statics" / "admin" / "config.html").read_text(
            encoding="utf-8"
        )
        main = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

        self.assertIn("[account.cleanup]", defaults)
        self.assertIn("deleted_retention_days = 7", defaults)
        self.assertIn('run_at = "03:30"', defaults)
        self.assertIn("batch_size = 5000", defaults)
        self.assertIn("vacuum = false", defaults)
        self.assertIn("section: 'account.cleanup'", config)
        self.assertIn("deleted_retention_days", config)
        self.assertIn("max: 50000", config)
        self.assertIn("run_daily_deleted_account_cleanup", main)
        self.assertIn('name="deleted-account-cleanup"', main)
        self.assertIn("deleted_cleanup_task.cancel()", main)

    def test_deleted_account_cleanup_helpers(self):
        _install_common_stubs()
        cleanup_mod = importlib.import_module("app.control.account.cleanup")

        now = 1_000_000_000
        self.assertEqual(
            cleanup_mod.cleanup_threshold_ms(now, 7),
            now - 7 * 86_400_000,
        )
        self.assertGreater(
            cleanup_mod.seconds_until_next_daily_run(
                now_ms_value=now,
                run_at="bad",
            ),
            0,
        )

        class _Repo:
            pass

        async def _run():
            return await cleanup_mod.purge_deleted_accounts_once(
                _Repo(),
                retention_days=7,
                batch_size=100,
                vacuum=False,
            )

        result = asyncio.run(_run())
        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, "repository_does_not_support_purge")

    def test_local_repository_token_payloads_and_invalid_fast_path(self):
        _install_common_stubs()
        local_mod = importlib.import_module("app.control.account.backends.local")
        commands = importlib.import_module("app.control.account.commands")
        enums = importlib.import_module("app.control.account.enums")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = local_mod.LocalAccountRepository(Path(tmpdir) / "accounts.db")
            asyncio.run(repo.initialize())
            asyncio.run(
                repo.upsert_accounts(
                    [
                        commands.AccountUpsert(token="tok-active", tags=["nsfw"]),
                        commands.AccountUpsert(token="tok-expired"),
                        commands.AccountUpsert(token="tok-disabled"),
                    ]
                )
            )
            asyncio.run(
                repo.patch_accounts(
                    [
                        commands.AccountPatch(
                            token="tok-active",
                            usage_use_delta=3,
                            usage_fail_delta=2,
                            usage_sync_delta=1,
                            last_fail_reason="rate_limited",
                            quota_grok_4_3={"remaining": 8, "total": 10},
                            ext_merge={"note": "keep-ui-field"},
                        ),
                        commands.AccountPatch(
                            token="tok-expired",
                            status=enums.AccountStatus.EXPIRED,
                        ),
                        commands.AccountPatch(
                            token="tok-disabled",
                            status=enums.AccountStatus.DISABLED,
                        ),
                    ]
                )
            )
            items = asyncio.run(repo.list_token_payloads())
            invalid = asyncio.run(repo.list_invalid_tokens())

        by_token = {item["token"]: item for item in items}
        active = by_token["tok-active"]
        self.assertEqual(active["use_count"], 3)
        self.assertEqual(active["fail_count"], 2)
        self.assertEqual(active["sync_count"], 1)
        self.assertEqual(active["last_fail_reason"], "rate_limited")
        self.assertEqual(active["quota"]["grok_4_3"], {"remaining": 8, "total": 10})
        self.assertEqual(active["tags"], ["nsfw"])
        self.assertEqual(active["ext"], {"note": "keep-ui-field"})
        self.assertEqual(invalid, ["tok-expired"])

    def test_local_repository_purges_deleted_accounts_in_batches(self):
        _install_common_stubs()
        local_mod = importlib.import_module("app.control.account.backends.local")
        commands = importlib.import_module("app.control.account.commands")
        import sqlite3
        from contextlib import closing

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "accounts.db"
            repo = local_mod.LocalAccountRepository(db_path)
            asyncio.run(repo.initialize())
            asyncio.run(
                repo.upsert_accounts(
                    [
                        commands.AccountUpsert(token="old-deleted-1"),
                        commands.AccountUpsert(token="old-deleted-2"),
                        commands.AccountUpsert(token="new-deleted"),
                        commands.AccountUpsert(token="live-token"),
                    ]
                )
            )
            asyncio.run(
                repo.delete_accounts(["old-deleted-1", "old-deleted-2", "new-deleted"])
            )
            cutoff = 2_000_000
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "UPDATE accounts SET deleted_at = ? WHERE token LIKE ?",
                    (cutoff - 1, "old-deleted%"),
                )
                conn.execute(
                    "UPDATE accounts SET deleted_at = ? WHERE token = ?",
                    (cutoff + 1, "new-deleted"),
                )
                conn.commit()

            purged = asyncio.run(
                repo.purge_deleted_accounts(
                    deleted_before_ms=cutoff,
                    batch_size=1,
                    vacuum=False,
                )
            )
            with closing(sqlite3.connect(db_path)) as conn:
                rows = {
                    row[0]: row[1]
                    for row in conn.execute(
                        "SELECT token, deleted_at FROM accounts ORDER BY token"
                    )
                }

        self.assertEqual(purged, 2)
        self.assertNotIn("old-deleted-1", rows)
        self.assertNotIn("old-deleted-2", rows)
        self.assertIsNotNone(rows["new-deleted"])
        self.assertIsNone(rows["live-token"])

    def test_admin_tokens_uses_fast_list_paths(self):
        tokens_mod = _load_admin_tokens_module()

        class _FastRepo:
            def __init__(self):
                self.list_called = False

            async def list_token_payloads(self):
                return [
                    {
                        "token": "tok-fast",
                        "pool": "basic",
                        "status": "active",
                        "quota": {},
                        "use_count": 0,
                        "fail_count": 0,
                        "sync_count": 0,
                        "last_used_at": None,
                        "tags": [],
                        "ext": {},
                    }
                ]

            async def list_accounts(self, _query):
                self.list_called = True
                raise AssertionError("list_tokens should use compact payloads")

        repo = _FastRepo()
        body = json.loads(asyncio.run(tokens_mod.list_tokens(repo)).body.decode("utf-8"))

        self.assertFalse(repo.list_called)
        self.assertEqual(body["tokens"][0]["token"], "tok-fast")

        class _InvalidRepo:
            def __init__(self):
                self.deleted = []
                self.payload_called = False

            async def list_invalid_tokens(self):
                return ["tok-expired"]

            async def list_token_payloads(self):
                self.payload_called = True
                return []

            async def delete_accounts(self, tokens):
                self.deleted = list(tokens)

        invalid_repo = _InvalidRepo()
        body = json.loads(
            asyncio.run(tokens_mod.delete_invalid_tokens(invalid_repo)).body.decode("utf-8")
        )

        self.assertFalse(invalid_repo.payload_called)
        self.assertEqual(invalid_repo.deleted, ["tok-expired"])
        self.assertEqual(body, {"deleted": 1})

    def _runtime_table_for_selection(self):
        _install_common_stubs()
        table_mod = importlib.import_module("app.dataplane.account.table")
        shared = importlib.import_module("app.dataplane.shared.enums")
        table = table_mod.make_empty_table()

        def _append(token: str, quota: int, *, inflight: int = 0, fails: int = 0, last_use_s: int = 0):
            idx = table._append_slot(
                token=token,
                pool_id=int(shared.PoolId.BASIC),
                status_id=int(shared.StatusId.ACTIVE),
                quota_auto=0,
                quota_fast=quota,
                quota_expert=0,
                quota_heavy=-1,
                quota_grok_4_3=-1,
                quota_console=20,
                total_auto=0,
                total_fast=30,
                total_expert=0,
                total_heavy=0,
                total_grok_4_3=0,
                total_console=20,
                window_auto=0,
                window_fast=86400,
                window_expert=0,
                window_heavy=0,
                window_grok_4_3=0,
                window_console=3600,
                reset_auto=0,
                reset_fast=0,
                reset_expert=0,
                reset_heavy=0,
                reset_grok_4_3=0,
                reset_console=0,
                health=1.0,
                last_use_s=last_use_s,
                last_fail_s=0,
                fail_count=fails,
                tags=[],
            )
            table.inflight_by_idx[idx] = inflight
            return idx

        return table, shared, _append

    def test_quota_selector_spreads_recent_and_inflight_heavy_accounts(self):
        selector = importlib.import_module("app.dataplane.account.selector")
        table, shared, append = self._runtime_table_for_selection()
        blocked_recent = append("tok-recent", 20, last_use_s=50)
        blocked_inflight = append("tok-inflight", 999, inflight=12)
        chosen = append("tok-ready", 20)

        selected = selector._quota_select(
            table,
            int(shared.PoolId.BASIC),
            int(shared.ModeId.FAST),
            exclude_idxs=None,
            prefer_tag_idxs=None,
            now_s=100,
        )

        self.assertEqual(selected, chosen)
        self.assertNotEqual(selected, blocked_recent)
        self.assertNotEqual(selected, blocked_inflight)

    def test_random_selector_filters_high_fail_accounts(self):
        selector = importlib.import_module("app.dataplane.account.selector")
        table, shared, append = self._runtime_table_for_selection()
        blocked_failed = append("tok-failed", 30, fails=5)
        chosen = append("tok-random-ready", 30)

        selected = selector._random_select(
            table,
            int(shared.PoolId.BASIC),
            exclude_idxs=None,
            prefer_tag_idxs=None,
            now_s=100,
        )

        self.assertEqual(selected, chosen)
        self.assertNotEqual(selected, blocked_failed)

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

    def test_incremental_sync_advances_to_batch_max_revision_while_paginating(self):
        _install_common_stubs()
        sync_mod = importlib.import_module("app.dataplane.account.sync")
        table_mod = importlib.import_module("app.dataplane.account.table")
        models = importlib.import_module("app.control.account.models")

        table = table_mod.make_empty_table()
        calls = []

        class _Repo:
            async def scan_changes(self, since_revision, *, limit=5000):
                calls.append(since_revision)
                if len(calls) == 1:
                    return models.AccountChangeSet(
                        revision=100,
                        batch_max_revision=7,
                        has_more=True,
                    )
                return models.AccountChangeSet(revision=100, has_more=False)

        changed = asyncio.run(sync_mod.apply_changes(table, _Repo()))

        self.assertFalse(changed)
        self.assertEqual(calls, [0, 7])
        self.assertEqual(table.revision, 100)

    def test_config_default_usage_concurrency_matches_sql_pool_capacity(self):
        defaults = (REPO_ROOT / "config.defaults.toml").read_text(encoding="utf-8")
        self.assertIn("usage_concurrency = 15", defaults)

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
