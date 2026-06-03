import base64
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_common_stubs() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _Logger:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        loguru.logger = _Logger()
        sys.modules["loguru"] = loguru

    if "curl_cffi.const" not in sys.modules:
        curl_cffi = types.ModuleType("curl_cffi")
        const = types.ModuleType("curl_cffi.const")

        class _CurlOpt:
            PROXY_SSL_VERIFYPEER = object()
            PROXY_SSL_VERIFYHOST = object()

        const.CurlOpt = _CurlOpt
        sys.modules["curl_cffi"] = curl_cffi
        sys.modules["curl_cffi.const"] = const


class _Config:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self._values = values or {}

    def get_str(self, key: str, default: str = "") -> str:
        value = self._values.get(key, default)
        return str(value) if value is not None else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self._values.get(key, default)
        return bool(value)


class StatsigHeadersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        from app.dataplane.proxy.adapters import headers

        cls.headers = headers

    def setUp(self):
        self._original_get_config = self.headers.get_config

    def tearDown(self):
        self.headers.get_config = self._original_get_config

    def test_manual_statsig_id_takes_precedence(self):
        self.headers.get_config = lambda: _Config({"features.statsig_id": "  manual-statsig\n"})

        self.assertEqual(self.headers._statsig_id(), "manual-statsig")

    def test_generated_statsig_matches_browser_fallback_prefix(self):
        self.headers.get_config = lambda: _Config()

        decoded = base64.b64decode(self.headers._statsig_id()).decode()

        self.assertTrue(decoded.startswith("x1:TypeError:"), decoded)
        self.assertFalse(decoded.startswith("e:"), decoded)


if __name__ == "__main__":
    unittest.main()
