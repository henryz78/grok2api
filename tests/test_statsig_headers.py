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


class StatsigHeadersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_common_stubs()
        from app.dataplane.proxy.adapters import headers

        cls.headers = headers

    def test_generated_statsig_matches_official_browser_fallback_prefix(self):
        decoded = base64.b64decode(self.headers._statsig_id()).decode()

        self.assertTrue(decoded.startswith("x1:TypeError:"), decoded)
        self.assertFalse(decoded.startswith("e:"), decoded)


if __name__ == "__main__":
    unittest.main()
