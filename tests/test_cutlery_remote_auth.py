import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.auth import build_auth_headers, configured_remote_token, is_authorized


class RemoteAuthTests(unittest.TestCase):
    def test_configured_remote_token_prefers_environment(self):
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": " secret-token "}, clear=False):
            self.assertEqual(configured_remote_token(), "secret-token")

    def test_build_auth_headers_uses_bearer_token(self):
        self.assertEqual(build_auth_headers("abc123"), {"Authorization": "Bearer abc123"})

    def test_is_authorized_accepts_matching_bearer_token(self):
        self.assertTrue(is_authorized({"Authorization": "Bearer abc123"}, "abc123"))

    def test_is_authorized_rejects_missing_or_wrong_token(self):
        self.assertFalse(is_authorized({}, "abc123"))
        self.assertFalse(is_authorized({"Authorization": "Bearer wrong"}, "abc123"))


if __name__ == "__main__":
    unittest.main()
