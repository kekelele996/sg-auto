"""Solo Manager login verification, password saving and the prompt's user name."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.common import ManagerApiError, MonitorError, SettingsStore, render_auto_trigger_prompt  # noqa: E402
from api.platform import PlatformProvider  # noqa: E402
from api.service import SchedulerService  # noqa: E402
from tests.support import SchedulerTestCase  # noqa: E402

GOOD = "right-pass"


def fake_manager(users=None):
    """A Manager that knows ``users`` (name -> password); records every call."""
    users = users if users is not None else {"gaobo": GOOD}
    calls = []

    def request(base_url, path, token="", method="GET", payload=None, timeout=0):
        calls.append((path, method))
        if path == "/auth/login":
            if users.get(payload["username"]) != payload["password"]:
                raise ManagerApiError('HTTP 500 {"detail":"Bad credentials"}', 500)
            return {"accessToken": f"access-{payload['username']}"}
        if path == "/auth/me":
            if not token.startswith(("access-", "durable-")):
                raise ManagerApiError("HTTP 401", 401)
            return {"username": token.split("-", 1)[1], "role": "ADMIN", "enabled": True}
        if path == "/tokens":
            return {"token": "durable-" + token.split("-", 1)[1], "id": "t1"}
        raise AssertionError(path)

    return request, calls


class VerifyLoginTests(unittest.TestCase):
    def setUp(self):
        self.provider = PlatformProvider({"platform": {"managerBaseUrl": "http://m", "username": "gaobo"}})

    def test_a_real_login_confirms_the_user(self):
        request, calls = fake_manager()
        with mock.patch("api.platform.manager_request_json", side_effect=request), \
                mock.patch.object(PlatformProvider, "_manager_password", return_value=GOOD):
            login = self.provider.verify_login()
        self.assertEqual(login["username"], "gaobo")
        self.assertEqual(calls[0], ("/auth/login", "POST"))

    def test_bad_credentials_are_reported_plainly(self):
        request, _calls = fake_manager()
        with mock.patch("api.platform.manager_request_json", side_effect=request):
            with self.assertRaises(MonitorError) as caught:
                self.provider.verify_login(password="wrong")
        self.assertIn("用户名或密码错误", str(caught.exception))
        self.assertNotIn("wrong", str(caught.exception))

    def test_no_saved_password_fails_instead_of_trusting_the_token(self):
        with mock.patch.object(PlatformProvider, "_manager_password", return_value=""):
            status = self.provider.connection_status()
        self.assertFalse(status["ok"])
        self.assertIn("未保存", status["error"])


class SaveCredentialsTests(SchedulerTestCase):
    def setUp(self):
        super().setUp()
        self.service = SchedulerService(self.config)
        self.service.settings = SettingsStore(self.root / ".state" / "settings.json")
        self.addCleanup(self.service.stop)
        self.keychain = {"solo-manager-password": "old-pass"}
        request, self.calls = fake_manager()
        for target in (
            mock.patch("api.platform.manager_request_json", side_effect=request),
            mock.patch("api.service.keychain_read", side_effect=lambda s: self.keychain.get(s, "")),
            mock.patch("api.platform.keychain_read", side_effect=lambda s: self.keychain.get(s, "")),
            mock.patch("api.common.keychain_write", side_effect=self.keychain.__setitem__),
            mock.patch("api.platform.keychain_write", side_effect=self.keychain.__setitem__),
            mock.patch("api.platform.atomic_write_json", create=True),
            mock.patch("api.common.atomic_write_json"),
            mock.patch.object(PlatformProvider, "_resolve_manager_connection",
                              side_effect=lambda pb: ("http://m", self.keychain.get("solo-manager-token", ""), "")),
            mock.patch.object(PlatformProvider, "_modules", return_value=(None, None)),
        ):
            target.start()
            self.addCleanup(target.stop)

    def test_a_wrong_password_is_not_saved(self):
        with self.assertRaises(MonitorError) as caught:
            self.service.save_manager_credentials(base_url="http://m", username="gaobo", password="typo")
        self.assertIn("未保存", str(caught.exception))
        self.assertEqual(self.keychain["solo-manager-password"], "old-pass")

    def test_a_good_password_is_saved_and_a_token_issued_for_that_user(self):
        result = self.service.save_manager_credentials(base_url="http://m", username="gaobo", password=GOOD)
        self.assertTrue(result["passwordSaved"])
        self.assertTrue(result["settings"]["manager"]["passwordSaved"])
        self.assertEqual(self.keychain["solo-manager-password"], GOOD)
        self.assertEqual(self.keychain["solo-manager-token"], "durable-gaobo")
        self.assertTrue(result["connection"]["ok"])
        self.assertEqual(result["connection"]["token"]["username"], "gaobo")
        self.assertEqual(self.config["platform"]["username"], "gaobo")
        self.assertNotIn(GOOD, repr(result))

    def test_a_token_of_another_user_is_flagged(self):
        self.keychain["solo-manager-password"] = GOOD
        self.keychain["solo-manager-token"] = "durable-admin"
        self.config.setdefault("platform", {}).update(managerBaseUrl="http://m", username="gaobo")
        status = self.service.platform.connection_status()
        self.assertTrue(status["login"]["ok"])
        self.assertFalse(status["token"]["ok"])
        self.assertIn("admin", status["warning"])

    def test_testing_does_not_save(self):
        result = self.service.test_manager_login(base_url="http://m", username="gaobo", password=GOOD)
        self.assertTrue(result["ok"])
        self.assertEqual(self.keychain["solo-manager-password"], "old-pass")


class PromptUserTests(unittest.TestCase):
    def test_prompt_names_the_configured_user(self):
        prompt = render_auto_trigger_prompt("", None, manager_username="gaobo")
        self.assertIn("gaobo 对应的有效登录态", prompt)
        self.assertNotIn("admin", prompt)
        self.assertIn("admin 对应", render_auto_trigger_prompt("", None))

    def test_shipped_config_template_uses_the_variable(self):
        import json

        template = json.loads((APP / "config.json").read_text(encoding="utf-8"))["automation"]["promptTemplate"]
        self.assertIn("{{manager_username}}", template)
        self.assertNotIn("admin 对应", template)


if __name__ == "__main__":
    unittest.main()
