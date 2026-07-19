from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cavo_bot.config import Settings


class SettingsTests(unittest.TestCase):
    def test_private_mode_requires_an_allowlist(self) -> None:
        env = {"TELEGRAM_BOT_TOKEN": "token", "SHEET_ID": "sheet"}
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(RuntimeError, "ALLOWED_USER_IDS"),
        ):
            Settings.from_env()

    def test_admins_become_default_allowlist(self) -> None:
        env = {
            "TELEGRAM_BOT_TOKEN": "token",
            "SHEET_ID": "sheet",
            "ADMIN_USER_IDS": "10,20",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.allowed_user_ids, frozenset({10, 20}))
        self.assertFalse(settings.allow_public)
        self.assertTrue(settings.drop_pending_updates)

    def test_invalid_boolean_is_rejected(self) -> None:
        env = {
            "TELEGRAM_BOT_TOKEN": "token",
            "SHEET_ID": "sheet",
            "ADMIN_USER_IDS": "10",
            "ALLOW_PUBLIC": "sometimes",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(RuntimeError, "true or false"),
        ):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()
