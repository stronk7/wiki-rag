#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Tests for wiki_rag.mcp_server.auth."""

import unittest

from unittest.mock import patch

from fastmcp.server.auth.auth import AccessToken

from wiki_rag.mcp_server.auth import WikiRagTokenVerifier


class TestWikiRagTokenVerifier(unittest.IsolatedAsyncioTestCase):
    """Tests for WikiRagTokenVerifier.verify_token()."""

    def setUp(self):
        """Create a fresh verifier for each test."""
        self.verifier = WikiRagTokenVerifier()

    async def test_valid_token_in_local_env_returns_access_token(self):
        with patch("wiki_rag.mcp_server.auth.check_token_with_local_env", return_value=True):
            with patch("wiki_rag.mcp_server.auth.check_token_with_service", return_value=False):
                result = await self.verifier.verify_token("valid-token")
        self.assertIsInstance(result, AccessToken)
        self.assertEqual("valid-token", result.token)
        self.assertEqual("wiki-rag-client", result.client_id)

    async def test_invalid_token_returns_none(self):
        with patch("wiki_rag.mcp_server.auth.check_token_with_local_env", return_value=False):
            with patch("wiki_rag.mcp_server.auth.check_token_with_service", return_value=False):
                result = await self.verifier.verify_token("invalid-token")
        self.assertIsNone(result)

    async def test_empty_token_returns_none(self):
        result = await self.verifier.verify_token("")
        self.assertIsNone(result)

    async def test_whitespace_token_returns_none(self):
        result = await self.verifier.verify_token("   ")
        self.assertIsNone(result)

    async def test_token_validated_via_remote_service_returns_access_token(self):
        with patch("wiki_rag.mcp_server.auth.check_token_with_local_env", return_value=False):
            with patch("wiki_rag.mcp_server.auth.check_token_with_service", return_value=True):
                result = await self.verifier.verify_token("remote-token")
        self.assertIsInstance(result, AccessToken)
        self.assertEqual("remote-token", result.token)

    async def test_local_auth_not_checked_again_when_already_authorised(self):
        """When local check passes, the remote service must not be called."""
        with patch("wiki_rag.mcp_server.auth.check_token_with_local_env", return_value=True) as mock_local:
            with patch("wiki_rag.mcp_server.auth.check_token_with_service") as mock_remote:
                result = await self.verifier.verify_token("local-token")
        self.assertIsInstance(result, AccessToken)
        mock_local.assert_called_once_with("local-token")
        mock_remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
