#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Bearer token authentication verifier for the wiki-rag MCP server."""

import logging

from fastmcp.server.auth import TokenVerifier
from fastmcp.server.auth.auth import AccessToken

from wiki_rag.server.util import check_token_with_local_env, check_token_with_service

logger = logging.getLogger(__name__)


class WikiRagTokenVerifier(TokenVerifier):
    """Bearer token verifier for the wiki-rag MCP server.

    Validates tokens using the same dual-method approach as the HTTP server:
    local token list (AUTH_TOKENS env var) and/or remote service (AUTH_URL env var).
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify a bearer token against local and remote sources.

        Args:
            token: The bearer token string to validate.

        Returns:
            AccessToken if the token is valid, None otherwise.

        """
        if not token or not token.strip():
            logger.debug("Rejecting empty or whitespace token")
            return None

        authorised = check_token_with_local_env(token)

        if not authorised:
            authorised = check_token_with_service(token)

        if not authorised:
            logger.debug("Token validation failed: not found in local list or remote service")
            return None

        return AccessToken(
            token=token,
            client_id="wiki-rag-client",
            scopes=[],
            expires_at=None,
        )
