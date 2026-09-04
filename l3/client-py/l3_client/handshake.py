"""Connection-time handshake for CAMA clients.

Shared by both TCP and RDMA transports. The handshake is a one-time
JSON exchange that communicates version info between client and server.
"""

import json
import logging
import warnings
from typing import Callable

from l3_client import protocol
from l3_client._version import (
    API_VERSION,
    MIN_SERVER_VERSION,
    __version__,
    compare_versions,
)

logger = logging.getLogger(__name__)


def perform_handshake(
    roundtrip_fn: Callable[[int, bytes, int], protocol.Message],
    transport: str,
) -> dict | None:
    """Perform a version handshake with the CAMA server.

    Args:
        roundtrip_fn: A callable with signature (opcode, body, flags) -> Message.
            This is the client's _roundtrip method.
        transport: Transport name for logging ("tcp" or "rdma").

    Returns:
        Server info dict on success, or None if the server does not
        support the handshake (legacy server).
    """
    request = {
        "api_version": API_VERSION,
        "client_version": __version__,
        "client_type": "python",
        "transport": transport,
        "capabilities": ["batch_ops", "rdma_read"],
    }
    body = json.dumps(request).encode()

    try:
        resp = roundtrip_fn(protocol.OP_HANDSHAKE, body, 0)
    except RuntimeError as exc:
        # Server returned RespError — could be legacy server (unknown opcode)
        # or an incompatibility error with JSON body
        err_msg = str(exc)
        # Try to parse JSON from the error for structured incompatibility info
        try:
            # Extract the body portion after "CAMA error: "
            json_part = err_msg.split("CAMA error: ", 1)[-1]
            err_data = json.loads(json_part)
            if err_data.get("status") == "incompatible":
                reason = err_data.get("reason", "unknown")
                warnings.warn(
                    f"CAMA handshake: server reports incompatibility: {reason}. "
                    f"Connection will proceed but may not work correctly.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning(
                    "[%s] handshake incompatible: %s", transport, reason
                )
                return err_data
        except (json.JSONDecodeError, IndexError):
            pass

        # Legacy server — unknown opcode error
        logger.warning(
            "[%s] server does not support handshake (legacy server), "
            "continuing without version check",
            transport,
        )
        return None

    # Parse server response
    try:
        server_info = json.loads(resp.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("[%s] handshake response was not valid JSON", transport)
        return None

    server_version = server_info.get("server_version", "unknown")
    logger.debug(
        "[%s] handshake OK: server=%s, client=%s",
        transport,
        server_version,
        __version__,
    )

    # Warn if server is too old
    if server_version != "unknown" and compare_versions(server_version, MIN_SERVER_VERSION) < 0:
        warnings.warn(
            f"CAMA server version {server_version} is older than minimum "
            f"{MIN_SERVER_VERSION} for cama-client {__version__}. "
            f"Please upgrade the server.",
            UserWarning,
            stacklevel=3,
        )
        logger.warning(
            "[%s] server %s < minimum %s",
            transport,
            server_version,
            MIN_SERVER_VERSION,
        )

    return server_info
