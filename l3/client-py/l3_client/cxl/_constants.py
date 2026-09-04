"""CXL client constants."""

from __future__ import annotations

import logging

logger = logging.getLogger("l3_client.cxl")

DEFAULT_CXL_PORT = 18002
DEFAULT_DEVDAX_PATH = "/dev/dax0.0"

# Protocol opcodes (must match server)
OP_CXL_REGION_MAP = 0x37
RESP_CXL_REGION_MAP = 0xF5
OP_CXL_READ_READY = 0x38
