"""RDMA client constants and WC status helpers."""

from __future__ import annotations

import logging

try:
    from l3_client._l3_rdma import DEFAULT_SEND_BUF_SIZE as _DEFAULT_SEND_BUF_SIZE
    from l3_client._l3_rdma import DEFAULT_RECV_BUF_SIZE as _DEFAULT_RECV_BUF_SIZE
    from l3_client._l3_rdma import DEFAULT_READ_BUF_SIZE as _DEFAULT_READ_BUF_SIZE
except ImportError:
    _DEFAULT_SEND_BUF_SIZE = 16 * 1024 * 1024  # 16 MB fallback
    _DEFAULT_RECV_BUF_SIZE = 16 * 1024 * 1024  # 16 MB fallback
    _DEFAULT_READ_BUF_SIZE = 32 * 1024 * 1024  # 32 MB fallback

# Preserve logger name for backward compatibility with existing log filters
logger = logging.getLogger("l3_client.rdma_client")

# Python-side WC status name fallback (used when _cama_rdma.wc_status_name unavailable)
_WC_STATUS_NAMES = {
    0: "SUCCESS",
    1: "LOC_LEN_ERR",
    2: "LOC_QP_OP_ERR",
    3: "LOC_EEC_OP_ERR",
    4: "LOC_PROT_ERR",
    5: "WR_FLUSH_ERR",
    6: "MW_BIND_ERR",
    7: "BAD_RESP_ERR",
    8: "LOC_ACCESS_ERR",
    9: "REM_INV_REQ_ERR",
    10: "REM_ACCESS_ERR",
    11: "REM_OP_ERR",
    12: "RETRY_EXC_ERR",
    13: "RNR_RETRY_EXC_ERR",
    14: "LOC_RDD_VIOL_ERR",
    15: "REM_INV_RD_REQ_ERR",
    16: "REM_ABORT_ERR",
    17: "INV_EECN_ERR",
    18: "INV_EEC_STATE_ERR",
    19: "FATAL_ERR",
    20: "RESP_TIMEOUT_ERR",
    21: "GENERAL_ERR",
}

try:
    from l3_client._l3_rdma import wc_status_name as _wc_status_name
except ImportError:
    def _wc_status_name(status: int) -> str:
        return _WC_STATUS_NAMES.get(status, f"UNKNOWN({status})")
