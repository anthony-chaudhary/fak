"""SGL (Scatter-Gather List) shim for TCP transport.

In PrisKV, SGL holds a registered RDMA memory region pointer for zero-copy
transfers. In TCP mode, we copy data through ctypes instead.
"""

import ctypes
import logging

logger = logging.getLogger(__name__)


class SGL:
    """Scatter-gather list wrapping a host memory pointer.

    In RDMA mode (PrisKV), this enables zero-copy DMA between GPU-host memory
    and the KV store. In TCP mode (this shim), we copy via ctypes.memmove.
    """

    def __init__(self, ptr: int, size: int, reg_handle: int = 1):
        self.ptr = ptr
        self.size = size
        self.reg_handle = reg_handle

    def to_bytes(self) -> bytes:
        """Copy data from the host pointer into Python bytes for TCP send."""
        buf = (ctypes.c_char * self.size)()
        ctypes.memmove(buf, self.ptr, self.size)
        return bytes(buf)

    def from_bytes(self, data: bytes) -> None:
        """Copy received TCP bytes into the host pointer."""
        n = min(len(data), self.size)
        if len(data) > self.size:
            logger.warning(
                "SGL.from_bytes: data truncated — received %d bytes but buffer is %d bytes. "
                "%d bytes discarded.", len(data), self.size, len(data) - self.size)
        ctypes.memmove(self.ptr, data, n)
