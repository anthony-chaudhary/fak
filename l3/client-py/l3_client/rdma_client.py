"""CAMA RDMA client — backward-compatible facade.

All public symbols remain importable from l3_client.rdma_client.
Implementation lives in the l3_client.rdma subpackage.

Usage:
    from l3_client.rdma_client import RDMAClient
    client = RDMAClient("10.0.0.1", port=18001)
"""

from l3_client.rdma._client import RDMAClient
from l3_client.rdma._pool import RDMAClientPool, _PooledConn

__all__ = ["RDMAClient", "RDMAClientPool", "_PooledConn"]
