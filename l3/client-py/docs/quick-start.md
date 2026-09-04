# Quick Start

## String Operations

```python
from cama_client import PriskvClient

client = PriskvClient("127.0.0.1", 18000)

# Store and retrieve strings
client.setstr("model:name", "llama-70b")
name = client.getstr("model:name")   # "llama-70b"

# Check existence and delete
client.exists("model:name")           # 1
client.delete("model:name")           # 0
client.exists("model:name")           # 0

client.close()
```

## SGL-Based Operations (Buffer Transfers)

```python
import ctypes
from cama_client import PriskvClient, SGL

client = PriskvClient("127.0.0.1", 18000)

# Create a buffer and wrap it in an SGL
buf = ctypes.create_string_buffer(4096)
ctypes.memmove(buf, b"KV cache page data...", 20)
sgl = SGL(ptr=ctypes.addressof(buf), size=4096)

# Store from SGL
client.set("page:0:0:k", sgl, ttl_ms=60000)

# Retrieve into SGL
out_buf = ctypes.create_string_buffer(4096)
out_sgl = SGL(ptr=ctypes.addressof(out_buf), size=4096)
rc = client.get("page:0:0:k", out_sgl)  # 0 = found, -1 = miss

client.close()
```

## RDMA with Memory Registration

```python
import ctypes
from cama_client import PriskvClient, SGL

client = PriskvClient("10.0.0.1", 18001)  # RDMA port

# Allocate and register a large buffer for zero-copy RDMA.
# Pass buf= to hold a GC reference — prevents the NIC from reading freed memory
# mid-RDMA-Read (one-sided DMA, no CPU involvement on either side).
buf = ctypes.create_string_buffer(64 * 1024 * 1024)  # 64 MB
handle = client.reg_memory(ctypes.addressof(buf), len(buf), buf=buf)

# Create SGL with the registration handle
sgl = SGL(ptr=ctypes.addressof(buf), size=5 * 1024 * 1024, reg_handle=handle)

# GET now uses RDMA Read directly into the registered buffer (zero-copy)
rc = client.get("kv_page_key", sgl)

# Clean up
client.dereg_memory(handle)
client.close()
```
