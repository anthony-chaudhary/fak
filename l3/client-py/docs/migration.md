# Migration from PrisKV

## Import Changes

| PrisKV | cama-client |
|---|---|
| `from priskv.priskv_client import PriskvClient` | `from cama_client import PriskvClient` |
| `import priskv; priskv.SGL(...)` | `from cama_client import SGL` |

## Behavioral Differences

| Behavior | PrisKV | cama-client |
|---|---|---|
| Default transport | RDMA only | RDMA first, TCP fallback |
| Default TCP port | 6379 | 18000 |
| Default RDMA port | — | 18001 |
| Batch ops (`mset`, `mget`, etc.) | Native pipelined batch | Loop-based (individual ops) |
| `exists()` return type | `bool` | `int` (`1` or `0`) |
| `mexists()` return type | `list[bool]` | `list[int]` (`1` or `0`) |
| `reg_memory()` in TCP mode | Not applicable | Returns `1` (no-op) |
| `get()` miss return value | Exception or None | `-1` (int) |
| Authentication | Password supported | Ignored (no auth) |

## Minimal Migration Example

```python
# Before (PrisKV)
from priskv.priskv_client import PriskvClient
from priskv import SGL
client = PriskvClient("10.0.0.1", 6379)

# After (CAMA)
from cama_client import PriskvClient, SGL
client = PriskvClient("10.0.0.1", 18000)

# Everything else stays the same
client.setstr("key", "value")
sgl = SGL(ptr=buf_ptr, size=buf_size)
client.set("tensor_key", sgl)
```
