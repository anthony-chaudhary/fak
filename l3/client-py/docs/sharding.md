# Server-Side Sharding

The CAMA server splits its keyspace into **shards** — independent slices that run in parallel with zero locking. Each shard is a self-contained unit with its own:

- **Swiss Table index** — hash map for key lookup
- **Slab allocator** — memory for key/value storage
- **Eviction engine** — W-TinyLFU, SIEVE, or LRU (configurable)
- **Lease table** — pin/lease protection
- **Metrics counters** — per-shard atomic stats
- **Ops channel** — buffered (cap 4096), async operation queue

## Key Routing

Every key deterministically maps to exactly one shard via hash masking:

```
keyHash = wyhash("user:123")        # deterministic 64-bit hash
shardID = keyHash & (numShards - 1) # fast bitmask (shards are power-of-2)
```

The client does not choose the shard — the server's dispatcher hashes the key and routes it. For batch operations like `MGET`, the dispatcher groups keys by target shard, fans out to each shard in parallel, then reassembles results in the original order.

## Concurrency Model

Each shard runs as **one goroutine pinned to one OS thread** (`runtime.LockOSThread`), reading operations from its buffered channel. Because only one goroutine ever touches a shard's state, there are no mutexes, no spinlocks, no contention. The default shard count is **8** (tuned for GPU nodes) — configurable via `num_shards`.

## Shards, NICs, and Clients Are Independent Axes

```
            ┌──────────── Shard Manager ────────────┐
            │  Shard 0  Shard 1  ...  Shard 15      │
            │  (32 GB)  (32 GB)       (32 GB)       │
            └────▲──────────▲──────────────▲────────┘
                 │          │              │
          hash-route   hash-route     hash-route
                 │          │              │
         ┌───────┴──┐  ┌───┴──────┐  ┌───┴──────┐
         │ TCP srv   │  │ RDMA srv │  │ RDMA srv │
         │ (eth0)    │  │ (mlx5_0) │  │ (mlx5_1) │
         └───────────┘  └──────────┘  └──────────┘
```

There is **one shard manager** shared by every transport server (TCP and RDMA). A request arriving on any NIC can touch any shard — the hash decides, not the NIC. Multiple clients hitting the same key always land on the same shard, where the single goroutine serializes their operations.

| Dimension | Shards | NICs | Clients |
|-----------|--------|------|---------|
| **Scales** | CPU parallelism | Network bandwidth | Concurrent users |
| **Default count** | 8 | Auto-detect RDMA devices | Unlimited |
| **Affinity** | None — fully independent axes | | |
| **Key routing** | Determined by hash | Irrelevant | Irrelevant |

Adding NICs gives more **network bandwidth**, not more shards. Adding shards gives more **CPU parallelism**, not more bandwidth. Adding clients increases **concurrent users** without affecting either.

## Configuration

```toml
num_shards = 16            # 0 = auto (default 16, tuned for GPU nodes)
max_memory_gb = 512        # split equally: 512 / 16 = 32 GB per shard
max_keys = 131072          # per-shard index capacity
eviction_policy = "wtinylfu"
```

See also: [Architecture](architecture.md) | [Configuration](configuration.md) | [Data Flow](data-flow.md)
