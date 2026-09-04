# Hashing and Key Handling — End-to-End Reference

## Overview

CAMA uses a **two-layer hashing** design:

| Layer | Function | Algorithm | Purpose |
|-------|----------|-----------|---------|
| 1 — Application | Token sequence → cache key string | SHA256 (chained) | Position-aware, content-addressable page identification |
| 2 — Server index | Key string bytes → slot index | wyhash v4 | O(1) Swiss table lookup |

Layer 1 runs in Python (SGLang). Layer 2 runs in Go (cama-server). They are **independent** — the server treats the SHA256 hex string as opaque key bytes and wyhash-es them for internal indexing.

---

## Layer 1: SHA256 Chained Hashing

### Core function: `get_hash_str`

Location: `sglang/python/sglang/srt/mem_cache/hicache_storage.py`

```python
def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))      # 32 raw bytes from parent
    for t in token_ids:
        hasher.update(t.to_bytes(4, "little", signed=False))  # 4 bytes per token
    return hasher.hexdigest()                           # 64-char lowercase hex
```

**Properties:**
- **Deterministic**: same tokens + same prior hash → same output, always.
- **Position-dependent**: chaining the parent hash means identical tokens at different positions in a prompt produce different hashes.
- **Fixed-length output**: always 64 lowercase hex characters.
- **Collision-resistant**: SHA256's 256-bit output space makes accidental collisions negligible.

### Chain structure

For a prompt split into pages P0, P1, P2, ...:

```
H0 = SHA256(                     tokens_P0)   # root page — no prior hash
H1 = SHA256(bytes.fromhex(H0) || tokens_P1)   # chains from H0
H2 = SHA256(bytes.fromhex(H1) || tokens_P2)   # chains from H1
...
```

Each hash encodes the **full prefix** up to that page. Two prompts sharing the same first N pages will share hashes H0..H(N-1) and diverge at H(N).

---

## Two Paths Compute Hashes

### Path 1: Backup (write to CAMA)

**Flow:** Radix tree node → `compute_node_hash_values()` → `write_backup_storage()` → `batch_set_v1()`

Location: `sglang/python/sglang/srt/mem_cache/radix_cache.py`

```python
def compute_node_hash_values(node, page_size):
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]   # last page hash of parent

    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start:start + page_size]
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val   # chain forward

    return hash_values
```

The resulting `node.hash_value` list is stored on the radix tree node and passed directly to `cache_controller.write_storage()` → `batch_set_v1()`.

### Path 2: Prefetch (read from CAMA)

**Flow:** Scheduler → `prefetch_from_storage()` → `_storage_hit_query()` → `batch_exists()`

Location: `cama-connector/patches/cache_controller.py`

```python
def _storage_hit_query(self, operation):
    last_hash = operation.last_hash   # from radix tree match boundary
    tokens_to_fetch = operation.token_ids

    for start in range(0, len(tokens_to_fetch), page_size * storage_batch_size):
        batch_tokens = tokens_to_fetch[start:end]
        for i in range(0, len(batch_tokens), page_size):
            last_hash = self.get_hash_str(batch_tokens[i:i+page_size], last_hash)
            batch_hashes.append(last_hash)
        hit_page_num = self.storage_backend.batch_exists(batch_hashes, ...)
```

### Why they produce identical hashes

The **chain seed at the boundary** is the same in both paths:

1. **Backup**: Node X's hashes were computed starting from `node.parent.hash_value[-1]`.
2. **Prefetch**: The scheduler gets `last_hash = req.last_host_node.get_last_hash_value()` which returns `node.hash_value[-1]` — the same value that was the chain seed when computing the *next* node's hashes during backup.

Both paths:
- Call the **same `get_hash_str` function**
- Use the **same `page_size`** to chunk tokens
- Use the **same chain seed** at the boundary

→ Hashes are guaranteed to match.

---

## Key Construction

After hashing, the SHA256 hex string becomes part of a structured key:

```
{extra_backend_tag}_{sha256_hex}_{tp_rank}_{pp_rank}_{k|v}
```

Example: `"model-a_a1b2c3d4...f0f1f2f3_0_0_k"` (~80 bytes)

### SET path (batch_set_v1)

```python
keys = self._apply_tag(keys)           # prepend extra_backend_tag
key_strs, ptrs, sizes = self._batch_preprocess(keys, host_indices)
# _batch_preprocess → _get_mha_buffer_meta:
#   key_list.append(f"{key_}_{self.mha_suffix}_k")
#   key_list.append(f"{key_}_{self.mha_suffix}_v")
```

### EXISTS path (batch_exists)

```python
if self.extra_backend_tag is not None:
    keys = [f"{self.extra_backend_tag}_{key}" for key in keys]
# Then:
query_keys.append(f"{key}_{self.mha_suffix}_k")
query_keys.append(f"{key}_{self.mha_suffix}_v")
```

**Both paths produce identical key strings** for the same hash input.

---

## Layer 2: Server-Side Wyhash

### Wire protocol

Client encodes key as: `keyLen(uint16 LE) + key_bytes`

Both `encode_key_body()` (for EXISTS/TEST) and `encode_kv_body()` (for SET) use this identical prefix encoding.

### Server dispatch

```go
keyHash := index.KeyHash(key)      // wyhash v4 of the raw key bytes
sh := d.Manager.Route(keyHash)     // shard routing
sh.Submit(ShardOp{Key: key, KeyHash: keyHash, ...})
```

### Swiss table lookup

```go
func (t *Table) Lookup(keyHash uint64, keyLen uint16) (Entry, uint64, bool) {
    // H1(keyHash) → slot position
    // H2(keyHash) → 7-bit ctrl fingerprint
    // Match requires: e.KeyHash == keyHash AND e.KeyLen == keyLen
}
```

The dual match (hash + length) eliminates false positives from wyhash collisions.

---

## Correctness Guarantees

| Property | How it's ensured |
|----------|-----------------|
| Same key bytes for SET and TEST | Both paths apply tag + suffix identically, encode with same `encode_key_body`/`encode_kv_body` prefix |
| Deterministic hashing | SHA256 and wyhash are both deterministic |
| Position awareness | SHA256 chain seeds each page hash with its predecessor |
| No false positives in lookup | Swiss table matches on both `keyHash` AND `keyLen` |
| No case sensitivity issues | Python `hexdigest()` always returns lowercase hex |
| No encoding issues | Python `.encode()` → UTF-8, Go `[]byte` treats as raw bytes. SHA256 hex is pure ASCII. |

---

## Known Limitations and Risks

### 1. Code duplication in tag application
`batch_set_v1` calls `_apply_tag()`. `batch_exists` does inline `f"{tag}_{key}"`. If one changes without the other, keys won't match. **Recommendation:** Consolidate into `_apply_tag()` everywhere.

### 2. Error-as-miss in dedup
In `batch_set_v1`, if `_batch_exist()` returns `_EXISTS_ERROR` for a key (timeout/network issue), it's treated the same as `_EXISTS_MISSING` — the key gets written. This is **safe** (duplicates are idempotent on the server) but wastes bandwidth.

### 3. Redundant check of first key
`batch_exists` checks `query_keys[0]` in the short-circuit, then rechecks it in `_batch_exist(query_keys)` (all keys). Minor inefficiency, not a bug.

### 4. No runtime key format validation
Neither client nor server validates key structure. If hash format ever changes (e.g., different digest length), lookups silently fail.
