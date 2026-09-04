# End-to-End Audit: Key Handling Across Connector, Client, and Server

**Date:** 2026-03-03
**Scope:** cama-connector (`cama_module/cama_storage.py`, `patches/cache_controller.py`), cama-client (`client.py`, `rdma_client.py`, `protocol.py`), cama-server (`dispatch.go`, `shard_ops.go`, `swisstable.go`, `hash.go`)

## Verdict: Core flow is SOUND

The key construction, wire encoding, server-side hashing, and lookup logic are all correct. Keys SET and TESTed for the same logical page produce byte-identical wire representations, and the server's Swiss table lookup matches on both wyhash and key length, eliminating false positives.

---

## Trace: SET Path

```
SGLang radix tree
  → compute_node_hash_values()     SHA256 chain → ["a1b2...", "c3d4..."]
  → write_backup_storage()          passes node.hash_value to cache_controller
  → cache_controller.write_storage() → batch_set_v1(keys=["a1b2...", "c3d4..."])

cama_storage.batch_set_v1:
  → _apply_tag(keys)                "tag_a1b2...", "tag_c3d4..."
  → _batch_preprocess(keys, indices)
    → _get_mha_buffer_meta()        "tag_a1b2..._0_0_k", "tag_a1b2..._0_0_v", ...
  → _put_batch_zero_copy(key_strs, ptrs, sizes)
    → conn.set(key_str, sgl)

cama_client (TCP or RDMA):
  → protocol.encode_kv_body(key.encode(), value)
    → struct.pack("<H", len(key_bytes)) + key_bytes + struct.pack("<I", len(value)) + value
  → send over wire

cama-server dispatch.go HandleSet:
  → protocol.DecodeKVBody(body, flags)  → key []byte, value []byte
  → keyHash = index.KeyHash(key)         wyhash(key_bytes)
  → Manager.Route(keyHash)               pick shard
  → shard.Submit(OpSet{Key, KeyHash, Value})

shard_ops.go handleSet:
  → allocate key+value in slab
  → idx.Insert(keyHash, Entry{KeyHash, KeyLen, offsets...})
    → Swiss table: H1(keyHash) for slot, H2(keyHash) for ctrl byte
```

## Trace: EXISTS Path

```
Scheduler._prefetch_kvcache:
  → last_hash = req.last_host_node.get_last_hash_value()   node.hash_value[-1]
  → new_input_tokens = req.fill_ids[matched_len:]
  → prefetch_from_storage(req_id, ..., new_input_tokens, last_hash)

cache_controller._storage_hit_query:
  → for each page in new_input_tokens:
      last_hash = get_hash_str(page_tokens, last_hash)     SAME SHA256 function
      batch_hashes.append(last_hash)
  → storage_backend.batch_exists(batch_hashes)

cama_storage.batch_exists:
  → inline tag: f"{tag}_{key}" for each key
  → suffix: f"{key}_{mha_suffix}_k", f"{key}_{mha_suffix}_v"   SAME format as SET
  → short-circuit: conn.exists(query_keys[0])
  → full check: _batch_exist(query_keys) → thread pool → conn.exists(k) for each

cama_client (TCP or RDMA):
  → protocol.encode_key_body(key.encode())
    → struct.pack("<H", len(key_bytes)) + key_bytes           SAME key encoding prefix
  → send OP_TEST over wire

cama-server dispatch.go HandleTest:
  → submitSingleKeyOp(body, OpTest)
    → protocol.DecodeKeyBody(body)      → key []byte         SAME decode
    → keyHash = index.KeyHash(key)       wyhash(key_bytes)   SAME hash
    → Manager.Route(keyHash)             SAME routing
    → shard.Submit(OpTest{Key, KeyHash})

shard_ops.go handleTest:
  → idx.Lookup(keyHash, uint16(len(key)))
    → Swiss table: match e.KeyHash == keyHash AND e.KeyLen == keyLen
  → return Found: true/false
```

---

## Findings

### PASS: Key bytes are identical for SET and TEST

Both paths produce the same key string `"{tag}_{sha256_hex}_{tp}_{pp}_k"` and encode it identically on the wire (`keyLen(uint16 LE) + key_bytes`). The server decodes and hashes them the same way.

### PASS: SHA256 chain is deterministic across backup and prefetch

Both paths use `get_hash_str()` from `hicache_storage.py` with the same chaining logic. The chain seed at the boundary (`node.hash_value[-1]`) is the same value used by both `compute_node_hash_values` (backup) and `_storage_hit_query` (prefetch).

### PASS: Return code handling is correct

```python
# CAMA backend:   _RC.EXISTS_FOUND=1,  _RC.EXISTS_MISSING=0
# PrisKV backend:  _RC.EXISTS_FOUND=0,  _RC.EXISTS_MISSING=1
```

All comparisons use `_RC.EXISTS_FOUND` / `_RC.EXISTS_MISSING` — never hardcoded ints. The `_RC` class is set at import time based on which backend is available.

### PASS: Swiss table dual match prevents false positives/negatives

```go
if e.KeyHash == keyHash && e.KeyLen == keyLen { return e, idx, true }
```

Since key strings are fixed-format (same tag length + 64-char hex + same suffix), both `keyHash` and `keyLen` will match if and only if the key bytes are identical.

### PASS: Wire encoding is symmetric

```python
# Client encode (Python):
struct.pack("<H", len(key)) + key    # uint16 LE key length + key bytes

# Server decode (Go):
keyLen = binary.LittleEndian.Uint16(body[0:2])
key = body[2 : 2+keyLen]
```

No endianness or padding issues.

### PASS: No string encoding issues

- `hashlib.sha256().hexdigest()` always returns lowercase hex ASCII
- Python `.encode()` defaults to UTF-8 (all key chars are ASCII, so UTF-8 = raw bytes)
- Go `[]byte` treats the key as raw bytes — no character encoding transformation

---

## Issues Found

### Issue 1: Tag Application Code Duplication (LOW risk)

**Location:** `cama_storage.py` lines 563-567 vs 988-989

`batch_set_v1` calls `_apply_tag()`:
```python
keys = self._apply_tag(keys)  # line 791
```

`batch_exists` does it inline:
```python
keys = [f"{self.extra_backend_tag}_{key}" for key in keys]  # line 989
```

Both produce the same result today, but if `_apply_tag` is modified (e.g., different separator, validation), `batch_exists` won't pick up the change.

**Recommendation:** Replace the inline tag application in `batch_exists` with `keys = self._apply_tag(keys)`.

### Issue 2: EXISTS_ERROR Treated as Miss in Dedup (LOW risk)

**Location:** `cama_storage.py` lines 820-828

```python
if exist_results[i] == _RC.EXISTS_FOUND:
    merged_results[i] = _RC.SET_OK    # skip write (dedup)
else:
    # EXISTS_MISSING or EXISTS_ERROR — write the key
    set_keys.append(key_strs[i])
```

If `_batch_exist` times out or errors on some keys, those keys get written even if they might already exist. This is **safe** because:
- CAMA SET is idempotent (overwrites the same value)
- The alternative (skipping writes on error) would risk data loss

**Recommendation:** Log a warning when writing keys due to EXISTS_ERROR specifically.

### Issue 3: Redundant First-Key Check (NEGLIGIBLE)

**Location:** `cama_storage.py` lines 1006-1023

`batch_exists` checks `query_keys[0]` via `conn.exists()` for short-circuit, then rechecks it in `_batch_exist(query_keys)` which submits ALL keys to the thread pool. The first key is checked twice.

**Impact:** One extra RPC per batch when the first key exists. Negligible in practice since the short-circuit saves N-1 RPCs when hit rate is ~0%.

**Recommendation:** Could pass `query_keys[1:]` to `_batch_exist` and prepend the known result, but the complexity isn't worth it.

### Issue 4: No Key Format Validation (LOW risk)

Neither client nor server validates the expected key structure. If a code change produces keys with different format (e.g., uppercase hex, different separator), lookups silently fail.

**Recommendation:** Add a debug-mode assertion in `batch_set_v1` and `batch_exists`:
```python
assert len(sha256_hex) == 64 and sha256_hex == sha256_hex.lower(), f"Bad hash: {sha256_hex}"
```

---

## Summary

| Check | Result |
|-------|--------|
| Key bytes identical for SET and TEST | PASS |
| SHA256 chain deterministic | PASS |
| Return code handling correct | PASS |
| Wire encoding symmetric | PASS |
| No string encoding issues | PASS |
| Swiss table lookup sound | PASS |
| Tag application consistent | PASS (with code duplication risk) |
| Suffix application consistent | PASS |
| Key length within limits | PASS (< 100 bytes, limit 65535) |
| Short-circuit logic correct | PASS |
| Thread pool ordering correct | PASS |
| Error handling in dedup | PASS (safe fallback) |
