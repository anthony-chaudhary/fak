# KV-Transport Governance: NIXL, Mooncake, LMCache integration

This doc describes the governance contract for external KV-transport systems (NIXL, Mooncake, LMCache) to report P/D disaggregation and KV-transfer events to fak for observability, invalidation, and trust governance.

**Scope:** fak **governs** KV-transport events but moves **zero KV bytes**. External systems handle the actual transport; fak provides the observability plane and invalidation governance.

## 1. Governance contract

External KV-transport systems emit events to fak by creating `cachemeta.Entry` records on the `PlaneKVTransfer` plane via `cachemeta.FromKVTransfer()`.

### 1.1 Event types

| Direction | Meaning | Example use |
|---|---|---|
| `KVOffload` | Offload KV span from HBM to a lower tier (DRAM/disk/remote) | Engine offloading idle cache to CPU memory |
| `KVRestore` | Restore a span from tier back to HBM | Engine re-materializing a span for a new request |
| `KVRoute` | Route a request to a replica holding the span | KV-aware router pinning request to cache-local worker |
| `KVMigrate` | Migrate span residency between instances | Live migration for load balancing or node drain |

### 1.2 Outcome semantics

| Outcome | Meaning | Governance action |
|---|---|---|
| `KVTransferOK` | Transfer succeeded | Record successful residency transition |
| `KVTransferMissed` | Span not found in tier | Log miss; may trigger recompute |
| `KVTransferFault` | Transfer error | Surface as `residency_fault`; quarantine if in DSA path |

## 2. Integration path for external systems

### 2.1 NIXL integration

NIXL moves KV point-to-point over RDMA/InfiniBand, RoCE/UCX, TCP, or NVMe-oF. To bridge NIXL events to fak:

```go
// In NIXL wrapper/adapter:
import "github.com/anthony-chaudhary/fak/internal/cachemeta"

func reportNIXLTransfer(ctx context.Context, spanID string, tokens int64, fromTier, toTier cachemeta.ResidencyTier, bytesMoved int64, outcome cachemeta.KVTransferOutcome) {
    entry := cachemeta.FromKVTransfer(cachemeta.KVTransfer{
        Direction:    cachemeta.KVMigrate,
        SpanDigest:   spanID,
        Tokens:       tokens,
        ModelID:      "model-name",
        TokenizerID:  "tokenizer-name",
        PositionMode: cachemeta.PositionPrefixAligned,
        FromTier:     fromTier,
        ToTier:       toTier,
        Owner:        "nixl",
        Lease:        "session-lease",
        Outcome:      outcome,
        BytesMoved:   bytesMoved,
    })
    // Emit entry to fak's cachemeta sink (HTTP, gRPC, or local write)
}
```

### 2.2 Mooncake integration

Mooncake exposes a KVCache-centric store with a Transfer Engine for RDMA/TCP/NVMe-oF. Mooncake events map as:

| Mooncake event | fak `Direction` |
|---|---|
| Prefill → decode KV transfer | `KVMigrate` |
| Distributed pool lookup | `KVRoute` |
| Remote KV materialization | `KVRestore` |

### 2.3 LMCache integration

LMCache supplies the disaggregated-prefill KV path for vLLM. LMCache offload/restore events map directly:

| LMCache operation | fak `Direction` |
|---|---|
| `lmcache.append()` → offload | `KVOffload` |
| `lmcache.lookup()` + materialize | `KVRestore` |

## 3. fak's governance responsibilities

1. **Observability:** Record all KV-transfer events with outcome metrics (`BytesMoved`)
2. **Invalidation:** Route `ExternalInvalidationDirective` to the owning system for cache reset
3. **Trust:** Apply admission verdict and taint to KV-transfer entries when in DSA paths
4. **No data plane:** fak does not move, serialize, or deserialize KV bytes

## 4. External system requirements

To integrate with fak's governance plane, an external KV-transport system must:

1. **Report span identity:** Emit a stable `SpanDigest` that identifies the KV span (e.g., hash of parent-hash + block tokens)
2. **Report tier:** Identify source (`FromTier`) and destination (`ToTier`) residency tiers
3. **Report outcome:** Distinguish OK/missed/fault outcomes so fak can surface faults
4. **Report bytes moved:** Populate `BytesMoved` for observability
5. **Handle invalidation:** Respond to cache-invalidation directives from fak

## 5. Example: vLLM + NIXL to fak bridge

A vLLM deployment with NIXL KV-connector can report transfer events to fak:

```go
// In vLLM adapter code (e.g., fak's vLLM adapter #40):
import "github.com/anthony-chaudhary/fak/internal/cachemeta"

func onNIXLTransferComplete(spanID string, tokens int, bytesMoved int64, success bool) {
    outcome := cachemeta.KVTransferOK
    if !success {
        outcome = cachemeta.KVTransferFault
    }
    entry := cachemeta.FromKVTransfer(cachemeta.KVTransfer{
        Direction:    cachemeta.KVMigrate,
        SpanDigest:   spanID,
        Tokens:       int64(tokens),
        ModelID:      vllmConfig.Model,
        TokenizerID:  vllmConfig.Tokenizer,
        FromTier:     cachemeta.TierGPU, // NIXL moves from source GPU
        ToTier:       cachemeta.TierRemote, // to target
        Owner:        "vllm+nixl",
        Outcome:      outcome,
        BytesMoved:   bytesMoved,
    })
    fakCachemetaSink.Emit(entry)
}
```

## 6. Status and roadmap

- **Shipped:** `FromKVTransfer()` seam in `internal/cachemeta/kvtransfer.go:54`
- **Shipped:** `KVTransferVerdict()` for fault/miss/hit classification at `kvtransfer.go:121`
- **Shipped:** `enginecache` invalidation client for external engines at `internal/enginecache/enginecache.go`
- **GAP:** Live wired integrations with NIXL/Mooncake/LMCache (this doc defines the contract; actual wiring is a later step)

---

**Parent issue:** #37 (Orchestrate external P/D disaggregation + govern KV-transport bridge)  
**Track:** Track A — RIDE (orchestrate + govern; do NOT fork engine internals)  
**Dual-track serving plan:** docs/serving/dual-track-serving-plan.md