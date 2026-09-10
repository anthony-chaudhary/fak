# Vulkan KV copy-on-write forks

<!-- fak-compute-key: vulkan-kv-contiguous-cow-fork -->

Tracking: [fak #12400](https://github.com/anthony-chaudhary/fak/issues/12400).
Observed 2026-09-09 against public source `816506b53c263edd44e4b2a151ac80cac3b44031`.

## Current state

At the baseline revision, `vulkanKV.Clone` allocates and copies every populated `K`, `Kraw`, and
`V` layer buffer. A session fork therefore performs device allocations and D2D
copies proportional to the entire cached prefix before either branch diverges.
The active Qwen3.8 sequence path then reserves and appends directly into these
contiguous buffers, and attention consumes `K` and `V` as contiguous device
storage. This makes the eager clone both an active serving cost and the correct
seam for a bounded storage change.

The replacement shares each existing contiguous device allocation through a
small backing object with a reference count and a written high-water mark. A
fork copies only slice metadata and positions. While holding the existing global
`vulkanMu`, the first branch may append into unused capacity at or beyond the
backing high-water mark without copying; that append advances the high-water
mark. Any branch whose write range overlaps bytes above its own visible length
must first detach into a private contiguous allocation. Reallocation, eviction,
and any rewrite also detach when the backing is shared. Releasing a cache drops
one reference and frees the Vulkan allocation exactly once at the final release.

This is contiguous-buffer copy-on-write with segment-aware high-water ownership.
It is not a paged KV cache, page table, radix-tree policy, or proof that arbitrary
prefix pages can be independently shared. Paged prefix storage remains future
work.

## Core through-line

Existing populated contiguous Vulkan KV buffers -> metadata-only fork -> shared
immutable prefix -> first non-conflicting append may consume untouched backing
capacity -> conflicting writer detaches once -> each branch continues with
independent contiguous `K`, `Kraw`, and `V` views -> eviction and teardown affect
only the selected branch -> final reference releases each device allocation once.

Keep all backing ownership and high-water transitions under `vulkanMu`. Update
both ordinary `growAppend` and the Qwen3.8 sequence reserve/append path so neither
can mutate or free shared storage without applying the same detach rule.

## Gold-plating boundary

- Do not introduce paged allocation, page tables, radix policy, prefix matching,
  allocator replacement, or scheduler changes.
- Do not change tensor layout or the contiguous pointer contract consumed by
  Qwen3.8 attention.
- Do not include recurrent GDN state cloning; those tensor clones remain
  unchanged and must be reported as a remaining fork cost.
- Do not claim Halo latency, memory savings, or default-promotion credit from the
  software witness. Warm-prefix timing and memory effects require matched physical
  appliance evidence.

## Done condition

- [ ] `vulkanKV.Clone` shares populated `K`, `Kraw`, and `V` allocations and
      performs zero device allocation and zero D2D bytes for eight forks.
- [ ] A safe first append into untouched capacity stays shared and advances the
      backing high-water mark without D2D copying.
- [ ] A second divergent writer detects overlap, detaches once, and preserves the
      exact prefix and both divergent continuations.
- [ ] Growth beyond capacity detaches or transfers ownership without freeing an
      allocation still referenced by another cache.
- [ ] Eviction and rewrite detach from shared storage and leave sibling positions
      and bytes unchanged.
- [ ] Freeing forks in either order releases every backing once, never frees it
      early, and tolerates repeated `Free` calls.
- [ ] Existing contiguous `KeysView`, `ValuesView`, Qwen3.8 attention, and
      resident-byte accounting semantics remain valid.
- [ ] Recurrent tensor clones are unchanged and explicitly remain outside this
      leaf.

## Witness

### Verification status (2026-09-09)

The default Vulkan clone implementation now uses shared backing. The independent
regression executed successfully on both Radeon RX 7600 and Radeon 8060S
(RADV STRIX_HALO), including first-writer append, sibling detachment, eviction,
attention parity, and physical allocation cleanup after draining the recycle pool.
The existing Qwen sequence geometric-reservation and output-parity test also passed.
The baseline implementation fails the same fork contract by eagerly copying device
buffers. This is component correctness and copy-elimination evidence; matched
end-to-end latency, recurrent-state cloning, and paged storage remain outside
the completed implementation. Tracking issue #12400 stays open for its broader
physical promotion requirements.

### Verifiable Witness

Run on a host with a real Vulkan device, Vulkan SDK, and working cgo C++
toolchain:

```text
FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/compute -run '^TestVulkanKVCopyOnWriteFork$' -count=1 -v
```

The test must execute rather than skip. It creates a populated cache with spare
capacity, records Vulkan allocation/D2D/free counters, creates eight forks,
checks zero allocation and zero D2D work during the fork operation, exercises a
safe first append and a conflicting second writer, compares device readback for
prefix and divergent suffix parity, evicts one branch, and releases branches in
adversarial order while asserting one final release per backing.

This is the deterministic software and physical-device correctness witness. A
separate matched Halo baseline/candidate run must report absolute warm-prefix
native prefill time, computed suffix tokens, cached tokens, reuse ratio, and
memory observations over repeated trials before any performance claim.

## Adversarial review cases

- Parent appends first versus child appends first; global serialization must not
  make ownership depend on a preferred branch.
- Two siblings begin at the same visible length: only one may claim untouched
  capacity, and the other must detach before overwriting that suffix.
- A multi-token append straddles the backing high-water mark; overlap of any byte
  requires detachment before the write.
- `K`, `Kraw`, and `V` detach consistently when one reserve or append fails; no
  mixed branch may retain partially shared, partially overwritten logical rows.
- Exact-capacity append, geometric growth, zero-length cache, and empty-layer
  buffers preserve pointer, length, capacity, and reference invariants.
- Eviction of a prefix, middle span, or tail in one fork cannot change sibling
  bytes, positions, RoPE reconstruction, or visible lengths.
- Free parent before children, children before parent, and repeat `Free`; no
  use-after-free, leak, negative reference count, or double free is allowed.
- Views created before a detach must not outlive ownership in a way that exposes
  freed storage; document or enforce the existing view lifetime boundary.
- Backend/device-loss cleanup must release bookkeeping without issuing duplicate
  frees against an invalid Vulkan allocation.

## Likely files

- `internal/compute/vulkan_kv_vulkan.go`
- `internal/compute/vulkan_qwen35_sequence.go`
- `internal/compute/vulkan_kv_cow_vulkan_test.go`

## Lane

```routing
lane: compute
paths: internal/compute/vulkan_kv_vulkan.go, internal/compute/vulkan_qwen35_sequence.go, internal/compute/vulkan_kv_cow_vulkan_test.go
expected_steps: 6
```
