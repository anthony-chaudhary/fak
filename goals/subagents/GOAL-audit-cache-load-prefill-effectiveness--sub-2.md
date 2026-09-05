---
parent_goal: goals/GOAL-audit-cache-load-prefill-effectiveness.md
sub_step: 2_radix_kv_and_inkernel_prefill
witness: "go test -v ./internal/radixkv -run 'TestRadixKV_Basic|TestTree_Split' -count=1 && go test -v ./internal/agent -run 'TestInKernelPlannerPrefixReuse' -count=1"
target_files:
  - internal/radixkv/radixkv.go
  - internal/agent/inkernel_decode.go
  - internal/agent/inkernel_planner.go
---
# Sub-Goal Objective
Audit how the radix KV tree (`internal/radixkv`) discovers cached prefixes and how the InKernel agent runtime (`internal/agent/inkernel_decode.go`) loads them back into sessions for suffix-only prefill.

Specifically investigate:
1. Radix prefix discovery & tiers in `internal/radixkv/radixkv.go`:
   - How `Lookup` and `LookupSnapshotTieredContext` walk the compressed trie.
   - How multi-tier snapshots work: Device L1, Host DRAM L2 (`host_l2.go`), Remote HTTP L3 (`remote_l3.go`).
   - How edge splitting and node truncation preserve exact prefix KV.
2. InKernel decode loop loading & prefill orchestration (`internal/agent/inkernel_decode.go`):
   - How `matchedKV` or `matchedSnapshot` is restored into `model.Session`.
   - The exact-hit handling: what happens when `matched >= len(ids)` (with vs without cached logits)?
   - Suffix prefill partitioning: how `prefillAt := matched` scopes `s.Prefill(ids[prefillAt:])`, and what role intermediate checkpoints play for hybrid models.
   - Post-prefill snapshot admission: how the extended prompt is re-inserted into the tree.
3. Run the witness commands and provide exact file:line references and audit findings.
