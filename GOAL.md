---
loop: goal
witness: "go test -v ./internal/amdgpu/... ./internal/deepseekv4moe/... ./internal/compute/..."
budget: { max_iters: 20 }
lane: amdgpu
---
# Objective
Complete, independently witness, and reconcile the top 5 GitHub tickets for AMD Strix Halo (gfx1151).

# Non-Goals
- Do not modify frozen ABI (`internal/abi`).
- Do not modify root build configuration (`go.mod`, `go.sum`, `dos.toml`).
- Do not commit unrelated dirty files or peer WIP.
- Do not force push or create branches (`main` only).

# Plan
- [x] 1. Identify top 5 AMD Strix Halo tickets: #11093, #11094, #10755, #11095, #10763
- [x] 2. Verify compilation and test suite for #11093 & #11094 (internal/amdgpu)
- [x] 3. Verify compilation and test suite for #10755 (internal/deepseekv4moe)
- [x] 4. Verify compilation and test suite for #11095 & #10763 (internal/compute)
- [x] 5. Reconcile and close the 5 GitHub issues with landed commit SHAs and witness receipts
- [x] 6. Emit final verification and completion report

# Results and Verification Evidence
- **Issue #11093** (`4ffbef402`): Native AQL and PM4 packet builder in `internal/amdgpu/aql_pm4.go`. 64B ABI alignment, atomic ring buffer queue, and PM4 type-3 packet encoding/decoding. Witness: `go test -v ./internal/amdgpu` (12/12 passing). Status: CLOSED.
- **Issue #11094** (`4ffbef402`): Standalone HSACO code-object assembler and ELF64 emitter in `internal/amdgpu/hsaco.go`. Zero-dependency generation of valid AMDGPU ELF objects with MsgPack metadata and gfx1151 / RDNA3.5 register blocks. Witness: `go test -v ./internal/amdgpu` (6/6 passing via stdlib `debug/elf`). Status: CLOSED.
- **Issue #10755** (`afdff95e5`, `82c79711a`): Contiguous selection range-based expert sharding for memory-constrained multi-node TP on dual Strix Halo in `internal/deepseekv4moe/expert_sharding.go`. Witness: `go test -v ./internal/deepseekv4moe` (35/35 passing, 0-byte unowned allocations, mathematical AllReduce parity). Status: CLOSED.
- **Issue #11095** (`cf3f1ee0b`): KPACK multi-target kernel package parser and dynamic architecture resolver in `internal/compute/kpack.go`. Stack-bounded MsgPack parsing and zstd decompression. Witness: `go test -v ./internal/compute` (TestKPACK* passing). Status: CLOSED.
- **Issue #10763** (`cf3f1ee0b`): Direct I/O and post-upload page cache eviction for large-model weight streaming in `internal/compute/disk_direct.go`. 4096B aligned buffers, O_DIRECT, and POSIX_FADV_DONTNEED. Witness: `go test -v ./internal/compute` (TestDirectIO* passing). Status: CLOSED.

# Scratch / last-refusal
All 5 top AMD Strix Halo tickets independently verified, witnessed, and closed.
