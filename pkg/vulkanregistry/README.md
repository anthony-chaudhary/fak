# pkg/vulkanregistry

Exports the identity of the Vulkan SPIR-V shader registry the fak builder
compiles and stamps, so consumers that admit fak Vulkan build receipts track
the builder instead of copying its constants. API: [`vulkanregistry.go`](vulkanregistry.go)
(`Current()` -> `Identity{ReceiptSchema, ID, SHA256, ModuleCount}`).

## Wiring audit (2026-11-03, flow ticket debtwave2-pkg-vulkanregistry)

Verdict: **ABSENT** — no production call path invokes this package yet
(presence != invokability).

Evidence (read-only, run from the repo root):

- `rg -n "github.com/anthony-chaudhary/fak/pkg/vulkanregistry"` over the whole
  tree: 0 hits — zero importers, test or otherwise.
- `rg -n "vulkanregistry" cmd/ internal/ -g '*.go'`: 0 hits outside this
  package; no verb/registry table entry references it.
- The underlying identity is consumed directly, bypassing this package: the
  `fak-dev build-vulkan verify` path reaches
  `computebuild.CurrentVulkanBuildIdentity()` at
  `internal/devcmd/build_vulkan.go:515`
  (`cmd/fak-dev/main.go` -> `devcmd.RunBuildVulkan` -> `runBuildVulkanVerify`
  -> `validateVulkanVerificationSuccess`).

Why it is not wired in this ticket: the minimal connection is a single-file,
~3-5 line swap in `internal/devcmd/build_vulkan.go` (call
`vulkanregistry.Current()` instead of the direct computebuild call, adapting
the `(schema, registry, error)` shape to `Identity`). The flow ticket boundary
restricts this ticket to files under `pkg/vulkanregistry/`, so per its work
item 1 the audit records ABSENT-with-reason and lands this evidence instead of
forcing new wiring.

Next checkable step: a follow-up ticket with a widened boundary lands the
devcmd swap; that flips the debt-lanes reader's `integrated` flag for lane
`vulkanregistry` (the reader caps non-integrated lanes at maturity 5.0, so
tests/benchmarks alone cannot raise the lane past the tested rung).
