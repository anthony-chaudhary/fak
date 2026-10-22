# Experimental native Vulkan completion hook

SOURCE PATCH ONLY. UNAPPLIED, UNCOMPILED, UNRUN. Not runtime qualification,
not a packed backend enablement, and not permission to release a failed owner.
Suggested landing: `experiments/vulkan-completion-seam/{native-hook.patch,README.md}`.

## Exact source and dependency

`native-hook.patch` targets only `internal/compute/vulkan_shim.cpp` from reviewed
combined tree `e2ea4b86a2909f57e64731fe56dfa6f6806dc694` (public3b790 lineage).
Input SHA256: `bb02bbb80c654d394d39a9b820dc9cbbb76b2b993a5706c3bc5864b2c493ed68`.
It depends on the separately packaged, unchanged headers:

- `internal/compute/_experimental/v41_packed_runtime/proposed_api.h`
- `internal/compute/_experimental/v41_packed_runtime/vulkan_v41_batch_lifetime_prototype.h`

No default source includes this patch. Applying it is a later review decision.
The existing prototype remains bookkeeping; this patch supplies actual lifecycle
call sites rather than another simulated store or telemetry-derived completion.

## Implemented source slice

- Native monotonic step owner plus per-window serial reservation before command
  begin; reopening a step window obtains a different serial. Exhaustion refuses
  admission. Existing allocation generation is required, never guessed.
- Step begin and every reopened window reject inflight/restore ownership before
  the checked begin helper can reach legacy unchecked drain. Checked batch flush
  also fails closed on an unexpected inflight owner. Pipeline eligibility stays
  unchanged and excludes these V4.1 windows.
- `v41CompletionRecordPackedZero` validates the exact current ticket before the
  private live handle, enrolls a dedicated allocation before barrier/fill commands,
  and records custody before submission. Allocation/enrollment failure emits no
  commands. Only whole-allocation zero is included; copy/append are deliberately
  not implemented here.
- The sole COMPLETE retirement call is inside the actual successful
  `vkWaitForFences` branch of `v41EndSubmitWaitChecked`, before command reuse/free.
  Identity and command are captured before submission. Empty windows cancel.
  End/reset/submit/wait errors quarantine the record; uncertain submit/wait retains
  existing command/descriptor/staging behavior; owning-window end/reset errors
  now also retain those resources rather than recycle them. No status/counter/Read boolean is
  accepted as proof.
- `destroyBuffer` and `fvk_free` independently refuse pinned allocation destruction
  or pool return, including complete but unacknowledged uses. Query is read-only;
  acknowledgement is exact-use and forbidden on an unhealthy/stale context.
  Receipts outlive command recycling. A later failing window does not erase an
  earlier successful fence fact.
- Re-init is deliberately refused once this store binds a context. This is a
  conservative experimental restriction, not a new context-retirement protocol.
  No generation/store reset is supplied.

Existing successful-wait accounting and transfer counters remain in their original
paths. Generic abort behavior is untouched. New fail-closed branches concern seam
admission/invariant failures. For identity-bound windows ONLY, end/reset errors
are deliberately stricter than baseline: they set pending failure and retain
command, descriptors and owners. This conservative behavior needs runtime review. Existing packed busy guards,
shader resolver, model callers and registry are not edited.

## Exact unwired boundary / missing proof

The static `v41CompletionRecordPackedZero` has NO CALLER. A private Go packed owner
adapter must first retain strong source/destination references and native identity,
prove fresh dedicated-allocation ownership, keep initialized/eligible metadata
private, and reject Free/Close until exact completion/publication/acknowledgement.
There is no exported raw-pointer enrollment API. This patch must not be used by
casting Tensor/halW or by relaxing the synchronous packed state guard.

`fvk_free` is void: a refused release preserves native custody but cannot tell an
existing Go owner not to discard its accounting/handle. That Go owner protocol is
missing. Acknowledgement must be atomic with metadata publication under vulkanMu;
no caller supplies that protocol yet. Every window receipt requires explicit
forget after acknowledgements; until a caller manages it the store grows. This
is another reason NOT to enable the patch as normal step runtime.

The full owner-teardown/device-lifetime/direct-allocation-destruction audit remains
required. The slice rejects arena/offload/mapped/suballocated destinations and
blocks known generic destroy/free paths; this is not proof about every future
alias, pool path, or external direct free. No device retirement API is invented.

Required before enablement: compilation against the real headers; independently
reviewed native fake with end/reset/submit/wait faults; A-completes/B-reopens/B-fails;
empty/exhausted/stale/foreign/command-reused identities; enrollment failure before
commands; duplicate acknowledgement; source/destination release and context
re-init attempts; retained accounting; real Vulkan fence/device witness. Simulated
bookkeeping tests cannot replace any actual completion proof.

## Checks performed

Read-only frozen-object extraction and source inspection. `git diff --no-index
--check` passed; patch parser reports 175 insertions / 3 deletions in one source
file. No executable checkout, active source edit, compiler, GPU, registry change,
publication, or runtime test was performed by this worker.
