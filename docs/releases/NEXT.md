# fak vNext (targeting v0.58.0): Work in Progress

This document tracks in-flight work on `main` targeting the upcoming `v0.58.0` release.
It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.

- **Projected version:** `0.58.0` (`minor` bump)
- **Base release tag:** `v0.57.0`
- **Commits in flight:** 174

## What changed

### Security & Governance
- Add CAS KV ownership directory with epoch fencing and stale-view refusal.
- Add DCGM health/thermal field seam with fail-closed absent-service refusal.
- Add Langfuse trace/observation-tree export projection over trajectory audit JSONL.
- Add Nomos action-review gate with allow/deny verdict grammar and overeager categories.
- Add serve-config knob inventory by policy goal, hot/cold, defect-hiding.
- Add signed-policy artifact model with admission-time enforcement and version binding.
- Add sliding-window and token-bucket per-route caps with cap-refusal grammar.

### Autonomous Agent & Multi-Model Harness
- Add nvbandwidth memory-transport measurement plans with d2d/h2d matrix.
- Add tool-name canonicalization map applied on every parse path.
- Add unreferenced-only session eviction that keeps referenced warm-prefix KV resident under pressure.
- Bind each closed agent Role to a predefined context-window type.
- Complete flow ticket memory-hierarchy-question-ledger (memory-hierarchy-question-ledger).
- Memory-estimation model ported from aiperf _memory_estimator.
- Record DFlash2 decode host-overhead window, exclude bootstrap as non-decode work.
- Session synthesizer reset ported from aiperf session_synthesizer.

### Serving Engine, Gateway & Kernel Acceleration
- Add block-table row-span contract with export, validation, and rebuild.
- Add CUTLASS tiling and NVFP4 kernel template study with upstream file pins.
- Add drafter embedding alias onto the target table instead of a second copy.
- Add DSpine adjacent-causal-injection drafter lineage.
- Add foreign quant-descriptor adapter translating to canonical scheme before binding.
- Add Kimi K3.1 SiTU (SoftCap-GLU) activation + reference parity.
- Add MPK SM-level task-graph schema and partition invariants with closed host oracle.
- Add nvbench axis/cartesian sweep model with warmup/stream states and CSV contract.
- Add Qwen3.8 QSA pluggable sparse-attention backend contract with closed host oracle.
- Add safetensors shard filter so draft loads open only shards holding needed keys.
- Add same-run end-to-end time breakdown reader + fak compute-trace breakdown verb.
- Add the Kimi K3.1 attention-residual bank + reference parity.
- Add the Kimi K3.1 latent-MoE host path (front/norm/tail + ordering).
- Add the Kimi K3.1 native MoE router (noaux_tc bias-ranked top-k).
- Add v41_head_readback host-fallback counter for the V4.1 head-projection readback.
- Billing-grade per-request usage ledger keyed by the authenticated principal on every route.
- Coordinate-keyed probabilistic sampling ABI.
- Count tool-call parse outcomes so a fail-closed turn is a scrapeable fact, not a log line.
- Define the scale-out state model - per-replica limits, state volume, pod-unique trace ids.
- Detect the PLE dtype from checkpoint metadata instead of assuming the model default.
- Expose ContinuousBatcher.FreeQueue live empty-slot free-queue for M2 free-queue batching.
- Give the K3.1 MoE block the dense-layer (first_k_dense_replace) branch.
- Histogram quantile estimator ported from aiperf histogram_percentiles.
- Hoist the DSA light-indexer top-k to layer-group scope and broadcast to aliasing layers.
- Keep captured graphs non-evictable while a replay is in flight.
- Lossless MXFP4 source-to-GGUF expert repack with bit-exact round-trip witness.
- Surface a fail-closed device-stage fault as its own 500 code and ledger kind.
- Wire Kimi K3.1 forward-path dispatch (L8) - classify kimi_k31 and run the KDA/DSA composite over the landed oracles.
- Wire per-tenant admission gates to fak serve flags.
- Wire the Kimi K3.1 decoder layer's FFN/MoE half into the composite forward.

### Shared Trunk & Workspace Lifecycle
- Add exact-path recovery of interrupted-reap residue with dead owner and released lease.
- Catch the site/cluster codename shape and redact the public tree.
- Port the DeepEP two-mode MoE dispatch split as a default-off phase-selection step.

### Developer Platform, Tooling & Evidence
- Add measured per-tick load accounting as fak-ops-load/1 (#14474).
- Register checkpoint-scorecard in the control-pane ratchet Closes.
- Register product-scorecard in the control-pane ratchet Closes.
- Add an H200 load row to the unified router board so per-replica serving and error state read at a glance.
- Add bounded per-lane KV checkpoint ring for save/restore at token boundaries.
- Add bounded sub-section validation planner (fak.scope-plan.v1).
- Add closed KV wire-encoding contract with stored-bytes and scale validators.
- Add DSpark unified-KV-arena contract with in-place spec-step reconciliation oracle.
- Add expand-once-share-many dequant ledger with reference counting and last-use release.
- Add forward-admission gate for Hopper MLA KV layout axes with per-block byte recompute.
- Add hierarchical-cache prefetch bound clamped to longest-prefix-match with admission charge.
- Add Inspect AI eval-log export projection over agentbench run results.
- Add ModelOpt NVFP4/W4A16 recipe semantics with scale layout and late-tensor-scale pins.
- Add per-sequence EOS terminal transition inside batched constrained generation.
- Adjusted-latency percentiles ported from aiperf derived_latency.
- APGR cost-aware router evaluation ported from RouteLLM.
- CI-width convergence check ported from aiperf ci_width.
- Exact-quantization admission gate.
- ISL budget compensation ported from aiperf composer/base.
- Layered prefix allocator ported from aiperf prefix_model.
- Make the launch readiness verdict terminal: a stopping probe never re-readies.
- Matrix-factorization win-rate router ported from RouteLLM.
- Monotonic search planner ported from aiperf search_planner.
- Multi-run early-stop ported from aiperf multi_run.
- Normalized interactivity metric ported from aiperf e2e_normalized_interactivity.
- One global quiet switch (FAK_SLACK_QUIET) mutes every outbound Slack surface.
- Pace requests on absolute deadlines so wake-up error does not accumulate.
- Percentile-band metric dict ported from aiperf metric_dicts.
- Prefix hit-rate analysis ported from aiperf prefix_analyzer.
- Price multi-GPU split (single/layer/tensor/expert) x host offload (cpu/stream) with PlanSplit.
- Qualification matrix + Vulkan MTP default-on gate with DisableVulkanMTP opt-out.
- Reuse isolated snapshots for selected execution checks.
- Size HTTP output for bounded coding tasks.
- Speculative-decoding metrics ported from aiperf spec_decode_metrics.
- Split admission into execution slots and a capped in-flight KV-transfer budget.
- Stable sweep seed derivation ported from aiperf plan_builder.
- Tool-call delta merge ported from aiperf openai_chat.
- Window fak-router lifetime totals + add caching/token throughput panels.
- Wire aiperf time-weighted + active-only throughput into Summarize.
- Wire packing-matrix reader over the loader packing coverage matrix (fak-private evergreen-tensor-layout-mismatch).

## Reliability and correctness

### Security & Governance
- Bound the drain: queue cap, amortized Stepper, windowed carryover seed / nothing bounded queue length and every step re-normalized the spec, so draining a Q-item stream was Theta(Q^2) allocation (18.4 GB of churn at Q=10000) with no structural refusal / StreamSpec gains MaxItems (default 1000, validated at load and enforced at navigation as a split-the-stream halt), NewStepper normalizes once per drain so a step is O(1) in Q (flat 634 B/op, 21 allocs across Q=1..10000 vs 1.85 MB/op for DecideStep at Q=10000), and the carryover seed windows CompletedItems to the last 32 receipts plus CompletedItemsTotal / drain-allocation gate TestStepperDrainAllocationLinear fails at O(Q^2) and passes linear; equivalence test pins Stepper == DecideStep over a scripted lifecycle (fak superstream) (Ref superstream-navigation-scale).
- Compare repository identities, not path spellings, in the land root guard / --root "." names the same repository as the absolute worktree owner, so sameHostPath refused every default-rooted land as a cross-repository one: 16 of 58 land attempts in 3h (47%, the largest single failure class) died on LAND_ROOT_WORKTREE_MISMATCH with detail "root=. worktree_owner=C:\work\fak-private" although both named the same repository / go test ./cmd/fak -run 'TestWorkerLandRelativeRoot|TestWorkerLandAbsoluteRoot|TestWorkerLandRefusesRootOfAnotherRepository|TestWorkerLandDerivesTrunk' -count=1: 5 tests PASS (relative-"." root admitted, absolute same-repo root admitted, relative and absolute other-repo roots still refused, omitted root still derived), and the mandatory red-then-green symptom witness fails the new relative-root test on the parent commit.
- Make issue-create scrub-gate root resolution -trimpath-safe.

### Autonomous Agent & Multi-Model Harness
- Enforce trace ownership on session, trace-reset, and A2A routes.
- Accept ready-only router healthz so fak agent auto-connects.
- Clamp Linux host free memory to cgroup memory.max headroom (fak-private hostmem-cgroup-clamp-20261120).
- Discover the served model with the proven router key so fak agent/chat stop defaulting to an unserved model.
- Honor explicitly closed tool catalogs.

### Serving Engine, Gateway & Kernel Acceleration
- Add the Kimi K3.1 exclusion to IsQwen35Hybrid.
- Implement a queue-independent decision step / implement the zero-copy step: the decision helpers read only limit defaults plus the caller's state.Queue, so the earlier one-copy normalization was removable rather than intrinsic, and the carryover seed preallocated len(queue) summaries it never filled / defaults move to normalizeLimits, the seed capacity becomes the completed count, and two allocation-bytes gates replace the single-copy bound (flat at 632-634 B/op over Q=1..10000; the old shapes fail at 373767 vs 41588 and 567395 vs 62320 B/op); the public cost page states the corrected model.
- Scope tenant observability routes and journal drains (gateway).
- Allowlist DSpine TerminalEnabled as a fail-closed contract gate in the test-only-enable detector.
- Emit typed pre-token stream failures.
- Gate the served-inline restore banner on the wire advertising a restore affordance.
- Reconcile WeightResidency ratio with the resident inventory so layout-forced Q8 expansions count.
- Release qwen35 cohort HAL state locks on panic or early exit so cleanup cannot deadlock.
- Restore per-tenant admission gates reverted by stale-base landing 94ac7067.
- Scope tenant keys to inference-only, reserve kernel principals, refuse --allow-lan x --key-principal.
- Start tool children from an allowlisted minimal env, denying gateway key vars.
- Stop proxy-mode /v1/embeddings from masquerading hash vectors as the chat model.

### Shared Trunk & Workspace Lifecycle
- Drain the kept-checkout pool on release so a land burst cannot park it permanently.
- Land a worktree against the repository that owns it, never the caller's cwd.
- Populate ActionItem.Lane so live propagation-debt dispatch clears the strict-scope gate.
- Restore 16 files clobbered by commit-tree parent-swap landings.
- Widen the startup tear repair to the whole locks namespace and stop it destroying symref leases.

### Developer Platform, Tooling & Evidence
- Add render_prometheus.go to the grafanacontract ops-exporter manifest (fak-private ops).
- Implement one queue copy per decision step / implement the one-copy step path: repeated NormalizedSpec passes made each decision cost 2-3 queue copies, so a Q-item drain burned twice the allocation the contract allows and nothing measured the slope / EvaluateContextSafety and BuildCarryoverSeed gain normalized-spec helpers used by DecideStep, ApplyStep drops a dead copy, and a Q=1..10000 scaling benchmark plus a deterministic one-copy allocation gate pin it (369->184 B/item at Q=10000; pre-fix gate 371 B/item > 276 bound).
- Abstain on unresolvable go.work in scratch checkouts instead of misrefuting.
- Accept typographic patch context.
- Admit POD_NAME trace pod-identity read to the post-freeze ledger.
- Close the three defects in the public secret-shape gate.
- Derive issue create --repo from declared paths and refuse cross-repo mismatch (fak-private devcmd).
- Drain in-flight streams across SIGTERM; ship grace period, preStop, drain env, and PDB in committed manifests.
- Repo-hygiene collector robust to UTF-16/BOM docs.

## Engineering quality and evidence

### Security & Governance
- Correct the cost page to the landed Stepper/MaxItems design / the page described a per-step zero-copy DecideStep that did not land, while HEAD routes drains through Stepper and caps queues, so the page overstated the one-shot primitive and omitted the queue cap / states DecideStep as Theta(Q) one-shot vs Stepper O(1) per step (656-662 B/op flat over Q=1..10000 vs 1.85 MB/op), the MaxItems=1000 cap, the 32-receipt seed window, and the four gates that pin them.
- Fragment-assemble the private markers in the gateup-slab witness guard.
- Retire debtwave2 lane debt - land the ABSENT-with-reason wiring audit record.

### Autonomous Agent & Multi-Model Harness
- Add the memory-hierarchy ladder - five rungs from registers to NVMe.
- Record SubagentSupervisor as a deliberately tested-but-unwired primitive, with TaskState as the executor-teardown owner.

### Serving Engine, Gateway & Kernel Acceleration
- Add connected diagnostic-boundary regression for the independent decode dispatcher.
- Add the navigation cost model page / the primitive's O(1)-context claim was measured on turns only, so the queue and multi-stream terms had no stated bound and repeated queue copies went unnoticed / states Theta(Q) per step, Theta(Q^2) per bounded drain and Theta(W*Q) per W-stream sweep, records the measured one-copy step cost (184 B/item at Q=10000, was 369) and the deterministic gate that pins it.
- Add v41_head_readback host-fallback counter witness.
- Make cuda-tag claim version-durable.
- Prove the K3.1 composite runs all 93 published layers with the full FFN/MoE order.
- Regenerate the llms-full digest for the workstream cost page / the discoverability gate compares committed bytes, so the page landed without its digest entry and the next reader of llms-full would not find the cost model / render at 1bf3e466 adds the page reference.
- Repoint tracked opencode default to fak-router/qwen/qwen3.8-27b.

### Shared Trunk & Workspace Lifecycle
- Commit to a 12-month relicensing notice period and map layers to concrete paths.
- File release-ci-reds-v0570 follow-up for the pre-existing reds behind the v0.57.0 --skip-ci waiver (fak docs) v0.57.0 shipped with a one-time operator-authorized --skip-ci waiver because ci-fast was red on prior tip 17892788c96 (4 deterministic whole-tree invariant reds + host-load timeouts, 0 data races, not a release regression); this ticket is the root-cause follow-up so the next ship is green without a waiver. witness: ticket committed by explicit pathspec; 4 deterministic reds (TestCheckedInPageIsFresh, TestLiveTreeUnderCeiling, TestTreeEnumExhaustivenessNotGrowing, TestRunTreeVerifyMicrobenchmarkMatrixAndOverhead) + 4 load-timeout flakiness reds each named with a root-cause fix.
- Give the explainers tree the entry point it never had.

### Developer Platform, Tooling & Evidence
- Add the accelerator ladder - five rungs from the CPU to Vulkan and ROCm.
- Add the six-rung kernel-development ladder and its question ledger.
- Assert closed contracts instead of error prose.
- Benchmark scenario invariants ported from aiperf scenario/base.
- Cut phantom coupling edges to clear the high-coupling surcharge.
- Diagram the Vulkan-vs-ROCm comparison, the rung that had none.
- Document the consumer tier as an anonymized 3-repo import DAG.
- Drop the retired router plane from the phantom-alert exemption.
- Format the 17 files the trunk gofmt gate refuses.
- Give the ladder the visuals its sibling already had.
- Index the inference ladder from the docs landing page.
- Index the kernel-development teaching cluster on every discoverability edge.
- Keep the published module zip under the 500 MiB proxy limit.
- Land the mixed-precision ladder and repair the inference ladder's front-door edges.
- Make the two new explainer ladders discoverable from the catalog.
- Move router plane to private companion.
- Name the ROCm artifacts one way in the accelerator ladder.
- Position execution witness snapshot identifier admission debt.
- Record fan-out reuse cadence tick.
- Record same-cadence milestone tick.
- Record the two retention findings the pool measurement surfaced / the campaign index still read 'all five solved' while a 42x-over-cap checkout cache and an unbounded witness-quarantine namespace were live / index now names both, with the pool fix at public 686b27a3 and the quarantine cure ticketed as witness-quarantine-retention-20261120.
- Redact the residual identity leaks the fixture exemption hid.
- Reduce consumer tier to a pointer, add routing row.
- Refresh stale committed self-inventory.
- Regenerate the concept scorecard the render-materialize corpus row left stale.
- Remove the throwaway land probe that a mis-targeted worker land put on public trunk.
- Replace the host label my own redaction wrote with a real needle.
- Ship license notices in container images and resolve Modular attribution.
- Teach the hardware instruction-set ladder, from SIMD to silicon.
- The inference ladder's link check is real, its citation check is not.
- The inference-and-parallelism ladder - five rungs from a forward pass to splitting one across workers.
- The ROCm backend has run on silicon - the ladder said it never compiled.
- The ROCm backend is written, not deferred - and never compiled.
- Unbold the naive-baseline multiple on the prove-faster rung.
- Wire the inference ladder into the front door and the question loop.
- Fak breath -all -> 0 findings naming any mixed-precision page; go test ./internal/promptlint/breath/... -count=1 -> ok EXIT=0; rg -n 'mixedprecision\.(Evaluate|ParseAndEvaluate)' -g '!*_test.go' -> no production caller; rg -l internal/covmatrix -g '!*_test.go' -> cmd/fak, milestonedoc, milestonereport, supportmaturity only; rg -n RunScheduledPass -> definition plus precision_schedule_test.go only.

## Upgrade and breaking changes

- No manual migration required unless specified above.
