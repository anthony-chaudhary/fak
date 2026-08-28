# CLAIMS.md — the fak honesty ledger


Every capability claim carries **exactly one** tag:

- `[SHIPPED]` — real code on the critical path, closed by a mechanical witness (a `go test`, a `go build`, a benchmark field, a file read-back). Reproducible now.
- `[SIMULATED]` — modeled with labeled stand-in data (no GPU / no live engine on the build box); the seam is real, the numbers are illustrative.
- `[STUB]` — plumbing present, behavior deferred; clearly labeled, returns a STUB/no-op result.

The tag answers "is it real". It does not answer the orthogonal question — **is it ON for an operator who configured nothing** — so every `[SHIPPED]` claim also carries a machine-readable **exposure** state, declared at the END of the line. It is the ledger's binding of Q6 of [the net-true-value standard](docs/standards/net-true-value.md) ("on by default, or honestly gated with a stated reason"), reusing `internal/claimcheck`'s `Realized` type rather than a parallel vocabulary — exposure is an axis beside the tag, never a fourth tag:

- `[exposure: default-on]` — on for an operator who set no flag. This is also the state a `[SHIPPED]` line with no marker asserts, so a missing marker is a claim, not silence.
- `[exposure: gated — <reason>]` — ships off, with the reason stated. A gated claim with no stated reason FAILS the lint; that is exactly the Q6 rule `claimcheck.gradeRealized` already encoded.
- A `[SIMULATED]`/`[STUB]` claim is PARKED by its tag (it is not on the critical path by definition) and carries no exposure marker.

The lint witness (unit 96): `go test -v -run TestCLAIMSLedger ./internal/claimcheck` (the `claims-lint` gate, in CI on every push) checks every line beginning with `- [` for one and only one of the three tags AND for a declared exposure on every `[SHIPPED]` line whose prose discloses gating, and EMITS the default-off count instead of leaving it to be grepped out of prose.


<!-- fak:document-set -->

This compact ledger indexes one addressable page per claim. Claim text and maturity tags are preserved on the linked pages.

## Claims

<a id="the-product"></a>
- [SHIPPED] [The product](docs/claims/the-product.md) [exposure: default-on]
<a id="the-syscall-subsystem-latency-check-not-the-headline-kpi-unit-82"></a>
- [SHIPPED] [The syscall subsystem latency check (not the headline KPI — unit 82)](docs/claims/the-syscall-subsystem-latency-check-not-the-headline-kpi-unit-82.md) [exposure: default-on]
<a id="adjudication-the-in-process-dos-reference-monitor"></a>
- [SHIPPED] [Adjudication (the in-process DOS reference monitor)](docs/claims/adjudication-the-in-process-dos-reference-monitor.md) [exposure: default-on]
<a id="tool-vdso-3-tier-local-fast-path"></a>
- [SHIPPED] [Tool vDSO (3-tier local fast path)](docs/claims/tool-vdso-3-tier-local-fast-path.md) [exposure: default-on]
<a id="pre-flight-ladder-grammar-rung"></a>
- [SHIPPED] [Pre-flight ladder + grammar rung](docs/claims/pre-flight-ladder-grammar-rung.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="context-mmu-write-time-result-admission"></a>
- [SHIPPED] [Context-MMU (write-time result admission)](docs/claims/context-mmu-write-time-result-admission.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="answer-shape-the-consumer-facing-degeneration-verbosity-witness"></a>
- [SHIPPED] [Answer-shape: the consumer-facing degeneration/verbosity witness](docs/claims/answer-shape-the-consumer-facing-degeneration-verbosity-witness.md) [exposure: default-on]
<a id="codelint-language-server-packs-over-agent-written-code"></a>
- [SHIPPED] [Codelint: language-server packs over agent-written code](docs/claims/codelint-language-server-packs-over-agent-written-code.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="session-core-dump-context-debugger-recall-cdb"></a>
- [SHIPPED] [Session core-dump + context debugger (recall + cdb)](docs/claims/session-core-dump-context-debugger-recall-cdb.md) [exposure: default-on]
<a id="portable-session-image-uniform-dump-restore-session-restore-sessionimage-snapshot"></a>
- [SHIPPED] [Portable session image + uniform dump/restore (session.Restore + sessionimage + snapshot)](docs/claims/portable-session-image-uniform-dump-restore-session-restore-sessionimage-snapshot.md) [exposure: default-on]
<a id="in-kernel-agent-to-agent-message-channel-a2achan"></a>
- [SHIPPED] [In-kernel agent-to-agent message channel (`a2achan`)](docs/claims/in-kernel-agent-to-agent-message-channel-a2achan.md) [exposure: default-on]
<a id="shared-task-record-fold"></a>
- [SHIPPED] [Shared task record fold](docs/claims/shared-task-record-fold.md) [exposure: default-on]
<a id="trajectory-observability-primitives-data-plane-reference-similarity-scorer-seam"></a>
- [SHIPPED] [Trajectory observability primitives (data plane + reference similarity + scorer seam)](docs/claims/trajectory-observability-primitives-data-plane-reference-similarity-scorer-seam.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="task-manager-snapshot"></a>
- [SHIPPED] [Task manager snapshot](docs/claims/task-manager-snapshot.md) [exposure: default-on]
<a id="s7-write-time-durability-gate-context-is-not-memory"></a>
- [SHIPPED] [S7 write-time durability gate (context is not memory)](docs/claims/s7-write-time-durability-gate-context-is-not-memory.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="in-kernel-model-the-model-fused-into-the-kernel"></a>
- [SHIPPED] [In-kernel model (the model fused into the kernel)](docs/claims/in-kernel-model-the-model-fused-into-the-kernel.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="security-substrate-the-kernel-stops-believing-the-model"></a>
- [SHIPPED] [Security substrate (the kernel stops believing the model)](docs/claims/security-substrate-the-kernel-stops-believing-the-model.md) [exposure: default-on]
<a id="gateway-fak-serve"></a>
- [SHIPPED] [Gateway (`fak serve`)](docs/claims/gateway-fak-serve.md) [exposure: default-on]
<a id="model-routing-per-aspect-ensemble-fak-route"></a>
- [SHIPPED] [Model routing (per-aspect + ensemble — `fak route`)](docs/claims/model-routing-per-aspect-ensemble-fak-route.md) [exposure: default-on]
<a id="turn-tax-benchmark-fak-turntax"></a>
- [SHIPPED] [Turn-tax benchmark (`fak turntax`)](docs/claims/turn-tax-benchmark-fak-turntax.md) [exposure: default-on]
<a id="self-ablation-sweep-fak-ablate"></a>
- [SHIPPED] [Self-ablation sweep (`fak ablate`)](docs/claims/self-ablation-sweep-fak-ablate.md) [exposure: default-on]
<a id="cross-agent-ablation-regime-b-bare-claude-p-vs-fak-guard-claude-p"></a>
- [SHIPPED] [Cross-agent ablation (Regime B — bare `claude -p` vs `fak guard -- claude -p`)](docs/claims/cross-agent-ablation-regime-b-bare-claude-p-vs-fak-guard-claude-p.md) [exposure: default-on]
<a id="fan-out-benchmark-fanbench-one-master-goal-n-sub-agents-n-1-1024"></a>
- [SHIPPED] [Fan-out benchmark (`fanbench` — one master goal → N sub-agents, N=1…1024)](docs/claims/fan-out-benchmark-fanbench-one-master-goal-n-sub-agents-n-1-1024.md) [exposure: default-on]
<a id="bounded-microagents-construct-harnesses-cmd-microharnessdemo"></a>
- [SHIPPED] [Bounded microagents construct harnesses (`cmd/microharnessdemo`)](docs/claims/bounded-microagents-construct-harnesses-cmd-microharnessdemo.md) [exposure: default-on]
<a id="in-process-microagent-host-internal-microagent-n-agent-loops-as-goroutines-behind-one-gateway"></a>
- [SIMULATED] [In-process microagent host (`internal/microagent` — N agent loops as goroutines behind ONE gateway)](docs/claims/in-process-microagent-host-internal-microagent-n-agent-loops-as-goroutines-behind-one-gateway.md)
<a id="ultra-long-context-work-floor-longctxbench-per-agent-context-100k-tokens"></a>
- [SHIPPED] [Ultra-long-context work floor (`longctxbench` — per-agent context > 100k tokens)](docs/claims/ultra-long-context-work-floor-longctxbench-per-agent-context-100k-tokens.md) [exposure: default-on]
<a id="engine"></a>
- [SHIPPED] [Engine](docs/claims/engine.md) [exposure: default-on]
<a id="stewards-rsi-ship-gate"></a>
- [SHIPPED] [Stewards + RSI ship-gate](docs/claims/stewards-rsi-ship-gate.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="vcache-chains-recall-m4"></a>
- [SHIPPED] [vCache Chains & Recall (M4)](docs/claims/vcache-chains-recall-m4.md) [exposure: gated — linked shipped mechanisms include an opt-in or default-off path; see the detail page for each gate]
<a id="vcache-governor-m5"></a>
- [SHIPPED] [vCache Governor (M5)](docs/claims/vcache-governor-m5.md) [exposure: default-on]
<a id="vcache-observability-per-sub-concept-lens"></a>
- [SHIPPED] [vCache observability (per-sub-concept lens)](docs/claims/vcache-observability-per-sub-concept-lens.md) [exposure: default-on]
<a id="what-fak-is-not"></a>
- [SIMULATED] [What fak is NOT](docs/claims/what-fak-is-not.md)
<a id="prior-art-posture"></a>
- [SHIPPED] [Prior-art posture](docs/claims/prior-art-posture.md) [exposure: default-on]
