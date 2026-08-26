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
- [The product](docs/claims/the-product.md)
<a id="the-syscall-subsystem-latency-check-not-the-headline-kpi-unit-82"></a>
- [The syscall subsystem latency check (not the headline KPI — unit 82)](docs/claims/the-syscall-subsystem-latency-check-not-the-headline-kpi-unit-82.md)
<a id="adjudication-the-in-process-dos-reference-monitor"></a>
- [Adjudication (the in-process DOS reference monitor)](docs/claims/adjudication-the-in-process-dos-reference-monitor.md)
<a id="tool-vdso-3-tier-local-fast-path"></a>
- [Tool vDSO (3-tier local fast path)](docs/claims/tool-vdso-3-tier-local-fast-path.md)
<a id="pre-flight-ladder-grammar-rung"></a>
- [Pre-flight ladder + grammar rung](docs/claims/pre-flight-ladder-grammar-rung.md)
<a id="context-mmu-write-time-result-admission"></a>
- [Context-MMU (write-time result admission)](docs/claims/context-mmu-write-time-result-admission.md)
<a id="answer-shape-the-consumer-facing-degeneration-verbosity-witness"></a>
- [Answer-shape: the consumer-facing degeneration/verbosity witness](docs/claims/answer-shape-the-consumer-facing-degeneration-verbosity-witness.md)
<a id="codelint-language-server-packs-over-agent-written-code"></a>
- [Codelint: language-server packs over agent-written code](docs/claims/codelint-language-server-packs-over-agent-written-code.md)
<a id="session-core-dump-context-debugger-recall-cdb"></a>
- [Session core-dump + context debugger (recall + cdb)](docs/claims/session-core-dump-context-debugger-recall-cdb.md)
<a id="portable-session-image-uniform-dump-restore-session-restore-sessionimage-snapshot"></a>
- [Portable session image + uniform dump/restore (session.Restore + sessionimage + snapshot)](docs/claims/portable-session-image-uniform-dump-restore-session-restore-sessionimage-snapshot.md)
<a id="in-kernel-agent-to-agent-message-channel-a2achan"></a>
- [In-kernel agent-to-agent message channel (`a2achan`)](docs/claims/in-kernel-agent-to-agent-message-channel-a2achan.md)
<a id="shared-task-record-fold"></a>
- [Shared task record fold](docs/claims/shared-task-record-fold.md)
<a id="trajectory-observability-primitives-data-plane-reference-similarity-scorer-seam"></a>
- [Trajectory observability primitives (data plane + reference similarity + scorer seam)](docs/claims/trajectory-observability-primitives-data-plane-reference-similarity-scorer-seam.md)
<a id="task-manager-snapshot"></a>
- [Task manager snapshot](docs/claims/task-manager-snapshot.md)
<a id="s7-write-time-durability-gate-context-is-not-memory"></a>
- [S7 write-time durability gate (context is not memory)](docs/claims/s7-write-time-durability-gate-context-is-not-memory.md)
<a id="in-kernel-model-the-model-fused-into-the-kernel"></a>
- [In-kernel model (the model fused into the kernel)](docs/claims/in-kernel-model-the-model-fused-into-the-kernel.md)
<a id="security-substrate-the-kernel-stops-believing-the-model"></a>
- [Security substrate (the kernel stops believing the model)](docs/claims/security-substrate-the-kernel-stops-believing-the-model.md)
<a id="gateway-fak-serve"></a>
- [Gateway (`fak serve`)](docs/claims/gateway-fak-serve.md)
<a id="model-routing-per-aspect-ensemble-fak-route"></a>
- [Model routing (per-aspect + ensemble — `fak route`)](docs/claims/model-routing-per-aspect-ensemble-fak-route.md)
<a id="turn-tax-benchmark-fak-turntax"></a>
- [Turn-tax benchmark (`fak turntax`)](docs/claims/turn-tax-benchmark-fak-turntax.md)
<a id="self-ablation-sweep-fak-ablate"></a>
- [Self-ablation sweep (`fak ablate`)](docs/claims/self-ablation-sweep-fak-ablate.md)
<a id="cross-agent-ablation-regime-b-bare-claude-p-vs-fak-guard-claude-p"></a>
- [Cross-agent ablation (Regime B — bare `claude -p` vs `fak guard -- claude -p`)](docs/claims/cross-agent-ablation-regime-b-bare-claude-p-vs-fak-guard-claude-p.md)
<a id="fan-out-benchmark-fanbench-one-master-goal-n-sub-agents-n-1-1024"></a>
- [Fan-out benchmark (`fanbench` — one master goal → N sub-agents, N=1…1024)](docs/claims/fan-out-benchmark-fanbench-one-master-goal-n-sub-agents-n-1-1024.md)
<a id="bounded-microagents-construct-harnesses-cmd-microharnessdemo"></a>
- [Bounded microagents construct harnesses (`cmd/microharnessdemo`)](docs/claims/bounded-microagents-construct-harnesses-cmd-microharnessdemo.md)
<a id="in-process-microagent-host-internal-microagent-n-agent-loops-as-goroutines-behind-one-gateway"></a>
- [In-process microagent host (`internal/microagent` — N agent loops as goroutines behind ONE gateway)](docs/claims/in-process-microagent-host-internal-microagent-n-agent-loops-as-goroutines-behind-one-gateway.md)
<a id="ultra-long-context-work-floor-longctxbench-per-agent-context-100k-tokens"></a>
- [Ultra-long-context work floor (`longctxbench` — per-agent context > 100k tokens)](docs/claims/ultra-long-context-work-floor-longctxbench-per-agent-context-100k-tokens.md)
<a id="engine"></a>
- [Engine](docs/claims/engine.md)
<a id="stewards-rsi-ship-gate"></a>
- [Stewards + RSI ship-gate](docs/claims/stewards-rsi-ship-gate.md)
<a id="vcache-chains-recall-m4"></a>
- [vCache Chains & Recall (M4)](docs/claims/vcache-chains-recall-m4.md)
<a id="vcache-governor-m5"></a>
- [vCache Governor (M5)](docs/claims/vcache-governor-m5.md)
<a id="vcache-observability-per-sub-concept-lens"></a>
- [vCache observability (per-sub-concept lens)](docs/claims/vcache-observability-per-sub-concept-lens.md)
<a id="what-fak-is-not"></a>
- [What fak is NOT](docs/claims/what-fak-is-not.md)
<a id="prior-art-posture"></a>
- [Prior-art posture](docs/claims/prior-art-posture.md)
