# Guard value under a kernel we control — the GPU server GLM-5.2 vs API ablation (2026-07-03)

**Goal.** Prove *real* `fak guard` value in the one regime where every number is ours:
the wrapped agent's upstream is **fak's own inference kernel** serving GLM-5.2 on GPU server class
compute, ablated against the same guarded workload pointed at **API-based sessions**
(Z.ai's GLM-5.2 API; the Anthropic API the guard proxies today). This note is the
benchmark design: the arms, the metrics with their trust classes, what only the
kernel arm can prove, and the honest gates.

**Status: DESIGN + dispatch wiring. No number in this note is a result.** The live
run is box-gated on GPU server access — the same residual as epic
[#1010](https://github.com/anthony-chaudhary/fak/issues/1010) child
[#1012](https://github.com/anthony-chaudhary/fak/issues/1012). Every claim below is
labeled runnable-today / not-yet.

---

## 1. Why "kernel we control" is the guard-value lever

`fak guard -- <agent>` today is a loopback proxy + supervisor: the gate
(`internal/gateway/gateway.go` `adjudicate()` → `k.Decide`) sits at the tool-call /
HTTP boundary, and the model upstream is someone else's. That shape caps what
"guard value" can *mean*:

- fak's own KV kernel is structurally absent in proxy mode — Track-1 in-kernel
  KV-prefix reuse is **0 by construction**
  ([GUARD-OWN-CACHE-VALUE-PATH.md](GUARD-OWN-CACHE-VALUE-PATH.md)); the only
  fak-authored cache value is compaction shed.
- The cache economics the exit summary reports are mostly the provider's own
  `cache_read` numbers — **OBSERVED**, relayed, unverifiable.
- The guard-hop latency RSI loop (`tools/guard_hop_rsi.py`) is stuck
  `PENDING_MEASUREMENT`: pricing the hop needs a serve baseline we control;
  against a provider, network + queue jitter swamps the ~µs adjudication cost.
- Enforcement is post-generation: a denied tool call was already fully decoded
  (and paid for) before `DEFAULT_DENY` fires.

Swap the upstream for `fak serve --engine inkernel` (the pure fak kernel,
`internal/model/` — GLM-5.2 = `glm_moe_dsa`, `glm_dsa_session.go`) and each cap
inverts. That inversion **is** the benchmark: the same guarded workload, upstream
ablated, and the guard-value deltas read off instruments we already shipped.

## 2. The ablation matrix

Two planes, fully crossed on one frozen workload (solved SWE-bench Verified
smoke instance(s), per the
[GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK](../benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md)
"solved ticket" discipline — correctness checkable from gold evidence, no
dependence on fast full-patch generation):

| Arm | Upstream (serving plane) | Who controls the kernel | Guard plane |
|---|---|---|---|
| **K1** | `fak serve --engine inkernel --backend cuda` GLM-5.2 on GPU server | fak (full: weights + kernel + KV + decode loop) | on / off |
| **K2** | vLLM or SGLang serving the same GLM-5.2 weights on the same box (`tools/glm52_sglang_vllm_serve.sh`) | us (weights + box), not fak (kernel) | on / off |
| **A1** | Z.ai GLM-5.2 API (same model family, provider-hosted) | nobody we trust | on / off |
| **A2** (optional) | Anthropic API via today's guard subscription posture | nobody we trust | on / off |

- **K1 vs A1** is the headline ablation: *same model*, kernel control as the only
  serving-plane variable. A2 is cross-model and is only read for guard-value
  *shape* (verdict mix, refusal behavior), never for model comparison.
- **K2 splits "self-hosted" from "fak-kernel"**: it shows which deltas come from
  merely owning the box (clean latency, no provider credential) vs from fak
  *being* the kernel (witnessed KV counts, deterministic decode, constrained-
  decode headroom).
- Guard-off arms (agent → upstream direct) price the hop and the refusal set.
- Within-arm discipline follows `internal/ablate` (one `WorkloadHash` across
  arms); the cross-upstream halves are the nondeterministic regime and follow
  `tools/cross_agent_ablate.py` rules (report distributions, never single-run
  deltas, for anything the model's sampling can move).

## 3. What ONLY the kernel arm can prove (the five axes)

Each axis names its seam (shipped today) and its witness.

1. **Witnessed economics.** K1's cache value is fak-authored:
   `kv_prefix.reused_tokens` / `prompt_tokens` / regime turns off `/metrics`,
   folded by `fak swebench cache-witness` (`internal/cachewitness/`, shipped,
   commit `52dfea0d`). In A-arms the same slot is provider `cache_read` —
   OBSERVED, and never verifiable. **Witness:** cache-witness.json per arm;
   guard exit summary `fak_share > 0` only in K1.
2. **Clean overhead pricing.** With the serve baseline ours, the guard-hop
   TTFB delta closes the `tools/guard_hop_rsi.py` loop for real (adjudication
   is ~1.3 µs/decide per
   [EXPLAINER-gate-down-the-stack](EXPLAINER-gate-down-the-stack-2026-06-22.md);
   the claim "guard costs ≈ nothing next to decode" becomes a measured ratio,
   not an argument). **Witness:** guard-on vs guard-off TTFB/turn distribution
   on K1/K2; the same measurement on A1 documents the provider-jitter noise
   floor that makes it unmeasurable there.
3. **Enforcement-depth headroom.** The engine already has logit-level
   constrained decode (`internal/model/constraint.go` `GenerateConstrained`).
   Today the guard's deepest move is post-generation `TRANSFORM`/`DENY`; under
   kernel control the deny can move *before* generation. The benchmark
   measures the gap as **tokens decoded for calls the guard then denied**
   (destructive-canary turns, per the `trychatdemo` `liveWitness` pattern:
   append a `delete_account`-class canary, assert `POLICY_BLOCK`). K1 makes
   that waste eliminable (wiring is the
   [grammar-constrained decoding research note](RESEARCH-grammar-constrained-tool-call-decoding-2026-06-22.md)
   — **not yet**, and this benchmark prices exactly what it would buy).
   **Witness:** per-arm canary refusal rows in the hash-chained journal +
   wasted-decode token counts.
4. **Credential-surface elimination.** In A-arms the guard's best posture is
   holding the OAuth/API key upstream and handing the child a placeholder
   (`guard_child.go`, #2358 secret-strip floor) — the secret still *exists* on
   the host. K1/K2 have **no provider credential at all**: the exfil class the
   guard defends (AUDIT-fak-guard-secret-exfil-2026-06-28.md) is structurally
   empty. **Witness:** child env dump per arm (names only), zero credential
   env-vars present in K-arms.
5. **Deterministic replay adjudication.** The fak kernel decodes greedy-argmax
   by default: replaying the identical session (`fak guard --replay-trace` /
   `internal/guardtrace`) should reproduce the identical journal —
   byte-comparable verdict streams, which turns guard-policy changes into
   regression-diffable artifacts. Impossible against an API (sampling +
   server-side drift). **Witness:** two identical K1 runs → journal diff = ∅
   (expected; to be witnessed, not asserted — batch-layout effects are the
   known risk to check).

## 4. Why GPU server specifically (the #996 wall is a fit problem)

The prior kernel run decoded at ~0.03–0.17 tok/s under `--cpu-offload-experts`
(#996/#971): GLM-5.2's ~0.45 TB UD-Q4_K_M can't fit older single/dual-GPU hosts,
and host-RAM expert GEMM is the wall. That figure is a reading of *that host's
offload posture*, not the kernel's ceiling. GPU server class boxes change the fit, not
the code:

- **The sm_80 8-GPU class (8×80 GB = 640 GB aggregate)**: Q4_K_M weights fit aggregate VRAM with
  `--expert-parallel` / `--tensor-parallel` sharding; KV + activation headroom is
  tight and is itself a first datum (staging script exists:
  `tools/glm52_stage_serve_dgx3.sh`, `tools/glm52_mgpu_serve.sh`,
  `tools/dgx_pure_kernel_bench.sh`, sm_80).
- **The sm_90+ 8-GPU class (8×141 GB = 1128 GB aggregate)**: fits with real headroom; also the
  lane for the K2 arm (`tools/glm52_sglang_vllm_serve.sh`) and the open
  `witness-glm52-vllm-throughput-parity` nightrun row.
- GLM-5.2 is the right subject: open weights (MIT, released 2026-06-13; 753B
  MoE, ~40B active — [models.dev](https://models.dev/models/zhipuai/glm-5.2/),
  [eigent.ai](https://www.eigent.ai/blog/glm-5-2)), API-comparable capability, and
  the *same model* is reachable in K1/K2/A1 — no cross-model confound on the
  headline ablation.

Guard-value proof does **not** require frontier decode speed: the solved-ticket
protocol (#1010) already routes around throughput, and axes 1/3/4/5 are
throughput-independent. Axis 2's ratio just gets more favorable as decode gets
real.

## 5. Metric sheet (with trust classes — conflation discipline)

| Metric | Arms | Class | Instrument |
|---|---|---|---|
| `kv_prefix.reused_tokens`, `prompt_tokens`, frozen/partial/cold turns | K1 | **WITNESSED** | `/metrics` → `fak swebench cache-witness` |
| provider `cache_read` tokens | A1/A2 | **OBSERVED** | gateway usage ledger (`internal/gatewayusageledger`) |
| vLLM/SGLang prefix-cache counters | K2 | **OBSERVED** (our box, not our kernel) | engine metrics, relayed |
| compaction-shed savings rows | all guard-on | **WITNESSED** | `internal/cachevaluereport/track2.go` → `docs/nightrun/cache-savings.jsonl` |
| guard-hop TTFB delta (on−off) | K1, K2 (A1 = noise-floor documentation) | **WITNESSED-derived** timing | `tools/guard_hop_bench.py` protocol per turn |
| verdict mix, `DEFAULT_DENY`/`POLICY_BLOCK` counts, canary blocks | all guard-on | **WITNESSED** | hash-chained journal (`.dispatch-runs/guard-audit/`), `fak guard-verdict-rsi fold` |
| wasted-decode tokens on denied calls | all guard-on | **WITNESSED** (K1) / OBSERVED usage (A) | journal + usage rows joined per turn |
| credential env-var names present in child | all | **WITNESSED** | child env audit (names only, never values) |
| replay journal diff | K1 | **WITNESSED** | two runs, byte diff |
| decode tok/s, wall-clock | all | **OBSERVED** (a reading of the box/provider) | never attributed to a fak action |
| $ per session | A vs K | **OBSERVED** both (API billing vs amortized box) | labeled, never mixed into witnessed rows |

No number sums across the WITNESSED/OBSERVED line; the packet inherits the
[GLM52 results-doc honesty fence](../benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md)
verbatim.

## 6. Runnable today vs not-yet

**Witnessed today — the CPU-scale pilot (all five axes, no box):**
A runnable stand-in for the GPU server matrix landed 2026-07-03 at
[`experiments/benchmark/runs/by-machine/desktop/20260703T124500Z-guard-kernel-pilot/`](../../experiments/benchmark/runs/by-machine/desktop/20260703T124500Z-guard-kernel-pilot/RESULTS.md).
Same seams, same witnesses, model swapped down to Qwen2.5-1.5B Q8_0 served by
`fak serve --engine inkernel` (the kernel we control), driver run guard-off vs
guard-on. Results, all WITNESSED on this host:
- **Axis 1:** fak's own KV-prefix cache BIT — 615/1370 prefill tokens reused
  (44.9%), provider `cache_read` = 0 (no provider).
- **Axis 2:** guard-on−off single-turn TTFB delta = −0.056 s median — the hop is
  within measurement noise of a ~4.5 s decode (the "guard ≈ free next to decode"
  claim, measured).
- **Axis 3:** a forced `wipe_disk` tool call DENIED (`DEFAULT_DENY/TERMINAL`,
  denied before a wasted round-trip), in a hash-chained journal that
  `fak audit verify` confirms intact.
- **Axis 4:** zero provider credentials in the K-arm child env.
- **Axis 5:** two independent gateway processes produced byte-identical KV
  counters on the identical workload — greedy-argmax determinism.

Also runnable-today without any box: the deterministic prefill-elimination floor
(A/C 17.9×→23.4×, dos-bound offline — the geometry ceiling the K1 cache value
chases), guard verdict-RSI folding on any guarded journal, and the nightrun
dispatch rows below.

**`not yet` (missing witness = a GPU server class box session, the #1012 residual):**
- The **GLM-5.2** per-arm numbers and the **K2 (vLLM/SGLang)** arm from §5 — the
  pilot proves the harness and every witness end-to-end at CPU scale, so the GPU server
  run is a substitution of model+box, not new wiring. Next checkable step: a box
  that satisfies the nightrun rows below picks them up (`fak nightrun plan`
  surfaces them as manual frontier recipes); first artifact is one K1 GLM-5.2
  guard-on run producing journal + cache-witness.json + usage rows.
- Moving axis-3 enforcement *before* generation (constrained decode) still needs
  the guard→kernel grammar wiring; the pilot prices what it would buy (today the
  deny fires pre-round-trip but post-decode).

## 7. Insertion points

- **Dispatch:** two overlay rows in `experiments/nightrun/backlog.json` —
  `witness-guard-kernel-ablation-mgpu` (the matrix, manual, frontier) and
  `witness-guard-hop-live-kernel` (axis 2 alone; cheap, any CUDA box with any
  supported weights — it does not need GLM-5.2).
- **Tracker:** [#2509](https://github.com/anthony-chaudhary/fak/issues/2509)
  (child of epic #1010; acceptance criteria live there); adjacent: #1012 (solved-ticket replay), #1846 (live-session
  ablation arm), #996/#971 (the offload wall this routes around by hardware).
- **Authority path:** artifacts graduate via BENCHMARK-AUTHORITY.md only after
  `dos commit-audit` on the results commit — self-reported numbers never ship.
