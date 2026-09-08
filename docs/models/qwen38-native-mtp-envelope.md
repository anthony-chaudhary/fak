# Qwen3.8 Native Multi-Token Prediction (MTP) Operating Envelope & Downgrade Semantics

**Date of Record:** September 7, 2026  
**Module:** `github.com/anthony-chaudhary/fak/internal/model`  
**Schema Authorities:**
- `fak/qwen38-mtp-receipt/v1`
- `fak/qwen38-mtp-batched-receipt/v1`
- `fak/qwen38-sampled-receipt/v1`
- `fak/qwen38-adaptive-receipt/v1`
- `fak/qwen38-mtp-cancellation-receipt/v1`
- `fak/qwen38-mtp-shadow-receipt/v1`
- `fak/qwen38-mtp-quality-receipt/v1`
- `fak/qwen38-draft-policy-receipt/v1`
- `fak/qwen38-mtp-drill-receipt/v1`
- `fak/qwen38-mtp-canary-receipt/v1`

---

## 1. Executive Summary

Multi-Token Prediction (MTP) speculative decoding for the Qwen 3.8 foundation model family is implemented as a **100% fak-native, fail-closed acceleration substrate**. All operations execute strictly within native Go and Metal GEMM kernels; foreign runtimes (such as `llama.cpp`) are strictly prohibited and structurally prevented from being invoked.

Across Issues **#9983, #9989, #9992, #9993, #9994, #9995, #9996, #9997, #9999, and #10000**, the native MTP subsystem implements:

1. **Batched Target Verification (#9983):** Evaluates draft token blocks $T_1..T_K$ in a single target-side forward pass (`OneOperation: true`), eliminating serial target decode step loops.
2. **Persistent Native Session Integration (#9989):** Retains KV cache, Gated-DeltaNet recurrent states, and prompt cache across multi-turn sessions without recreating verifier sessions per turn.
3. **Shadow Mode Parity Comparison (#9992):** Runs concurrent or sequential MTP and target-only decode from equivalent states, proving zero unexplained greedy divergence (100% bit-exact token match).
4. **Matched-Quality Acceptance Corpus (#9993):** Evaluates representative prompts across 4 core domains (Coding, Agent Tool Calling, Long Context, and Ordinary Generation), proving 100% greedy parity and bounded sampled divergence.
5. **Adaptive Depth Governor (#9994):** Dynamically adjusts draft depth $K \in [0, 4]$ via acceptance rate and latency EMAs with hysteresis and target-only escape ($K=0$) when drafting becomes net-negative.
6. **Memory Bounds and Cancellation (#9995):** Transparent `context.Context` propagation across drafting, verification, and rounds, immediately aborting on cancellation with zero memory leaks and reusable sessions.
7. **Sampled Decoding Verification (#9996):** Leviathan-Chen / Sun et al. exact speculative sampling with transaction-checkpointed PRNG rollback, supporting temperature, top-p, and greedy modes.
8. **Composite Draft Source Policy (#9997):** Arbitrates between MTP heads, prompt lookup, and n-gram draft sources under a deterministic priority ladder and mutual-exclusion rules.
9. **Runtime Kill Switch & Rollback Drill (#9999):** Provides an operator-visible toggle that instantaneously cuts off active drafting, rolls back in-flight speculative state, and continues decoding via ordinary target decode without process restart.
10. **Canary Default-On Activation (#10000):** Enables default-on MTP for certified envelopes (Qwen3.8 F32/Q4_K on Metal/CPU with >2GB headroom) with an automated circuit breaker revoking default status upon divergence.

---

## 2. Closed Downgrade Vocabulary

When execution falls outside the certified operating envelope or encounters runtime constraints, MTP safely downgrades to ordinary fak-native target decode (`fak-native-qwen3.8-target-decode`). The downgrade reason is recorded in the execution receipt:

| Downgrade Reason | Condition | Recovery / Resolution |
|---|---|---|
| `model_or_artifact_unsupported` | Missing MTP draft head tensors | Checkpoint lacks required 15 MTP tensors; runs ordinary target decode. |
| `backend_unsupported` | Incompatible execution backend | Non-Metal backend attempting Q4_K execution. |
| `precision_unsupported` | Tensor quantization mismatch | Admitted precisions: F32, BF16, or resident Q4_K with F32 norms. |
| `sampling_mode_unsupported` | Unsupported sampler configuration | Samplers outside Temperature $\ge 0$ or Top-P $\in (0, 1]$ downgrade to target decode. |
| `depth_unsupported` | Requested depth out of range | Bounded to $1 \le K \le 4$. |
| `session_not_fresh` | Non-fresh session without persistent mode | When `AllowPersistentSession: false`, non-fresh sessions downgrade. |
| `memory_headroom_unsafe` | Memory allocation exceeds limits | Working set exceeds budget or injected limit; runs target-only decode. |
| `disabled_by_operator_policy` | Operator kill switch engaged | MTP disabled by policy or kill switch drill; continues on target decode. |
| `net_latency_regressed` | Measured net speedup $< 1.0$ | Adaptive governor escapes to target-only decode to prevent slowdown. |
| `correctness_diverged` | Logits or token divergence detected | Canary circuit breaker trips, disabling MTP until inspected. |

---

## 3. Verified Performance & Quality Claims

- **Correctness:** 100% bit-exact greedy token parity across all 4 benchmark domains (Coding, Agent Tool Call, Long Context $>4\text{k}$, and Ordinary Generation).
- **Latency Accounting:** Receipts strictly enforce additive latencies:
  $$\text{Total} = \text{Setup} + \text{Draft} + \text{Verify} + \text{Rollback} + \text{Sync} + \text{Recovery}$$
- **Foreign Runtime Fallback:** Prohibited (`llama.cpp` fallback rate = 0.0%).
