# Qwen3.6-27B on this M3 Pro — parity & measurement status (2026-06-20)

Three claims about the Qwen3.6-27B (hybrid Gated-DeltaNet) bring-up on this node, each
**proved or refuted against evidence the author did not author** — captured artifacts,
the live machine, and the public arch record — rather than asserted. The honesty rule of
[`../../docs/proofs/00-METHOD.md`](../../docs/proofs/00-METHOD.md) applies: a counter-witness
is recorded with its counterexample, not rounded away.

**Box (all three claims):** `Mac15,7` — Apple M3 Pro, 12 core (6P+6E), **36 GiB**
(`hw.memsize = 38654705664`), arm64, darwin, go1.26. Model:
`~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf` (hybrid GDN, 64 layers = 48 linear_attn
+ 16 full_attn).

---

## 1 — REFUTED: fak's in-kernel decode is **not** at correctness parity yet (token-3 drift)

**Claim under test.** "fak's in-kernel greedy decode of Qwen3.6-27B reproduces the
llama.cpp reference token-for-token (multi-token greedy parity) on the q4_k_m GGUF."

**Verdict: REFUTED.** First-token parity holds, but the greedy continuation **diverges at
the third token**. Same GGUF file, same prompt, same decode policy (`temperature=0`,
`top_k=1`):

| step | fak (in-kernel, `cmd/qwen35check`) | llama.cpp b9707 Metal oracle | match |
|---|---|---|---|
| 0 | `248068` `<think>` (logit 28.30) | `248068` `<think>` (logprob −0.0008, top) | ✅ |
| 1 | `198` `\n` (logit 27.15) | `198` `\n` (logprob −0.647, top) | ✅ |
| **2** | **`8160` `Here`** (logit **23.18**), 2nd `90700` (21.43) | **`90700` `Thinking`** (logprob **−0.547**), 2nd `8160` (−0.945) | ❌ |

fak generated `[248068, 198, 8160]`; the oracle generated `[248068, 198, 90700]`. At step 2
**both engines surface the same top-2 set `{8160, 90700}` but rank it oppositely** — fak puts
`8160` ≈1.75 logits above `90700`; llama.cpp puts `90700` ≈0.40 nats above `8160`. It is a
**near-tie argmax flip**, the signature of accumulated numerical drift reaching the decision
boundary by token 3, not a gross error — token 0's **argmax** matches (fak logit 28.30; the
oracle pins it at logprob −0.0008, i.e. near-certain, but its artifact reports no raw logit to
compare digit-for-digit).

**Counter-witnesses (on disk, deterministic):**
- fak side — [`native-gguf-q8-multitoken-parity-20260619.json`](native-gguf-q8-multitoken-parity-20260619.json)
  (`expect_match: false`, `generated_ids:[248068,198,8160]`, `expected_ids:[248068,198,90700]`;
  `model.source` is the **q4_k_m** GGUF — the "q8" in the filename names fak's Q8-resident decode
  series, not the file).
- oracle side — [`llamacpp-qwen36-multitoken-oracle-20260619.json`](llamacpp-qwen36-multitoken-oracle-20260619.json)
  (`llama-server -m Qwen3.6-27B.q4_k_m.gguf -ngl 99`, `tokens:[248068,198,90700]`).

**Reproduce (this box; needs the gated 27B GGUF, so this is a weight-backed / "Optional" rung).**
`-expect` makes `qwen35check` a self-checking witness — it **exits non-zero** here because fak
produces `8160` where the oracle pins `90700` at step 2:
```bash
cd fak
go run ./cmd/qwen35check -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  -n 3 -topk 5 \
  -ids 248045,8678,198,2523,513,264,10631,17313,13,248046,198,248045,846,198,44240,10092,13,248046,198,248045,74455,198 \
  -expect 248068,198,90700   # -> mismatch at step 2 (fak 8160 != oracle 90700), exit != 0
```

**Scope (what this does and does NOT refute).**
- It refutes **Qwen3.6 (hybrid-GDN) multi-token greedy parity** specifically. It does **not**
  touch the llama-family forward parity that *is* PROVEN
  ([`../../docs/proofs/model-forward-parity.md`](../../docs/proofs/model-forward-parity.md)
  THEOREM 1, smollm2-135m, single-step argmax + greedy ids), nor the **first-token** parity
  (which holds here and in the batched-prefill witness `q4kdiag`).
- The **root cause is not yet isolated** — candidates are the f32 Gated-DeltaNet recurrent scan,
  the int8 SDOT decode reduction rounding, or attention/softmax accumulation order vs
  llama.cpp's Metal kernels. The refutation is of the *parity claim*, not a diagnosis. Closing
  it to PROVEN requires fak to reproduce the oracle's `[248068,198,90700]` greedily on the same
  GGUF; until then the obligation is an honest **REFUTED**, recorded in the proof ledger as a
  finding (not deleted).

This is the operative meaning of "**fak isn't correctness-parity yet**": the shipped decode
path is *forward-parity-PROVEN on llama and first-token-parity on Qwen3.6*, but its **greedy
continuation on the hybrid-GDN model fails at token 3**.

---

## 2 — INVALID BAR: MLX is **not** a valid measured throughput bar for this hybrid-GDN arch

**Claim under test.** "MLX (Apple `mlx-lm`) is a valid measured performance bar for
Qwen3.6-27B on this Apple-Silicon node."

**Verdict: REFUTED / not a valid bar.** Qwen3.6 is a **hybrid Gated-DeltaNet** model — a 3:1
interleave of linear-attention (GDN recurrence) and full-attention blocks. `mlx-lm`'s hybrid
**cache is broken for this architecture class: it silently recomputes the full context every
turn for Qwen3-Next-style hybrids** (a cache-class defect in mlx-lm, not a Qwen problem). A
tok/s figure produced under a silently-recomputing cache does **not** measure the architecture
the model actually runs — it measures an O(n²)-per-turn fallback — so it cannot be an
apples-to-apples bar for the cached hybrid decode fak (and llama.cpp) implement. The public
guidance is explicit that, until that is fixed, **GGUF / llama.cpp is the practical path on
Apple Silicon** for this family.

**Consequences for our numbers.**
- We have **never** used MLX as a bar here — **zero** `mlx` references in any fak code or
  measurement artifact (`grep -rli mlx fak/` now matches only these status docs, which name MLX
  to say we don't use it). The only measured bar is **llama.cpp b9707 Metal** (`-ngl 99`),
  which *does* implement the cached hybrid GDN: **pp22 = 51.55 tok/s**
  ([`mac-hybrid-prefill-20260620/`](mac-hybrid-prefill-20260620/README.md)). That bar stays.
- Adding an MLX column would not strengthen the comparison; it would inject a number whose
  denominator (recompute-every-turn) is a different computation. So MLX is recorded as an
  **invalid bar for this arch**, not merely "not yet run."

**Evidence:** vLLM's Qwen3-Next writeup and the LM Studio / community Apple-Silicon reports of
the mlx-lm hybrid-cache recompute behavior (see Sources). This is a *public arch-support* fact,
independent of our box.

---

## 3 — FLAGGED: the swap-contaminated live tok/s is **not** trusted over the clean on-disk numbers

**Claim under test.** "The live, co-resident tok/s figures are a trustworthy measure of fak's
Qwen3.6 throughput."

**Verdict: FLAGGED (contaminated).** The live prefill figures (0.5 → 0.6 → 0.8 tok/s across the
per-token / batched-f32 / batched-int8 paths) were captured **with the llama-server oracle
co-resident in the same session** (stated in the hybrid-prefill README). On a **36 GiB** box that
is a guaranteed memory-pressure / swap regime:

- fak's resident footprint for the 27B q4_k_m is **≈ 23.8 GiB** (`resident_split_27b`: q4k
  7942.5 MiB + q8 11604.4 MiB + f32 4860.1 MiB = 24407 MiB).
- The co-resident `llama-server` holds the **same 27B** on Metal (≈ another ~16 GiB of weights +
  KV/overhead). Two copies ⇒ **~40 GiB demand on a 36 GiB box**.
- **Live witness on this machine, right now:** `sysctl vm.swapusage` →
  `total = 7168.00M used = 5995.81M free = 1172.19M` — the box is **actively paging ~5.9 GiB to
  swap**. A tok/s sampled under that pressure is dominated by page-fault stalls, not fak compute.

So the live co-resident tok/s is a **floor under swap**, not fak's compute-bound throughput, and
**must not override the clean on-disk numbers**:
- the **per-phase `FAK_QPROFILE` compute breakdown** (mlp_gate_up 43%, mlp_down 20%, GDN
  recurrence 0.5%, …) — attributes the wall to specific GEMM kernels; the *relative* structure is
  swap-independent and is the honest "where the time goes."
- the **isolated kernel-latency microbenchmarks** in
  [`../mac-m3pro-kernel-20260620/`](../mac-m3pro-kernel-20260620/) — measured without a second
  27B resident.
- the **bar** (llama.cpp Metal 51.55 tok/s pp22) is itself a GPU-resident figure and is the
  comparison anchor; the CPU-batched fak path's *kernel-speedup* deltas (1.58× int8 on
  mlp_gate_up, etc.) are the trustworthy fak-side claims, because they are ratios measured within
  the same (contaminated) session and so cancel the swap tax.

**Rule of record:** when a live co-resident tok/s and a clean on-disk per-phase/kernel number
disagree, **trust the clean on-disk number**; cite the live tok/s only as a swap-bounded floor,
labelled as such.

---

## One-line status

fak Qwen3.6: **correctness — REFUTED at token 3** (not parity yet); **MLX — invalid bar** for the
hybrid GDN (use llama.cpp Metal, 51.55 tok/s pp22); **live tok/s — swap-contaminated floor**, trust
the clean on-disk per-phase/kernel numbers. Token-3 finding is bound into the proof ledger
([`../../docs/proofs/model-forward-parity.md`](../../docs/proofs/model-forward-parity.md) THEOREM 3,
REFUTED).

### Sources (claim 2, public arch-support facts)
- vLLM — *Now Supports Qwen3-Next: Hybrid Architecture with Extreme Efficiency* — https://blog.vllm.ai/2025/09/11/qwen3-next.html
- Qwen3-Next on Apple Silicon (hybrid + SWA, cache behavior) — https://github.com/QwenLM/Qwen3.6/discussions/139
- Qwen3 Next (GGUF + MLX availability / Apple-Silicon path) — https://lmstudio.ai/models/qwen/qwen3-next-80b
