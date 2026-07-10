---
title: "Concept study: DFlash — block-diffusion speculative decoding (NVIDIA / Z Lab) → fak (2026-07-10)"
description: "A mechanics-first, source-graded read of DFlash — the single-pass block-diffusion drafter shipped into vLLM via Speculators v0.5.0 as a config-only EAGLE-3 swap. Honest fak lens: no code borrow (fak never touches a KV tensor); the transferable surface is the *parallel-block-speculation* shape and the *config-only method swap* ergonomic, mapped to served-inline (`--vdso-proxy-fill`) and the vLLM adapter lane."
---

# Concept study: DFlash — block-diffusion speculative decoding → fak

A **mechanics-first** read of **DFlash** (Z Lab; highlighted by NVIDIA in June 2026), graded by source
strength the way this repo grades serving studies (**confirmed** = primary docs/repo/paper; **vendor** =
NVIDIA/first-party benchmark; **reported** = secondary/press; **unverified** = hedge). It closes with the
honest fak lens: DFlash is a **GPU-internal drafter** and fak is a prompt / tool-page / session /
cache-value + gateway layer that **never touches a KV tensor**, so there is **no code borrow** — only a
*conceptual* borrow (parallel block speculation) and an *ergonomic* one (config-only method swap), plus a
*consumption* path through the vLLM adapter lane.

Companion to the spec-decode section of the vLLM study
([`docs/serving/vllm-internals-study.md` §2.7](../serving/vllm-internals-study.md)), which currently only
*lists* `dflash` among the checkpoint-based methods — this note is the full treatment.

## 1. What DFlash is (confirmed)

DFlash replaces the **autoregressive draft head** of EAGLE-3 with a **lightweight block-diffusion draft
model** that predicts an **entire block of B tokens in a single forward pass**, conditioned on **hidden
states extracted from the target (verifier) model**. The target then verifies all B+1 positions in one pass,
exactly as in any speculative-decoding scheme — so DFlash is **lossless-in-expectation** like EAGLE, and the
change is confined to *how the draft is produced*, not the verify/accept rule.

The bottleneck it attacks: in EAGLE-3 the draft head is **sequential** — to draft K tokens you pay K draft
forward passes, each depending on the last. DFlash's block-diffusion drafter is **single-pass**, so K draft
tokens cost roughly the same as 1. This is the whole thesis: *drafting is no longer serialized against draft
depth.* (Source: Z Lab `z-lab/dflash`; arXiv:2602.06036 "DFlash: Block Diffusion for Flash Speculative
Decoding," Jian Chen / Yesheng Liang / Zhijian Liu; vLLM Speculators DFlash algorithm docs.)

## 2. The pipeline inside Speculators / vLLM (confirmed)

DFlash rides the **Speculators** library, which links the drafter to the target's hidden states in the vLLM
inference path. Stages, per the Speculators DFlash algorithm page:

1. **Hidden-state extraction** — the verifier processes the prompt/context and yields intermediate hidden
   states (Speculators v0.5.0 migrated this to vLLM's **native hidden-states extraction system**).
2. **Anchor points + masking** — DFlash picks anchor points in the sequence and appends **mask tokens** for
   the block positions to be predicted. *The block structure is realized entirely through the attention
   mask* — no bespoke kernel surgery beyond the mask.
3. **Parallel draft** — the draft layers process context features **and** the mask tokens together in **one
   forward pass**; output is projected through the **target LM head** to produce vocabulary logits for the
   speculated block.
4. **Verify** — standard target forward over the B+1 positions; standard rejection/accept.

Reference eval configuration (confirmed, Speculators docs): **block size 16, a single denoising step**;
the EAGLE-3 baseline it is compared against uses speculation length 7 (`RedHatAI/Qwen3-8B-speculator.eagle3`).

## 3. Config-only integration — the actual seam (confirmed)

The headline ergonomic: **swapping EAGLE-3 → DFlash is a config-only change** (vLLM project's own X post,
2026; NVIDIA blog). In vLLM it is selected via `--speculative-config` with `"method": "dflash"`:

```bash
vllm serve <target-model> \
  --speculative-config '{"method": "dflash", "model": "<dflash-checkpoint>", ...}'
```

Z Lab's own Gemma-4 example ships as a Docker image because Gemma-4 currently needs a **temporary vLLM
build**:

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130 \
  google/gemma-4-26B-A4B-it --host 0.0.0.0 --port 8000 \
  --speculative-config '{"method": "dflash", ...}'
```

SGLang reached DFlash support earlier and uses different flags:
`--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-Coder-30B-A3B-DFlash
--tp-size 1 --dtype bfloat16 --attention-backend fa3` (confirmed, Z Lab / SGLang docs).

**Tuning shift (confirmed):** because drafting is single-pass, **higher K is cheaper on DFlash than on
EAGLE-3** — with autoregressive drafting a larger K means more sequential draft steps; with DFlash the K
tokens are one pass. So the K/acceptance tradeoff moves in DFlash's favor.

## 4. Performance — graded by source

- **Paper / Z Lab (paper-reported):** up to **6× lossless acceleration on Qwen3-8B**, ≈**2.5× faster than
  EAGLE-3**. Reasoning/thinking mode: ≈**4.5× greedy** on Qwen3-4B/8B, ≈**3.9× under sampling** (reported —
  hedge; workload- and decode-mode-specific).
- **NVIDIA (vendor-reported):** up to **15× higher throughput on Blackwell**; on **Gemma-4 31B, single
  Blackwell Ultra GPU**, **5.8× throughput at the same concurrency** vs autoregressive decode — Math500 5.8×,
  GSM8K 5.3×, HumanEval 5.6×, MBPP 4.4×.
- **Third-party (reported, weak):** Spheron ("6× on GPU cloud"), press writeups (MarkTechPost / TechTimes) —
  echo the vendor/paper numbers; **do not lean on these independently**.

**Grading discipline (same as vLLM §2.7):** speculative decoding is a **latency** win that **collapses toward
zero at high concurrency** once the GPU is compute-bound — every DFlash speedup number is therefore
**concurrency- and hardware-specific**, and the biggest figures are **Blackwell-gated vendor benchmarks**.
State the batch/hardware with any claim; treat 15× as a vendor ceiling, not a portable expectation.

## 5. Requirements & caveats (confirmed)

- **vLLM nightly** (or a release with DFlash support) — until it lands in a stable tag, use
  `vllm/vllm-openai:nightly`; Gemma-4 needs the temporary `ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130`
  build.
- **Hardware validation is partial** — "not all hardware configurations have been validated yet; refer to
  model cards." The marquee numbers are **Blackwell**.
- **Online training OOM hazard** — DFlash training is typically **online**: hidden states are extracted
  on-the-fly from a *running* vLLM server to train the speculator, so trainer + server share the GPU and you
  must **isolate GPU resources to avoid OOM** (Speculators v0.5.0 tutorial).
- **Checkpoints:** pretrained DFlash speculators exist on HuggingFace (RedHatAI speculator collection).

## 6. The fak lens — is there a borrow?

**Verdict: no direct/code borrow. Two candidate *conceptual/ergonomic* borrows + one consumption path.
None shipped — all `not yet`, with a checkable next step.**

fak never computes a draft token or touches a KV tensor (see the M2 lens in
[`CONCEPT-STUDY-VLLM-M2-2026-07-10.md`](CONCEPT-STUDY-VLLM-M2-2026-07-10.md): the transferable surface is
always *one level up*). So DFlash's diffusion kernel is out of scope by construction. What maps up:

- **(a) Conceptual — parallel-block speculation over autoregressive.** fak already *speculates*: the
  **served-inline** path (`fak serve --vdso-proxy-fill`) answers read-only-shaped tool calls from a warmed
  cache instead of executing them (gate `readOnlyPrefix` at `internal/gateway/adjudicate_proposed.go:66`;
  flag `vdsoProxyFill` at `internal/gateway/gateway.go:697`; served-inline path at
  `internal/gateway/adjudicate_proposed.go:224`). Today that path is **per-call** (one tool → one inline
  fill). DFlash's shape suggests a **block** analogue: draft a *block* of anticipated read-only tool-fills in
  one speculative pass and let the turn "verify" (accept the prefix that the model actually asks for),
  amortizing the per-call adjudication overhead — exactly DFlash's single-pass-vs-sequential win, one level
  up. **Status: `not yet` — conceptual only.** Next checkable step: measure whether real transcripts issue
  read-only tool calls in *runs* (blocks) frequently enough to make block-drafting pay (per the served-inline
  measurement in [[served-inline-name-gate-blocks-claude-native]]: today served-inline is 0% on Claude
  Code's native Read/Grep/Glob because those don't match `readOnlyPrefix` — so this borrow is *gated on
  fixing the name gate first*, and should not be pursued before that).

- **(b) Ergonomic — config-only method swap.** DFlash's "swap the checkpoint reference, one config line, no
  code" is a design pattern for **method selection by config**. fak's `dispatch_model_policy` selects models
  by policy; a spec-decode-*method* knob (were fak ever to front a spec-decode backend) belongs in that same
  config surface, not in code. **Status: `not yet` — no fak surface consumes a spec-decode method today.**

- **(c) Consumption — the vLLM adapter lane.** If/when fak serves through a vLLM backend (adapter lane
  #1729–#1734; epic #40 vLLM adapter), `method: dflash` is a **serving-config passthrough**, not a fak
  feature. Any perf claim from it is **Blackwell-hardware-gated** and must follow the #40 acceptance-split
  discipline ([[issue-40-vllm-adapter-acceptance-split]]): the live-serving / parity items are
  hardware-gated — **witness them on a Blackwell box or leave them `not yet`; do not fabricate the gated
  numbers.**

**No leaf filed** this pass: (a) is blocked on the served-inline name gate, (b) has no consuming surface,
and (c) is a passthrough on an already-tracked lane. Re-filing any of these as an issue now would be a
false-bound rider, not a borrow — anti-re-file discipline. This note is the record; a leaf follows only when
(a)'s name-gate blocker clears and transcripts show block-shaped read-only runs.

## 7. Source ledger

Primary (confirmed):
- Z Lab — project page `z-lab.ai/projects/dflash`, repo `github.com/z-lab/dflash`.
- arXiv:2602.06036 — "DFlash: Block Diffusion for Flash Speculative Decoding" (Chen, Liang, Liu).
- vLLM Speculators docs — DFlash algorithm page `docs.vllm.ai/projects/speculators/.../algorithms/dflash/`.
- vLLM blog — "Speculators v0.5.0: DFlash Support and Online Training" (2026-05-28);
  Red Hat Developer mirror (2026-06-04).
- vLLM project on X — DFlash config-only-swap announcement (2026).

Vendor:
- NVIDIA Technical Blog — "Boost Inference Performance up to 15x on NVIDIA Blackwell Using DFlash Speculative
  Decoding" (2026-06-24).

Secondary (reported — weak, do not lean on):
- MarkTechPost (2026-06-24), TechTimes (2026-06-27), Spheron blog, regolo.ai tutorial, Allen Kuo (Medium).

fak seams cited (verified in-tree at this HEAD):
- `internal/gateway/adjudicate_proposed.go:66` (`readOnlyPrefix` gate), `:224` (served-inline path).
- `internal/gateway/gateway.go:697` (`--vdso-proxy-fill` flag), `:1460` (`vdsoProxyFill`).
