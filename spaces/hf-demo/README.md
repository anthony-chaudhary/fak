---
title: fak — Agent Kernel, Proven Offline
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
tags:
  - agents
  - llm-security
  - prompt-injection
  - kv-cache
  - mcp
short_description: Adjudicate every tool call like a syscall; provably evict a poisoned result. Three witnesses, no key, no GPU.
---

# fak — the agent kernel, proven offline

`fak` treats every tool call like a **syscall**: it is adjudicated at a real boundary
*before* it runs, and the kernel **owns the KV cache**, so a poisoned result can be
*provably* evicted — the surviving context comes out byte-identical to a run that never
saw it (`max|Δ|=0`).

This Space runs three of fak's own witnesses **live, offline — no API key, no model
weights, no GPU.** Each tab prints the literal command it ran, the raw output, and the
one honest fence that bounds what that output proves.

| Tab | Command | What it shows |
|---|---|---|
| **1 · Policy floor** | `fak preflight --policy examples/customer-support-readonly-policy.json --tool <t> --args {}` | `refund_payment` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW` — the same runtime-loaded JSON policy, two structural verdicts. |
| **2 · Provable deletion** | `deletioncert -selfcheck` | Evict a secret span, prove `evicted == never-saw` (`max|Δ|=0`), mint a tamper-evident certificate, watch verification **fail closed** on a forged cert/journal. |
| **3 · Turn tax** | `turntaxdemo -print -suite turntax-airline` | A tuned 2026 SOTA agent is **forced** into 5 recovery round-trips; fak resolves each inside the syscall and stays flat at **0**. |

## Lead with the fence

fak's credibility is its fences and its zeros. Read these before the demo, not after:

- The prompt-injection detector is **~100% evadable by design** — it is explicitly *not*
  the floor. The floor is the **policy** (tab 1): a declarative, version-tagged manifest
  that decides *which* tools may be called, validated against a closed reason vocabulary.
- The deletion certificate is a **self-signed v1** receipt: it attests the integrity of
  the recorded facts, not independence from the recorder, and `evicted_count` is a
  self-report. The **numeric** bit-exactness is proven separately against a **HuggingFace
  oracle** — a pure-Go SmolLM2-135M forward pass with embedding exact, per-layer
  **cos=1.000000**, final-logits **max|Δ|≈4.4e-5**, and KV-decode / KV-evict
  **token-for-token identical (max|Δ|=0)** — in `go test ./internal/model`.
- The turn-tax headline is fak vs a **tuned 2026 SOTA agent** (the forced round-trips it
  cannot elide), never a naive baseline; the safety floor is a **separate axis**. fak is
  **not a faster token engine** — the win is deleted model round-trips, not tok/s.
- Third-party model weights are **not fak's artifact**: the oracle above validates fak's
  own forward pass *against* HuggingFace `transformers`; it makes no claim about any
  uploaded checkpoint's quality, licence, or provenance.

## Run it locally

```bash
docker build -t fak-hf-demo spaces/hf-demo
docker run --rm -p 7860:7860 fak-hf-demo
# open http://localhost:7860
```

The build git-clones the public repo and compiles the three binaries from source, so
this image is reproducible from just `Dockerfile` + `app.py` + this README. Pin a release
with `--build-arg FAK_REF=<release-tag>` — the newest published tag at the time of writing
is `v0.44.0`.

## More

- **Repo & honesty ledger:** <https://github.com/anthony-chaudhary/fak> ·
  [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every number
  here traces to a `[SHIPPED]` line with a test witness.
- **Live demos:** <https://anthony-chaudhary.github.io/fak/demos.html>
- **Runnable notebooks:**
  [Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb)
  (free T4) · in-kernel decode on Lightning AI / RunPod.

Apache-2.0.
