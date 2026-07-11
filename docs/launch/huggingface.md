---
title: "fak on Hugging Face — the Space as the offline front door"
description: "The owner-gated playbook for shipping fak's Hugging Face Space (Docker SDK): what to deploy, the copy-paste card metadata, the three witnesses it runs, and the fences that keep it honest. fak is not a model — the Space is the artifact."
---

# Hugging Face — the Space is the demo, not a model

Hugging Face is where the local-LLM / agent-security crowd already lives, and its **Spaces**
are the one surface that lets a skeptic *run* a claim in-browser instead of reading about it.
That's the fit for fak: not a model upload (fak ships no weights — see the fence below), but a
**Docker Space** that runs three of fak's own witnesses live, **offline, no API key, no GPU.**

The scaffold is committed at [`spaces/hf-demo/`](https://github.com/anthony-chaudhary/fak/tree/main/spaces/hf-demo)
— `Dockerfile`, `app.py`, `README.md`. Deploying it is owner-gated (it needs *your* HF
account), so this page is the paste-ready payload, exactly like
[`directory-submissions.md`](directory-submissions.md).

## The one rule (same as every channel)

**Lead with the fence — it's the hook, not the caveat.** The Space's own `README.md` already
opens with a "Lead with the fence" block; keep it there. The HF audience is the same
perf-literate, overclaim-allergic crowd as r/LocalLLaMA — the fences *are* the credibility:

- the injection detector is **~100% evadable by design** — explicitly *not* the floor (the
  floor is the policy, tab 1);
- the deletion certificate is a **self-signed v1** receipt (`evicted_count` is a self-report);
  the **numeric** bit-exactness is proven separately against a **HuggingFace oracle** in
  `go test ./internal/model`;
- fak is **not a faster token engine** — the turn-tax win is deleted model round-trips, not tok/s;
- **third-party weights are not fak's artifact** (the HF-specific fence, below).

## What the Space runs (three tabs, three witnesses)

Each tab shells out to one binary the image built from source and prints the literal command,
the raw stdout, and the one honest fence. All three were re-verified against the live binaries
(2026-07):

| Tab | Command | Verified output | Witness in `CLAIMS.md` |
|---|---|---|---|
| **1 · Policy floor** | `fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args '{}'` | `verdict=DENY reason=POLICY_BLOCK by=monitor`; `--tool search_kb` → `verdict=ALLOW reason=NONE` | "Deployable capability floor" |
| **2 · Provable deletion** | `deletioncert -selfcheck` | `PROVEN: evicted == never-saw (max\|Δ\|=0)` … `OK — provable-deletion certificate minted, verified, and tamper-rejected.` | "Provable-deletion certificate" |
| **3 · Turn tax** | `turntaxdemo -print -suite turntax-airline` | `tuned SOTA agent: 5 forced round-trips` / `fak: 0 extra round-trips` | `cmd/turntaxdemo`, `testdata/turntax/*.json` |

The Space is the **60-second front door** to the heavy proof, not the proof itself. The
HuggingFace-oracle parity behind tab 2 — a pure-Go SmolLM2-135M forward pass, embedding exact,
per-layer **cos=1.000000**, final-logits **max|Δ|≈4.4e-5**, KV-decode/KV-evict
**token-for-token identical (max|Δ|=0)** — is in `go test ./internal/model` (`IN-KERNEL-MODEL-RESULTS.md`).

## The HF-specific fence (say it once, plainly)

fak uploads **no model** to the Hub. The oracle above validates **fak's own** pure-Go forward
pass *against* HuggingFace `transformers` as the reference; it makes **no claim** about any
uploaded checkpoint's quality, licence, or provenance. If anyone reads the Space as "fak is a
model," correct it: fak is the **kernel in front of** a model. This is why the artifact is a
**Space** (a running program) and not a **model card** (a weights listing).

## Deploy it (owner-gated — you run these)

**Create the Space**

1. <https://huggingface.co/new-space> → Owner = your account, **Space name** `fak-demo`,
   **License** `apache-2.0`, **SDK** = **Docker** (blank template), visibility **Public**.
2. The three files in `spaces/hf-demo/` map 1:1 onto the Space repo root:

   | Space repo file | Source |
   |---|---|
   | `README.md` | `spaces/hf-demo/README.md` (already carries the HF YAML front-matter: `sdk: docker`, `app_port: 7860`) |
   | `Dockerfile` | `spaces/hf-demo/Dockerfile` (builder git-clones the public repo → builds `fak`, `deletioncert`, `turntaxdemo`) |
   | `app.py` | `spaces/hf-demo/app.py` (the Gradio front door) |

3. Push them to the Space remote:
   ```bash
   git clone https://huggingface.co/spaces/<you>/fak-demo && cd fak-demo
   cp /path/to/fak/spaces/hf-demo/{README.md,Dockerfile,app.py} .
   git add . && git commit -m "fak offline demo (docker space)" && git push
   ```
   The Space builds automatically. First build is slow (Go compile + `pip install gradio`);
   subsequent pushes are cached.

**Verify before you announce**

- [ ] The Space **builds green** and the three tabs each return output (not a build error).
- [ ] Tab 1 shows **both** verdicts (DENY *and* ALLOW) — the same policy file, two answers.
- [ ] Tab 2 ends with `tamper-rejected` and tab 3 shows `5 forced round-trips` / `0`.
- [ ] The card's **short description** and **lead-with-the-fence** block render above the fold.

Reproduce the whole image locally first if you want:
```bash
docker build -t fak-hf-demo spaces/hf-demo
docker run --rm -p 7860:7860 fak-hf-demo   # http://localhost:7860
```

## Copy-paste card metadata

Already set in `spaces/hf-demo/README.md`; repeated here for the submission record.

| Field | Value |
|---|---|
| Space SDK | `docker` (`app_port: 7860`) |
| Short description | Adjudicate every tool call like a syscall; provably evict a poisoned result. Three witnesses, no key, no GPU. |
| License | `apache-2.0` |
| Tags | `agents`, `llm-security`, `prompt-injection`, `kv-cache`, `mcp` |

## What NOT to do on HF

- **Don't upload a strawman model** to pad a model card — fak has no weights to ship, and a
  filler checkpoint would invite exactly the licence/provenance questions the fence pre-empts.
- **Don't let the Space imply tok/s SOTA.** It runs offline witnesses; it measures deleted
  round-trips and bit-exact eviction, never throughput.
- **Don't claim the Space proves the oracle.** The Space runs the *offline* witnesses; the
  numeric HF-oracle parity is `go test ./internal/model`. Link it; don't conflate it.
- **Don't ask for likes.** Same reflex as upvote-begging elsewhere — it reads as manipulation.

## Cross-links

- **Runnable notebooks** (the other in-browser front door):
  [Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb)
  (free T4) and the in-kernel-decode notebook — see [`notebooks/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/notebooks/README.md).
- **Live web demos:** <https://anthony-chaudhary.github.io/fak/demos.html>
- **Other owner-gated listings:** [`directory-submissions.md`](directory-submissions.md)
  (mcpservers.org, mcp.so, Smithery, AlternativeTo, …).
- **Honesty ledger:** [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
  — every number on the Space traces to a `[SHIPPED]` line with a test witness.

---

*Provenance & fact-check: the three demo commands in the table were run against the live
binaries from a clean `C:\work\fak` checkout (2026-07) and reproduced the quoted output
verbatim. The Space `Dockerfile` mirrors the repo-root `Dockerfile` builder (golang:1.26,
`CGO_ENABLED=0`, `-trimpath -s -w`). Oracle-parity numbers quoted from `CLAIMS.md` (the
in-kernel SmolLM2-135M row). Keep this appendix for your audit trail; it is not part of the
Space.*
