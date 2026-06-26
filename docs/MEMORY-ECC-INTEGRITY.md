---
title: "ECC-style memory integrity for agent recall — the mapping, and the non-goals"
description: "How fak applies the check/syndrome/scrub/erasure discipline of error-correcting memory to its persisted agent-memory cells — what is shipped, what is proposed, and the boundary that keeps the analogy honest."
---

# ECC-style memory integrity — the mapping, and where the analogy stops

*fak is not inventing error-correcting codes for language. It is applying the
**discipline** of error-correcting memory — a check word per cell, a syndrome computed
at page-in, a typed repair-or-erase decision, and an off-path patrol scrub — to the
cells of agent **recall**: the pages of a persisted session core image.*

## TL;DR

A finished session persists as a durable **core image** — `manifest.json` (the page
table) over `cas.json` (the content-addressed swap device). The CAS already
self-verifies a page's **body**: a blob must hash to its key or load refuses the whole
image. But the page **table** carries security-critical *metadata* the body digest does
not cover — `Quarantined`, `QID`, `Taint`, `Digest`, `Len`. A bit-rot or tamper that
flips `Quarantined` true→false, or repoints `Digest` at a benign blob, sails past the
body check. That silent metadata fault is exactly what ECC exists to catch, and the
mechanism below closes it.

This doc maps the ECC vocabulary to the fak mechanisms that realize it, marks each
`[SHIPPED]` or `[PROPOSED]` against the honesty ledger ([`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)),
and — most importantly — draws the boundary that keeps the analogy from overclaiming.
It is the integrity companion to the *temporal/durability* axis
([`CONTEXT-IS-NOT-MEMORY.md`](CONTEXT-IS-NOT-MEMORY.md), the S7 write-time gate, **#82**)
and the *spatial/trust* axis ([`MEMORY-LAYERS-EXPLAINER.md`](MEMORY-LAYERS-EXPLAINER.md),
S1–S6). The tracked epic is **#782**; the shipped substance lives in
[`internal/recall/syndrome.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/recall/syndrome.go).

---

## The mapping — ECC term → fak mechanism

| ECC concept | What it means | fak mechanism | Status |
|---|---|---|---|
| **Check bits** | redundant information stored *alongside* a cell so corruption is detectable | the per-page check fields — `Digest` (CAS address), `Taint`, `Quarantined`/`QID`, `Witness`, `TrustEpoch`, `Durability` — **plus** the `Syndrome` check word (`computeSyndrome`) that binds the integrity-critical subset into one hash | `[SHIPPED]` (#783) |
| **Syndrome** | the value computed at read that says *whether, and how,* a cell is corrupted | the page-in / scrub **verification result**: `ClassifyFault(page, body)` → `FaultClass`, surfaced over a whole image by `Session.Verify() []PageFault` | `[SHIPPED]` (#783/#785) |
| **Single-error correction** | recover the true value from the redundancy, in place | **typed mechanical repair only** — re-derive the metadata from the still-authoritative CAS body and re-stamp; `FaultRepairable` is the classification that *gates* such a repair (the body hashes to its digest, only the metadata disagrees) | classification `[SHIPPED]` (#785); the repair **action** `[PROPOSED]` |
| **Double-error detection / uncorrectable** | the damage exceeds what the code can fix — fail safe, do not guess | `FaultErasure` (body absent or does not hash to its digest) → **quarantine** (`ctxmmu`), **tombstone** (`recall.Session.RequestContextChange`), or **refuse page-in** (load fails closed on a corrupt CAS entry) | `[SHIPPED]` |
| **Patrol scrub** | an *offline* pass that reads memory back and verifies it before a fault is ever demanded | an off-path scrub over the persisted `recall` / `sessionimage` images; `Session.Verify()` is the read-only classifier such a pass consumes | classifier `[SHIPPED]`; the offline patrol **driver** `[PROPOSED]` (#784) |
| **Parity (cross-replica)** | a disagreement between redundant copies is itself a fault signal | cross-**witness** / cross-**agent** disagreement checks before reuse; `Witness` + `TrustEpoch` are the per-page substrate the check would read | `[PROPOSED]` (#786) |

The shipped core (`syndrome.go`) is witnessed end to end: a freshly persisted image
verifies clean, flipping **any** integrity-critical field moves the syndrome, a
post-load metadata tamper is caught as `FaultRepairable`, a missing/rotted body is
`FaultErasure`, and a pre-rung page (no syndrome) classifies `FaultUnchecked` — honest
absence of evidence, never a false fault. Witness:
`go test ./internal/recall -run 'Syndrome|ClassifyFault|FaultClass|Verify'`
(`TestSyndrome_CatchesEachIntegrityField`, `TestClassifyFault`, `TestVerify_EndToEnd`,
`TestSyndrome_DefaultNeutral`).

---

## Non-goals — the boundary that keeps the analogy honest

The ECC frame is useful precisely because it is *disciplined*, and the discipline
includes knowing what it is **not**. Three explicit non-goals:

1. **No semantic rewrite of poison.** "Correction" here is **typed and mechanical** —
   re-deriving a page's *metadata* from its own authoritative bytes — never an attempt
   to fix the *meaning* of a corrupted or malicious result. `computeSyndrome`
   deliberately excludes presentation/learning fields (`Descriptor`, `Reason`,
   `Utility`) so a benign descriptor repair does not invalidate the check; it never
   touches content. A cell whose data is genuinely gone or untrusted is **erased**
   (quarantine / tombstone / refuse), not paraphrased. fak does not summarize poison
   into trusted text — the same boundary the trust gate already holds (#782 non-goal:
   *do not weaken quarantine by summarizing poison into trusted text*).

2. **No correctness dependency on provider prompt-cache.** The integrity discipline is
   over fak's **own** persisted images (`manifest.json` / `cas.json` / `.faksession`),
   which it owns and can re-derive deterministically. A provider's prompt-cache reuse is
   a latency/cost optimization; correctness never depends on it, and nothing here makes
   it load-bearing (#82 / #782 non-goal: *do not make provider prompt-cache reuse a
   correctness dependency*).

3. **No literal ECC novelty claim.** fak is **not** inventing error-correcting codes for
   language. The syndrome is a truncated SHA-256 over canonical metadata — a
   corruption/tamper-**evidence** check word — not a Hamming code; there is no algebraic
   per-bit *correction*, and (because a local image on the operator's own disk makes no
   confidentiality claim) it is **not** a secret-keyed MAC either. The contribution is
   the **systems discipline** — check / syndrome / scrub / repairable-vs-erasure — applied
   to agent memory cells, which is a place that discipline had not been put, not a new
   code.

---

## See also

- [`CONTEXT-IS-NOT-MEMORY.md`](CONTEXT-IS-NOT-MEMORY.md) — the **temporal/durability**
  axis (S7, #82). Durability is one of the per-page check fields the syndrome binds.
- [`MEMORY-LAYERS-EXPLAINER.md`](MEMORY-LAYERS-EXPLAINER.md) — the **spatial/trust**
  axis (S1–S6): routing / addressing / fusion / semantics.
- [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — the
  honesty ledger; the `[SHIPPED]` syndrome + fault-classification row, and the
  `[PROPOSED]` patrol-scrub (#784) and cross-witness parity (#786) rungs.
- [`internal/recall/syndrome.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/recall/syndrome.go)
  — the shipped substance: `computeSyndrome`, `ClassifyFault`, `Session.Verify`.
