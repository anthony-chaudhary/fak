---
title: "Enterprise positioning: the runtime enforcement layer security signs off on"
description: "How fak maps to the governance gaps enterprises actually name in 2026 — runtime enforcement, prove-it provenance, cost kill-switches, non-human identity, tamper-evident audit, and air-gapped single-binary deployment. Every market statistic is provenance-labeled to its source; every fak claim is fenced SHIPPED or TICKETED so a buyer knows exactly what runs today."
slug: enterprise-positioning
keywords:
  - agent runtime enforcement
  - AI TRiSM guardian agents
  - non-human identity agents
  - tamper-evident agent audit
  - agent cost kill switch
  - air-gapped agent deployment
  - enterprise agent governance
date: 2026-07-09
---

# Enterprise positioning: the layer security signs off on

Enterprises are not stalled on model capability. They are stalled on the **sign-off layer** —
the runtime that lets a security, risk, or platform team say *yes* to putting an agent in
production. This page maps the governance gaps the market itself named across mid-2026 to
the specific fak surface that answers each one, and it is deliberately honest about which of
those surfaces **ship today** and which are **ticketed**.

Two rules govern this page, so it can be quoted without a lawyer:

- **Every external statistic is provenance-labeled** to the source that reported it, tagged
  `[EXTERNAL]`. These are secondary figures collected in the research brief attached to the
  program epic ([#3256](https://github.com/anthony-chaudhary/fak/issues/3256)); they are
  reproduced with their original attribution, not independently re-derived here.
- **Every fak claim is fenced** `[SHIPPED]` (with the in-repo witness that proves it) or
  `[TICKETED #NNNN]` (named, planned, not yet built). Where a surface is real but partial,
  the note says exactly where the line is.

The neighbouring [capability matrix](adoption/compare/matrix.md) scores fak against
guardrails libraries, gateways, and inference servers. This page is the other axis: not
"which layer owns this capability," but "which named 2026 buying trigger does each fak
surface answer, and is it live yet."

## Why now: the market's own vocabulary

In mid-2026 the language buyers use lines up almost one-to-one with fak's founding thesis.
Each trigger below is labeled with the source that named it.

- **"Runtime enforcement," not written policy, is the named gap.** `[EXTERNAL — Gartner AI
  TRiSM; *Market Guide for Guardian Agents*, Feb 2026]` framed guardian agents as "the
  runtime enforcement mechanism for AI TRiSM" and predicted ~50% of agent-deployment
  failures by 2030 will trace to insufficient runtime enforcement.
- **"From 'trust me' to 'prove it'."** `[EXTERNAL — a16z, *Big Ideas 2026*]` makes execution
  provenance and "Know Your Agent" the frame; a self-reported "done" is treated as
  untrustworthy.
- **Runaway agent cost went board-level.** `[EXTERNAL — TechCrunch reporting, Jun 5 2026]`:
  a large engineering org burned its 2026 AI budget by April on coding agents; a four-agent
  loop ran ~11 days and cost ~$47k before anyone noticed. The stated lesson: budget
  enforcement must live **outside** the agent, as a hard kill/pause at the control plane.
- **Governance, not model capability, blocks pilot→production.** `[EXTERNAL — widely-cited
  adoption splits]`: ~85% of organizations experimenting, ~5% in production. The stall is
  the sign-off layer.
- **Agents are being reclassified as first-class non-human identities (NHI)**, and audit is
  not native to MCP/A2A — there is no tamper-evident event log at the agent-to-agent
  boundary. `[EXTERNAL — industry NHI framing; open standards-body gap, 2026]`
- **One static binary, two `golang.org/x` deps, air-gapped is a regulated-industry lever**, sharpened by a
  `[EXTERNAL — 2026 supply-chain attack on the most-downloaded OSS LLM proxy]`.
- **The AI-gateway market is consolidating up-stack to governance.** `[EXTERNAL — a major
  security vendor acquired a leading LLM gateway, Apr 2026]`. Gateway alone is table stakes;
  the differentiation buyers pay for is *agent-runtime* governance.

**Honesty fence (the one we lead with):** the EU AI Act high-risk / GPAI enforcement
timeline **slipped to 2027–2028**. We therefore sell on *operational* pain — cost,
breach-avoidance, unblocking a stalled pilot — **not** a compliance countdown. Any deck that
leads with an EU deadline is overclaiming.

## The gap → surface map

Each row names a market gap, who named it, the fak surface that answers it, and the honest
status. The `Fence #` column points to the note below that cites the in-repo witness (for
`[SHIPPED]` rows) or the tracking issue (for `[TICKETED]` rows).

| # | Named market gap | Named by | fak surface | Status | Fence |
|---|---|---|---|---|---|
| 1 | Runtime **enforcement**, not written policy | Gartner AI TRiSM / Guardian Agents | `fak guard` + default-deny capability floor (`fak preflight --policy`, closed 12-reason refusal vocabulary) | `[SHIPPED]` | [1] |
| 2 | "Prove it," not "trust me" (Know Your Agent) | a16z Big Ideas 2026 | Commit-level verify — a claimed "done" refused from git evidence (`dos_verify` / `dos commit-audit`) | `[SHIPPED]` | [2] |
| 3 | Budget enforcement must live **outside** the agent (hard kill/pause) | TechCrunch, Jun 5 2026 | `serve` cost lever + pre-exhaustion budget webhook/reset; hard per-scope kill/pause | **Partial** — lever `[SHIPPED]`, hard kill/pause `[TICKETED #3273]` | [3] |
| 4 | Agents as first-class **non-human identities** + kill-switch | Industry NHI framing | First-class agent identity record + fleet kill-switch | `[TICKETED #3274]` | [4] |
| 5 | **Audit is not native to MCP/A2A** (no tamper-evident event log at the boundary) | Open standards-body gap | Hash-chained audit journal (`internal/journal`) + A2A ingress quarantine + structured refusals | `[SHIPPED]` | [5] |
| 6 | Prompt-injection via **poisoned tool results** | Guardrails / result-admit category | `QUARANTINE` verdict — suspicious tool *results* held out of context by structure | `[SHIPPED]` | [6] |
| 7 | **One static binary, air-gapped**, minimal dependency supply chain | Regulated industry; OSS-proxy supply-chain attack | Single static Go binary, runs offline (`--gguf`/mock); [air-gapped deployment kit](air-gapped-deployment-kit.md) + [SPDX SBOM](sbom/fak.spdx.json) | **Partial** — binary/offline, kit doc, SBOM, the zero-network governed-session witness (mock-planner seam), and the `UNAUTHENTICATED_OFF_HOST_BIND` startup refusal `[SHIPPED]`; `--gguf` model-backed air-gap witness `[TICKETED #3279]` | [7] |
| 8 | Fleet agents **clobbering the same files** concurrently | Concurrency governance | File-lease arbitration (`dos_arbitrate`, lock-mode tree-disjointness rule) | `[SHIPPED]` (decision kernel) | [8] |
| 9 | **PII/secret redaction** before bytes leave the box | Gateway governance parity | Pre-send wirescreen redactor (`[REDACTED:<kind>]`, original pinned in CAS) | `[SHIPPED]`, default-inert; flagship passthrough `[TICKETED #555]` | [9] |

Nine named gaps. Five are answered by a surface that is **fully live today** (rows 1, 2, 5,
6, 8); row 9's redactor also ships but is fenced on the flagship route; rows 3 and 7 ship a
partial answer with the hard part ticketed; row 4 is **entirely ticketed**. No row is left
as a bare "yes" — each carries a note that names exactly where the line is.

## Notes and sources

fak cells cite the in-repo artifact that proves them (a `CLAIMS.md` row, a witnessing test,
or a doc). Ticketed cells cite the tracking issue. External cells reproduce the original
attribution collected in the epic research brief.

1. **Default-deny capability floor.** Every tool call gets an `ALLOW`/`DENY` verdict against
   a reviewable allow-list before it runs — **362 ns**, in-process, no model in the loop.
   The policy is a declarative, version-tagged JSON manifest loaded at runtime
   (`--policy FILE`), so an adopter configures *which* tools an agent may call by editing a
   file, never by forking the kernel; unknown fields/reasons/versions are a fatal load error
   (fail-loud, never silently more-permissive). This is a **capability lock, not a text
   classifier** — a tool off the allow-list cannot be called no matter what the model was
   told. Witness: [`CLAIMS.md`](../CLAIMS.md) (deployable-capability-floor + structured-refusal
   rows), [`POLICY.md`](../POLICY.md), `internal/policy` tests.
2. **Commit-level verify.** `dos_verify` answers "did (plan, phase) actually ship?" from a
   run-registry row or a git-log grep over the ship-commit grammar — never from an agent's
   narration; `dos commit-audit` grades whether a commit's *diff* matches its *claim*.
   Honest fence: this verifies the **shape** of a claim (did the diff do the kind of thing
   claimed), not whether the code is correct — run the tests for that. Witness:
   [`CLAIMS.md`](../CLAIMS.md) (effect-verifying witness gate), the `dos_verify` /
   `dos_commit_audit` tools.
3. **Cost outside the agent.** Shipped today: `fak serve` sheds old turns to a resident-token
   budget (`--compact-history-budget`, default-on) and fires a pre-exhaustion warning +
   reset directive via `--context-budget-tokens` / `--budget-webhook` so an operator is
   notified before a session exhausts its budget. **Not yet shipped:** a hard per-scope
   token budget that *kills or pauses* the loop at the control plane — that is the explicit
   deliverable of [#3273](https://github.com/anthony-chaudhary/fak/issues/3273). Today's
   surface *observes and warns*; it does not hard-stop. Witness: `cmd/fak/serve.go`
   (`--compact-history-budget`, `--budget-webhook`, `--budget-warn-fraction`).
4. **Non-human identity.** First-class agent identity (an NHI record) plus a fleet
   kill-switch is **ticketed, not built**:
   [#3274](https://github.com/anthony-chaudhary/fak/issues/3274). Today an agent is
   identified by its per-session trace and everything it does is in the audit journal (note
   5), but there is no first-class identity object with a one-switch revoke. Claiming NHI
   support today would be overclaiming.
5. **Tamper-evident audit at the boundary.** `internal/journal` is a hash-chained,
   tamper-evident verdict ledger (a verdict over a digest); the in-kernel A2A channel
   (`a2achan`) refuses a quarantined body at send and *holds* it on ingress, gated by the
   same default-deny floor a tool call crosses; refusals carry a reason from a closed
   vocabulary. Honest fence: the journal is a **corruption/tamper-EVIDENCE** chain on the
   operator's own disk — it is not a secret-keyed MAC and makes no confidentiality claim
   against an operator with disk access. Witness: [`CLAIMS.md`](../CLAIMS.md) (audit-journal,
   `a2achan`, deletion-certificate rows).
6. **Poisoned-result quarantine.** A suspicious tool *result* is held out of context by
   structure (`QUARANTINE` verdict), not by a classifier that has to catch every attack —
   the poisoned bytes never reach the model's context. Witness: [`CLAIMS.md`](../CLAIMS.md)
   (result-admit / KV-quarantine bridge rows), [objections card](adoption/objections.md)
   items 1 and 4.
7. **Single binary, air-gapped.** fak is one static Go binary that drops in with a single
   base-URL change (41 of 47 surveyed harnesses; see the
   [compatibility matrix](integrations/compatibility-matrix.md)) and runs fully offline with
   a local model (`--gguf`) or the mock planner — no external dependency on the request path.
   The packaged kit now ships: [air-gapped deployment kit](air-gapped-deployment-kit.md)
   carries the hardened bring-up, a captured zero-network governed-session witness, the
   regulated-deployment checklist, and a generated [SPDX SBOM](sbom/fak.spdx.json).
   **Honest dependency posture:** two `golang.org/x` extended-stdlib modules and a 4-line
   `go.sum` — *not* the older "zero external deps, no `go.sum`" phrasing, which is stale.
   **Bind safety is now a kernel refusal, not a convention** (#5373): `fak serve` exits 2 with
   `UNAUTHENTICATED_OFF_HOST_BIND` rather than binding an off-host interface with no token
   door — the 175,108-auth-less-server shape is refused at startup. Honest edge: it classifies
   by IP value and admits a DNS `--addr` it cannot prove off-host.
   **Still ticketed on [#3279](https://github.com/anthony-chaudhary/fak/issues/3279):** the
   `--gguf` *model-backed* air-gap witness (the captured one uses the mock-planner seam),
   tracked as [#5372](https://github.com/anthony-chaudhary/fak/issues/5372).
8. **File-lease arbitration.** `dos_arbitrate` is a pure admission kernel: given the leases
   already held, it decides whether a new worker may take a file-tree lane, using a lock-mode
   tree-disjointness rule (shared/shared may overlap; anything with an exclusive holder must
   be tree-disjoint) — so two agents do not edit the same files at once. Honest fence: it is
   the **decision** kernel (state in, verdict out, no I/O); the host wires the lease store and
   enforcement around it. Witness: the `dos_arbitrate` tool and its lane taxonomy.
9. **Pre-send redaction.** The wirescreen redactor proposes byte spans (credit cards, SSNs,
   AWS/GitHub/Slack/Stripe/Google keys, emails, bearer tokens, PEM keys via high-precision
   regex + Luhn) and replaces each with `[REDACTED:<kind>]` before bytes leave the box,
   pinning the original in the shared CAS so an authorized restore returns it byte-exact.
   Honest fences: it is **default-inert** (`FAK_WIRE_REDACT`), it is a **compliance floor, not
   a token saver**, and on the flagship `fak guard -- claude` Anthropic passthrough the
   redaction cannot reach the wire until the cache-prefix-preserving `req.Raw` transform
   ([#555](https://github.com/anthony-chaudhary/fak/issues/555)) lands — it *does* reach the
   wire on the non-passthrough proxy/serve routes today. Witness: [`CLAIMS.md`](../CLAIMS.md)
   (pre-send PII/secret redaction row), `internal/wirescreen`.

## What we are NOT claiming

The honest scope, stated once, so nothing above has to be walked back:

- **No compliance countdown.** The EU AI Act high-risk deadline slipped to 2027–2028 (see the
  fence above). We sell on operational pain, not a regulatory clock.
- **No hard cost kill-switch yet.** Row 3 observes and warns; the hard kill/pause is #3273.
- **No non-human-identity product yet.** Row 4 is entirely ticketed (#3274).
- **No model-backed air-gap witness yet.** The kit doc, SBOM, and the kernel bind-safety
  refusal all ship, and the captured zero-network governed session is real — but at the
  mock-planner seam. The `--gguf` model-backed air-gapped run remains #3279 / #5372.
- **Redaction is off by default and gated on the flagship route.** Row 9 is real but fenced
  on both axes.
- **No market-share or "most secure" claim.** This page maps gaps to surfaces; it makes no
  benchmark or adoption claim about fak itself. The novelty is assembly, not invention (a
  29-claim prior-art audit scored 0/29 novel; see [`CLAIMS.md`](../CLAIMS.md)).

## Where the depth lives

- [Capability matrix](adoption/compare/matrix.md) — fak scored yes/partial/no against
  guardrails libraries, gateways, and inference servers, with a sourced note on every cell.
- [The fak pitch ladder](adoption/pitch-ladder.md) — the canonical pitch at three zoom
  levels (one sentence, one paragraph, one page), each honest, self-consistent, and quotable.
- [Program epic #3256](https://github.com/anthony-chaudhary/fak/issues/3256) — the workstream
  and the research brief every `[EXTERNAL]` figure above is drawn from.
- [`CLAIMS.md`](../CLAIMS.md) — the tagged shipped/simulated/stub ledger behind every
  `[SHIPPED]` fence on this page.

## Verify

```
test -f docs/enterprise-positioning.md            # this artifact exists
fak score seo                                     # new doc does not red the SEO scorecard
go run ./cmd/fak claim-check --self-test          # the honesty grader passes green
```
