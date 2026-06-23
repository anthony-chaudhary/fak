---
title: "idea-scout triage: MCPPrivacyDetector — protocol-induced MCP-server leakage; prior art + a real threat fak already gates at RUNTIME (result-admit), but the cross-language static analyzer is not adopted (2026-06-23)"
description: "Triage of the idea-scout candidate arXiv:2606.21338 (Yan, Xu, Yang, Ma, Dai, Li, 'What Happens Locally, Leaks Globally: Detecting Privacy Leakage Risks in MCP Servers'): a context-aware cross-language static-analysis framework (MCPPrivacyDetector) that taint-tracks credentials / API keys / PII to protocol-specific implicit sinks (@mcp.tool returns, logs, exceptions) in MCP-server source, with no explicit outbound request — >10% leakage across 10,655 real servers. Verdict: prior art to cite + a real threat fak ALREADY gates at runtime on the result-admit boundary (ctxmmu/normgate secret-shaped quarantine), with honest fences (the gate is best-effort not the floor; the 'logged' channel is out of fak's context-admit scope). The static analyzer itself is NOT adopted — it is source-side pre-deploy tooling outside fak's runtime-substrate scope and would re-import the evadable-detector ceiling fak keeps off the floor."
---

# idea-scout triage — MCPPrivacyDetector / protocol-induced MCP leakage (issue #552)

> Closes the daily idea-scout candidate [#552](https://github.com/anthony-chaudhary/fak/issues/552)
> (`tools/idea_scout.py`, filed 2026-06-23). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + a real threat fak already gates at RUNTIME on the
> result-admit boundary. The cross-language static-analysis framework is NOT adopted.**

**Source:** https://arxiv.org/abs/2606.21338 — "What Happens Locally, Leaks Globally:
Detecting Privacy Leakage Risks in MCP Servers", Biwei Yan, Minghui Xu, Yijun Yang, Boyang
Ma, Xuelong Dai, Jingku Li (submitted 2026-06-19). Read from the arXiv abstract via WebFetch
on 2026-06-23; this is a surface read of the abstract, not a paper audit or a reproduction.

## The paper, in one pass

The thesis is that MCP-server leakage is **protocol-induced**, not a conventional exfiltration
bug: credentials, API keys, and PII cross the **local/LLM boundary** *simply by being
returned, logged, or raised inside a tool handler* — there is **no explicit outbound request
in the source** for a classic taint-to-network analyzer to flag. The contribution is
**MCPPrivacyDetector**, a context-aware **cross-language static analysis** framework that:

1. **lifts** heterogeneous server code (Python is the named example) into a unified program
   representation;
2. applies **context-aware semantic filtering** to isolate genuinely sensitive values and the
   **protocol-specific *implicit* sinks** — the `@mcp.tool` handler's return, its log calls,
   its raised exceptions — that conventional tools don't model as sinks; and
3. runs **taint analysis** to enumerate feasible flows from a sensitive source to one of those
   implicit sinks.

Applied to **10,655 real-world MCP servers** it reports a **leakage rate above 10%**, with
case studies of concrete exposures — leaked Bearer tokens, propagated API keys, plaintext
authentication credentials (per the abstract; not re-measured here).

## Where fak actually stands

The lens that resolves this paper cleanly is fak's **two-boundary** picture of the same
threat — and fak already owns the **runtime** half of it.

- **The paper's boundary is source-side / pre-deploy.** It reads the *server's source* and
  proves a taint flow exists from a secret to an `@mcp.tool` sink *before* the server runs. It
  is a linter for the MCP-server author.
- **fak's boundary is runtime / write-time, at the gateway.** When a tool *result* tries to
  enter the model's context, the **result-admit gate** (`ctxmmu`, `CLAIMS.md` units 61–70)
  screens the bytes and **QUARANTINES secret-shaped** payloads — paging them to a stub pointer
  so they never cross into the LLM's context. `normgate` (rank-5 normalize-and-rescan,
  `CLAIMS.md` unit 50) broadens the secret vocabulary to exactly the paper's case-study
  classes — `AIza…`, `github_pat_`, JWTs, Slack/`ASIA` keys, Bearer tokens — and strips
  zero-width / homoglyph / base64 obfuscation first.

So fak and the paper describe **the same leak from opposite ends**: the paper proves "this
server *can* return a secret"; fak's gate catches "a tool result *is* returning a secret" as
it crosses the boundary.

| The paper's frame | fak's position |
|---|---|
| Leakage is **protocol-induced** — a secret crosses the local/LLM boundary by being *returned / logged / raised*, with no explicit outbound request | This is a precise external statement of the threat fak's **result-admit gate** exists to contain at runtime. The gate doesn't look for an outbound *request* either — it screens the *result bytes* themselves at the write-into-context boundary (`ctxmmu.ScreenBytes`), which is exactly where a "just returned" secret shows up. |
| Implicit sinks: `@mcp.tool` **return** + **raised exception** | fak's gate screens the tool *result payload* regardless of whether it's a success return or an error — both arrive as bytes the gateway admits. (Residual recorded below: assert the gate runs on the *error/exception* payload identically, since the paper makes "raised" a first-class sink.) |
| Implicit sink: **log** call inside the handler | **Out of fak's scope, honestly.** A log write goes to the *server's own* log sink, not into the LLM's context — it never crosses fak's gateway, so fak's context-admit gate cannot and does not see it. This channel is the server author's responsibility and squarely the static analyzer's domain, not the gateway's. |
| Static analysis **enumerates feasible flows** with high recall | fak's runtime gate carries the documented `≈100% evadable + FP-prone` ceiling (`CLAIMS.md` units 50, 71): a paraphrased or semantically-hidden secret slips it, and it false-positived on benign base64 images in a real session. The gate is a **best-effort rung that never load-bears** — the floor is the capability lock, not this detector (the standing *detector-is-not-the-floor* thesis). |

## Triage decision

- **Adopt the cross-language static analyzer as a fak capability?** **No.** fak is a *runtime*
  trust substrate / gateway, not a static linter for arbitrary multilingual MCP-server source.
  Building a unified-IR, taint-analysis framework over 10k-plus servers' source is a different
  product on a different boundary. fak's nearest code-analysis surface, `codelint`
  (`CLAIMS.md` units 60–62), is deliberately a **zero-false-positive HARD-error** checker
  (parse/compile only, off the hot path); turning it into a privacy *taint* analyzer would
  import precisely the evadable + FP-prone detector ceiling fak documents and keeps **off the
  load-bearing path**. Not adopted.
- **Defend against (is the threat real for fak)?** **Real — and already gated at runtime, with
  named residuals.** fak's result-admit gate quarantines secret-shaped tool results (Bearer
  tokens, API keys — the paper's exact case studies) before they cross into the model's
  context, so the *returned/raised* channels into fak's own context are covered on a
  best-effort basis. The honest fences worth not regressing on:
  1. **The gate is best-effort, never the floor** — the `≈100% evadable + FP-prone` ceiling
     applies; a semantically-hidden secret is not caught, and that is by design (the floor is
     the capability lock).
  2. **The "logged" channel is out of context-admit scope** — logs don't cross the gateway
     into the LLM, so fak neither sees nor claims to gate them; that is the server author's and
     the static analyzer's domain.
  3. **fak gates ingress to *its* context**, not leakage at MCP servers it does not front.
- **Cite as prior art?** **Yes.** The "protocol-induced leakage, no explicit outbound request"
  framing and the **implicit-sink taxonomy** (`@mcp.tool` return / log / exception) are the
  cleanest external statement of *why* runtime result-admit screening is necessary, and the
  **10,655-server, >10% leakage** measurement quantifies the threat surface. It belongs
  alongside fak's existing prior-art discipline (`CLAIMS.md` `[0/29 novel — the contribution is
  the assembly]`). fak and the paper agree on the diagnosis (a secret leaks just by being
  returned) and split the defense by boundary: the paper proves it static at the source; fak
  catches it dynamic at the gateway.

**Action:** close #552 as triaged → **prior art cited + a real threat fak already gates at
runtime (result-admit), with the 'logged' channel out of scope and the gate explicitly off the
floor; the cross-language static analyzer is not adopted** (this note). No code change in this
increment: `tools/idea_scout.py` surfaced and scored the candidate correctly (topic
`mcp-security`, score 57), and the right small artifact for a research / security triage is the
recorded verdict + the named residuals, not a half-built static-analysis framework.

**Next step (the smallest honest follow-on, if pursued):** a one-time test asserting the
result-admit gate screens an **error / raised-exception** tool-result payload identically to a
success payload — closing the one residual the paper surfaces that touches fak's *own*
boundary (the "raised" implicit sink) — filed as its own scoped test against `internal/ctxmmu`.
It guards the invariant this note names rather than importing the paper's source-side
detector-and-taint framework.
