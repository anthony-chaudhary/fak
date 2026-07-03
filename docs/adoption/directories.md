---
title: "Where to submit fak — a curated directory & awesome-list checklist"
description: "The maintained, honest list of the awesome-lists, tool directories, and registries where fak actually belongs — agent infrastructure, MCP, harness engineering, LLM security, Go, and self-hosted. Each venue carries why-it-fits, a submission note, and a current status (wired / live / not yet / blocked / declined). Padded with nothing: venues that do not fit are in the Declined section with the reason."
slug: where-to-submit-directories
keywords:
  - where to submit fak
  - awesome list submission
  - MCP registry
  - agent infrastructure directory
  - awesome-harness-engineering
  - awesome-mcp
  - directory listing checklist
  - go-to-market distribution
date: 2026-07-02
---

# Where to submit fak — the directory & awesome-list checklist

> The maintained answer to "where should fak be listed?" — one page so outreach is
> tracked, deduplicated, and honest instead of scattered. This is the **map**; the
> copy-paste submission payloads for the form/account-gated venues live in
> [`docs/launch/directory-submissions.md`](../launch/directory-submissions.md), and the
> Indian-language channel research is in
> [`docs/notes/INDIA-I18N-DISTRIBUTION-CHANNELS-2026-07-01.md`](../notes/INDIA-I18N-DISTRIBUTION-CHANNELS-2026-07-01.md).
> Dimension J of the [concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md);
> the measurement half is the [adoption-signals dashboard](signals.md).

## What fak is (so a listing describes it accurately)

`fak` is an **agent kernel / reference-monitor**: an in-process, default-deny
tool-call gate fused with an addressable, bit-exact KV cache, shipped as **one static
Go binary, Apache-2.0**. You put it in front of an agent you already run (Claude Code,
Codex, Cursor, any OpenAI / Anthropic / MCP client) by repointing one base URL; every
tool call it proposes crosses the capability floor first, suspicious tool *results* are
held out of context (prompt-injection / tool-poisoning containment), and every verdict
is audited. The same boundary is the trust substrate for a **fleet of autonomous
agents**. It fronts a token engine; it does not replace one — so it belongs on
**governance / agent-infra / security / MCP / self-hosted** surfaces, **not** on
"fastest inference engine" lists.

Reusable one-liner (≈140 chars, capability-first, no perf multiplier):
*In-process default-deny permission gate for AI agents — fronts OpenAI/Anthropic/MCP
wires and adjudicates every tool call like a syscall (prompt-injection / tool-poisoning
containment). One static Go binary, Apache-2.0.*

## Honesty fences (apply to every submission)

- **No fabricated status.** A row is `merged` only when its evidence is the upstream
  PR/listing URL, never this doc's say-so. Where the status is uncertain it says so.
- **No duplicate submissions.** The launch campaign records awesome-list PRs already
  filed but does not enumerate the exact merged set in-repo; **check the target repo for
  an existing `fak` entry before opening any PR** (a `fak` grep of its README/list file).
  When you merge one, flip the row to the PR URL.
- **No padding.** Only venues that genuinely fit fak's domain are listed as targets;
  near-misses are in [Declined](#declined--consulted-and-not-a-fit) with the reason.
- **Read the venue's `CONTRIBUTING.md` first**, one venue per PR, disclose author
  affiliation where the venue requires it (awesome-selfhosted does), never resubmit a
  rejected entry without addressing the reason.
- **Capability-first copy, no perf headline.** Never lead a listing with the naive
  8.8–9.7× number; the tuned figure is ~1.5–4.1× vs a warm-cache stack, and most
  listings need no number at all. Trace claims to [`CLAIMS.md`](../../CLAIMS.md).

Status vocabulary: **`wired`** = a committed manifest auto-indexes it (no PR needed) ·
**`live`** = already listed/indexed · **`not yet`** = human/owner action pending ·
**`blocked`** = venue gate not yet met · **`declined`** = consulted, not a fit ·
**`verify`** = link or exact fit unconfirmed, check before acting.

## Tier 1 — awesome-lists that are a precise category match

These are the highest-ROI awesome-lists: the maintainer audience *is* fak's ICP.

| Venue | Why it fits | Submission note | Status |
|---|---|---|---|
| [ai-boost/awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering) | Literally "permissions, MCP, observability, orchestration" — fak is a textbook capability-gate/harness entry | One PR under a permissions/capability-gate heading; lowest-effort, most durable move | not yet |
| [korchasa/awesome-mcp](https://github.com/korchasa/awesome-mcp) | fak ships an MCP server (`fak serve --stdio`, five `fak_*` adjudication tools) | PR under a security/gateway category; link `examples/mcp/README.md` | not yet |
| [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) `verify` | The largest MCP-servers list; fak is a governance MCP server | Follow its category + one-line format exactly | not yet · verify |

## Tier 2 — agent-infrastructure & agent-catalog lists

Broader agent lists. Fit is real (fak is agent infra) but signal-to-noise is lower;
pick the governance/security/infra category explicitly, never "framework".

| Venue | Why it fits | Submission note | Status |
|---|---|---|---|
| [e2b-dev/awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents) | Long-standing, widely-cited agent list | fak as agent kernel/gateway (guard + cache + audit); link README + integration index | not yet |
| [kyrolabs/awesome-agents](https://github.com/kyrolabs/awesome-agents) | Curated open-source agent tooling | Same one-liner; lead with the open-source/self-host angle | not yet |
| [slavakurilyak/awesome-ai-agents](https://github.com/slavakurilyak/awesome-ai-agents) | 300+ agentic resources, tracks stars | Same one-liner | not yet |
| [ashishpatel26/500-AI-Agents-Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) | Use-case-organized catalog, large reach | List under an infra/governance use case, per its `CONTRIBUTION.md` metadata format | not yet |

## Tier 3 — LLM / agent security lists

fak's floor is a security boundary (capability lock + result quarantine), so the
security lists are a genuine fit — lead with the boundary, not the cache.

| Venue | Why it fits | Submission note | Status |
|---|---|---|---|
| [corca-ai/awesome-llm-security](https://github.com/corca-ai/awesome-llm-security) `verify` | Catalog of LLM-security tooling; fak addresses OWASP Agentic Top-10 + MCP Top-10 by structure | PR under a tools/defense heading; lead with prompt-injection / tool-poisoning containment | not yet · verify |
| [tensorblock/awesome-mcp-security](https://github.com/tensorblock/awesome-mcp-security) `verify` | MCP-security-specific list; fak is an MCP tool-call gate | Confirm the repo/format, then PR under a gateway/defense category | not yet · verify |

## Tier 4 — ecosystem front doors (Go & self-hosted)

Real category fits, but each has a maturity/quality gate fak has not yet cleared —
tracked here so we submit the day the gate is met, not before.

| Venue | Why it fits | Submission note | Status |
|---|---|---|---|
| [avelino/awesome-go](https://github.com/avelino/awesome-go) | fak is a pure-Go project; this is the Go ecosystem front door | Gate: ≥5-month commit history + Go Report Card grade. Repo created 2026-06-21 → earliest eligibility ~2026-11-21. Do not force it | blocked (age gate) |
| [awesome-selfhosted/awesome-selfhosted](https://github.com/awesome-selfhosted/awesome-selfhosted) | Genuinely self-hosted, Apache-2.0, single binary | Strict rules: maturity window, category fit, **author affiliation must be disclosed**. Read `CONTRIBUTING.md` first | not yet |

## Registries & marketplaces (MCP / AI-tool directories)

Programmatic ones auto-index from a committed manifest (no PR). The form/account ones
have copy-paste payloads in
[`docs/launch/directory-submissions.md`](../launch/directory-submissions.md) — this
table is the status roll-up, not a duplicate of the payloads.

| Venue | Why it fits | Submission note | Status |
|---|---|---|---|
| [Glama](https://glama.ai) `verify` | MCP-server directory; auto-indexes from a committed manifest | Wired via [`glama.json`](https://github.com/anthony-chaudhary/fak/blob/main/glama.json); approves in minutes | wired |
| [Official MCP Registry](https://github.com/modelcontextprotocol/registry) | The canonical MCP registry | Wired via [`server.json`](https://github.com/anthony-chaudhary/fak/blob/main/server.json) (@0.34.0) + ghcr image; one interactive `mcp-publisher publish` step is owner-only | not yet (owner publish) |
| [Smithery](https://smithery.ai) | MCP-server marketplace | Wired via [`smithery.yaml`](https://github.com/anthony-chaudhary/fak/blob/main/smithery.yaml); GitHub-connect + Deploy (owner). Richest listing wants a hosted HTTPS endpoint | not yet (owner deploy) |
| [mcpservers.org](https://mcpservers.org/submit) | wong2's MCP list | Web form (payload in launch doc); **do not open a PR — the repo refuses them** | not yet |
| [mcp.so](https://github.com/chatmcp/mcpso) | High-traffic MCP directory | GitHub issue (payload in launch doc); `gh`-doable but posts to a third-party tracker → owner go-ahead | not yet |
| [Cline MCP Marketplace](https://github.com/cline/mcp-marketplace) | Drives installs to the Cline IDE-agent audience | GitHub issue with the Server Submission template. **Missing asset: a 400×400 PNG icon** (web-UI upload only) | not yet (needs 400×400 icon) |
| [AlternativeTo](https://alternativeto.net) | Community software directory; "alternative to" discovery | Free account; list as an alternative to LangChain guardrails, NeMo Guardrails, E2B, vLLM (governance layer). Payload in launch doc | not yet |

## Package & discovery registries (already handled)

Durable machine-facing surfaces driven from the repo, listed for completeness.

| Venue | Why it fits | Submission note | Status |
|---|---|---|---|
| [pkg.go.dev](https://pkg.go.dev/github.com/anthony-chaudhary/fak) | Canonical Go package index; `go install` front door | Indexes automatically on first fetch | live |
| [GitHub topics](https://github.com/anthony-chaudhary/fak) | Topic-based discovery | 20 topics set on the repo | live |
| [Homebrew core](https://github.com/Homebrew/homebrew-core) | macOS package reach | Gate: ~225 stars, not yet met. A personal tap (`homebrew-fak`) is available now if Mac reach matters sooner | blocked (star gate) |

## Related catalog fak maintains

- [Awesome Token Efficiency](../awesome-token-efficiency.md) — fak's own maintained
  awesome-list of token/context/KV-cache efficiency methods. Not a submission target;
  it is the reference catalog fak *publishes*, and a natural cross-link from any listing.

## Declined — consulted and not a fit

Honest non-targets. Listing fak here would be off-topic; kept so nobody re-litigates them.

- **[AI4Bharat indicnlp_catalog](https://github.com/AI4Bharat/indicnlp_catalog)** — a
  catalog of *Indic NLP resources*. fak is agent infrastructure, not an Indic NLP model
  or dataset; an entry would be off-topic. (Full reasoning in the
  [India i18n note](../notes/INDIA-I18N-DISTRIBUTION-CHANNELS-2026-07-01.md).)
- **GitHub topic [`indian-languages`](https://github.com/topics/indian-languages)** —
  same fence; fak must not tag itself into Indic-NLP discovery just for having i18n
  entry pages.
- **"awesome-python" / language-model *framework* lists** — fak is not a Python package
  and not an agent framework; framework lists reject infra that isn't a framework.
- **"fastest inference engine" / serving-benchmark lists** — fak fronts a token engine
  and does not compete on tokens/sec; listing it there would invite an apples-to-oranges
  comparison it would (correctly) lose. Use vLLM/SGLang/llama.cpp for that surface.

## Verify

```
# front-matter present and this doc is indexed
grep -n "^title:" docs/adoption/directories.md
grep -n "where-to-submit" INDEX.md

# before opening ANY awesome-list PR, confirm no existing fak entry upstream
#   (clone/grep the target list file for "fak" — a merge you forgot is the #1 dup risk)
```

When a submission merges, flip that row's status to the upstream PR/listing URL — the
row's evidence is the third-party artifact, never this doc's assertion.
