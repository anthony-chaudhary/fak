---
title: "AGENTS.md instruction-pulled floor - measured baseline"
description: "The measured per-section baseline (#5445) for AGENTS.md, the instruction-pulled per-agent floor that epic #3229's resident-floor surfaces cannot see."
---

# The AGENTS.md instruction-pulled floor — measured baseline (#5445)

Part of epic **#3229** (shrink the always-sent token floor). This is the measured
baseline for the epic's largest *unmeasured* slice.

## The category this names: instruction-pulled, not resident

The two floors this directory already prices are **resident**: bytes the harness
seats in context before turn 1, which `/context` and `fak footprint` (#3230) can
both see and `internal/mcpfootprint` can ratchet.

`AGENTS.md` is not one of those, and it is bigger than all of them. `CLAUDE.md` is
resident and genuinely lean — **2,227 B ≈ 556 est. tokens** — but its third line is
an instruction: *read `AGENTS.md` first*. Every agent that obeys pays
`AGENTS.md`'s **66,904 B ≈ 16,726 est. tokens** as a turn-1 `Read`. Because those
bytes are *pulled by an instruction* rather than *seated in the system prompt*,
they appear in **neither** surface this epic built to make the floor visible: not
in `/context`, not in `fak footprint`. A floor in effect but not in form.

Call it the **instruction-pulled floor**, and price it separately from the resident
floor. The two compose but do not substitute, and they arrive differently: a
resident byte is in the stable prefix from turn 0, so it is a cache-read on every
turn after the first. An instruction-pulled byte is paid at full price on the turn
it is read, then joins the message history — riding every later turn and occupying
window until compaction sheds it. Neither surface the epic built reports the
second, which is why 16,726 tokens per agent have gone unpriced.

Where it sits next to the slices the epic already gates:

| Slice | est. tokens | form | gated? | measured by |
|---|---:|---|---|---|
| **AGENTS.md (turn-1 `Read`)** | **16,726** | instruction-pulled | **no** | `fak footprint --doc AGENTS.md` |
| `.claude/skills` resident descriptions | 11,809 | resident | not yet — #5444 | `fak skill footprint` |
| fak MCP tool schemas | 5,888 | resident | yes — `floorgate.go` | `fak footprint` |
| ‣ of which description prose | 1,966 | resident | yes — `descbudget.go` | `fak footprint` |
| `CLAUDE.md` | 556 | resident | no (already lean) | `fak footprint --doc CLAUDE.md` |

Every row is a command a reader can re-run, not a quoted estimate; the skills row
is 47,236 B across 58 skills at the same ~4 B/token divisor.

The per-agent pull chain is `CLAUDE.md` + `AGENTS.md` = **17,282 est. tokens**, of
which 96.8% is the pulled half.

## Regenerate it

`fak footprint --doc` prices any markdown doc's instruction-pulled floor per
section. It reuses `agent.RequestFootprint` verbatim — the same char-walk and the
same ~4-byte/token divisor as `EstimateAnthropicTokens` — so a doc floor can never
drift from the estimator #3230 and the gateway already use.

```
fak footprint --doc AGENTS.md            # human table, heaviest section first
fak footprint --doc AGENTS.md --json     # schema fak-doc-footprint/1
fak footprint --doc AGENTS.md --top 5    # just the heaviest N
fak footprint --doc CLAUDE.md            # the resident half of the same pull chain
```

The section is the report's unit because the lever this epic wants is *"page this
subsection out to a queryable store"*, and a section is the smallest thing that can
be relocated without breaking a link.

Two reading rules for the table below:

- **`Bytes` is inclusive** — the heading line, its prose, and every nested
  subsection. That is what a "move this section out" lever actually removes, so it
  is the number a cut line is chosen against. It therefore double-counts across
  levels and must never be summed down the column.
- **The inventory partitions the file exactly.** The preamble plus every section's
  *own* bytes (excluding nested subsections) reconstruct the file byte-for-byte —
  that invariant is what makes each percentage mean something, and it is witnessed
  by a test rather than asserted here.

## Baseline (measured)

`fak footprint --doc AGENTS.md`, measured 2026-07-28:

```
doc-footprint: AGENTS.md · 16726 est. tokens (66904 bytes, ESTIMATED, instruction-pulled) · 15 section(s)
```

| est. tokens | bytes | % of file | level | section |
|------------:|------:|----------:|:-----:|---|
| 16,726 | 66,904 | 100.0% | L1 | AGENTS.md — orientation for coding agents |
| 10,800 | 43,201 | 64.6% | L2 | Hard rules (these WILL bite an agent …) |
| 6,303 | 25,215 | 37.7% | L3 | ‣ If the kernel refuses you (recover, don't fight it) |
| 1,367 | 5,471 | 8.2% | L2 | Build / test / run |
| 1,124 | 4,496 | 6.7% | L2 | New work defaults: spine first, then fan out |
| 1,079 | 4,319 | 6.5% | L3 | ‣ Which build am I asking about? (shared trunk …) |
| 706 | 2,826 | 4.2% | L2 | Planning: two kinds of work … |
| 522 | 2,091 | 3.1% | L2 | Releasing (cut, publish, roll back) |
| 441 | 1,766 | 2.6% | L2 | Where to go next |
| 404 | 1,617 | 2.4% | L2 | Proof by default (every issue fix ships its evidence) |
| 402 | 1,611 | 2.4% | L2 | The local machine is the control point, not the compute boundary |
| 261 | 1,047 | 1.6% | L2 | Version everything: cite `module@rev`, not just a bare SHA |
| 217 | 871 | 1.3% | L2 | The 60-second proof (no key, no model, no GPU — verified) |
| 211 | 846 | 1.3% | L2 | Repo layout (where things live) |
| 163 | 655 | 1.0% | L2 | What this project is |

Preamble (before the first heading): 0 B — the file opens on its `#` title.

**The whole file is one section's problem.** `## Hard rules` is 64.6% of it, and
inside that, the single `### If the kernel refuses you` subsection is **25,215 B —
37.7% of the entire per-agent floor**. That one subsection outweighs the other
eleven top-level sections *put together* (23,297 B). It is a flat table: 63
refusal-token rows, no `####` subheadings anywhere in the file, no index. An agent
that trips one token reads sixty-three.

## The 25 kB duplicates a store that is already queryable

The recovery vocabulary has a structured home and fak already ships the readers.
Measured against the same working tree:

```
# the section's current line span comes from the inventory itself:
#   fak footprint --doc AGENTS.md --json   ->  .sections[] | {title, line}
# it was 442..529 at the measurement above.
cookbook rows      awk 'NR>=442 && NR<=529' AGENTS.md | grep -oE '^\| `[A-Z][A-Z0-9_]+`'  ->  63 tokens
dos.toml records   grep -oE '^\[reasons\.[A-Za-z0-9_]+\]' dos.toml                        ->  68 records
cookbook \ dos.toml                                                                       ->   0 tokens
dos.toml \ cookbook                                                                       ->   5 records
```

**All 63 of the cookbook's refusal tokens already resolve as `[reasons.*]`
records**, live over MCP through `dos_check_reason` / `dos_refuse_reasons` and on
the CLI as `dos man wedge <TOKEN> --explain` — each with a `summary` and a `fix`. Live
spot-check of two rows, a commit-lane one and the table's last: `BARE_COMMIT_SWEEP`
and `WEBHOOK_URL_NOT_ALLOWLISTED` both return `known: true` over MCP; the CLI reports
each as a `VALID reason` with its category and fix. Both carry richer recovery text in
the store than the row inlines.

Two consequences, one of which corrects the framing #5445 was filed with:

- The duplication is **100%, not 84%**. #5445's "95 tokens, 80 registered, 15
  missing" counts every ALL-CAPS string in the section — which sweeps in prose
  (`STOP`, `OPEN`, `JSON`, `RISK`, `DENY`), an OS errno (`ENOENT`), and env knobs
  (`FAK_CHURN_BURST_THRESHOLD`, `FAK_RATELIMIT_MIN_429`). Under the refusal-token
  denominator the gap is zero. **#5445's third done-condition — "all tokens resolve
  via `dos_check_reason`" — is already satisfied**, so the backfill it scoped is a
  no-op and #3980's drift gate has nothing to catch up on here.
- That makes the paging-out case *stronger*, not weaker: 37.7% of the per-agent
  floor is prose whose every load-bearing token is already served by query, so
  relocating it costs no recoverability. The store is not a migration target; it is
  the incumbent.

The store also carries five reasons the cookbook never mentions —
`OVERLAY_WOULD_GATE`, `RELAY_IDLE_PARKED`, `RELAY_NO_PROGRESS`,
`RELAY_ORPHANED_FOLLOWON`, `SESSION_CEILING_SATURATED` — which is the ordinary
failure mode of a hand-maintained mirror and a second argument for querying it.

## Why this matters more under fan-out

`fak footprint` and `/context` both price **one** context. A `Workflow`/ultracode
run pays the per-agent floor **once per subagent**, so the bill is `floor × N`. At
the medium-workflow guideline of 15 agents:

| Term | per agent | × 15 |
|---|---:|---:|
| AGENTS.md turn-1 `Read` | 16,726 | 250,890 |
| ‣ of which the refusal cookbook | 6,303 | 94,545 |
| CLAUDE.md (resident) | 556 | 8,340 |

Both columns are ESTIMATED (~4 bytes/token), never provider-billed counts. The
general fan-out axis deserves its own issue; this is the single largest term in it.

## What is deliberately NOT gated here

There is **no ratchet on AGENTS.md bytes**, and adding one now would be a mistake.
`internal/mcpfootprint/floorgate.go` earns its `FLOOR_BUDGET_STALE` direction
precisely because a ceiling pinned at today's number *banks* today's bloat: the
gate would then defend 66,904 B as acceptable. Measure first, trim, and pin the
ceiling at the post-trim number. Until then this page is a dated measurement, not a
contract, and the regeneration command above is how a reader gets a current one.

## Witness

- `cmd/fak.TestDocFootprintPartitionsFile` proves the inventory is a faithful
  partition (preamble + every section's own bytes == the file), so every percentage
  above is a share of something real.
- `cmd/fak.TestDocFootprintIgnoresHeadingsInsideFences` is the mutation witness for
  the fenced-code-block tracker. AGENTS.md is command-dense; a naive `^#+ ` scan
  invents sections out of shell comments at column 0 and shifts every byte
  percentage below them. Against a fence-less mutant of `footprint_doc.go` this
  test and the partition test both fail.
- `cmd/fak.TestDocFootprintTokensMatchEstimator` proves the token column is
  `EstimateAnthropicTokens`' own answer, not a private divide-by-four — without it
  the resident floor (#3230) and this instruction-pulled floor would not be
  comparable quantities.
- `cmd/fak.TestDocFootprintRanksHeaviestFirst` pins the ordering a cut line is
  chosen from.
- `cmd/fak.TestDocFootprintVerbJSON` runs the verb against the **real** AGENTS.md
  and re-checks the partition there, so the table above is reproducible rather than
  hand-typed. It deliberately pins no byte total — see the section above.

## Open follow-ons (#5445 in scope, not yet done)

1. **Serve the cookbook by query.** Replace the 25,215 B table with a short pointer
   (*you were refused `X` → `dos man wedge X --explain`, or `dos_check_reason` over MCP*)
   plus the handful of rules that must be known *before* a refusal rather than
   after. The recovery text is already in the store — see above — so this is a
   deletion, not a migration. Two gates block doing it blind:
   - The cut line must be ranked by **observed refusal frequency**, not byte size.
     #5445's own risk note is the binding one: if a routine refusal now costs three
     extra tool calls, the lever loses. Keep the commit-lane family inline and page
     out the long tail.
   - A restructure of this size must be checked against the structural doc gates
     (`INDEX_SYNC`, `CONCEPT_FRESHNESS`, `File:`-pinned path inventories) and the
     `llms-full` mirror before it lands, per the same risk note.
2. **Pin the ceiling after the trim**, not before (above).
3. **A frequency source.** Nothing in-tree currently ranks refusal tokens by how
   often agents actually hit them; the guard decision journal is the obvious
   candidate and is what makes step 1 checkable rather than a guess.

## Cross-links

- **#3229** — epic: shrink the always-sent context budget.
- [MCP tool-schema floor — committed baseline](mcp-tool-floor.md) (#3230) — the
  *resident* systemic floor and its ratchet; the estimator this page reuses.
- [Gateway cold-tool deferral](gateway-cold-tool-deferral.md) (#3232) — the 10×
  lever on that resident floor.
- [The Footprint Ladder](../footprint-ladder.md) — the doctrine both floors serve:
  add a capability at the highest rung that works.
- **#5444** — the `.claude/skills` resident description floor, the other userland
  slice `/context` shows but nothing gates.
- **#3980** — register every fak-emitted refusal token in `dos.toml` with a
  structural drift gate; the measurement above says the AGENTS.md cookbook is
  already fully covered.
