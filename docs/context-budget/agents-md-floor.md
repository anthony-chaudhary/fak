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
`AGENTS.md`'s **49,882 B ≈ 13,301 est. tokens** as a turn-1 `Read`. Because those
bytes are *pulled by an instruction* rather than *seated in the system prompt*,
they appear in **neither** surface this epic built to make the floor visible: not
in `/context`, not in `fak footprint`. A floor in effect but not in form.

Call it the **instruction-pulled floor**, and price it separately from the resident
floor. The two compose but do not substitute, and they arrive differently: a
resident byte is in the stable prefix from turn 0, so it is a cache-read on every
turn after the first. An instruction-pulled byte is paid at full price on the turn
it is read, then joins the message history — riding every later turn and occupying
window until compaction sheds it. Neither surface the epic built reports the
second, which is why 13,301 tokens per agent have gone unpriced.

Where it sits next to the slices the epic already gates:

| Slice | est. tokens | form | gated? | measured by |
|---|---:|---|---|---|
| **AGENTS.md (turn-1 `Read`)** | **13,301** | instruction-pulled | **no** | `fak footprint --doc AGENTS.md` |
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

## Measured inventory after paging the refusal cookbook

Regenerate from the repository root:

```bash
fak footprint --doc AGENTS.md
fak footprint --doc AGENTS.md --json
```

Post-trim result on 2026-08-23:

```text
doc-footprint: AGENTS.md · 13301 est. tokens (49882 bytes, ESTIMATED, instruction-pulled) · 17 section(s)
```

| EST. tokens | Bytes | Share | Section |
|---:|---:|---:|---|
| 6,075 | 22,782 | 45.7% | Hard rules |
| 1,528 | 5,733 | 11.5% | New work defaults |
| 1,465 | 5,495 | 11.0% | Build / test / run |
| 1,158 | 4,343 | 8.7% | Which build am I asking about? |
| 754 | 2,830 | 5.7% | Planning |
| 557 | 2,091 | 4.2% | Releasing |
| 431 | 1,617 | 3.2% | If the kernel refuses you |

The refusal section fell from **36,028 B / 9,608 estimated tokens** to **1,617 B /
431 estimated tokens**: **22.3× smaller**. Whole-file AGENTS.md fell from the
immediately-preceding **83,681 B / 22,314 estimated tokens** to **49,882 B / 13,301
estimated tokens**: **1.68× smaller**. The section now carries only the preventive
commit-lane rules, one-hop query commands, setup check, and appeal route.

## Recovery stays queryable

The removed table duplicated the authoritative `[reasons.*]` records in `dos.toml`.
Every actual refusal row in the old table already resolved through:

```bash
dos man wedge <TOKEN> --explain
fak recover <TOKEN>
```

`dos man wedge OFF_TRUNK --explain` returns the category, detailed fix, and references;
`fak recover OFF_TRUNK` returns concrete dry-run commands. Apparent missing items in
older counts were not refusal reasons: `ENOENT` is an OS error, `DENY` is a journal
outcome, and `FAK_CHURN_BURST_THRESHOLD` / `FAK_RATELIMIT_MIN_429` are tuning
environment variables. No reason-record migration was necessary.

## Fan-out effect

A 15-agent run pays the instruction-pulled floor once per agent. At that width, the
whole-file floor drops from about **334,710** to **199,515** estimated tokens, while
the refusal slice drops from **144,120** to **6,465** estimated tokens. These are
house-estimator values, not provider-billed measurements.
## What is deliberately NOT gated here

There is **no ratchet on AGENTS.md bytes**, and adding one now would be a mistake.
`internal/mcpfootprint/floorgate.go` earns its `FLOOR_BUDGET_STALE` direction
precisely because a ceiling pinned at today's number *banks* today's bloat: the
gate would then defend 49,882 B as acceptable. Measure first, trim, and pin the
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

## Open follow-ons

The cookbook paging is complete. A separate ratchet can now pin the post-trim ceiling without banking the old bloat.

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
