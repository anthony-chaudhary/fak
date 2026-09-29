---
title: "AGENTS.md instruction-pulled floor - measured baseline and byte ratchet"
description: "The measured per-section baseline (#5445) for AGENTS.md, the instruction-pulled per-agent floor that epic #3229's resident-floor surfaces cannot see, and the byte ratchet pinned on it at 52,301 bytes on 2026-09-28."
---

# The AGENTS.md instruction-pulled floor — measured baseline and byte ratchet (#5445)

Part of epic **#3229** (shrink the always-sent token floor). This is the measured
baseline for the epic's largest *unmeasured* slice.

## The category this names: instruction-pulled, not resident

The two floors this directory already prices are **resident**: bytes the harness
seats in context before turn 1, which `/context` and `fak footprint` (#3230) can
both see and `internal/mcpfootprint` can ratchet.

`AGENTS.md` is not one of those, and it is bigger than all of them. `CLAUDE.md` is
resident and far smaller — **8,303 B ≈ 2,214 est. tokens** on 2026-09-28 — but it
instructs: *read `AGENTS.md` first*. Every agent that obeys pays
`AGENTS.md`'s **52,301 B ≈ 13,946 est. tokens** (2026-09-28) as a turn-1 `Read`. Because those
bytes are *pulled by an instruction* rather than *seated in the system prompt*,
they appear in **neither** surface this epic built to make the floor visible: not
in `/context`, not in `fak footprint`. A floor in effect but not in form.

Call it the **instruction-pulled floor**, and price it separately from the resident
floor. The two compose but do not substitute, and they arrive differently: a
resident byte is in the stable prefix from turn 0, so it is a cache-read on every
turn after the first. An instruction-pulled byte is paid at full price on the turn
it is read, then joins the message history — riding every later turn and occupying
window until compaction sheds it. Neither surface the epic built reports the
second, which is why ~14k tokens per agent went unpriced.

Where it sits next to the slices the epic already gates:

| Slice | est. tokens | form | gated? | measured by |
|---|---:|---|---|---|
| **AGENTS.md (turn-1 `Read`)** | **13,946** | instruction-pulled | **yes — [byte ratchet](#the-byte-ratchet-2026-09-28)** | `fak footprint --doc AGENTS.md` |
| `.claude/skills` resident descriptions | 11,809 | resident | yes — `skillfootprint/descbudget.go` (#5444) | `fak skill footprint` |
| fak MCP tool schemas | 5,888 | resident | yes — `floorgate.go` | `fak footprint` |
| ‣ of which description prose | 1,966 | resident | yes — `mcpfootprint/descbudget.go` | `fak footprint` |
| `CLAUDE.md` | 2,214 | resident | no | `fak footprint --doc CLAUDE.md` |

Every row is a command a reader can re-run, not a quoted estimate. The AGENTS.md
and CLAUDE.md rows were re-measured on 2026-09-28 with the doc estimator (3.75
B/token). The skills and MCP rows are the 2026-07-28 figures, and their own pages
carry the current pins. The skills row was 47,236 B across 58 skills at a flat 4
B/token.

The per-agent pull chain is `CLAUDE.md` + `AGENTS.md` = **16,160 est. tokens**, of
which 86.3% is the pulled half.

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

## The byte ratchet (2026-09-28)

`AGENTS.md` now has a one-way byte floor, the same two-sided ratchet its siblings
run (`floorgate.go`, `skillfootprint/descbudget.go`):

- **Floor:** `DOC_BYTES	AGENTS.md	52301` in [`agents-md-floor.tsv`](agents-md-floor.tsv),
  the file's size at `HEAD` on 2026-09-28 (`git cat-file -s HEAD:AGENTS.md`).
- **Gate:** `internal/agentsindex/bytefloor.go`, enforced by
  `TestAgentsMDByteFloorAtHEAD` in every `go test ./...` run, which CI's
  `ci-fast` shards run.
- **Growth** past the floor refuses as `AGENTS_MD_FLOOR_EXCEEDED`.
- **A trim** of more than 200 B (`FloorSlackBytes`, less than the smallest
  Hard-rules bullet, 229 B) refuses as `AGENTS_MD_FLOOR_STALE` until the win is
  banked. Deleting any whole bullet must therefore be banked.
- **Re-pin** in the same commit as the change that justifies it, then update the
  figure on this page:

```bash
go test ./internal/agentsindex -run TestAgentsMDByteFloorAtHEAD -update-agents-md-floor
```

### Why the floor is 52,301 and not a smaller number

The floor is today's honest size. It does not flatter the file.

- **Not 49,882.** That figure, quoted below as the post-cookbook baseline, is
  working-copy bytes from a CRLF checkout. The committed blob at `a7474d30f0` was
  49,241 B (plus 641 line-ending bytes = 49,882). Pinning at either number would
  record the regrowth since then as already paid, when it is 3,060 B of real growth
  in like-for-like bytes.
- **Not 21,538.** That was the low after the #8705 trim. A floor there would red
  every commit until 30,763 B are trimmed, which makes it a trim task. The ratchet
  lands first and stops the growth. The trim is a separate change, and the STALE
  direction makes that change re-pin the floor down.

### How it counts

The floor counts LF-normalized bytes, which equals the committed blob size on every
platform. Raw working-tree bytes would refuse a byte-identical commit on a CRLF
checkout. They are also why some of this page's older figures read high.

The baseline follows the counted-ratchet contract of `internal/promptlint/breath`:

- **Stable keys.** Rows are `KIND<TAB>path<TAB>count`. A section is keyed by its
  heading slug (the same slug `agentsindex` pages by), never by a line number, so
  inserting a line renumbers nothing. `agentsindex` numbers repeated slugs by
  position (`notes`, `notes-2`), which would shift keys. The gate therefore refuses
  two headings that share a slug, and names both lines.
- **Counts, not presence.** Trimming a section and re-pinning tightens its count.
  Growth past the count is caught even though the key already exists.
- **Strict parsing.** A row that does not parse is a hard error naming its line.
  So is a duplicate key, a missing `DOC_BYTES` row, or `SECTION_BYTES` rows that do
  not sum to `DOC_BYTES`. That last check refuses a hand-raised ceiling, so a raise
  must go through the regenerate command.
- **Honest claim.** A green run says only that AGENTS.md is not growing.

Only `DOC_BYTES` gates. The `SECTION_BYTES` rows are each heading's own bytes at pin
time (the level-1 title and lede are the `(preamble)` row). They exist so a refusal
names where the bytes came back. Moving prose between sections is not refused.

A real refusal, from adding one Hard-rules bullet and one new section to a scratch
copy of the file:

```text
AGENTS_MD_FLOOR_EXCEEDED: AGENTS.md is 52892 bytes, 591 bytes over its committed floor of 52301. Every agent that obeys CLAUDE.md reads this file whole on turn 1, so each byte is paid in every session (`fak footprint --doc AGENTS.md` prices it in est. tokens).
  Sections most likely to have regrown (own bytes vs the pinned baseline):
      +419 B  L392  Hard rules (enforced below the agent layer)  [hard-rules: 10977 -> 11396]
      +172 B  L502  Session etiquette  [new heading "session-etiquette", not in the baseline; a rename shows here too]
  Fix, in order: (1) move the new prose one hop away (a linked doc, `dos man wedge <TOKEN>`, or a fak verb) and leave a one-line pointer; (2) trim elsewhere in AGENTS.md to pay for it; (3) only if every agent needs it on turn 1, re-pin in the SAME commit with `go test ./internal/agentsindex -run TestAgentsMDByteFloorAtHEAD -update-agents-md-floor` and update the figure in docs/context-budget/agents-md-floor.md.
```

Because both the baseline and the measurement partition the file, the named growths
sum to the overage (419 + 172 = 591).

### Size history

Committed blob bytes, `git cat-file -s <commit>:AGENTS.md`:

| Date | Commit | Bytes | Event |
|---|---|---:|---|
| 2026-06-21 | `1029e37cf` (v0.30.0) | 5,563 | |
| 2026-08-23 | `b0ff8778a3` | 83,681 | peak |
| 2026-08-23 | `a7474d30f0` | 49,241 | refusal cookbook paged out (#5445) |
| 2026-08-23 | `2edab7a264` | 32,723 | Hard-rules de-duplication (#8698) |
| 2026-08-23 | `8ef75f513b` | 21,538 | whole-file trim (#8705); 5,743 est. tokens |
| 2026-09-28 | `75c870b62` | **52,301** | floor pinned; 13,946 est. tokens |

The file was cut 3.9× and then regrew 2.4× over the next 34 days (the last growth
commit landed on 2026-09-26). By section, the +30,763 B
since `8ef75f513b` landed as follows:

| Growth | Section |
|---:|---|
| +16,889 | eight sections that did not exist at the #8705 low (largest: *Scope discipline for smaller models* 5,261, *Divide and conquer* 3,907, *Track work in GitHub* 1,843, *Focus on "move forward"* 1,581) |
| +6,576 | *Hard rules* (own bytes 4,401 → 10,977) |
| +2,315 | *If the kernel refuses you* (1,589 → 3,904) |
| +1,743 | *Proof by default* (1,594 → 3,337) |
| +1,220 | *Native inference performance invariant* (839 → 2,059) |
| +2,020 | seven other sections, net of a −92 B trim |

Those rows are where a trim pays most. The sectioned loader from #3535 (transferred;
now `fak-private#2046`) would stop charging them to every session; see
[the wiring estimate](../notes/2026-09-28-agents-md-sectioned-loader-estimate.md).

## Measured inventory after paging the refusal cookbook

Regenerate from the repository root:

```bash
fak footprint --doc AGENTS.md
fak footprint --doc AGENTS.md --json
```

Post-trim result on 2026-08-23. The figures in this section mix two measurements.
The 83,681 B / 22,314-token peak is the LF blob. Most other byte figures here
(49,882, 22,782, 33,136, 6,036) are working-copy bytes from a CRLF checkout, so
their ratios run slightly low. The [size history](#size-history) above has
like-for-like blob bytes, and [how it counts](#how-it-counts) explains the
difference.

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


### Follow-on Hard-rules trim (#8698)

The next pass reduced `Hard rules` from **22,782 B / 6,075 estimated tokens** to
**6,036 B / 1,609 estimated tokens**: **3.77× smaller**. Whole-file AGENTS.md is now
**33,136 B / 8,836 estimated tokens**, a **2.53× reduction** from the 83,681 B /
22,314-token pre-cleanup baseline. Preventive invariants remain inline; detailed
runbooks stay behind their linked project verbs and documents.

### Whole-file 3× threshold (#8705)

The final orientation, build, new-work, release, and planning pass reduced whole-file
`AGENTS.md` to **21,769 B / 5,805 estimated tokens**: **3.84× smaller** than the
83,681 B / 22,314-token pre-cleanup baseline and below the 7,438-token 3× target.
Shared-trunk, native-inference, proof, commit, scratch, private-control, Windows, and
external-write invariants remain inline; detailed procedures remain one hop away.
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
  hand-typed. It pins no byte total; the ratchet below does.
- `internal/agentsindex.TestAgentsMDByteFloorAtHEAD` is the ratchet: the real
  AGENTS.md must sit inside `[floor − 200, floor]` of the committed
  [`agents-md-floor.tsv`](agents-md-floor.tsv).
- The rest of `internal/agentsindex/bytefloor_test.go` covers the contract on
  synthetic docs. Growth past the floor fails and names the section. A doc at the
  floor passes. Fixing one of two findings tightens the floor, and re-growth is then
  caught. Every corrupt baseline row is a hard error naming its line. Keys stay
  stable when a line is inserted, and two headings sharing a slug are refused. A
  CRLF checkout measures the same as the blob. `CheckFloor` is bound to
  `FloorSlackBytes`, and a refusal ranks the regrown sections, flags new headings,
  and counts any beyond the first five. `TestAgentsMDFloorDocPinsTheCeiling` fails
  if this page drops the floor figure, the regenerate command, or either refusal
  token.
- On 2026-09-28 an out-of-tree mutation run applied 21 mutants to `bytefloor.go`
  (flipped band edges, a lenient parser, dropped checks, broken attribution). The
  suite killed all 21.

## Open follow-ons

- **Trim and re-pin.** The ratchet stops growth; it does not shrink the file. Trim
  the sections in the history table above, then run the regenerate command so
  `AGENTS_MD_FLOOR_STALE` banks the win.
- **Wire the sectioned loader** (#3535, now `fak-private#2046`). `internal/agentsindex`
  already parses and pages the file through `fak dev index agents`, but nothing on the
  session path uses it. What wiring it would take and save is estimated in
  [the 2026-09-28 note](../notes/2026-09-28-agents-md-sectioned-loader-estimate.md).
- **Register the refusal tokens.** `AGENTS_MD_FLOOR_EXCEEDED` and
  `AGENTS_MD_FLOOR_STALE` are not yet `[reasons.*]` rows in `dos.toml`, the step
  #5444 took for the skill floor's tokens.

## Cross-links

- **#3229** — epic: shrink the always-sent context budget.
- [MCP tool-schema floor — committed baseline](mcp-tool-floor.md) (#3230) — the
  *resident* systemic floor and its ratchet; the estimator this page reuses.
- [Gateway cold-tool deferral](gateway-cold-tool-deferral.md) (#3232) — the 10×
  lever on that resident floor.
- [The Footprint Ladder](../footprint-ladder.md) — the doctrine both floors serve:
  add a capability at the highest rung that works.
- **#5444** — the `.claude/skills` resident description floor, the other userland
  slice `/context` shows, gated by `internal/skillfootprint/descbudget.go`.
- **#3980** — register every fak-emitted refusal token in `dos.toml` with a
  structural drift gate; the measurement above says the AGENTS.md cookbook is
  already fully covered.
