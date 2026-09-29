---
title: "AGENTS.md sectioned loader: what wiring it would take and save (#3535 estimate)"
description: "Estimate only; nothing implemented. The internal/agentsindex loader library exists and fak dev index agents serves it, but nothing on the session path uses it. This note lists the wiring work and the ESTIMATED forced-read savings per session and per 15-agent fan-out."
---

# 2026-09-28: AGENTS.md sectioned loader, wiring estimate (#3535)

**Estimate only. This note implements nothing.**

#3535 (transferred; now `fak-private#2046`, still open) asks to replace the whole
turn-1 read of `AGENTS.md` with a small resident table of contents (TOC) plus
section lookup. The loader itself was built on 2026-07-09. What is missing is the
session path: nothing tells an agent to use it. Wiring it is about 15 small file
edits plus one generated 29-line block. For typical worker profiles it would cut
the turn-1 forced read from **16,160** to **2,859–7,940 est. tokens**, a saving of
**8,220–13,301 per session** and **123,300–199,515 per 15-agent fan-out**.

All token figures are ESTIMATED with the house prose estimator that
`fak footprint --doc` prints (bytes × 4 / 15). They are not provider-billed.
Measured at `HEAD 75c870b62`.

## What exists today

| Piece | Evidence | State |
|---|---|---|
| Section parser, resident TOC renderer, marker splice | `internal/agentsindex/agentsindex.go`, `toc.go` (`2e23234833`) | Built and tested |
| CLI | `fak dev index agents [--section S \| --full \| --write-resident]` via `cmd/fak-dev/main.go:30` and `internal/devcmd/index_agents.go` | Dev binary only. Runtime `fak index` was moved out by `5de6df6c96` (#6022) and now prints `DEV_COMMAND_MOVED` |
| Resident block in `CLAUDE.md` | `grep -c agents-toc CLAUDE.md` prints `0` | Never written, so `TestResidentBlockMatchesAgents` always skips |
| Pointer | `CLAUDE.md:10-11`: "read it first" | Forces the whole 52,301 B read |

Two stale strings would mislead the first agent that uses the TOC:

- `toc.go:37` renders `fak index agents --section <slug>` into the TOC body. That
  runtime verb no longer exists.
- `agentsindex.go:2` still says "~41.7 KB (~10.4k EST. tokens)" and `toc.go:6`
  says "~10.4k-token". Today the file is 52,301 B and 13,946 est. tokens.

`docs/_plans/gateshare-followon.md:100-101` says the loader "does not exist". The
library and the dev verb exist; only the wiring is missing.

## What wiring it would take

In order, each with its own witness:

1. **Point the TOC at a verb that runs.** Change the `fak index agents` strings in
   `toc.go`, `resident_drift_test.go`, and `internal/devcmd/index_agents.go` to
   `fak dev index agents`, and fix the stale sizes. About 5 files and 20 LOC, with
   no behavior change. Witness: `./fak dev index agents | sed -n 2p` shows the
   runnable verb.
2. **Add a fallback for hosts without `fak-dev`.** Put each section's line range
   in its TOC row so a ranged `Read` works with no binary. About 1–2 files and
   30 LOC. Witness: a test that each row's range equals the `--section` output.
3. **Write the block and change the pointer.** Put the markers next to the
   pointer in `CLAUDE.md`, run `fak dev index agents --write-resident`, and change
   "read it first" to "map below; fetch a section by slug; `--full` for the
   whole file". This takes `CLAUDE.md` from 8,303 to 10,722 B (2,859 est. tokens).
   Witness: `TestResidentBlockMatchesAgents` goes from SKIP to PASS.
4. **Retarget the other whole-file pointers.** These are `GEMINI.md:8`,
   `.github/copilot-instructions.md:3`, `.junie/guidelines.md:4`,
   `.continue/rules/fak.md:4`, `llms.txt:5`, and two `.claude/goal-prompts`.
   That is 7 files at 1–3 lines each. The native auto-loaders (`opencode.json`,
   `AGENT.md`, `.aider.conf.yml`) are out of scope for this lever.
5. **Measure it.** Run one guarded worker before and after, and compare turn-1
   input with `fak trajectory audit`. This is the step that turns the estimate
   below into a measurement.

## What it would save (ESTIMATED)

The baseline is `CLAUDE.md` (2,214) + `AGENTS.md` (13,946) = 16,160 est. tokens per
session, or 242,400 per 15-agent fan-out. With the loader, each session pays the
post-write `CLAUDE.md` (2,859) plus each section it fetches.

| Worker profile | Sections fetched | Loader | Saved / session | Saved / 15 agents |
|---|---|---:|---:|---:|
| TOC only | none | 2,859 | 13,301 (82%) | 199,515 |
| Light | `build-test-run` | 3,423 | 12,737 (79%) | 191,055 |
| Commit-only | `hard-rules` | 6,827 | 9,333 (58%) | 139,995 |
| Code | `hard-rules`, `build-test-run` | 7,391 | 8,769 (54%) | 131,535 |
| Native-perf | the above + `native-inference-performance-invariant` | 7,940 | 8,220 (51%) | 123,300 |

Turn 1 is only part of the saving. The whole-file read stays in history and is
re-sent on every later turn, mostly as cache reads. That multiplier depends on turn
count and cache pricing, so it is not quantified here.

## Risks and open questions

- **Heavy sections.** `hard-rules` alone is 3,968 est. tokens (28% of the file),
  and a fetch always includes its children. Step 5 should check whether
  commit-only workers need it at all, since `CLAUDE.md` already summarizes the
  hard rules.
- **No gist.** `AGENTS.md` has no `> **In one breath:**` block
  (`docs/ONE-BREATH-CONTRACT.md`). A TOC-only agent sees titles and prices but not
  the gist. A one-breath line at the top of the resident block would fit the
  2,000-token `TOCBudgetTokens`.
- **Gate coupling.** The byte ratchet
  ([`agents-md-floor.md`](../context-budget/agents-md-floor.md)) already makes an
  `AGENTS.md` change re-pin a baseline. Once step 3 arms the drift gate, the same
  edit must also regenerate `CLAUDE.md`. Land step 3 as its own commit so that each
  gate's first failure has one cause.
- **Two estimators.** The TOC header prints bytes/4 (13,076), while
  `fak footprint --doc` prints 13,946 for the same file.

## Next checkable step

Land step 1. It is done when `./fak dev index agents | sed -n 2p` prints
`fak dev index agents --section <slug>` and
`go test ./internal/agentsindex/ ./internal/devcmd/` is green.

## Reproduce

```bash
./fak footprint --doc AGENTS.md          # 13946 est. tokens, 52301 bytes
./fak footprint --doc CLAUDE.md          # 2214 est. tokens, 8303 bytes
./fak footprint --doc AGENTS.md --json   # per-section est. tokens for the table
./fak dev index agents --section hard-rules | wc -c   # 14881
W=$(mktemp -d) && cp AGENTS.md CLAUDE.md dos.toml "$W"/ \
  && ./fak dev index agents --root "$W" --write-resident \
  && ./fak footprint --doc "$W"/CLAUDE.md   # 2859 est. tokens, 10722 bytes
grep -c agents-toc CLAUDE.md             # 0
```
