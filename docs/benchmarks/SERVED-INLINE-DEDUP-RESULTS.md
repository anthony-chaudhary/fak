---
title: "Served-inline proxy-dedup hit-rate — results"
description: "WITNESSED measurement of how often the vDSO served-inline fast path (--vdso-proxy-fill) actually fires on a named, reproducible agent read pattern: the dedup rate served_inline / read-only-proposed, decomposed by win class, with a default-on recommendation. Closes #1350."
---

# Served-inline proxy-dedup hit-rate (`served_inline`) — results

**Issue:** [#1350](https://github.com/anthony-chaudhary/fak/issues/1350) ·
**Mechanism:** serve duplicate read-only tool calls inline from the vDSO (`36122bc9`) +
proxy cache-warming (`f2d1bec5`) · **Metric:** `fak_gateway_served_inline_total`.

> Numbers are authoritative only in [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
> This sheet keeps its own provenance labels. The committed artifact is
> [`internal/gateway/testdata/served_inline_dedup_report.json`](../../internal/gateway/testdata/served_inline_dedup_report.json);
> re-run the witness with
> `go test ./internal/gateway -run TestServedInlineDedupMeasurement -count=1`.

## Headline

**WITNESSED: on a redundant read pattern with read-only-*shaped* tool names, served-inline
deduplicates 5 / 14 read-only proposals (35.7%) — and every served hit is a cross-turn
re-read (W3). On the dominant Claude Code path (native `Read`/`Grep`/`Glob` names) the same
pattern serves 0 / 14, because the read-only NAME gate does not recognize those names.**

served-inline saves the *EXECUTION* of a duplicate/repeat read (the client tool round-trip),
**not** the surviving call's tokens (the honesty fence from `36122bc9`).

## Provenance

- **The serve is WITNESSED** — fak's real `adjudicateProposedServed` seam served these calls
  in-process (the exact per-turn order `messages.go` drives: `admitInboundResults` warms the
  vDSO from the client's returned tool_results, then `adjudicateProposedServed` probes/serves
  the next turn's proposals). Not a model of what it *would* do.
- **The trace redundancy is MODELED** — a named, reproducible representative coding transcript
  (`representative-coding-session-v1`, 6 turns) embedding the three win classes + non-repeated
  reads + one write + one force-fresh re-read. It is not a capture of one specific real session.

## Measured decomposition (eligible-name regime)

Tool names read-only-*shaped* (`get_`/`read_`/`search_`/`list_`/`lookup_`/`find_`/`calc`).

| Win class | Proposed | Served | Note |
| --- | --- | --- | --- |
| `cross_turn_reread` (W3) | 5 | **5** | A read whose result a PRIOR turn already returned — the win the mechanism is built for. |
| `within_turn_parallel` (W1/W2) | 1 | 0 | Two identical reads in ONE turn are cold (no prior fill), so this mechanism does not dedup them. |
| `force_fresh_declined` | 1 | 0 | `_fak_fresh` re-read: the model opted out of the serve (escape hatch). |
| `first_occurrence` | 7 | 0 | Nothing warm to serve (expected — this is the denominator). |
| **Total** | **14** | **5** | **dedup rate = 35.7%** |

## Claude-native-name regime (the dominant path today)

Same 6-turn read pattern, tools renamed to Claude Code natives (`Read`/`Grep`/`Glob`/…):

| Read-only proposed | Served inline | Dedup rate |
| --- | --- | --- |
| 14 | **0** | **0%** |

`readOnlyPrefix` matches `get_`/`read_`/`search_`/`list_`/`lookup_`/`find_`/`calc` (case-sensitive,
underscore). `Read` does not start with `read_`, `Grep`/`Glob` match nothing — so on real Claude
Code traffic no native read is ever *eligible* for the serve, regardless of how redundant it is.

## Recommendation — should `--vdso-proxy-fill` default on?

**No, not yet — keep it opt-in.** A miss is byte-identical to today
(`TestServedInline_Miss`), so the lever is safe to leave ON where it pays, and it delivers a
real ~36% execution saving on a redundant read pattern **with read-only-shaped names**. But it
fires **zero** on the dominant Claude Code path because the read-only NAME gate does not
recognize `Read`/`Grep`/`Glob`.

**Gate for default-on:** first extend the serve eligibility classifier to recognize the
harness's real read-tool names (widen `readOnlyPrefix` / add a shape classifier for
`Read`/`Grep`/`Glob`), OR scope default-on to MCP / snake_case tool deployments whose reads are
already `get_`/`read_`/`search_` shaped.

## Assumptions, promotion, demotion

- **Assumption (most likely to flip the result):** the tool-NAMING regime. On real Claude Code
  traffic the read tool is `Read` (not `read_*`), so `served_inline` is 0 regardless of
  redundancy until the name gate is widened.
- **Promotion:** feed the SAME harness a captured real `/v1/messages` session (`fak manage --
  claude`, or a tau2/SWE run) as the trace. If the real read pattern is redundant AND uses
  eligible tool names, the WITNESSED rate promotes the lever toward default-on.
- **Demotion / retirement:** if a captured real session shows near-zero cross-turn re-read, or
  the dominant harness keeps native `Read`/`Grep` names unrecognized, the lever is demoted to
  opt-in and the value claim is retired for that path.
