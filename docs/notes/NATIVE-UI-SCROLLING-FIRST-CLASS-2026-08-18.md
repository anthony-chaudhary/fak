---
title: "Native UI scrolling is session state, not terminal residue — 2026-08-18"
description: "First-class viewport, resume-tail, follow-tail, history, and render-witness contract for a fak-native agent UI."
---

# Native UI scrolling is session state, not terminal residue — 2026-08-18

## Verdict

> **TL;DR:** Resume at the latest meaningful block with the composer visible. Preserve an
> operator's history position when new output arrives, and test both behaviors as rendered UI.

A fak-native UI must own scrolling as durable, explicit view state. On launch or resume, the
normal default is **the latest meaningful content with the composer visible**, not a replay of
the whole transcript through terminal scrollback. History remains available on demand; it does
not seize the viewport merely because a session was reconstructed.

This is a `Core` UX requirement for the native session command center, not terminal polish.
An operator who cannot predict where a resumed session will land cannot quickly tell whether the
agent is current, blocked, or still producing output.

## Value frame

- For: an operator resuming a long-running agent session.
- Problem: transcript reconstruction can flood the visible terminal and leave the viewport far
  from the latest turn, forcing manual scrolling before the operator can act.
- Today: terminal and harness behavior decide where the viewport lands. Codex CLI can mitigate
  this with its alternate screen. That containment setting is not a native UI contract.
- Better because: resume opens at a stable tail anchor, new output follows only when appropriate,
  and deliberate history reading is never yanked back to the bottom.
- Witness: a captured render test resumes a transcript longer than the viewport and proves the
  latest meaningful block and composer are visible without replaying earlier blocks into scrollback.

## All-work problem check

| Check | Effect of this requirement |
|---|---|
| P1 managed context | The UI materializes only the viewport plus bounded overscan while keeping the full transcript addressable. |
| P2 net-true efficiency | Resume reaches actionable state without transcript repaint, manual page-downs, or an unbounded render cost. |
| P3 bounded adaptation | Follow-tail changes through explicit user actions and deterministic output events, not inferred intent. |
| P4 integrated operations | The same session identity, event cursor, unread marker, and viewport anchor survive pause, resume, reconnect, and handoff. |

## First-class state model

Scrolling must not be an accidental side effect of appending rendered lines. The view model owns:

- `anchor`: stable event or block identity, plus an intra-block visual offset.
- `follow_tail`: whether arriving output may advance the viewport.
- `unread_after_anchor`: content appended while follow-tail is off.
- `resume_target`: latest meaningful block by default. Use the persisted anchor when the operator
  deliberately left the session away from the tail.
- `layout_revision`: renderer inputs used to re-resolve the anchor after resize or reconnect.

Persist semantic anchors, never raw row numbers. Wrapping, folded tool output, status rows, and
terminal width make row offsets unstable across resume.

## Interaction contract

1. Fresh open and ordinary resume land at the tail. Show the latest meaningful agent/tool block,
   its status, and the composer. Do not animate or stream historical reconstruction through the
   viewport.
2. Manual upward scroll disables follow-tail. New events increase an unread marker but do not
   move the viewport.
3. Explicit End / jump-to-latest restores follow-tail. Reaching the bottom by deliberate scrolling
   may also restore it, but receipt of new output alone may not.
4. History loading preserves the visual anchor. Prepending older events must not shift the block
   currently under the operator's eye.
5. Resize and reflow preserve semantic position. Resolve the same anchored block after wrapping;
   do not approximate with the old terminal row.
6. Large tool output is bounded and foldable. The initial resume viewport prefers the latest
   summary/status edge, with full bytes reachable through expansion or raw-output mode.
7. Resume exposes recency. Distinguish persisted history from events received after reconnect,
   and expose truncation, compaction, or a missing event range instead of silently jumping.
8. Every navigation input uses one state machine. This includes keyboard, wheel, touchpad,
   scrollbar, and screen reader. Input adapters may differ; follow-tail and anchoring semantics may not.

## Required witnesses

The native UI scrolling spine is not complete until captured tests prove:

- a transcript at least 10 viewports long resumes on the latest meaningful block;
- the first frame includes the composer and no historical replay frames;
- scrolling up, then appending output, leaves the anchor pixel/row stable and shows unread content;
- jump-to-latest clears unread state and resumes following output;
- prepending history and resizing across two widths preserve the same semantic anchor;
- a long streaming tool call cannot yank an operator out of history-reading mode;
- reconnect with a compacted or missing range renders an explicit boundary;
- terminal and graphical adapters pass the same behavior suite, plus adapter-specific render captures.

Record first-frame bytes or pixels and the ordered view-state transitions. A unit test that only
checks transcript data is not evidence for this visual defect.

## Current fak gap and product decision

As observed on 2026-08-18, fak has session-control and transcript-oriented pieces, while the native
desktop substrate and its `TerminalAdapter` scrollback contract remain **ABSENT**. This requirement
is therefore `DEFAULT` for the native UI spine: not an optional module, recipe, or terminal-specific
preference. It extends the gap and staged-delivery rows in
[`NATIVE-DESKTOP-DURABLE-SESSION-COMMAND-CENTER-2026-08-13.md`](NATIVE-DESKTOP-DURABLE-SESSION-COMMAND-CENTER-2026-08-13.md).

## Immediate Codex CLI containment

The reported resume behavior occurred in Codex CLI 0.147.0 on Windows. Codex's supported control is:

```toml
[tui]
alternate_screen = "always"
```

This keeps the TUI in an alternate screen instead of preserving/replaying inline terminal scrollback.
The inverse one-run switch, `--no-alt-screen`, explicitly disables alternate-screen mode and preserves
terminal scrollback, so it is unsuitable for this symptom. This setting reduces terminal residue; it
does not replace the native state model above. `/raw` is likewise a copy-friendly history mode, not a
resume-tail policy.

## Source ledger

Observed 2026-08-18:

- OpenAI Codex configuration reference, `tui.alternate_screen` (`auto | always | never`) and
  `tui.raw_output_mode`: <https://developers.openai.com/codex/config-reference>
- OpenAI Codex CLI reference, `--no-alt-screen` and `/raw`:
  <https://developers.openai.com/codex/cli/reference>
- Local native command-center gap audit and staged plan:
  [`NATIVE-DESKTOP-DURABLE-SESSION-COMMAND-CENTER-2026-08-13.md`](NATIVE-DESKTOP-DURABLE-SESSION-COMMAND-CENTER-2026-08-13.md)



