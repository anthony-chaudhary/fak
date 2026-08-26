---
title: "Frame loss between master agents and subagents"
description: "Measured 2026-08-14 from local Claude Code transcripts. This finding is recomputable with:"
---
# Frame loss between master agents and subagents

**Measured 2026-08-14 from local Claude Code transcripts.** This finding is
recomputable with:

```powershell
go run ./cmd/framevisibility --homes C:\Users\USER --project C--work-fak --out <scratch>\fold.json
```

The committed verb emits aggregate numbers plus per-session counts; raw transcript
text, prompts, paths inside tool arguments, and account credentials are never copied
into the output. The raw fold remains local and untracked.

## Result

| Question | Measured result |
|---|---:|
| Subagent events reconstructable in the master frame | **13,485 of 636,730 (2.12%)** |
| Generous upper bound, crediting events whose existence is only countable | **233,970 of 636,730 (36.75%)** |
| Decision-relevant events in the master's verbose stream | **117,902 of 319,028 (36.96%)** |

Operating envelope: 25 Claude homes, 777 root sessions, 9,572 subagent
transcripts, 3,356 explicit spawn events, and 317,191 classified tool calls
(96,706 master; 220,485 subagent). This exceeds the issue's requested envelope
of 20 transcripts, 50 spawns, and 500 classified calls.

## Reproducible rubric

The exact executable rubric is `internal/framevisibility/fold.go`:

- A subagent event is **visible** only when its normalized text is represented in
  the master's transcript. A tool call or tool result that exists only in a
  descendant transcript is not visible.
- The generous bound additionally credits events whose existence can be inferred
  from a returned count without reconstructing their content.
- A master event is **decision-relevant** when it is a tool error, repository
  mutation, mutating shell command, delegation control event, or terminal answer.
  The broad shell matcher deliberately makes 36.96% an upper bound.
- Thinking, progress narration, read-only calls, and successful read results are
  not decision-relevant under this operator-action rubric.

A synthetic transcript test in `internal/framevisibility/fold_test.go` proves that
hidden descendant calls stay outside master visibility while the returned result
is credited.

## Interpretation

This **CONFIRMS** #6572's claim that watching only the master's verbose tool stream
is a strongly biased sample: the strict view reconstructs only 2.12% of descendant
events. It also confirms that verbose master output is not a proxy for the whole
fleet: even with an intentionally generous rubric, 63.04% of master events are not
decision-relevant.

The result weakens any stronger claim that the master knows nothing about its
subagents: under the generous existence-only interpretation, up to 36.75% of
subagent events leave some trace. That is still not enough to reconstruct the
actual calls or dead ends.

## Limits and invalidating assumptions

- The corpus is local fak work, not a random sample of every agent workload.
- Text normalization may credit coincidental repeated text; therefore 2.12% is an
  upper estimate of semantic reconstructability, not proof of causal forwarding.
- Subagent files are deduplicated by basename to avoid nested copies. If basenames
  are not unique within a session, the denominator is understated.
- A provider that starts forwarding structured descendant events into the master
  transcript would invalidate the central result; rerun the verb after such a
  format change.
- If the operator's true decision rubric treats ordinary read calls as relevant,
  master relevance rises and this classification should be revised before reuse.
