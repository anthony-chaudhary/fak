# Session-observability corpus (scrubbed seed)

`corpus.jsonl` is a **scrubbed** session-observability corpus: one
[`sessionobs.Record`](../../internal/sessionobs/sessionobs.go) per line, the committable
unit of the [session-observability RSI ladder](../../docs/fak/session-observability-rsi-loop.md).

It exists so the **learn** rung has a durable corpus to read across hosts instead of one
host's transient `~/.claude*` transcripts. It is consumed by:

- `fak sessions learn --corpus experiments/sessionobs/corpus.jsonl` — the value-vs-waste
  behavior contrast (which behaviors separate sessions that shipped from sessions that stalled), and
- the `sessions_learn` member of `fak garden` — the registered loop that reads this corpus
  on every garden tick.

## It is scrubbed by construction

A `Record` carries only **structured signal** — turn/tool counts, output-token totals, an
outcome class, and behavior-feature counts. It NEVER carries raw prompt or result prose.
That is what makes the corpus safe to commit and fold across hosts: the prose stays on the
host that produced it; only the signal travels. `session_id` is the opaque transcript UUID,
not content.

## Regenerating / extending it

Fold this host's own sessions, witness their commits against git history, and write the
scrubbed corpus:

```bash
fak sessions score --corpus experiments/sessionobs/corpus.jsonl
```

The fold is deterministic and the outcome link is evidence, not a guess: a session NAMES
the commits it landed in its own transcript (the `git commit` success marker), and a
committed SHA is classified `shipped` only when it is still an ancestor of `HEAD`
(witnessed, not reverted).
