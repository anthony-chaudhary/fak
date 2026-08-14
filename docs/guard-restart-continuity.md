---
title: "Guard restart continuity: the contract, the modes, and the seed-handback orphan trap"
description: "What happens to your conversation when fak manage --restart-on-budget relaunches its child: the restart chain, the closed continuity modes, and how to diagnose 'conversation was compacted' right after a guarded budget restart."
---

# Guard restart continuity — the contract, the modes, and the seed-handback orphan trap

*Epic: [guard-lifecycle #1193](https://github.com/anthony-chaudhary/fak/issues/1193). Issue:
[#3058](https://github.com/anthony-chaudhary/fak/issues/3058). Code rungs this documents:
[#3055](https://github.com/anthony-chaudhary/fak/issues/3055) (`--continue` reattach),
[#3056](https://github.com/anthony-chaudhary/fak/issues/3056) (seed-prompt handback, reserved),
[#3057](https://github.com/anthony-chaudhary/fak/issues/3057) (restart-chain observability, shipped).*

A `fak manage --restart-on-budget` session can hide any number of child relaunches, and each
one is a continuity cliff: the wrapped agent either resumes the conversation it was working
in, or it boots cold and silently loses the task. This page is the operator contract for
that cliff — what the restart chain does, the closed vocabulary for how continuity was (or
was not) handed back, and how to diagnose the one symptom operators actually see.

**The symptom to remember: if you see "conversation was compacted" (or a cold "I don't have
the task" reply) right after a guarded budget restart, it is the seed-handback path — check
`fak manage restart-audit`.**

## The restart chain

When the wrapped child exhausts its `--context-budget-tokens` envelope, the supervision loop
walks one fixed chain:

```
 budget exhausted ──► seed written ──► relaunch ──► handback
 (context budget      (carryover seed   (child is      (does the new child
  for the current      JSON lands in     started under   actually receive the
  trace is spent;      a private         the fresh       captured context? the
  a continuation       fak-guard-        continuation    continuity modes below
  trace is minted)     reset-* dir)      trace)          are the closed answers)
```

Each hop of that chain is recorded as ONE correlated `RESTART_HOP` row (schema
`fak.guard.restart_chain.v1`) in the guard-audit journal, plus a single stderr line carrying
the same fields — from/to trace, seed file and approximate size, handback mode, child
session id, continuity status — so a captured log alone can reconstruct the chain (#3057).

The `FAK_RESET_*` env vars the relaunch sets (`FAK_RESET_FROM_TRACE`, `FAK_RESET_TRACE_ID`,
`FAK_RESET_SEED_FILE`, …) are **advisory only** — no in-child reader consumes them. Real
continuity comes from the *handback*: either the agent's own resume flag, or (future) the
seed re-injected as a prompt. That gap is exactly where the orphan trap lives.

## The continuity modes (closed vocabulary)

| Operator mode | Handback (`handback=`) | Status (`status=`) | What actually happened |
|---|---|---|---|
| **resumed (`--continue`)** | `continue` | `ok` | Recognized agent (Claude Code) was relaunched with its own resume flag appended, so the child reattaches the captured conversation (#3055). Idempotent — repeated restarts never stack the flag. |
| **seed-prompt** | `seed-prompt` | `ok` | *Reserved, not shipped:* the #3056 rung for headless/no-continue agents — the carryover seed re-injected as a prompt. Today's emitter never produces it; if you see it, you are on a build newer than this page. |
| **orphaned** | `ORPHANED` | `inert` (live) / `loss` (audit backfill) | Unrecognized agent: fak cannot guess a safe resume syntax, so the child relaunched **cold** while the seed sits unread on disk. The session keeps running; the task context did not ride along. A seed file with no recorded hop at all is backfilled by the audit as `loss` — outcome unknowable, presumed lost. |
| **blocked** | — | `break` | The handover itself failed: no continuation trace was minted or no seed survived the write, so there was nothing to hand the relaunched child. Live analogue: the reset-limit status line's `continuity=blocked`. |

Two vocabularies interlock here, both closed: the **handback** names *how* continuity was
handed to the new child (`continue` / `seed-prompt` / `ORPHANED`), and the **status** folds
*whether it engaged* (`ok` / `inert` / `break` / `loss`). Closed sets keep the audit buckets
stable instead of exploding into free text.

## Diagnosing "conversation was compacted"

Right after a guarded budget restart, the wrapped agent may report that the conversation was
compacted, or answer as if it never had the task. Do not debug the agent's compaction
settings first — this is the seed-handback path. Ask the chain:

```
fak manage restart-audit            # every hop, journals joined against seed files on disk
fak manage restart-audit --json     # the fak.guard.restart_audit.v1 report
fak session status <id>            # per-session restart chain appended to the status verb
```

Read the report by status:

- `ok` hops — continuity engaged; the compaction message is the agent folding the resumed
  transcript, not a loss. Nothing to do.
- `inert` hops — the **orphan trap**: the child booted cold and the seed never rode along.
  The seed JSON named in `seed_file=` still holds the carryover context; re-inject it by
  hand (or relaunch with the agent's resume flag) and file the agent name for a #3056-style
  handback.
- `break` / `loss` hops — the handover failed or predates the observability rung; treat the
  new child as a fresh session and recover the task from your own records. `loss` rows are
  rendered red on a terminal for exactly this reason.

The audit is read-only and never caps: every unreadable journal or unparseable seed becomes
a note in the report, and a seed with no recorded hop is reported honestly as
`handback=ORPHANED / status=loss` rather than guessed at. It does not verify hash chains
(`fak audit verify` owns tamper-evidence) and always exits 0 when the scan ran — an orphan
is a reported fact, not a process failure; gate on the report, not the exit code.

## Where the evidence lives

- **Journal rows** — `RESTART_HOP` rows in the guard-audit journals
  (`.dispatch-runs/guard-audit/*.jsonl` under the repo root; override with `--journal-dir`).
- **Seed files** — `reset-*.json` under private `fak-guard-reset-*` directories in the OS
  temp dir (or your `--restart-seed-dir`); scan extras with `--seed-dir`.
- **Live stderr** — one correlated line per restart:
  `fak manage: restart #N from=… to=… seed=…tok handback=… child=… status=…`.

Related pages: [managed-context continuous usage semantics](managed-context-continuous-usage.md)
(the product contract a reset must uphold), [managed-context glossary](managed-context-glossary.md),
and [long sessions: keep the cache hit](explainers/long-sessions-keep-the-cache-hit.md) (why the
restart preserves the continuation trace instead of rewriting history).
