# `docs/questions/` — the question-loop ledger

This directory is the durable home of the **question loop**: a super-loop-family
member whose whole job is to **ask hard, honest questions** about what this repo
is doing — and *not* to answer them, fix them, or ship anything. Asking is the
work. A **separate** next-step loop, in a **separate** context window, is what
turns a question into a GitHub ticket for actual work.

The front door is the [`/question-loop`](../../.claude/skills/question-loop/SKILL.md)
skill. This page is the schema, the labeling, and — most importantly — the
**disambiguation**, because "loop" and "super loop" mean several precise things in
this repo and a question loop is none of the obvious ones.

## What it is NOT (read this first)

| Not this | That surface | This surface (`/question-loop`) |
|---|---|---|
| `/super-loop` | launches detached workers that **ship fixes** to issues | launches detached workers that **ask questions** |
| `/run-it-all-night` | collects the next most important **datum** on the box | asks the next most important **question** about the plan |
| idea-scout (`fak idea-scout`) | mines **external** feeds (arXiv/GitHub/HN) for related **ideas** | generates **internal** provocations about **our own** choices |
| `research` issues | design-shaped **work items** already on the backlog | **questions**, which are explicitly *not yet work* |
| Go `superloop.Super` (`internal/superloop`) | an **interior** node that walks *other* loops worst-first and **mutates nothing at its own altitude** | a **leaf** that **produces questions** at its own altitude |

The last row is the sharp one. `superloop.Classify` grades a loop on five ordered
properties — `has_members`, `walks_first`, `selects_worst_first`,
`exits_on_aggregate`, `interior_node` — and reports the **first** one it fails. The
question loop is a bare **leaf**: it has no member loops and it acts at its own
altitude (it emits questions). So witness it the way the repo witnesses everything
and `Classify(question-loop).Reason` reads **`has_members does not hold`** (the
first rung — `LeafFacts` fails all five); it is *separately* not an `interior_node`
because it acts directly. Either way it is deliberately **not** registered as a
`superloop.Super`. It is a "super loop" only in the operator sense that
`/super-loop` is — a bulk detached launcher — never the `internal/superloop` sense.
Mislabeling it as a registered super loop would be exactly the confusion this page
exists to prevent.

(A genuine future `superloop.Super` *could* sit above these two loops — an interior
node that walks `[ask-loop, next-step-loop]` and picks worst-first between "we are
under-questioned" and "the question backlog is un-triaged". That would be a real
interior node. It is noted as an honest extension, not built here.)

## The two loops (why they live in separate context windows)

1. **ASK loop** (role: *questioner*) — fuel:
   [`ask-hard-questions.md`](../../.claude/goal-prompts/ask-hard-questions.md).
   Each tick appends 5–10 `open` questions to the ledger and stops. It never
   proposes fixes, files tickets, or answers itself.
2. **NEXT-STEP loop** (role: *ticketer*) — fuel:
   [`questions-to-tickets.md`](../../.claude/goal-prompts/questions-to-tickets.md).
   Each tick takes **one** `open` question, decides its fate, and — only if it
   earns it — files **one** gh ticket. It never invents new questions.

Keeping them in different workers / different context windows is load-bearing, not
tidiness: a questioner that also holds the ticketing pen slides into solution mode
— it starts writing questions it already knows how to close and quietly stops
asking the uncomfortable ones. Reading the open backlog is fine and *necessary*
(the `UNASKED` category literally means "not on any issue"); what corrupts the ASK
role is *triaging* it. A fresh context per ASK tick keeps the provocations honest.

## The ledger — `asked.jsonl`

Durable and **append-mostly**, one JSON object per line: the ASK loop only appends
`open` rows; the NEXT-STEP loop rewrites exactly one existing row's `status`/`ticket`
in place. Schema:

| field | meaning |
|---|---|
| `id` | `Q-YYYYMMDD-NNN` — the day's date + a per-day counter. Never reused. |
| `ts` | ISO-8601 UTC timestamp the question was asked. |
| `category` | one of the four below. |
| `target` | the claim / area / invariant the question interrogates. |
| `question` | one sharp sentence, ending in `?`. |
| `why` | one line: what it challenges / why it matters (or, once resolved, the evidence). |
| `status` | `open` → `answered` \| `ticketed` \| `dismissed`. |
| `ticket` | the issue number once `ticketed`, else `null`. |

### The four categories (the labeling taxonomy)

- **UNASKED** — a blind spot no agent has raised; the thing not on any issue.
- **AFRAID** — the elephant in the room everyone's thinking but avoids saying.
- **CONTRARIAN** — directly challenges something the repo **claims** is correct.
- **STEELMAN** — the strongest form of the **opposite** side of a design choice.

A row that merely requests a feature, or that an existing ticket already asks, is
not a question — it is work. It does not earn a row.

### Enforced, not just documented

The schema above is not honor-system prose — it is machine-checked by
`fak question-ledger` (Go leaf `internal/questionledger`, the labeling authority
both loops defer to; tests auto-run by `make ci`). It refuses an unknown category
or status, a malformed or duplicate id, a `ticketed` row without a positive-int
`ticket` (or a non-ticketed row that has one), a non-question (no `?`), and an
absolute-path/host/email leak:

```
fak question-ledger next-id                 # continue today's Q-…-NNN
fak question-ledger dedupe-check --question "…"
fak question-ledger lint                    # exit 1 on any violation
fak question-ledger stats                   # counts by category/status
```

## The GitHub label — `question-loop`

When the NEXT-STEP loop files a ticket, it tags it **`question-loop`** as a
**provenance** label ("this work originated from a witnessed question"), *plus* a
real priority/area/class so a human can triage fast. Provenance only — a
question-ticket is meant to be *actual work*, so it is **not** hidden from the
dispatch surface. Find them all through the `question-tickets`
[issue view](../../.github/issue-views.json).

## The honesty boundary (do not cross)

- Asking is not solving. The ASK loop ships **only** the ledger, never code.
- A launch is not a ship. A question-ticket's witness is the filed issue **and**
  the committed `ticketed` status change, not the fact that a worker started.
- A thin, honest batch beats a padded one. If there are not 5 non-obvious
  questions to ask this tick, ask fewer.
- Public-safe: no machine-absolute path, host, or personal identifier in a
  question or a ticket body (the `PUBLIC_LEAK` gate refuses it).
