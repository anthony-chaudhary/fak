---
title: "concept(maturity): the feature-maturity lifecycle ladder — completeness is not maturity, and every gap is the next work item"
description: "A critical subsystem for the long-horizon 'agentic culture' problem: a v1 prototype can be legitimately COMPLETE yet not tested, not dogfooded, not benchmarked, not the default — and the system and the operator should SEE exactly where each capability sits and what would mature it next. internal/maturity places every declared fak capability (one per internal/<leaf> lane) on a closed lifecycle ladder (proposed → prototyped → tested → dogfooded → default, with a benchmarked badge), gated by evidence the author did not write, and renders the first unmet rung of each as a concrete next work item. Immaturity is never a defect; only a ladder-skip (looking more mature than the evidence supports) is. This is the shipped fak-tree binding of grammar G1 / readiness ladder (#582), the lifecycle sibling of tools/product_scorecard.py and internal/conceptusage. The `fak maturity` / `fak maturity next` verb is live; this note is the plan to make it critical."
---

# The feature-maturity lifecycle ladder

> Concept + ship note for `/goal` 2026-06-29. The subsystem is **shipped**
> (`internal/maturity`, the `fak maturity` / `fak maturity next` verb); this note
> states the thesis, the design, the honest first-run numbers, and the plan to
> make it a *critical* subsystem. It is the lifecycle counterpart of the durable
> innovations catalog ([`docs/INNOVATIONS-INDEX.md`](../INNOVATIONS-INDEX.md)) and
> the shipped fak-tree binding of grammar **G1**
> ([`CONCEPT-AGENT-PROGRAMMING-GRAMMAR`](CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md),
> the readiness ladder of [#582](https://github.com/anthony-chaudhary/fak/issues/582)).

## 1. The thesis — completeness is not maturity

A fleet of agents building software is very good at making things *complete*: a
ticket closes, an epic lands, a v1 prototype works. It is far worse at the slow,
unglamorous, long-horizon work that turns a complete thing into a *mature* one —
writing the tests, making fak itself use it, benchmarking it, promoting it from an
opt-in to the default. That second kind of work is real, but it is invisible:
nothing in the tree says "this v1 is done but nobody has dogfooded it yet," so
nobody picks it up, and the fleet accretes a long tail of complete-but-immature
capabilities.

This is a **culture** problem, not a correctness problem. The fix is not to get
each feature right the first time — a v1 that is *only* prototyped is a perfectly
honest, legitimate state. The fix is to make the lifecycle **visible and
actionable**:

1. Place every capability on a closed maturity ladder, where each rung is gated by
   evidence the author did not write — so the operator (and the system) can see, at
   a glance, that a thing is `prototyped` but not yet `dogfooded`.
2. For each capability, mechanically derive the **next work item** that would
   mature it — so the desire to "create the next ticket" is not a virtue an agent
   must remember, but a fact the tree itself produces.

That is what `internal/maturity` is. It is the same invariant the rest of the fak
kernel carries — *a decision no participant can move by narrating a number* — aimed
at the question "how mature is this, really, and what is owed next?"

## 2. The lifecycle ladder

Every declared fak capability is one `internal/<leaf>` lane in
[`dos.toml`](https://github.com/anthony-chaudhary/fak/blob/main/dos.toml) `[lanes.trees]`. Each is placed on a closed,
total-ordered ladder, best last:

```text
proposed → prototyped → tested → dogfooded → default        (+ benchmarked badge)
```

| # | Rung | Reached when (evidence the author did NOT write) |
|---|---|---|
| 0 | `proposed` | a declared capability with no code on disk yet |
| 1 | `prototyped` | a non-test `.go` file exists in the leaf — **a complete v1** |
| 2 | `tested` | the leaf carries a `*_test.go` (the QA rung) |
| 3 | `dogfooded` | the leaf is on the running binary's **transitive import graph** — *fak itself runs it* |
| 4 | `default` | the capability is a documented `fak` verb (named in `docs/cli-reference.md`) — the default surface |

`benchmarked` (a `func Benchmark*` in the leaf, or a `BENCHMARK-AUTHORITY.md` row)
is deliberately **not a ladder rung** — measurement is orthogonal (a capability can
be measured at any rung), and forcing it into the total order manufactures false
inversions. It is tracked as a **badge** and is the natural next step *after*
`default`: a documented surface that has never been measured still owes a number.

The four ladder rungs *are* naturally ordered: a documented default surface
presupposes fak runs it, which presupposes it is tested, which presupposes it has
code. The current rung of a capability is the **monotonic** one — the highest rung
`R` such that every promotion predicate up to and including `R` holds. A gap caps
it: a leaf fak runs but never tested is honestly at `prototyped`, not `dogfooded`.

### Why it cannot be gamed

Every predicate reads ground truth the capability's author could not fabricate by
editing a claim: code on disk, a `*_test.go`, an edge in the import graph, a
`Benchmark` func, a verb in the CLI reference. To move a capability up the ladder
you change the **real tree**, not a data file. This is the same "no rung reachable
by editing the claim" property as the product scorecard and the readiness ladder —
the maturity sibling of `dos_verify`'s "shipped comes from git, not the worker."

## 3. Immaturity is not a defect — only a ladder-skip is

The single most important design decision: **immaturity is never counted against
anyone.** A capability honestly sitting at `prototyped` is a complete v1 that has
simply not been matured yet. That is the normal, expected, *fine* state — and it is
the whole point that the operator can see it without it reading as a failure.

The one thing this subsystem refuses is a **ladder-skip**: a capability that looks
more mature than its evidence supports — concretely, one that fak *relies on*
(dogfooded, a default surface, or benchmarked) yet has **no tests**. "We depend on
it but never QA'd it" is a real maturity inversion, and it is the maturity sibling
of the product scorecard's verdict-overclaim and the readiness ladder's
`READINESS_OVERCLAIM`. The headline metric — `maturity_debt` — counts exactly these
inversions, and nothing else. The CI-relevant signal is therefore *"no capability
appears more mature than its evidence supports,"* not *"everything must be mature."*

## 4. The agentic-culture engine — every gap is the next work item

For each capability, the **first unmet rung** is rendered as a concrete, checkable
next work item that mirrors the [CLAUDE.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAUDE.md) "not yet" idiom — the
gap, the missing witness, and the step that closes it:

| Capability is at | The next work item | Closed when |
|---|---|---|
| `prototyped` (no tests) | *test it: add unit tests covering the leaf* | a `*_test.go` exists |
| `tested` (fak doesn't run it) | *dogfood it: wire it onto the running binary's path* | the leaf is imported transitively from a `cmd/` binary |
| `dogfooded` (no verb) | *default it: promote it to a documented `fak` verb* | named in `docs/cli-reference.md` |
| `default` (never measured) | *benchmark it: prove the surface with a number* | a `Benchmark` func or an authority row |

`fak maturity next` is that backlog — the queue an agent (or the issue-dispatch
loop) pulls from to advance the fleet one rung at a time, ladder-skips first
(the real debt), then the least-mature capabilities (the most leverage). The
"desire to create the next work item" stops being a virtue an agent must remember
and becomes a fact the tree emits. The recursive proof: this subsystem scores
itself, and the first thing it says it owes is its own next rung.

## 5. Honest first run (dogfooded on fak itself)

`fak maturity` over fak's own declared capabilities, 2026-06-29 (a snapshot — the
roster grows as new leaf lanes land):

```text
maturity_debt (ladder-skips): 0    index 78/100 [C]    over 111 capabilities
lifecycle ladder (count of capabilities per rung):
  0 proposed       1
  1 prototyped     0
  2 tested        18
  3 dogfooded     56
  4 default       36     (+ 35 carry the benchmarked badge)
```

Read this right: `maturity_debt = 0` is the *honest* result — fak's test discipline
is strong enough that there is no capability fak relies on yet leaves untested. The
**value is the distribution and the backlog**, not the debt: 18 capabilities are
`tested` but not `dogfooded` (complete and QA'd, but the product binary does not run
them yet — exactly the "complete but not dogfooded" gap this exists to surface), and
the gap between 56 `dogfooded` and 36 `default` is the long tail of capabilities fak
runs internally that have not been promoted to a documented surface. The one
`proposed` capability is a declared lane with no code (a coverage gap). The backlog
names the next rung for each. (The subsystem scores *itself*: `maturity` sits at
`default` — a documented verb fak runs — and the first thing it says it owes is its
own next rung, a benchmark.)

## 6. Where it sits among the siblings (not a duplicate)

| Scorecard | The question it asks | Unit |
|---|---|---|
| `internal/conceptusage` | does the fleet's *development* route through fak's own concepts? | commits + journals |
| `tools/product_scorecard.py` | is a concept a durable *product* a person can use today? | CLAIMS concepts |
| agent-readiness | can an *agent* discover / adopt / build on fak? | the agent journey |
| **`internal/maturity`** | **where is each *capability* in its lifecycle, and what matures it next?** | **per-leaf lane** |

`maturity` is the per-capability lifecycle view none of the others provide. It is a
**Go leaf + a `fak` verb**, not a new `tools/*.py` — per [AGENTS.md](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md)
("a new verb is Go in a leaf"), and mirroring the `internal/conceptusage` idiom
(`Build`/`Render`/`Markdown`/`Compare`, evidence re-derived from disk, deterministic
`Options` seam). It is the **shipped fak-tree binding** of grammar G1 / the readiness
ladder; the domain-free `dos maturity` kernel verb + a minted `MATURITY_OVERCLAIM`
reason remain the dos-kernel follow-up (out of this tree), exactly as #582 sequenced.

## 7. The plan to make it a critical subsystem

Shipping the verb is rung 3 of this subsystem's own ladder (it is dogfooded). The
plan to make it *default* and *critical* — the long-horizon program:

1. **Generated scorecard doc** — `docs/MATURITY-SCORECARD.md`, regenerable by
   `fak maturity --markdown`, so the lifecycle is a durable, reviewable page next to
   the other scorecards. *(landed with this note.)*
2. **Cadence + control-pane fold.** Add `maturity` to the cadence report
   (`internal/cadencereport`) and the scorecard control pane so the maturity index
   and the rung distribution trend over time in `docs/cadence/history.jsonl` — the
   ratchet then holds the one honest invariant (`maturity_debt` may not regress above
   baseline; no new ladder-skips land). *(landed: `fak cadence` now carries a
   maturity dimension and the unified scorecard pane folds `fak maturity --json`.)*
3. **Wire `fak maturity next` into the issue-dispatch loop.** The backlog is now a
   feeder for the same GitHub-issue surface the dispatcher already drains:
   `fak maturity route` turns the ranked top rows into stable, marker-keyed issue
   create/update plans, and `--live` files them with done-conditions/witnesses.
   The issue titles carry `maturity(<lane>)`, so the existing issue router sends
   each item back to the capability lane. Private-boundary lanes remain visible in
   `fak maturity next`, but `route` reports them as skipped instead of filing
   public issues. This is the literal mechanization of "each agent has the desire
   to create the next work item." *(landed.)*
4. **Per-capability targets + surface ceilings (v2).** Not every capability should
   reach `default` — an internal building block legitimately tops out at
   `dogfooded`. Add an optional, curatable target/surface-class per lane (the
   product scorecard's surface-ceiling `cap`), so the backlog stops nagging an
   internal mechanism to become a verb, and a `MATURITY_OVERCLAIM` fires when a
   capability *declares* a rung its evidence does not support. *(v2.)*
5. **Lift to the dos kernel.** Promote the domain-free core — closed rung ladder,
   evidence-bound adjudication, structured overclaim-refusal — into a `dos maturity`
   verb with a minted `MATURITY_OVERCLAIM` reason, so any agent fleet can score its
   own tree's lifecycle. fak's `internal/maturity` stays the reference binding.
   *(dos-kernel follow-up, out of this tree — sequenced, not done here.)*

## 8. Honest fences

- The subsystem is **shipped and dogfooded** (`fak maturity` runs over fak's own
  109 lanes today); items 2–5 in §7 are `not yet` and labelled as such.
- The evidence predicates are honest **proxies**, named so a reader can judge them:
  `dogfooded` = transitive reachability from the binary's import graph (a CI-only
  linter fak does not import reads as not-dogfooded, which is the intended signal);
  `default` = a documented verb in `docs/cli-reference.md`; `benchmarked` = a
  `Benchmark` func or an authority row. Sharper proxies (e.g. "appears in the `.dos`
  run journals" for *actually-exercised*) are a later calibration.
- `maturity_debt = 0` today is a true zero, not an unmeasured one — it means no
  relied-upon capability is untested. The gate is live: a future dogfooded-but-
  untested leaf moves it to 1 and reds the ratchet.
- This is a **governance/measurement** band over the tree. It does not replace the
  token engine, the model, or the harness — it is the same scope fak already owns.

## See also

- [`docs/INNOVATIONS-INDEX.md`](../INNOVATIONS-INDEX.md) — the durable catalog; this is its lifecycle cross-cut.
- [`CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md`](CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) — grammar G1 (readiness ladder), which this ships as the fak-tree binding.
- [`CONCEPT-DOS-READINESS-VERDICT-LADDER-2026-06-26.md`](CONCEPT-DOS-READINESS-VERDICT-LADDER-2026-06-26.md) — the readiness-ladder concept (#582) this lifecycle ladder instantiates.
- [`tools/product_scorecard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/product_scorecard.py) · [`internal/conceptusage`](https://github.com/anthony-chaudhary/fak/tree/main/internal/conceptusage) — the sibling scorecards it sits beside.
