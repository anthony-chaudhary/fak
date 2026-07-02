---
title: "DOS `salience` usefulness audit (2026-06-24): is the keep-but-park verdict actually useful, and where does fak use it?"
description: "Audit of whether `dos salience` — the 'is this true thing LIVE or true-but-PARKED?' verdict — earns its place. The verb is a sound, fail-safe-to-RETAIN primitive that fills a real verdict-space gap (the keep-but-park dual of `retire`), and its acceptance behaviour matches spec (PARKED→exit 3, LIVE→exit 0, INDETERMINATE→exit 4, exposed under `dos doctor --json .exit_codes.salience`). But its usefulness today is LATENT: nothing routes on the verdict, the CLI is single-item only (the no-loss `partition` fold is import-only), and the evidence bits are host-produced. Dogfooded over 10 real true-but-parked fak findings."
---

# DOS `salience` usefulness audit — 2026-06-24

**Question:** `dos salience` (docs/391) is the "is this TRUE thing LIVE, or true-but-PARKED
out of the hotpath?" verdict — the prevent-silent-loss dual of `retire`. Is it *genuinely
useful*, or a well-built verdict in search of a user? And where, if anywhere, does fak use it?

**Verdict:** the verb is **sound and fills a real gap** — a per-item, clock-free,
deterministic `LIVE` / `PARKED` / `INDETERMINATE` verdict whose fail-safe always points at
RETAIN (no state ever means delete; absence of evidence never parks). Its acceptance
behaviour matches the spec exactly. **But its usefulness today is LATENT, not realized**:
nothing in the dos kernel *or* fak routes on a salience verdict, the CLI judges only one
hand-asserted item at a time (the no-loss `partition` fold is import-only), and the evidence
bits are host-produced — so the headline "never silently lose" guarantee currently lives in
the *type system*, not in any wired consumer. This audit gives the verb its **first concrete
fak foothold**: a dogfood register of 10 real true-but-parked fak findings (below), and the
honest wiring fak still needs for the guarantee to bite.

Every load-bearing claim below was **adversarially re-derived by an independent verifier**
that was told to refute it; both survived (consumer-gap and gap-is-real, refuted=false).

## Acceptance — the verb exists and behaves to spec

The behaviour the goal asked for is already shipped in the `dos` kernel
(`dos-kernel-public`, `src/dos/salience.py` + the CLI handler `cmd_salience` in
`src/dos/cli.py`). Verified directly (exit codes captured with no trailing pipe — a pipe to
`head`/`jq` hides the real exit code):

```
$ dos salience --label F12 --default-off
PARKED  not on the default execution path (off by default) — parked, RETAINED + surfaced; recoverable …
exit=3                                        # PARKED(NOT_IN_HOTPATH), recoverable

$ dos salience --label F12 --reachable
LIVE  no park-reason fired — kept in the default hotpath (NOT a claim of importance; that is a JUDGE/HUMAN question)
exit=0                                        # LIVE

$ dos salience --label F12                     # no evidence supplied
INDETERMINATE  no usefulness evidence supplied — cannot judge salience; abstain → RETAIN + surface
exit=4                                        # abstain → RETAIN (never a silent drop)

$ dos doctor --json | jq '.exit_codes.salience'
{ "INDETERMINATE": 4, "LIVE": 0, "PARKED": 3, "contract_error": 2, "unknown": 5 }
```

The exit map is consistent with the rest of `dos exit-codes` (0 = the happy verdict, 3 = a
recoverable not-happy verdict — the same slot as `breaker` OPEN, `complete` INCOMPLETE,
`cooldown` RECENTLY_ATTEMPTED, `efficiency` COSTLY — 4 = indeterminate, 2 = contract error).
(`jq` was not on this box; `dos doctor --json` already exposes `.exit_codes.salience` — the
data was present, only the pipe tool was missing.)

## What it is — the keep-but-park dual of `retire`

A fleet's findings, claims, code paths, and lessons are not only TRUE-or-FALSE; they are
also USEFUL-or-not, and **those are orthogonal axes**. A thing can be perfectly true and not
currently useful: a real bug off the default execution path, a correct note about a feature
behind a disabled flag, a lesson that still holds but no longer decides. The danger is the
**silent loss** of such a thing — dropping it *as if it were false* when it is merely *not in
the hotpath* — which costs nothing today and bites the day the path goes live. `salience`
converts that silent drop into a **recorded, recoverable park** under a typed reason, each
carrying a re-entry line. It sits in the keep-only-what-a-witness-confirms family as the
KEEP-side dual of `retire`'s measured DROP:

| verb | question | terminal disposition |
|---|---|---|
| `retire` | does this item still EARN ITS PLACE? (measured) | **DROP** to archive (proposal) |
| `salience` | is this true thing LIVE or PARKED? (per item, mechanical) | **KEEP** — park in place, recoverable |

The four park rungs are `NOT_IN_HOTPATH` (`--default-off`), `UNREACHABLE` (`--unreachable`),
`SUPERSEDED` (`--superseded`), and the MEASURED `LOW_CONTRIBUTION`
(`--min-contribution`/`--min-trials`). The fail-safe always points at RETAIN: `None`-is-unknown
never parks, thin evidence never parks, and `is_retained` is `True` for *every* state.

## The honest usefulness verdict — real, but latent

The verb is well-built; what it lacks is *realization*. Three bottlenecks, each evidenced:

1. **No consumer routes on the verdict.** Across both repos, `salience.classify` /
   `salience.partition` / `SalienceVerdict` appear in exactly three files: the module, the
   lone CLI handler `cmd_salience` (`cli.py`, which turns the verdict straight into an exit
   code), and the unit tests. No picker, assembly policy, reviewer, MCP tool, hook, skill, or
   CI step reads a salience verdict or branches on its exit code. The docstring's named
   consumer — "a picker, an assembly policy, a reviewer decides whether to route the hotpath"
   — is **aspirational**. By contrast `retire` *is* wired: a real driver
   (`drivers/memory_recall.py:retire_sweep`) folds `RETIRE` into surfaced operator proposals.
   So salience's "never silently lose" guarantee currently lives in the **type system**
   (`SaliencePartition`'s no-loss invariant: `len(live)+len(parked)+len(indeterminate)` ==
   the input count), not in any behaviour a consumer exercises.

2. **The CLI is a single-item operator-assertion wrapper.** `cmd_salience` builds one
   `SalienceEvidence` from flags and calls `classify` — it judges exactly one item per
   invocation. The no-loss batch fold `salience.partition` (the part that actually *prevents
   silent loss across a corpus*) is **not reachable from the CLI**: no `--file`/`--items`/
   stdin path exists. To partition many items you must `import dos.salience` and call
   `partition` yourself. So the standalone CLI records a single park; it cannot sweep.

3. **The evidence is host-produced.** The kernel cannot and must not compute reachability —
   "a host hands the bit in." From the CLI, that host is the operator: `--default-off` is the
   operator asserting a bit they already know. The verb's value over a plain note is that it
   makes the assertion **typed, exit-coded, and carry a re-entry line** — real, but modest
   until a producer (a static-analysis pass, a flag-table reader, a contribution meter) feeds
   the bits automatically.

Net: salience earns its place as **(a) a discipline** — turn a silent drop into a typed,
recoverable, reason-classed record — and **(b) a library primitive ready to be wired**. It is
the verdict that has the vocabulary but not yet the wiring. This is the same "useful but
latent" shape as the [multi-sink cache lifecycle](MULTI-SINK-CACHE-LIFECYCLE-2026-06-23.md)
finding (one registrant today, the fan never exercised with >1 sink).

## Overlap — does it duplicate a neighbour? (mostly no)

Read against the actual source of its five nearest neighbours:

| neighbour | axis | distinct from salience? |
|---|---|---|
| `retire` | measured DROP-to-archive over (contribution, trials) | **mostly** — see the nuance below |
| `cooldown` | clock-windowed "have I TRIED this unit recently?" over attempt history | **yes** — salience has no clock, judges findings not work-units |
| `lifecycle` PARK | semantic, whole-plan, **judge-gated** class taxonomy (no per-item `classify`) | **yes** — salience is the mechanical per-item floor one altitude *under* it |
| `lint` dead-policy | self-consistency of the authored *config* (shadowed lanes, dangling aliases) | **yes** — disjoint subject (config) and input (registries, not item facts) |
| `pickable` HoldReason | pre-dispatch GATE over work-unit dispatchability (a PEP) | **yes** — salience is advisory (a PDP); it borrows the typed-reason+next-action *idiom*, not the function |

**The one real nuance (retire).** salience's `LOW_CONTRIBUTION` rung and retire's
`UNDERPERFORMED` rung use a **byte-identical predicate** — the same `(contribution, trials)`
pair behind the same `min_trials` thin-evidence floor. The distinction is therefore *not* in
the test; it is in the **verdict**: retire emits a DROP-to-archive proposal with no
recoverability field, while salience emits a PARK that is `is_retained==True` and carries a
`reactivation` re-entry line. Same question, opposite disposition — which is exactly the
declared design (the keep-but-park dual). So the gap is real at the verdict/vocabulary/field
layer, even where the predicate is shared.

## Dogfood — fak's own true-but-parked findings, recorded as a recoverable register

The "add in": fak is full of true-but-parked findings that today live only as prose in
`CLAIMS.md` (`[STUB]`/`[SIMULATED]`) and the [DOS effective-usage
audit](DOS-EFFECTIVE-USAGE-AUDIT-2026-06-22.md)'s "Remaining gaps". Run through `dos
salience`, each becomes a typed, exit-coded, recoverable park instead of a line that could be
silently dropped on the next cleanup. All ten classify `PARKED` (exit 3) with the right
reason class and a re-entry line; a genuinely-live finding and a no-evidence one are shown as
controls:

| finding (label) | reason class | flags | state / exit |
|---|---|---|---|
| empty-plan-portfolio-inert-dos-surfaces | UNREACHABLE | `--unreachable` | PARKED / 3 |
| commit-stamp-doctor-orphaned-automation | UNREACHABLE | `--unreachable` | PARKED / 3 |
| supervise-target-presumes-dead-watchdog-loop | NOT_IN_HOTPATH | `--default-off` | PARKED / 3 |
| polymodel-serving-core-off-mainline | NOT_IN_HOTPATH | `--default-off` | PARKED / 3 |
| wirescreen-model-arm-blocked-on-latency | NOT_IN_HOTPATH | `--default-off` | PARKED / 3 |
| xenginekv-no-live-external-engine-transport | NOT_IN_HOTPATH | `--default-off` | PARKED / 3 |
| headroom-rust-pageout-codec-optional-seam | NOT_IN_HOTPATH | `--default-off` | PARKED / 3 |
| custom-reasons-vocabulary-never-fires-via-dos | SUPERSEDED | `--superseded` | PARKED / 3 |
| vdso-realworld-hit-rate-below-threshold | LOW_CONTRIBUTION | `--contribution 0.007 --min-contribution 0.1 --min-trials 50 --trials 50` | PARKED / 3 |
| dos-mcp-tools-low-signal-advisory-cautions | LOW_CONTRIBUTION | `--contribution 0.0 --min-contribution 0.1 --min-trials 100 --trials 18500` | PARKED / 3 |
| *control:* adjudicator-default-deny-on-hot-path | — | `--reachable --default-on` | **LIVE / 0** |
| *control:* some-finding-with-no-evidence | — | *(no flags)* | **INDETERMINATE / 4** |

This table *is* the prevent-silent-loss register the verb is for: each row is a true fak
finding that is not in the default hotpath today, now recorded with a typed reason and a
recovery affordance, not dropped. The `LOW_CONTRIBUTION` rows carry illustrative
contribution/floor numbers derived from the prose (vDSO ~0.7 % real-world purity vs a 0.1
floor; the `dos_*` advisory cautions at ~nil net value over the ~18.5 k-call window); the
measured magnitudes are real, the exact floor values are operator-set.

## How to make it non-latent in fak (operator decisions / next steps)

The verb only stops being latent when something *routes* on its output. Concrete, low-risk
options, smallest-first — recorded as decisions, **not** built here (building a producer with
no consumer would just add another latent surface):

> **UPDATE 2026-06-24 — #1 is now BUILT and WIRED.** The first option below shipped as
> [`tools/claims_salience_register.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/claims_salience_register.py): it routes
> every `CLAIMS.md` claim through `dos.salience.partition` (`[SHIPPED]`→LIVE,
> `[SIMULATED]`/`[STUB]`→host-declared PARKED), asserts the no-loss invariant + cross-checks
> the live/parked counts against the ledger, and surfaces the recoverable register at
> [`docs/claims-salience-register.md`](../claims-salience-register.md). It uses the REAL
> kernel verb (`import dos.salience`) and is wired into the green gate (`make ci` +
> `scripts/ci.ps1`) — a real gate where the dos kernel is importable, an advisory SKIP where
> it is not; the hermetic logic is gated in `ci.yml` by `claims_salience_register_test.py`.
> So `dos salience` is no longer latent: a consumer now routes on the verdict. Options #2
> (auto-derive the `default_on` bit) and #3 (fold into `fak doctor`) remain open.

- **A `[STUB]`/`[SIMULATED]`-claim park register (CI advisory).** A fak tool reads every
  `[STUB]`/`[SIMULATED]` line in `CLAIMS.md` (each is true-but-not-LIVE by definition), feeds
  them through `salience.partition`, and asserts the no-loss invariant — every claim lands in
  exactly one bucket, none dropped — surfacing the `parked` bucket. This is the first *real
  consumer*: it routes on the verdict (it prints the parked register) and the no-loss count
  becomes a watched number, not a guarantee that only the type system knows.
- **An evidence producer for the boolean rungs.** The honest bottleneck is that the bits are
  hand-asserted. `default_on` is mechanically derivable for fak's flag-gated leaves (the
  `FAK_*` env gates + the `internal/registrations` defconfig membership the corpus already
  cites), so a small reader could supply `default_on=False` for off-mainline leaves
  automatically instead of by operator assertion.
- **Surface the parked register in `fak doctor` or the DOS audit re-run.** A reviewer reading
  the parked bucket *is* the "assembly policy / reviewer" consumer the docstring names; the
  cheapest realization is to fold this register into the existing audit cadence.

None of these needs a kernel change — salience is host-wired by design (the kernel ships the
fold + the floor; the host supplies the signals + the consumer).

## How to re-run this audit

```bash
# acceptance (exit codes captured WITHOUT a trailing pipe — a pipe hides the real code):
dos salience --label F12 --default-off ; echo exit=$?     # PARKED  / 3
dos salience --label F12 --reachable   ; echo exit=$?     # LIVE    / 0
dos salience --label F12               ; echo exit=$?     # INDETERMINATE / 4
dos doctor --json | jq '.exit_codes.salience'             # {LIVE:0, PARKED:3, INDETERMINATE:4, …}

# consumer scan — confirm nothing routes on a salience verdict (only the module, its CLI
# handler, and tests should match) in the dos kernel and in fak:
rg -n 'salience\.(classify|partition)|SalienceVerdict|SaliencePartition' <dos-kernel-public>/src
rg -ni 'dos salience|salience\.(classify|partition)' .   # fak: only this note + unrelated AWQ/memory hits

# dogfood — re-record fak's parked-truths as a recoverable register (see the table above):
dos salience --label vdso-realworld-hit-rate-below-threshold \
  --contribution 0.007 --min-contribution 0.1 --min-trials 50 --trials 50 --json ; echo exit=$?
```
