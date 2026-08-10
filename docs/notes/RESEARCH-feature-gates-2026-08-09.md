# RESEARCH: feature gates in fak — what exists, what doesn't, and what to build

Observed 2026-08-09 against trunk `56f14a1472`. Question asked: does fak want a feature-flag
system, and what would it be for — landing work in progress, and keeping the result consistent?

Filed from this note: **#6090** (unify the rollout-ladder vocabulary) and **#6091** (a
feature-gate registry). Adjacent and NOT duplicated: **#2862** / **#2896** (config-surface
budget + hard cap), **#2863** (the CONFIG_NOT_ENV ratchet, shipped).

## The short answer

fak does not need a feature-flag *product*. It already has the two hard halves — a precedence
lattice and a staged rollout ladder — and it built each of them once, in the wrong scope, under
a private name. What is missing is small: **one vocabulary, and a registry that makes a gate
temporary by construction.**

The compile-time half of "land work in progress" is genuinely solved. The runtime half is not
built at all, and the knob authors reached for has since been banned.

## What exists

### 1. The compile-time WIP path — solved, leave it alone

`//go:build wip_<slug>` fences a not-yet-compiling file so the shared trunk stays green
(AGENTS.md §"Leave the shared tree buildable"). `fak wip fence|unfence` mechanizes it,
`internal/buildwitness` enforces it by running `go build ./cmd/fak` under default tags from a
package that compiles independently of `cmd/fak` — so the guard still reports a precise
undefined symbol when `cmd/fak` itself is red.

Measured: `git grep -l '^//go:build wip_'` over tracked files returns **0**. The fence is used
heavily and always resolves back to zero. That is the mechanism working, not evidence it is
unused.

`fak wip` is a whole subsystem beyond fencing — 14 subcommands over `refs/fak/wip/<session>`
checkpoints (`checkpoint`, `land`, `adopt`, `reconcile`, `remote-drain`, `sweep-guard`, …),
with the pure fold in `internal/wipref`. The *git* axis of landing WIP is well developed.

### 2. The rollout ladder — invented three times, three ways

Three packages independently built the same "land it now, expose it later" ladder:

| package | vocabulary | default |
|---|---|---|
| `internal/dispatchtick/rollout.go` | `off` / `shadow` / `canary` / `default` | `off` |
| `internal/promptmmu/config.go` | `on` / `shadow` / `off` | `on` |
| `internal/memorycotravel` | `shadow` / `live` / `off` | `shadow` |

The applied rung is spelled `default`, `on`, and `live`. Only `dispatchtick` has a canary rung,
and only `dispatchtick` encodes the property actually worth keeping: the guard **refuses** to
promote itself to default-on, because "a rollout guard that would silently promote to default-on
is no guard at all." Promotion needs a separate witness.

This is not a niche pattern. **142 non-test `.go` files** carry default-off / opt-in gating
language (`git grep -liE 'default[- ]off|off by default|opt-in'`, non-test Go). Staged exposure
is fak's dominant way of landing incomplete work. It has no shared noun.

### 3. The precedence lattice — already built, wrong scope

`internal/policy/orgprecedence.go` resolves

```
compiled-in FROZEN floor  >  central org overlay  >  operator overlay  >  agent-self
```

with three knob kinds — `FROZEN`, `RATCHET` (deny if ANY channel denies), `GATED_WIDEN`
(min-fold of every ceiling, each clamped by the channel above). Enumerated row-by-row in
`org_precedence_test.go`.

**This is the "defaults + user preferences" answer, and it already exists.** Its central
insight is the one a naive flag system gets wrong: authority means *caps*, not *wins*. A higher
channel cannot un-deny a lower one — an operator who locked their box down further than the org
requires keeps that lock. Any feature-gate resolution order should delegate here rather than
grow a fresh override chain.

### 4. A feature registry — exists, but for benchmarking

`internal/ablate` has `Concept` (registry.go) and `FeatureCard` (catalog.go) with
owner / plane / component / dependency / fidelity classification, a closed catalog, and
registration that panics on a duplicate or malformed token. It is a real, disciplined feature
registry — scoped to sweepable cache levers for N-arm ablation benchmarks. Its tokens are A/B
arm selectors, not exposure gates.

`internal/experiments` is an ML experiment ledger (models × backends). Unrelated despite the name.

`fak feature query` (`cmd/fak/feature.go` → `internal/selfquery`) answers "what can fak do" from
dev facts, live MCP tools, memory drivers, and capability cards. Gated features are not
answerable there.

### 5. The generation model — already assumes a gate it cannot check

`internal/devindex/generation.go` defines `gen/now` / `gen/next` / `gen/second-next`. The
`gen/next` rows read:

- promotion: "Promote toward now when a gate, dogfood run, schema contract, or **default-exposure
  proof** lands with a focused witness."
- demotion: "Demote if … **a feature gate proves the work is not ready for default exposure**."

So the maturity model already expects a feature gate to exist as a checkable object. Nothing
binds a real gate to a generation row, so that promotion evidence cannot be verified — only
asserted in prose.

## What is missing, and why it got that way

The knob a gated feature would historically have used is a `FAK_*` env var. There are **780
distinct `FAK_*` names** read in non-test Go (992 including tests).

`internal/envconfiglint` (#2863, shipped) is the ratchet that stops this growing: an env read
must name a declared secret or it is a config-surface violation. Correct, and deliberately a
ratchet rather than a ban — the count may only go down.

But the ratchet's own debt ledger, `admittedPostFreeze` in `internal/envconfiglint/admitted.go`,
says the quiet part directly:

> Each entry is behavioral configuration that genuinely belongs on a config surface; none can
> move there yet **because that surface does not exist** — it is #2862's deliverable.

And #2862, read closely, is a **budget gate** — a metric plus a hard ceiling on knob count. It
bounds the surface. It does not provide one. So:

- the refusal shipped before the destination existed;
- new gated work either burns an `admittedPostFreeze` exception, or hides behind a helper the
  regex scanner cannot see (the SCANNER VISIBILITY note in `admitted.go` records four reads that
  went **ungated** exactly this way — `FAK_CHATOPS_BOT_USER`, `FAK_CHATOPS_CHANNEL`, and two
  `FAK_PRECOMMIT_*_BUDGET_MS` — still read at runtime, still unpaid debt, now invisible), or
  invents a fourth private mode enum;
- the ledger is designed to only shrink and currently cannot.

`envconfiglint`'s own doc.go names the same missing piece from the other side: what would move
the ratchet toward `gen/now` is "pairing it with the config-surface budget gate (#2862) so a
refused read has **a named destination to move to**."

## The one design constraint that matters

A feature-gate registry *is* a config surface, and this repo's explicit fear — the whole of
#2862 and #2896 — is config surfaces. Hermes' 72KB `cli-config.yaml.example` and 23KB
`.env.example` are the named anti-goal.

So the distinguishing property has to be structural, not cultural:

> **A gate row is temporary by construction. A config knob is permanent.**

Concretely: every row carries an **expiry / review-by** and a **promotion witness** — the
specific evidence that would move it up a rung, in the shape `gen/next` already asks for. A row
that cannot state either is refused by a test.

That is also what makes #2862's ceiling *enforceable* rather than aspirational: retiring a knob
becomes a dated event with an owning leaf, not a cleanup nobody owns. The two issues compose —
#6091 provides the surface, #2862 bounds its size.

If rows can live forever, #6091 has built the thing #2862 exists to prevent, and should be
closed rather than shipped. That is written into the issue as its own kill condition.

## Recommendation

1. **#6090 first** (small, no behavior change). One closed vocabulary in a shared package,
   three call sites migrated, each keeping its current default. Watch the fail-open/fail-closed
   trap: `promptmmu.ParseMode` resolves an *unrecognized* value to default-**on** with
   `ok=false`, which is deliberate for that leaf; `dispatchtick` fails closed. The shared parser
   must let the caller pick its fallback, or the migration silently flips one of them.
2. **#6091 second**, delegating resolution to `orgprecedence` rather than inventing an override
   chain, and answerable through `fak feature query` so "what is landed-but-off right now" is a
   query instead of a grep.
3. **Do not touch** the `//go:build wip_` path or `fak wip`. Both work.

## Unproven / open

- Whether to extend `internal/ablate`'s `Concept` registry or sit beside it is a genuine open
  call. Its classification vocabulary is good and its registration discipline (panic on
  duplicate) is right; but its tokens mean "which arm of a benchmark", not "who may see this".
  Conflating them would make an ablation sweep able to flip an exposure gate. Not decided here.
- The 142-file count measures *prose* claiming a default-off posture. It is a lower bound on the
  practice and not a census of real gates — some of those files describe a gate rather than
  implementing one. The three-ladder finding is exact; the 142 is directional.
- No measurement was taken of how often a default-off feature is actually promoted later versus
  quietly left off forever. That number would decide how much the expiry mechanism is worth, and
  it is the first thing to measure if #6091 stalls.
