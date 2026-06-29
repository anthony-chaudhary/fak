# Context-safety CI gate (#1228, C10 of epic #1217)

_Research / design only. This is the **C10 spec** for the context-safety epic
[#1217](https://github.com/anthony-chaudhary/fak/issues/1217) — the build-time
enforcement that turns C9's per-visual re-derivation verdict into a green gate
on `make ci`. C9
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md))
defines the per-visual `RederivationVerdict`: a rendered visual is RED unless
every number on it re-derives from a tamper-evident source. C10 defines **when a
RED verdict fails the build** — and, the load-bearing fence, when it must **not**
(a single transient scrape skew may not false-red the build; only a *persistent*
mismatch reds it). It closes doctrine mechanism **D** of the C1 note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
at the build boundary, and is the gate side of the C2 catalog's **G-A** gap
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
No code ships here — the deliverable is this committed contract: the gate's
inputs, its pass/fail rule, and the persistence / change-point fence._

---

## The gate, in one line

> C9 turns a rendered visual into a *witness* (RED unless every number
> re-derives). C10 makes that witness **load-bearing at build time**: a context-
> safety visual whose numbers stop re-deriving from source **reds `make ci`** —
> but only once the mismatch is **persistent**, never on a single transient
> tick, so a flaky scrape can never wedge the trunk.

The discipline already exists in the tree for *other* numbers — the scorecard
ratchet reds `make ci` when debt rises (`scorecard-ratchet` target,
`Makefile:229-233`), the tool-test no-blackhole runner reds when a hermetic test
regresses (`gated-tests`, `Makefile:280-283`), the strict diff-witnessed
`closure_rate` (`tools/dispatch_status.py:578`) refuses to count a self-reported
close. C10 is the same shape aimed at the *rendered context-safety visual*: a
deterministic, no-network Go gate that reds the build when the picture no longer
equals its tamper-evident source.

---

## Inputs — the C9 verdicts, nothing else

The gate consumes **only** C9 `RederivationVerdict` values
([`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md)).
It computes no new number and trusts no rendered artifact — it folds the C9
verdicts that were already produced against tamper-evident sources (the
hash-chained journal `internal/journal/journal.go:577` `Verify` /
`internal/journal/journal.go:619` `VerifyRows`; the WITNESSED `/metrics`
family; git; CAS digests — the four sources C9 enumerates).

Each `RederivationVerdict` carries `Visual`, `Numbers []NumberCheck`, `AllGreen
bool`, and `Refusal string`. The gate reads exactly three things off each:

| Field (from C9) | What the gate reads it for |
|---|---|
| `AllGreen` | the per-visual pass bit for one check tick |
| `Refusal` (`SAFETY_UNRECOVERED` / `VALUE_UNRECONCILED` / `SAFETY_UNWITNESSED`) | a *named* RED, routed to a replan rather than a silent fail — verified via `dos_check_reason` (C13, [#1231](https://github.com/anthony-chaudhary/fak/issues/1231), [`context-safety-dos-refusal-tokens-1217.md`](context-safety-dos-refusal-tokens-1217.md)) |
| `Numbers[].Status` (`GREEN`/`RED`/`REFUSED`) | which number on which visual failed, for the gate's defect detail |

The gate carries **zero new trust** over C9: it adds no derivation, only the
*when-does-this-fail-the-build* decision. A `REFUSED` number is a RED tick for
this visual the same as a `RED` mismatch — an un-re-derivable number is a `not
yet` gap (fence c), never a green pass.

---

## The pass/fail rule — a tick, a streak, then the build

The gate is a pure function of a **window of recent check ticks**, exactly like
the crashloop fence (`internal/loopmgr/restart.go`) is a pure function of recent
attempts. One tick is one full C9 re-derivation pass over every shipped
context-safety visual.

```
CIGateVerdict {
  Reds        bool      // true => `make ci` exits non-zero
  Reason      string    // "" | a C13 DOS token | "CTX_SAFETY_VISUAL_DRIFT"
  Visual      string    // the offending primitive / panel id (1..6)
  Streak      int       // how many CONSECUTIVE ticks this visual has been RED
  Threshold   int       // the persistence floor (Streak >= Threshold reds)
  Detail      string    // the NumberCheck label + Rendered vs Rederived
}
```

**The rule, stated as the acceptance property:**

1. **One green tick is a green pass for that visual.** A C9 verdict with
   `AllGreen == true` and no `Refusal` clears the visual's RED streak to zero
   (the dual of `loopmgr`'s `AttemptsAfterSuccess` resetting the attempt counter
   after a healthy run, `internal/loopmgr/restart.go:172`).
2. **One RED tick does NOT red the build.** A single tick where `AllGreen ==
   false` (a RED or REFUSED number) increments the visual's RED streak but
   **does not** fail `make ci`. This is the false-red fence (below) — a transient
   scrape skew is one tick, and one tick is never the build verdict.
3. **A persistent RED streak reds the build.** When a visual's RED streak
   reaches the persistence threshold — `Streak >= Threshold` across consecutive
   ticks — the gate sets `Reds = true` and `make ci` exits non-zero. The
   `Reason` is the C13 DOS token from the deciding `RederivationVerdict.Refusal`
   when one is present, else the generic `CTX_SAFETY_VISUAL_DRIFT`.
4. **The roll-up is conservative: any one persistently-RED visual reds the
   whole gate** (the conservation discipline — the gate is only as green as its
   weakest visual, the same rule C9 applies within a visual and `covmatrix`
   applies within its grid, `internal/covmatrix/covmatrix.go:290`).

The gate is **deterministic and no-network**: it folds already-emitted C9
verdicts, so it joins the deterministic local gate the same way `cuda-check`
(`Makefile:290-292`) and `scorecard-ratchet` (`Makefile:229-233`) do — a pre-push
`make ci` reds on the same drift CI would.

---

## The false-red fence — change-point / persistence, not a single tick

This is the load-bearing requirement of #1228: **do not red on noise.** A C9
number re-derives from a live source — the WITNESSED `/metrics` accumulator, a
journal re-read. A single scrape can skew (a counter read mid-increment, a
race between render and re-derive, a clock-edge), and a single skewed tick would
RED a C9 verdict for one pass while the underlying value is fine. Reding the
build on that one tick would wedge the shared trunk on noise.

The fence is a **persistence / change-point rule**, lifted from two in-tree
precedents that already separate a sustained signal from a transient one:

- **Crashloop persistence (`internal/loopmgr/restart.go`).** `RestartPolicy`
  reds (`WATCHDOG_RESTART_EXHAUSTED`, `internal/loopmgr/restart.go:43`) only
  after `MaxAttempts` **consecutive** failures in one streak
  (`internal/loopmgr/restart.go:12-15`), and a healthy run that lasts the
  `ResetAfter` clear window resets the streak to zero
  (`internal/loopmgr/restart.go:28-30`, `:159-180`). A single flap does not give
  up; a *sustained* one does. C10 is the identical contract with "failed restart
  attempt" replaced by "RED re-derivation tick" and "give up" replaced by "red
  the build."
- **Change-point clustering (`internal/rsiloop/metarsi.go`).** The meta-RSI fold
  acts only when adverse decisions **cluster** in a recent window: it scans the
  most-recent `cfg.Window` cycles and fires only if the adverse decisions reach
  `cfg.MinEscalations` (`internal/rsiloop/metarsi.go:102-107`, default
  `Window: 20, MinEscalations: 2`, `internal/rsiloop/metarsi.go:99`). The breaker
  K — "consecutive non-keeps before ESCALATE" (`internal/rsiloop/metarsi.go:47`)
  — is the same consecutive-streak fence in a sibling loop.

C10 adopts the **consecutive-streak** form (the `loopmgr` shape) as the floor: a
visual reds the build only after `Threshold` *consecutive* RED ticks, and any
one green tick resets the streak. A change-point variant (RED for a sustained
*fraction* of a recent window, the `metarsi` cluster shape) is a permitted
stricter overlay where a visual's check cadence is noisy enough that a strict
consecutive run is too brittle — but the **default is consecutive-streak**,
because it is the simplest rule a reader can falsify by counting ticks.

**Why a streak and not an average:** an average smears a real, just-started
drift into the noise floor and delays the red; a consecutive-streak reds the
moment the drift is *sustained* (every tick RED for `Threshold` ticks) yet
absorbs an isolated skew (one RED tick between greens never accumulates). The
fence is conservative in the safe direction — it can only **delay** a red by at
most `Threshold-1` ticks, never **hide** a persistent one. A truly broken visual
stays RED every tick and reds the build on schedule.

**The fence never widens into a blind spot.** The gate records the live
`Streak` and `Threshold` on every `CIGateVerdict` (above), so a reader can see
*how close* a visual is to reding even while it is still green — the same way
the scorecard ratchet surfaces an `early_warning` list without yet tripping the
gate (`tools/scorecard_control_pane.py:445-449`). A persistently-RED visual that
has not yet reached `Threshold` is a visible, named, climbing streak, not a
silent pass.

---

## Where the gate plugs into the build

The gate is a Go-side check, mirrored into both `make ci` and `scripts/ci.ps1`
so a Windows pre-push and a WSL/CI run red on the same drift (the local↔CI
alignment the `Makefile` header calls out, `Makefile:5-10`). Two wirings are
available; both are in-tree precedents:

- **As a `*_test.go` gate folded by `go test ./...`** — the cheapest wiring,
  since `go test ./...` is already the first hard step of both `make ci`
  (`Makefile:37-38`) and `scripts/ci.ps1` (`scripts/ci.ps1:14-16`). A
  `ctxsafety_gate_test.go` that loads the recent C9 verdict ticks and asserts no
  visual's RED streak has reached `Threshold` reds the build with no new Makefile
  step — the same way `internal/architest`'s tier gate and the
  `internal/pythongate` / `internal/windowgate` hygiene tests
  (`Makefile:169-170`) already red `go test`.
- **As a scorecard `--check` ratchet** — fold the persistent-RED count into a
  `scorecard.Payload` with a debt key (the shape `internal/covmatrix`
  `covmatrix.go:239-296` uses, `DebtKey = "growth_debt"`,
  `internal/covmatrix/covmatrix.go:36`), so it rides the existing
  `scorecard-ratchet` gate (`Makefile:229-233`,
  `tools/scorecard_control_pane.py --check`) with the persistent-RED-visual count
  pinned at 0. This makes the context-safety gate a first-class member of the
  portfolio floor — debt may hold or fall, never rise
  (`tools/scorecard_control_pane.py:427-428`).

The `*_test.go` wiring is the recommended default (deterministic, no Python, reds
the step that already runs first). The scorecard wiring is the upgrade once the
context-safety visuals are a tracked debt line in the control pane.

`fak audit verify` (`cmd/fak/audit.go:53-71`, calling
`internal/journal/journal.go:577` `Verify`) is the on-demand analog this gate
generalizes: where `audit verify` reds when the *chain* is tampered, the C10 gate
reds when the *rendered picture* persistently stops equaling the chain.

---

## Honest `not yet` gaps (per fence c)

- **No C9 checker exists yet, so C10 has no verdicts to fold today.** C10 is the
  build-time *enforcement* of C9; until the C9 re-derivation checker
  ([#1227](https://github.com/anthony-chaudhary/fak/issues/1227)) is built, this
  gate has no `RederivationVerdict` inputs. This note is the *gate contract*; the
  implementation is the follow-on build epic, gated on review of the #1217 design
  notes. Named, not assumed green.
- **The persistence threshold and tick cadence are unpinned.** The fence is
  specified as a consecutive-streak rule, but the concrete `Threshold` (how many
  consecutive RED ticks reds the build) and the check cadence (one tick per CI
  run? per scrape window?) are tuning the implementation must pin against the
  observed false-red rate of the live `/metrics` scrape — the same way `loopmgr`
  pins `MaxAttempts`/`ResetAfter` and `metarsi` pins `Window`/`MinEscalations`.
  Until pinned, the gate's exact red point is a `not yet`, not a fixed number.
- **`CTX_SAFETY_VISUAL_DRIFT` is not yet a declared DOS reason.** The generic
  persistent-drift token the gate emits when no C9 `Refusal` is present must be
  added to the closed refusal vocabulary (C13's `dos.toml [reasons]` set,
  [#1231](https://github.com/anthony-chaudhary/fak/issues/1231)) and verified via
  `dos_check_reason` before emission. It does not exist yet; the gate may not
  emit an unclassified reason.
- **G-B (harness-rewrite-defeats-the-shed) is observed, not yet gated.** The
  second-compactor event the C2 catalog names (G-B) is *observed* today —
  `internal/gateway/harness_coherence.go:99` counts `harnessRewrites` and
  `internal/gateway/harness_coherence.go:165` classifies the
  `EventHarnessRewrite` — but no C9 visual yet re-derives a "value preserved
  through the shed" number that a harness rewrite would falsify, so C10 cannot
  gate on it yet. The observation seam exists; the gated visual is a `not yet`.

---

## Acceptance check (against #1228)

The issue's acceptance: *a gate spec — when a RED C9 verdict fails the build —
plus the false-red fence (change-point / persistence, not a single tick).*

- **The inputs are pinned to the C9 verdicts** — the gate consumes only
  `RederivationVerdict` (`AllGreen`, `Refusal`, `Numbers[].Status`), computes no
  new number, and trusts no rendered artifact. It carries zero new trust over
  C9.
- **The pass/fail rule is stated** as `CIGateVerdict`: one green tick clears a
  streak, one RED tick does not red the build, a RED streak reaching `Threshold`
  reds `make ci` non-zero, and any one persistently-RED visual reds the whole
  conservative roll-up.
- **The false-red fence is the consecutive-streak / change-point rule**, lifted
  from `internal/loopmgr/restart.go` (consecutive `MaxAttempts` + `ResetAfter`
  clear window) and `internal/rsiloop/metarsi.go` (the `Window`/`MinEscalations`
  cluster). A single transient scrape skew is one tick and never the build
  verdict; the fence can only delay a red by `Threshold-1` ticks, never hide a
  persistent one, and the live streak is always visible.
- **The build wiring is grounded** in real in-tree gate precedents — a
  `*_test.go` folded by `go test ./...` (the `architest` / `pythongate` /
  `windowgate` shape) or a scorecard `--check` ratchet (the `covmatrix`
  `growth_debt` shape riding `scorecard-ratchet`), mirrored into both `Makefile`
  and `scripts/ci.ps1`.
- **The honesty fences hold** — the gate is kept distinct from the security floor
  and the #1147 self-tax plane (it reds on *value-preservation* drift only), a
  REFUSED number is a `not yet` RED never a green pass (fence c), and every
  un-built piece (no C9 verdicts yet, unpinned threshold, undeclared
  `CTX_SAFETY_VISUAL_DRIFT` token, ungated G-B) is named as a gap.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Consumes the
C9 re-derivation verdict (#1227,
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md))
as its sole input; closes doctrine mechanism D of the C1 note (#1218) at the
build boundary and the gate side of the C2 catalog's G-A gap (#1219); emits the
C13 DOS refusal tokens (#1231,
[`context-safety-dos-refusal-tokens-1217.md`](context-safety-dos-refusal-tokens-1217.md)).
Design-only — no implementation ships under #1217 until the notes are reviewed._
