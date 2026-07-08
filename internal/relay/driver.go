// Rung H1 (issue #1894): the relay driver loop — the first thing that runs a relay
// leg END TO END out of the already-shipped pieces. Every part of a leg exists as its
// own rung (reload D1/D2, ledger progress D3, safe point E1–E4, externalize gate F2,
// arm/fire G2 with the G3 trigger axes, baton C1/C2/C4) but nothing orchestrated a
// full leg; this rung is that orchestration, the spine's Phase 6 loop
// (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "One leg's loop"):
//
//	reload+verify -> done-check -> work hook -> arm -> safe-point -> externalize
//	  -> write baton -> Recontinue
//
// The driver ADDS no policy of its own: every verdict is computed by the rung that
// owns it, and every effect (doing work, persisting the baton, minting the successor
// leg) happens through a caller-supplied hook. Like every relay rung it does NOT
// import internal/session — the live session.Recontinue is wired in by a later floor
// through the Recontinue hook, exactly the way armtriggers.go takes its axis numbers
// as scalars so the session package can wire the relay without a cycle. The driver
// itself reads no clock and does I/O only through the injected seams.
//
// Deliberately out of scope, per the issue: lease continuity across legs is H2 (#1895
// — held_region is CARRIED verbatim here, never re-acquired), the park path is H5
// (#1898 — a leg that cannot reach a safe point errors instead of parking), and the
// status CLI is H7 (#1900).
package relay

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ReasonGoalDone is the relay reason token emitted when the done-check finds the
// leg's done_when already satisfied against the durable store
// (docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md): the relay ended normally —
// close it, do not launch another leg. It is the token that makes a relay's restart
// idempotent: a relay that is done stays done.
const ReasonGoalDone = "RELAY_GOAL_DONE"

// BudgetUsage is one boundary's reading of the four Envelope trigger axes the G3
// evaluator folds (armtriggers.go): context tokens (primary), turns, wall-clock,
// spend. The work hook reports the live numbers; the driver only passes them through
// to ArmTriggers.Crossed, keeping the axis math where it already lives.
type BudgetUsage struct {
	Context AxisUsage
	Turns   AxisUsage
	Wall    AxisUsage
	Spend   AxisUsage
}

// Orientation is what the driver derived from the incoming baton BEFORE any work ran
// — the reload+verify half of the leg, handed to the work hook so a successor starts
// from verified state, never from trust.
type Orientation struct {
	// FirstLeg is true when there was no incoming baton (Baton.IsZero): the relay is
	// starting, so there is nothing to reload and Stale/Progress stay zero.
	FirstLeg bool
	// Stale is the D2 verdict over the incoming baton (stale.go). When Stale.Stale is
	// true the cursor no longer matches ground truth: the driver has NOT read ledger
	// progress (Progress stays zero) and the work hook must re-derive from the durable
	// store rather than trust anything the baton carried.
	Stale StaleOutcome
	// Progress is the D3 ledger-verified progress read through the incoming cursor
	// (progress.go) — populated only when the baton verified fresh and a LedgerReader
	// was configured, so the hook is never handed progress that did not survive
	// re-verification.
	Progress VerifiedProgress
}

// BoundaryObs is what the work hook reports at each candidate boundary: the evidence
// every safe-point/trigger rung needs, as values, in the same caller-supplies-the-
// signal style those rungs already use. The driver derives every verdict from these
// observations; the hook asserts nothing directly.
type BoundaryObs struct {
	// Usage is the live reading of the G3 trigger axes at this boundary.
	Usage BudgetUsage
	// TurnInFlight is the E2 boundary signal: true while a tool call/decode is still
	// mid-turn (inflightguard.go) — a rotation is never permitted in flight.
	TurnInFlight bool
	// Tree is the E3 working-tree evidence (treegate.go): dirty paths and the paths
	// this leg explicitly parked.
	Tree TreeStatus
	// NextSteps are the leg's own candidate next actions for the E4 extractor
	// (nextaction.go); a boundary needs them to collapse to exactly one line.
	NextSteps []string
	// Facts are the load-bearing facts this leg relies on, for the F2 externalize gate
	// (externalize_gate.go): any transcript-only fact holds the rotation.
	Facts []LoadBearingFact
	// AtSHA is the git commit observed at this boundary — the anchor the written
	// baton's ProgressCursor.StartSHA and Tombstone.AtSHA carry when the leg rotates
	// here. A fired boundary with no anchor is an error: the successor could never
	// re-verify the handoff.
	AtSHA string
	// Artifacts, OpenQuestions and DoNotRederive are the durable pointers the leg has
	// accumulated, in baton form (pointers, never bytes). DoNotRederive entries are
	// merged into the carried-forward index (C4) so a rediscovered dead end dedupes.
	Artifacts     []Artifact
	OpenQuestions []string
	DoNotRederive []string
}

// LegConfig wires one relay leg: the incoming baton (zero for the first leg), the
// carried identity for a first leg, the rotation policy, the verification seams, and
// the three effect hooks. Everything the driver cannot compute purely arrives here.
type LegConfig struct {
	// Incoming is the predecessor's baton. The zero baton (IsZero) means this is the
	// relay's first leg: RelayID, Objective, DoneWhen, LedgerRef and HeldRegion below
	// seed the identity instead of being carried from the baton.
	Incoming Baton
	// RelayID / Objective / DoneWhen / LedgerRef / HeldRegion seed a FIRST leg's
	// identity; on a successor leg they are ignored and the incoming baton's values
	// are carried verbatim instead.
	RelayID    string
	Objective  ctxplan.ObjectivePin
	DoneWhen   string
	LedgerRef  string
	HeldRegion []string
	// TraceID is THIS leg's trace id; the written baton records it as ParentTrace so
	// the successor's lineage links back here.
	TraceID string
	// Triggers is the G3 soft-mark policy (armtriggers.go). Policy data, not code: an
	// unset/invalid SoftMark never arms, so a leg with no rotation policy never
	// rotates on a budget axis.
	Triggers ArmTriggers
	// MaxBoundaries is the driver's own fail-closed loop bound: the maximum number of
	// work-hook boundaries the leg may evaluate before the driver refuses with an
	// error instead of spinning forever. Must be positive.
	MaxBoundaries int
	// Resolver re-verifies the incoming baton against git (D1/D2). Required whenever
	// Incoming is non-zero; unused on a first leg.
	Resolver Resolver
	// Ledger optionally reads verified progress through the incoming cursor (D3).
	// Nil skips the read (Orientation.Progress stays zero).
	Ledger LedgerReader
	// DoneCheck evaluates the leg's done_when predicate against the durable store —
	// the idempotent-restart check run BEFORE any work. Nil skips the check. An error
	// is treated as NOT done (fail closed: done is never claimed on an unverifiable
	// predicate; the leg keeps working instead).
	DoneCheck func(doneWhen string) (bool, error)
	// Work advances the leg by one unit of work and reports the boundary it stopped
	// at. It receives the driver's orientation (so a stale baton is re-derived, not
	// trusted) and the boundary index. An error aborts the leg.
	Work func(o Orientation, boundary int) (BoundaryObs, error)
	// WriteBaton persists the canonical baton wire bytes (the C2 Marshal output) —
	// the C6-shaped persistence seam. Runs after the rotation fires, before
	// Recontinue.
	WriteBaton func(wire []byte) error
	// Recontinue mints the fresh successor leg from the written baton and returns its
	// trace id — the seam a later floor wires to session.Recontinue. It is called
	// only after the baton is durably written, so a successor always has a handoff to
	// reload.
	Recontinue func(b Baton) (successorTrace string, err error)
}

// LegOutcome is the typed result of driving one leg to a clean end: which closed
// reason ended it, the orientation the leg started from, and — on a rotation — the
// written baton, the successor's trace id, and the boundary trail.
type LegOutcome struct {
	// Reason is the closed relay token that ended the leg: ReasonGoalDone (the
	// done-check ended the relay before any work) or the ArmFire tokens' terminal
	// "RELAY_ROTATED" (the leg rotated into a fresh leg).
	Reason string
	// Orientation echoes what the leg started from (first leg / stale / verified
	// progress) so a supervisor can read the reload verdict off the outcome.
	Orientation Orientation
	// Baton is the canonical baton the closing leg wrote; zero unless the leg
	// rotated.
	Baton Baton
	// SuccessorTrace is the fresh leg's trace id as returned by Recontinue; empty
	// unless the leg rotated.
	SuccessorTrace string
	// Boundaries is how many work-hook boundaries the leg evaluated.
	Boundaries int
	// Holds are the closed reasons (in boundary order) for which an evaluated
	// boundary declined to rotate — IN_FLIGHT_TOOL_CALL, TREE_DIRTY_UNPARKED,
	// NO_NEXT_ACTION / AMBIGUOUS_NEXT_ACTION, RELAY_NOT_EXTERNALIZED — the operator
	// readout of why the leg kept working. Boundaries that were simply not armed and
	// not blocked record nothing.
	Holds []string
}

// DriveLeg runs ONE relay leg end to end and returns its typed outcome. The loop is
// the spine's, with each verdict computed by the rung that owns it:
//
//  1. Reload+verify (D1/D2): a non-zero incoming baton is re-checked against git;
//     a stale baton is NOT trusted — ledger progress is skipped and the stale
//     outcome rides into the work hook's orientation so the leg re-derives from
//     durable state.
//  2. Done-check: done_when is evaluated against the durable store BEFORE any work;
//     satisfied means the relay is already done (ReasonGoalDone) and no work or
//     successor leg is launched.
//  3. Work loop, at most MaxBoundaries times: the work hook advances and reports a
//     boundary; the G3 triggers decide arming, the E2/E3/E4 rungs derive the three
//     safe-point axes, the F2 gate holds any boundary with transcript-only state,
//     and the G2 arm/fire machine folds it all — FIRE remains reachable only
//     through a safe point every gate admitted.
//  4. On fire: the baton is built from carried identity plus the fired boundary's
//     durable pointers, canonically encoded (C2), persisted through WriteBaton, and
//     only then handed to Recontinue to mint the fresh leg.
//
// A leg that exhausts MaxBoundaries without firing is an error naming the last hold
// (the park path that would absorb it is rung H5, out of scope here) — fail closed,
// never a fabricated rotation.
func DriveLeg(cfg LegConfig) (LegOutcome, error) {
	if cfg.Work == nil || cfg.WriteBaton == nil || cfg.Recontinue == nil {
		return LegOutcome{}, fmt.Errorf("relay: driver needs Work, WriteBaton and Recontinue hooks")
	}
	if cfg.MaxBoundaries <= 0 {
		return LegOutcome{}, fmt.Errorf("relay: driver needs a positive MaxBoundaries loop bound, got %d", cfg.MaxBoundaries)
	}
	first := cfg.Incoming.IsZero()
	if !first && cfg.Resolver == nil {
		return LegOutcome{}, fmt.Errorf("relay: a successor leg needs a Resolver to reload+verify the incoming baton")
	}

	// Carried identity: a successor leg carries the incoming baton's identity fields
	// verbatim (objective drift is corruption, not evolution — baton.go); a first leg
	// seeds them from the config.
	relayID, leg := cfg.RelayID, 0
	objective, doneWhen := cfg.Objective, cfg.DoneWhen
	ledgerRef, heldRegion := cfg.LedgerRef, cfg.HeldRegion
	if !first {
		relayID, leg = cfg.Incoming.RelayID, cfg.Incoming.Leg+1
		objective, doneWhen = cfg.Incoming.Objective, cfg.Incoming.DoneWhen
		ledgerRef, heldRegion = cfg.Incoming.ProgressCursor.LedgerRef, cfg.Incoming.ProgressCursor.HeldRegion
	}
	if relayID == "" {
		return LegOutcome{}, fmt.Errorf("relay: driver needs a relay id (first leg: LegConfig.RelayID; successor: incoming baton relay_id)")
	}

	// 1. Reload+verify. Fail closed both ways: a stale baton yields no ledger read,
	// and the stale outcome itself is surfaced so the hook re-derives.
	o := Orientation{FirstLeg: first}
	if !first {
		o.Stale = CheckBatonStale(cfg.Incoming, cfg.Resolver)
		if !o.Stale.Stale && cfg.Ledger != nil {
			o.Progress = ReadVerifiedProgress(cfg.Incoming.ProgressCursor, cfg.Ledger)
		}
	}

	// 2. Done-check — idempotent restart: a relay that is done stays done. An
	// evaluator error never claims done; the leg keeps working.
	if cfg.DoneCheck != nil {
		if done, err := cfg.DoneCheck(doneWhen); err == nil && done {
			return LegOutcome{Reason: ReasonGoalDone, Orientation: o}, nil
		}
	}

	// 3. The work loop, bounded fail-closed.
	var af ArmFire
	var holds []string
	lastHold := ""
	for b := 0; b < cfg.MaxBoundaries; b++ {
		obs, err := cfg.Work(o, b)
		if err != nil {
			return LegOutcome{}, fmt.Errorf("relay: work hook at boundary %d: %w", b, err)
		}

		// Derive the three E1 axes from the boundary evidence, each by its own rung:
		// E2 (boundary), E3 (tree), E4 (next action).
		nv := ExtractNextAction(LegState{NextSteps: obs.NextSteps})
		sp := SafePoint{
			NoInFlightTurn:        !obs.TurnInFlight,
			TreeGreenOrParked:     obs.Tree.GreenOrParked(),
			NextActionExpressible: nv.Expressible,
		}
		// Pre-rotation assertions: the E2/E3 guards re-derive their axis from the raw
		// evidence and defer to the full conjunction; the F2 gate holds any boundary
		// still carrying transcript-only load-bearing state.
		inflight := GuardRotation(sp, obs.TurnInFlight)
		tree := TreeGate(sp, obs.Tree)
		gate := CheckExternalizeGate(obs.Facts)
		permitted := inflight.Permit && tree.Permit && gate.Admit
		if !permitted {
			switch {
			case inflight.Reason == ReasonInFlight:
				lastHold = ReasonInFlight
			case tree.Reason == ReasonTreeDirty:
				lastHold = ReasonTreeDirty
			case !nv.Expressible:
				lastHold = nv.Reason
			case !gate.Admit:
				lastHold = gate.Reason
			default:
				lastHold = ReasonNotAtSafePoint
			}
			holds = append(holds, lastHold)
		}

		// Arm/fire (G2) over the G3 trigger fold. The externalize gate is a
		// precondition of the safe point the machine may fire at (F2, fail-closed): a
		// boundary it refuses is presented as unsafe, so the machine can still ARM
		// there but can never FIRE through it. The E2/E3 refusals need no mask — they
		// are already false axes of sp.
		fireSP := sp
		if !gate.Admit {
			fireSP = SafePoint{}
		}
		crossed := cfg.Triggers.Crossed(obs.Usage.Context, obs.Usage.Turns, obs.Usage.Wall, obs.Usage.Spend)
		if af.Step(crossed, fireSP) != RotationFired {
			continue
		}

		// 4. Fired: drain the leg. The baton needs a git anchor the successor can
		// re-verify (reload.go fails closed on an empty start_sha), so a fired
		// boundary that observed none is an error, not a blind handoff.
		if obs.AtSHA == "" {
			return LegOutcome{}, fmt.Errorf("relay: rotation fired at boundary %d but the boundary observed no commit SHA; refusing to write an unverifiable baton", b)
		}
		dnr := NewDoNotRederiveIndex(cfg.Incoming.DoNotRederive)
		for _, p := range obs.DoNotRederive {
			dnr.Add(p)
		}
		baton := project(Baton{
			Schema:      Schema,
			RelayID:     relayID,
			Leg:         leg,
			ParentTrace: cfg.TraceID,
			Objective:   objective,
			DoneWhen:    doneWhen,
			ProgressCursor: ProgressCursor{
				StartSHA:   obs.AtSHA,
				LedgerRef:  ledgerRef,
				HeldRegion: heldRegion,
			},
			NextAction:    nv.NextAction,
			OpenQuestions: obs.OpenQuestions,
			Artifacts:     obs.Artifacts,
			DoNotRederive: dnr.Pointers(),
			Tombstone: Tombstone{
				Reason: af.Reason(),
				AtSHA:  obs.AtSHA,
				Note:   fmt.Sprintf("leg %d rotated at boundary %d", leg, b),
			},
		})
		wire, err := Marshal(baton)
		if err != nil {
			return LegOutcome{}, fmt.Errorf("relay: encode baton for leg %d: %w", leg, err)
		}
		if err := cfg.WriteBaton(wire); err != nil {
			return LegOutcome{}, fmt.Errorf("relay: write baton for leg %d: %w", leg, err)
		}
		successor, err := cfg.Recontinue(baton)
		if err != nil {
			return LegOutcome{}, fmt.Errorf("relay: recontinue after leg %d: %w", leg, err)
		}
		return LegOutcome{
			Reason:         af.Reason(),
			Orientation:    o,
			Baton:          baton,
			SuccessorTrace: successor,
			Boundaries:     b + 1,
			Holds:          holds,
		}, nil
	}
	return LegOutcome{}, fmt.Errorf(
		"relay: leg %d exhausted %d boundaries without reaching a rotation (state=%s, last hold: %s); the park path is rung H5",
		leg, cfg.MaxBoundaries, af.State(), lastHold)
}
