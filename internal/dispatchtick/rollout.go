package dispatchtick

import (
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/rolloutmode"
)

// ---------------------------------------------------------------------------
// THE ROLLOUT GUARD — the safety gate between "tier-aware routing is computed"
// and "tier-aware routing changes a live worker choice" (#3047, C10).
// ---------------------------------------------------------------------------
//
// This is the ROLLOUT node of the model-tier working path. Everything upstream
// already exists and is DEAD-SAFE (nothing routes through it): C5
// (RouteAccountForTier) decides which account a tier would pick; C7
// (Authorize/Envelope) bounds a cheap orchestrator's authority; C6
// (superloop.Evaluate) grades whether a cheaper model can do routine meta work.
// C10 answers the last, load-bearing question: given all of that, WHEN may a
// cheaper tier actually replace the live selection, and how does it back out?
//
//	tier route (C5)  ->  ROLLOUT GUARD (THIS)  ->  live worker selection
//
// THE WORKING PATH IS shadow -> canary -> on, and the guard makes each step
// refuse to become the next until parity is witnessed:
//
//   - OFF (this leaf's default): the live path is untouched. No would-choose is
//     computed, nothing is applied. This proves "default behavior is unchanged".
//   - SHADOW: compute would_choose_tier BESIDE the current selection and report
//     the delta. It NEVER switches a worker — Applied is false by construction, so
//     the readout can prove the shadow fields without launching anything.
//   - CANARY: apply the cheaper tier route, but ONLY to low-risk routine (T2)
//     watchdog/meta work, and roll back automatically on a quality or refusal
//     regression. Any non-routine class stays on the live selection untouched.
//   - ON (broad default-on routing): OUT OF SCOPE for this leaf. The guard
//     REFUSES it — a rollout guard that would silently promote to default-on is no
//     guard at all. Promotion past canary needs a separate witness.
//
// TWO CONFUSION RISKS #3047 names, made structural:
//
//   - "Do not count cheaper launches as success unless outcome parity holds." A
//     cheaper canary route is reported as RolloutReasonCanaryApplied — a
//     candidate saving PENDING PARITY — never as a win. The QualitySignal is what
//     realizes or revokes it; a cheaper tier alone never books a success here.
//   - "Do not expand canary from T2 meta work to implementation work without a
//     separate witness." Canary scope is exactly modelroute.ClassRoutine. Any
//     ClassNormalImpl / ClassUltraHard / ClassSecurityRelease work is refused
//     canary application and left on the live selection — no score, no cheaper
//     price, and no pass rate widens that scope.
//
// PURITY & THE NUMBERING TRAP. A pure fold — no I/O, no launch, no git — so the
// dry-run readout runs anywhere. Tiers are compared as ACCOUNT ModelTier ints
// (1 = frontier/most capable .. 3 = cheapest); "cheaper" is a HIGHER number.
// Work-tier comparisons still route through modelroute so the T0<T1<T2 inversion
// (C3's trap) can never leak in.

// RolloutMode is this leaf's name for the repo-wide closed rollout vocabulary
// (internal/rolloutmode) — the semantics below are where that ladder was first
// written out, and #6090 lifted the type so three leaves stop spelling the same
// rungs three ways. An unrecognized mode fails closed here (treated as no
// application) rather than silently routing; the shared parser leaves that
// direction to the caller precisely so this guard can choose it.
type RolloutMode = rolloutmode.Mode

const (
	// RolloutOff is this leaf's default: the live selection is untouched and no
	// would-choose is computed. The gen/next posture — gated until promotion.
	RolloutOff = rolloutmode.Off
	// RolloutShadow computes would_choose_tier beside the current selection and
	// reports the delta. It never switches a worker.
	RolloutShadow = rolloutmode.Shadow
	// RolloutCanary applies the cheaper tier route to routine (T2) meta work only,
	// rolling back on a quality/refusal regression.
	RolloutCanary = rolloutmode.Canary
	// RolloutOn is the shared ladder's APPLIED rung (broad default-on routing) —
	// OUT OF SCOPE for this leaf and NOT reachable here. Spelling it the shared way
	// does not implement it: EvaluateRollout answers RolloutOn with
	// RolloutReasonOnOutOfScope and applies nothing, exactly as it answered the old
	// private `default` spelling. Promotion past canary needs a separate witness.
	RolloutOn = rolloutmode.On
)

// Closed-vocabulary reason strings — a status/dry-run surface renders these
// verbatim, so every rollout verdict is explainable without free text.
const (
	RolloutReasonOffCurrentSelection = "rollout-off-live-selection-unchanged"
	RolloutReasonShadowObserveOnly   = "shadow-would-choose-observed-not-applied"
	RolloutReasonCanaryOutOfScope    = "canary-out-of-scope-non-routine-class-unchanged"
	RolloutReasonCanaryApplied       = "canary-cheaper-tier-applied-pending-parity"
	RolloutReasonCanaryNoChange      = "canary-would-choose-matches-current-no-change"
	RolloutReasonCanaryNotCheaper    = "canary-would-choose-not-cheaper-live-selection-kept"
	RolloutReasonCanaryRouteRefused  = "canary-tier-route-refused-live-selection-kept"
	RolloutReasonCanaryRolledBack    = "canary-rolled-back-on-quality-or-refusal-regression"
	RolloutReasonOnOutOfScope        = "default-on-routing-out-of-scope-needs-promotion-witness"
	RolloutReasonUnknownMode         = "unknown-rollout-mode-refused-fail-closed"
)

// Delta directions between the current and would-choose ACCOUNT tiers. "cheaper"
// is a higher ModelTier number (less capable, less costly); "more-capable" is a
// lower number. These are rendered verbatim in the shadow readout.
const (
	DeltaSame        = "same"
	DeltaCheaper     = "cheaper"
	DeltaMoreCapable = "more-capable"
	DeltaRefused     = "refused"    // the tier route could not choose an account
	DeltaNoCurrent   = "no-current" // no live selection tier was supplied to compare against
)

// QualitySignal carries the observed regression bits the canary rollback watches.
// Both default to false, so a caller that measures nothing gets the conservative
// "no regression observed" — but note that is NOT the same as "parity witnessed":
// the guard only ever APPLIES a cheaper route as pending-parity, it never books a
// success from the absence of a regression.
type QualitySignal struct {
	// RefusalRegression is set when a cheaper tier dropped or papered over a
	// refusal/DENY reason a faithful run would have preserved (the C6 honesty trap).
	RefusalRegression bool `json:"refusal_regression"`
	// QualityRegression is set when outcome parity failed — the cheaper launch did
	// not match the reference outcome (the "don't count cheaper as success" trap).
	QualityRegression bool `json:"quality_regression"`
	// Note is optional free text surfaced to a human, never parsed.
	Note string `json:"note,omitempty"`
}

// Regressed reports whether any regression bit is set — the rollback trigger.
func (s QualitySignal) Regressed() bool {
	return s.RefusalRegression || s.QualityRegression
}

// RolloutInput is one rollout decision's inputs: the mode, the work class (which
// fixes the canary scope), the issue's tier metadata and account pool (which the
// C5 route consumes to compute would_choose), the live selection tier to compare
// against, and the observed quality signal.
type RolloutInput struct {
	Mode        RolloutMode
	Class       modelroute.WorkClass
	Issue       IssueTier
	Rows        []AccountRow
	Product     string
	CurrentTier int // live-selected ModelTier (1..3); 0 = unknown/no current selection
	Signal      QualitySignal
}

// RolloutDecision is the guard's verdict, carrying every field a dry-run/status
// surface needs to explain the rollout without free text. Applied and AppliedTier
// are the load-bearing pair: Applied answers "did tier-aware routing change the
// live selection?" and AppliedTier is the ModelTier in effect AFTER the rollout —
// which equals CurrentTier in every mode except an applied canary.
type RolloutDecision struct {
	Mode            RolloutMode          `json:"mode"`
	ModeValid       bool                 `json:"mode_valid"`
	Class           modelroute.WorkClass `json:"class"`
	InCanaryScope   bool                 `json:"in_canary_scope"`
	CurrentTier     int                  `json:"current_model_tier"`
	WouldChooseTier int                  `json:"would_choose_model_tier"` // 0 when the route refused
	WouldChooseWork modelroute.WorkTier  `json:"would_choose_work_tier"`
	Differs         bool                 `json:"differs"`
	Delta           string               `json:"delta"`
	Applied         bool                 `json:"applied"`
	AppliedTier     int                  `json:"applied_model_tier"`
	RolledBack      bool                 `json:"rolled_back"`
	Reason          string               `json:"reason"`
	Route           TierRouteResult      `json:"route"`
}

// CanaryScopeClass is the ONLY work class canary may apply to: routine, low-trust
// watchdog/meta work (floor T2). Naming it once keeps the scope single-sourced
// with the #3047 assumption and the C6 read-only eval.
const CanaryScopeClass = modelroute.ClassRoutine

// EvaluateRollout is the rollout guard: a pure fold from a mode + the tier-route
// inputs to a decision that NEVER changes the live selection except in an applied
// canary on in-scope routine work with no regression. It is the same verdict for
// every caller, so a cheap model cannot talk the guard past its scope.
func EvaluateRollout(in RolloutInput) RolloutDecision {
	d := RolloutDecision{
		Mode:          in.Mode,
		ModeValid:     in.Mode.Valid(),
		Class:         in.Class,
		InCanaryScope: in.Class == CanaryScopeClass,
		CurrentTier:   in.CurrentTier,
		// Default posture: the live selection stands. Every early return below that
		// does NOT apply leaves AppliedTier at the current selection.
		AppliedTier: in.CurrentTier,
		Delta:       DeltaNoCurrent,
	}

	if !in.Mode.Valid() {
		d.Reason = RolloutReasonUnknownMode
		return d
	}

	switch in.Mode {
	case RolloutOff:
		// The default: nothing computed, nothing applied — the live path is exactly
		// what it was before the model-tier chain existed.
		d.Reason = RolloutReasonOffCurrentSelection
		return d

	case RolloutOn:
		// The rollout guard's refusal: broad default-on routing is out of scope for
		// this leaf. Named and visible, never a silent promotion.
		d.Reason = RolloutReasonOnOutOfScope
		return d
	}

	// SHADOW and CANARY both need the would-choose route. Compute it once.
	route := RouteAccountForTier(in.Rows, in.Product, in.Issue)
	d.Route = route
	if route.OK {
		d.WouldChooseTier = route.ChosenModelTier
		d.WouldChooseWork = route.ChosenTier
	}
	d.Delta = deltaDirection(in.CurrentTier, route)
	d.Differs = route.OK && in.CurrentTier > 0 && route.ChosenModelTier != in.CurrentTier

	if in.Mode == RolloutShadow {
		// Observe-only: report the delta, switch nothing. Applied stays false and
		// AppliedTier stays at the current selection BY CONSTRUCTION.
		d.Reason = RolloutReasonShadowObserveOnly
		return d
	}

	// --- CANARY ---
	if !d.InCanaryScope {
		// The scope wall: non-routine work is never canaried, however cheap the
		// route looks. Live selection kept.
		d.Reason = RolloutReasonCanaryOutOfScope
		return d
	}
	if in.Signal.Regressed() {
		// Automatic rollback: a quality or refusal regression revokes the canary and
		// restores the live selection.
		d.RolledBack = true
		d.Reason = RolloutReasonCanaryRolledBack
		return d
	}
	if !route.OK {
		// Fail closed: the tier route refused (no account meets the floor). Keep the
		// live selection rather than launch on nothing.
		d.Reason = RolloutReasonCanaryRouteRefused
		return d
	}
	switch d.Delta {
	case DeltaCheaper:
		// Apply the cheaper route — but this is a candidate saving PENDING PARITY,
		// not a booked success. The QualitySignal is what confirms or rolls it back.
		d.Applied = true
		d.AppliedTier = route.ChosenModelTier
		d.Reason = RolloutReasonCanaryApplied
	case DeltaSame:
		d.Reason = RolloutReasonCanaryNoChange
	default:
		// more-capable, or no current tier to compare: canary only ever moves routine
		// work DOWN in cost; it never upgrades. Keep the live selection.
		d.Reason = RolloutReasonCanaryNotCheaper
	}
	return d
}

// deltaDirection classifies the would-choose account tier against the live
// selection. A LOWER ModelTier number is more capable (and costlier); a HIGHER
// number is cheaper — so would > current is a saving.
func deltaDirection(currentTier int, route TierRouteResult) string {
	if !route.OK {
		return DeltaRefused
	}
	if currentTier <= 0 {
		return DeltaNoCurrent
	}
	switch {
	case route.ChosenModelTier == currentTier:
		return DeltaSame
	case route.ChosenModelTier > currentTier:
		return DeltaCheaper
	default:
		return DeltaMoreCapable
	}
}

// ---------------------------------------------------------------------------
// THE DRY-RUN SHADOW READOUT — fold many rollout inputs into one operator report
// of current-vs-would_choose deltas, WITHOUT launching a worker. This is the
// #3047 acceptance artifact: a status dry-run that proves the shadow fields.
// ---------------------------------------------------------------------------

// ShadowReportSchema is the versioned tag a serialized shadow readout carries, so
// a forked or older report shape fails loud rather than silently mis-reading.
const ShadowReportSchema = "fak.dispatch-rollout-shadow.v1"

// ShadowItem names one unit of work to shadow: an id plus the same tier-route
// inputs a live dispatch would carry. The report reads each in shadow mode.
type ShadowItem struct {
	ID          string
	Class       modelroute.WorkClass
	Issue       IssueTier
	Rows        []AccountRow
	Product     string
	CurrentTier int
}

// ShadowRow is one item's shadow verdict in the readout: the current vs
// would-choose tiers and their delta, plus the closed reason. Applied is always
// false in a shadow readout — the field is carried so a reader can SEE it never
// flips, which is the proof that the dry-run launched nothing.
type ShadowRow struct {
	ID              string               `json:"id"`
	Class           modelroute.WorkClass `json:"class"`
	CurrentTier     int                  `json:"current_model_tier"`
	WouldChooseTier int                  `json:"would_choose_model_tier"`
	WouldChooseWork modelroute.WorkTier  `json:"would_choose_work_tier"`
	Differs         bool                 `json:"differs"`
	Delta           string               `json:"delta"`
	InCanaryScope   bool                 `json:"in_canary_scope"`
	Applied         bool                 `json:"applied"`
	Reason          string               `json:"reason"`
}

// ShadowReport is the folded dry-run readout: one row per item plus the delta
// tally an operator reads to decide whether canary is safe to enter. Cheaper is
// the count of routine (canary-eligible) items where a cheaper tier would serve —
// the candidate savings, never a claim they succeeded.
type ShadowReport struct {
	Schema         string      `json:"schema"`
	Mode           RolloutMode `json:"mode"`
	Items          int         `json:"items"`
	Same           int         `json:"same"`
	Cheaper        int         `json:"cheaper"`
	MoreCapable    int         `json:"more_capable"`
	Refused        int         `json:"refused"`
	NoCurrent      int         `json:"no_current"`
	CanaryEligible int         `json:"canary_eligible"` // routine items with a cheaper route — the canary candidates
	AnyApplied     bool        `json:"any_applied"`     // MUST stay false: a shadow readout launches nothing
	Rows           []ShadowRow `json:"rows"`
}

// FoldShadowReport reads every item in SHADOW mode and folds the deltas into the
// dry-run readout. It is a pure, deterministic fold — same items, same bytes — so
// the acceptance gate (shadow fields proven without launching a worker) runs
// anywhere. AnyApplied can only ever be false; if it were true the fold would be
// lying, which is why the field is reported rather than assumed.
func FoldShadowReport(items []ShadowItem) ShadowReport {
	rep := ShadowReport{Schema: ShadowReportSchema, Mode: RolloutShadow, Items: len(items)}
	for _, it := range items {
		d := EvaluateRollout(RolloutInput{
			Mode:        RolloutShadow,
			Class:       it.Class,
			Issue:       it.Issue,
			Rows:        it.Rows,
			Product:     it.Product,
			CurrentTier: it.CurrentTier,
		})
		row := ShadowRow{
			ID:              it.ID,
			Class:           it.Class,
			CurrentTier:     d.CurrentTier,
			WouldChooseTier: d.WouldChooseTier,
			WouldChooseWork: d.WouldChooseWork,
			Differs:         d.Differs,
			Delta:           d.Delta,
			InCanaryScope:   d.InCanaryScope,
			Applied:         d.Applied,
			Reason:          d.Reason,
		}
		rep.Rows = append(rep.Rows, row)
		if d.Applied {
			rep.AnyApplied = true
		}
		switch d.Delta {
		case DeltaSame:
			rep.Same++
		case DeltaCheaper:
			rep.Cheaper++
			if d.InCanaryScope {
				rep.CanaryEligible++
			}
		case DeltaMoreCapable:
			rep.MoreCapable++
		case DeltaRefused:
			rep.Refused++
		default:
			rep.NoCurrent++
		}
	}
	return rep
}
