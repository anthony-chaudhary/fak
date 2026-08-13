package dispatchtick

// preflight_focus.go -- the focus WIP-breadth backpressure term: fold the MEASURED
// fleet breadth (the count of ACTIVE trajectory-control objectives vs the pinned WIP
// cap, re-derived by internal/focusscore from the docs/nightrun/trajctl.jsonl ledger)
// into the spawn decision so a fleet already at/over its work-in-progress ceiling WARNS
// (default) or HOLDS opening YET ANOTHER concurrent objective, instead of fanning out
// ever broader while nothing converges. It matches the repo's warn-first idiom (cf. the
// normgate warn-first redaction and the advisory supersession rungs).
//
// This term is DELIBERATELY DISTINCT from the rate_budget term (preflight_ratelimit.go)
// and the gate-health term. Those are cap terms: they lower the effective spawn cap. The
// focus term does NEITHER -- it does not touch the cap and it does not throttle work that
// CONTINUES an already-open objective. It throttles ONLY the act of OPENING a NEW
// objective while breadth is saturated. A worker resuming an issue the fleet already
// declared an open objective for is NEVER held: convergence needs those to finish, and
// starving them would defeat the whole point (bounded breadth, not a hard cap).
//
// Posture is WARN by default (advise + still spawn) so the live fleet stays
// byte-identical to today until an operator opts into HOLD via --focus-hold /
// FLEET_DISPATCH_FOCUS_HOLD. The advisory carries the closed FOCUS_WIP_SATURATED refusal
// token so `dos man wedge <TOKEN> --explain` can verify it and a loop can route/report on it distinctly
// from a rate-limit or collision hold.
//
// Pure: state in, decision out; no I/O. The impure shell (cmd/fak) reads the focusscore
// fold + the trajctl ledger and feeds Active/WIPCap/Present/NewObjective in.

import "fmt"

// FocusWIPSaturated is the closed-vocabulary refusal token the focus backpressure term
// emits on a new-objective spawn while breadth is at/over the WIP cap. It MUST stay
// byte-identical to the dos.toml [reasons.FOCUS_WIP_SATURATED] declaration so the token
// this fold emits is the one `dos man wedge <TOKEN> --explain` verifies and a loop routes on. It also
// doubles as the refusal VERDICT the tick reports under the HOLD posture (mirroring how
// the collision term reuses COLLISION_RISK as both token and verdict).
const FocusWIPSaturated = "FOCUS_WIP_SATURATED"

// Focus posture tokens. WARN advises but still spawns (the default); HOLD skips opening
// the new objective while over cap. Continuation of an already-open objective is never
// affected by either posture.
const (
	FocusPostureWarn = "warn"
	FocusPostureHold = "hold"
)

// FocusCheck carries the MEASURED breadth signal the focus term folds. Active, WIPCap and
// Present come straight from the focusscore fold over the trajctl ledger (never a worker
// self-report); NewObjective and Hold are the tick's per-candidate + posture inputs. The
// zero value (WIPCap 0 / Present false) is a no-op: a caller that wires nothing keeps
// today's behavior.
type FocusCheck struct {
	// Active is the count of ACTIVE (concurrently live) objectives -- the breadth measure.
	Active int
	// WIPCap is the pinned work-in-progress ceiling; <= 0 disables the term (no-op).
	WIPCap int
	// Present is whether the trajctl ledger folded >= 1 objective (there is a real signal
	// to grade). With no ledger the term abstains -- no slander of an empty fleet.
	Present bool
	// NewObjective is whether THIS candidate OPENS a new concurrent objective (true) vs
	// CONTINUES one already open in the ledger (false). A continuation is never advised or
	// held: the term only bounds fan-out, not convergence work.
	NewObjective bool
	// Hold is the posture: true HOLDs a new-objective spawn while over cap; false WARNs
	// only (still spawns). Default is WARN.
	Hold bool
}

// FocusAdmission is the closed verdict of the focus WIP-breadth term.
type FocusAdmission struct {
	Active    int    `json:"active"`     // echoed breadth (active objective count)
	WIPCap    int    `json:"wip_cap"`    // echoed pinned cap
	ExcessWIP int    `json:"excess_wip"` // max(0, Active-WIPCap): objectives beyond the cap
	Saturated bool   `json:"saturated"`  // Present && WIPCap>0 && Active>=WIPCap (at/over cap)
	Advise    bool   `json:"advise"`     // Saturated && NewObjective: advisory fires this candidate
	Hold      bool   `json:"held"`       // Advise && posture==HOLD: skip opening the new objective
	Posture   string `json:"posture"`    // "warn" | "hold": the posture graded under
	Token     string `json:"token"`      // FOCUS_WIP_SATURATED when Advise, else ""
	Reason    string `json:"reason"`     // dos_check_reason-legible citation when Advise, else ""
}

// EvaluateFocusAdmission grades the focus WIP-breadth term. Pure: state in, decision out.
// It NEVER fires on a continuation (NewObjective false), NEVER fires below the cap
// (Active < WIPCap), and NEVER fires with no ledger signal (Present false) -- so an
// under-cap, no-ledger, or continuation tick is byte-identical to today (no advisory
// attached, no hold), and continuing an already-open objective is never blocked.
func EvaluateFocusAdmission(f FocusCheck) FocusAdmission {
	a := FocusAdmission{Active: f.Active, WIPCap: f.WIPCap, Posture: FocusPostureWarn}
	if f.Hold {
		a.Posture = FocusPostureHold
	}
	if f.WIPCap > 0 && f.Active > f.WIPCap {
		a.ExcessWIP = f.Active - f.WIPCap
	}
	// At/over the cap is the saturation predicate (Active >= WIPCap), matching the DoD:
	// the fleet is already declaring as many live goals as its ceiling allows, so opening
	// one MORE is fan-out. ExcessWIP (the > case) is surfaced as extra detail.
	a.Saturated = f.Present && f.WIPCap > 0 && f.Active >= f.WIPCap
	a.Advise = a.Saturated && f.NewObjective
	if a.Advise {
		a.Hold = f.Hold
		a.Token = FocusWIPSaturated
		a.Reason = focusSaturatedReason(a)
	}
	return a
}

// focusSaturatedReason names the closed FOCUS_WIP_SATURATED token and cites the measured
// breadth (active vs cap) plus the posture, so a reader -- and `dos man wedge <TOKEN> --explain` -- can
// bind both the refusal class and its evidence.
func focusSaturatedReason(a FocusAdmission) string {
	verb, tail := "WARNING", "still spawning (warn-first, bounded breadth not a hard cap) -- set --focus-hold / FLEET_DISPATCH_FOCUS_HOLD=1 to hold new-objective spawns while over cap"
	if a.Hold {
		verb, tail = "HOLDING", "not opening a new objective this tick -- pause or meet an active objective to get back under the cap, or continue an already-open objective (continuations are never held)"
	}
	return fmt.Sprintf("%s: %s a NEW objective while fleet breadth is at/over the WIP cap (%d active objective(s) vs cap %d, %d over) -- %s",
		FocusWIPSaturated, verb, a.Active, a.WIPCap, a.ExcessWIP, tail)
}

// Map renders the focus admission as the tick-payload/status block. It is attached to the
// dispatch payload ONLY when Advise is true, so an under-cap / continuation tick stays
// byte-identical; when attached it lets `fak dispatch status` and the tick JSON surface a
// focus hold/warn distinctly from the rate-limit and collision holds.
func (a FocusAdmission) Map() map[string]any {
	return map[string]any{
		"token":      a.Token,
		"posture":    a.Posture,
		"held":       a.Hold,
		"active":     a.Active,
		"wip_cap":    a.WIPCap,
		"excess_wip": a.ExcessWIP,
		"saturated":  a.Saturated,
		"reason":     a.Reason,
	}
}
