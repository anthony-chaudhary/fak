package main

// trajctl_nudge.go — issue #2765, the trajctl half of the out-of-band operator
// control epic (#2753): the AUTOMATIC-producer bridge from the regime-gated
// re-anchor nudge (internal/trajctl's DecideNudge, #2540) to the #2753 control bus
// (internal/sessionctl's redirect op, #2755). trajctl is tier-1 and CANNOT import
// sessionctl(3) — the same layering that keeps GatewaySteer's HTTP delivery injected
// from outside the pure fold — so this bridge lives at the top of the tree where both
// packages are importable.
//
// What it closes. The re-anchor rung already DECIDES (DecideNudge) and already
// DELIVERS onto the freeform steer channel (GatewaySteer, POST /v1/fak/session/{id}/
// steer). #2765 routes that same regime-gated decision onto the FIRST-CLASS control
// plane instead: a degrading curve emits a `redirect` op the loop consumes as a
// standing OBJECTIVE directive (WitnessDirective), not injection-shaped prose.
//
// The regime GATE is reused verbatim — DecideNudge is the single source of the
// "a HEALTHY curve is never nudged" rule (interventions harm high-scoring sessions,
// #2533). This bridge never re-derives the gate; it only routes the gate's nudge
// decisions onto the bus, so the healthy-suppression cannot drift out of agreement
// with the shipped rung.
//
// Op-semantics choice (the #2765 confusion risk). A nudge is a re-ANCHOR, not a goal
// change; `annotate` would fit a one-shot re-anchor better, but redirect is the op
// #2753 has SHIPPED. So the nudge uses redirect HONESTLY: it re-asserts the
// objective's OWN declared statement as the goal (re-establish the real objective as
// a fresh standing directive), never landing the transient re-anchor prose as the
// permanent goal — that would be the redirect-vs-annotate confusion #2765 warns of.
// The composed re-anchor packet stays on the ledgered SteerDecision row (the
// calibration child's evidence), out of the objective state.

import (
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// trajctlNudgeBus runs the trajctl regime gate over every OPEN objective in st and,
// for each objective the gate elects to nudge, emits a first-class redirect op onto
// the #2753 out-of-band control bus keyed by trace (the run's trace id the loop's
// applyRedirect drains at its next turn boundary). A HEALTHY — or episode-held, or
// never-declared — curve is left alone because DecideNudge returns ActionNone for it,
// so the #2533 non-intervention rule holds by construction. Every decision, nudged
// and suppressed alike, is returned for the caller to ledger (AppendSteerDecisions),
// mirroring trajctl.SteerSweep's contract: the no-action rows are what let the
// calibration child score the policy from evidence.
//
// A closed-reason refusal from the bus (a terminal current objective →
// REDIRECT_NO_REDIRECTABLE_STATE, an empty statement → REDIRECT_MALFORMED) is
// captured on the decision row (Delivered=false, DeliverErr) exactly as a failed
// steer delivery is, so the episode stays armed and the next boundary retries — the
// bus op costs the nudge, never the turn.
func trajctlNudgeBus(st trajctl.State, trace string, stamp trajctl.Stamp, nowMillis int64) []trajctl.SteerDecision {
	out := make([]trajctl.SteerDecision, 0)
	for _, oc := range st.OpenCurves().Objectives {
		d := st.DecideNudge(oc)
		d.UnixMillis = nowMillis
		d.SessionID = stamp.SessionID
		d.RunID = stamp.RunID
		if d.Action == trajctl.ActionNudge {
			switch {
			case trace == "":
				d.DeliverErr = "no session trace to redirect onto the control bus"
			default:
				// Re-anchor by re-asserting the objective's OWN statement as the
				// redirect goal — never the re-anchor packet (see the op-semantics
				// note above). The packet remains on d.Packet for the ledger.
				obj := st.Objectives[oc.ObjectiveID]
				if ref := sessionctl.EnqueueRedirect(trace, sessionctl.Redirect{
					ObjectiveID: obj.ID,
					Goal:        obj.Statement,
				}); ref != nil {
					d.DeliverErr = ref.Error()
				} else {
					d.Delivered = true
				}
			}
		}
		out = append(out, d)
	}
	return out
}
