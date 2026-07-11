package main

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardrotate"
)

// guard_account_autorehome.go — the MINIMAL SPINE for the automatic "go" push: doing on a
// background tick what an operator does by hand with `fak accounts rehome` + prompting "go"
// (see docs/notes/ACCOUNT-REHOME-INSTANT-SWITCH-2026-07-09.md).
//
// Today a session only leaves a near-capped seat two ways: it 403/429s mid-turn and
// accountFailover.failover heals REACTIVELY, or a human notices, runs `fak accounts rehome`,
// and types "go" to flush the staged swap onto the wire. The gap is the middle: when the active
// seat is *about to* wall (headroom already TierWalled, or a still-active daily cap flagged
// usage_soon), nothing moves until the wall actually hits or a human intervenes.
//
// This file is the decision seam that closes that gap, kept PURE and DORMANT:
//   - proactiveWantsSwitch is a pure, table-testable predicate over the active seat's condition.
//   - proactiveRehomeTick is the thin adapter that, when the predicate says go, reuses the
//     EXISTING forceRehome (so target selection, the moved-set exclusion, and the typed
//     no-target reasons are shared, never re-implemented).
//   - proactiveSignalsForSeat is the "read the active seat's status" CONSUMER the rehome design
//     note named as the one missing input: it projects the annotated fleet roster + the rotation
//     headroom map (the same fold rotationHeadroom already builds) down to the ONE seat the
//     session is on, yielding exactly the (headroom, usage_soon) pair the predicate branches on.
//   - proactiveTickFromRoster composes the two: resolve the active seat's signals, then drive the
//     proven proactiveRehomeTick. It performs no I/O — the caller passes roster + headroom in — so
//     the whole roster->decision->swap path is testable end to end without a live fleet.
//
// It is deliberately NOT wired into any live loop yet: nothing in guard.go calls
// proactiveTickFromRoster (or proactiveRehomeTick), so live behavior is byte-identical. The
// signal-source gap the earlier note flagged is now closed at the read layer; the only step left
// is the live driver — a guard-side tick cadence + an enable flag that loads the roster/headroom
// and calls proactiveTickFromRoster. The spine and its signal reader are proven first; the loop
// that drives them lands on top.

// proactiveReason is the CLOSED vocabulary for a proactive-rehome tick's decision — mirroring
// failoverNoTargetReason's discipline: the tick explains itself once, at the point of decision,
// with a token a surface can render and a test can assert. A "go_*" reason means the predicate
// wants a switch (the tick then attempts forceRehome); a "hold_*"/disabled reason means it does
// not, and no roster is touched.
type proactiveReason string

const (
	// ProactiveDisabled — the feature is off; the tick never touches the roster.
	ProactiveDisabled proactiveReason = "disabled"
	// ProactiveHoldRoom — the active seat still has room (TierOfferable/TierUnknown, no
	// usage_soon flag); there is nothing to pre-empt.
	ProactiveHoldRoom proactiveReason = "hold_room"
	// ProactiveGoWalled — the active seat's headroom is already TierWalled; switch now rather
	// than wait for the next turn to 403/429.
	ProactiveGoWalled proactiveReason = "go_walled"
	// ProactiveGoUsageSoon — the active seat carries a still-active daily cap a fresh probe
	// reopened over (usage_soon); switch before it re-walls mid-turn.
	ProactiveGoUsageSoon proactiveReason = "go_usage_soon"
)

// wantsSwitch reports whether this reason is one the tick acts on (a "go_*" reason).
func (r proactiveReason) wantsSwitch() bool {
	return r == ProactiveGoWalled || r == ProactiveGoUsageSoon
}

// proactiveWantsSwitch is the PURE decision core: given whether the feature is enabled, the
// active seat's headroom score (nil => no runtime signal), and its usage_soon advisory flag,
// decide whether the tick should pre-emptively switch seats — and say why with a closed reason.
// It knows nothing about the roster or whether a target exists; that is forceRehome's job. A
// walled active seat is the strongest signal (it will hard-fail on the next turn), so it wins
// over the softer usage_soon advisory.
func proactiveWantsSwitch(enabled bool, activeHeadroom *float64, usageSoon bool) proactiveReason {
	if !enabled {
		return ProactiveDisabled
	}
	if activeHeadroom != nil && accounts.Classify(*activeHeadroom) == accounts.TierWalled {
		return ProactiveGoWalled
	}
	if usageSoon {
		return ProactiveGoUsageSoon
	}
	return ProactiveHoldRoom
}

// proactiveRehomeOutcome is one tick's result: the decision reason, whether a swap was actually
// applied, the seat-identity metadata when it was, and — when the predicate wanted a switch but
// forceRehome refused — the shared typed no-target reason so a surface can name the same fix the
// operator path names. It carries no token.
type proactiveRehomeOutcome struct {
	Reason   proactiveReason        // why the tick did (or did not) act
	Acted    bool                   // true iff a seat swap was applied
	Rehome   gateway.AccountRehome  // from->to identity when Acted; zero otherwise
	NoTarget failoverNoTargetReason // set when the predicate wanted a switch but no sibling qualified
	Err      error                  // forceRehome's refusal error when NoTarget is set
}

// proactiveRehomeTick is the thin, DORMANT adapter behind the automatic push. It runs the pure
// predicate and, only when it wants a switch, delegates to the existing forceRehome so the target
// pick, the operator-moved exclusion, and the typed no-target reasons are reused verbatim rather
// than re-implemented. A refusal (no sibling qualifies) is reported, not raised: a near-cap seat
// with nowhere to go simply holds until failover or a reset, exactly as the operator path does.
//
// Nothing calls this yet — it is the seam a future tick loop will drive. Keeping the call site
// out of guard.go for now means the spine changes no live behavior.
func (a *accountFailover) proactiveRehomeTick(enabled bool, activeHeadroom *float64, usageSoon bool) proactiveRehomeOutcome {
	reason := proactiveWantsSwitch(enabled, activeHeadroom, usageSoon)
	if !reason.wantsSwitch() {
		return proactiveRehomeOutcome{Reason: reason}
	}
	res, err := a.forceRehome(string(reason))
	if err != nil {
		return proactiveRehomeOutcome{Reason: reason, NoTarget: a.lastNoTargetReason(), Err: err}
	}
	return proactiveRehomeOutcome{Reason: reason, Acted: true, Rehome: res}
}

// proactiveSignalsForSeat resolves the two live inputs proactiveRehomeTick consumes for the
// ACTIVE seat — its per-bucket headroom score (nil => no runtime signal) and its usage_soon
// advisory — out of the annotated fleet roster and the rotation-headroom map the launch/next path
// already builds (rotationHeadroom -> headroomFromRoster). It is the CONSUMER the rehome design
// note names as the one missing piece (docs/notes/ACCOUNT-REHOME-INSTANT-SWITCH-2026-07-09.md,
// "no new signal source is required, only a consumer"): a reader that projects the fleet surfaces
// down to the single seat the session is sitting on. Pure over its inputs — the shell supplies the
// roster and headroom — so the projection is table-testable without a live fleet.
//
// The active dir is matched to its roster row by NORMALIZED path (guardrotate.NormalizeDir — the
// same Clean + platform case/separator fold the launch-time rotation matches seats with), then:
//   - headroom: the row's account bucket key (accounts.UUIDBucketKey over its AccountUUID) indexes
//     hr; a PRESENT score is returned by pointer so a walled (<0) or offerable (>0) bucket reads
//     through accounts.Classify exactly as the launch path sees it, while an ABSENT bucket returns
//     nil — the honest "no runtime signal" state, distinct from a 0 (TierUnknown) score.
//   - usageSoon: true iff the row carries a non-empty UsageSoonReset (a still-active daily cap a
//     fresh probe reopened over — the seat is Available but flagged about to re-wall).
//
// A dir that names no claude roster row (a hermetic test home, an undiscovered seat) yields
// (nil, false): no signal, so the predicate holds — fail-safe, never a spurious switch.
func proactiveSignalsForSeat(activeDir string, roster []fleetaccounts.Account, hr accounts.RotationHeadroom) (*float64, bool) {
	want := guardrotate.NormalizeDir(activeDir)
	if want == "" {
		return nil, false
	}
	for i := range roster {
		r := roster[i]
		if r.Product != "claude" || guardrotate.NormalizeDir(r.Dir) != want {
			continue
		}
		var headroom *float64
		if r.AccountUUID != nil && *r.AccountUUID != "" {
			if score, ok := hr[accounts.UUIDBucketKey(*r.AccountUUID)]; ok {
				s := score
				headroom = &s
			}
		}
		usageSoon := r.UsageSoonReset != nil && strings.TrimSpace(*r.UsageSoonReset) != ""
		return headroom, usageSoon
	}
	return nil, false
}

// proactiveTickFromRoster is the DORMANT composition a future guard tick will call: it resolves
// the active seat's live signals off the supplied roster + headroom map (proactiveSignalsForSeat
// over its own sticky currentConfigDir) and drives the proven proactiveRehomeTick with them. It
// does no I/O of its own — the caller loads the roster and headroom (the same fleetaccounts fold
// rotationHeadroom builds) and passes them in — so the full roster->decision->swap path is
// exercisable end to end without a live fleet.
//
// Nothing calls this yet. It is the one-line seam the guard loop plugs into once the tick cadence
// and the enable flag land; keeping the call site out of guard.go means the foundation still
// changes zero live behavior.
func (a *accountFailover) proactiveTickFromRoster(enabled bool, roster []fleetaccounts.Account, hr accounts.RotationHeadroom) proactiveRehomeOutcome {
	headroom, usageSoon := proactiveSignalsForSeat(a.currentConfigDir(), roster, hr)
	return a.proactiveRehomeTick(enabled, headroom, usageSoon)
}
