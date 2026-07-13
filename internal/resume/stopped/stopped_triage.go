package stopped

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// This file is the decenter-the-human fold at the resume-stopped triage seam. The
// DEFER bucket that Decide produces mixes two very different blocks: a session
// stopped on an auth/subscription wall genuinely needs a PERSON to re-auth before
// it can resume, but a session deferred behind an account or session THROTTLE is
// not a person's call at all — the wall clears on its own and the fleet simply
// waits behind it. Rendering both under one "operator, look at these" heading
// trains an operator to babysit throttles that would have cleared without them.
//
// PartitionDefer folds each deferred row's block reason through
// internal/choicetriage and splits the bucket: only a HUMAN_RESIDUAL disposition
// (an auth / credential / permission wall) waits on a person. It is the
// resume-seam analogue of operatorbrief.TriageHumanBucket, and it soaks behind a
// mode switch in the shell (FAK_RESUME_TRIAGE_GATE) so the render change can be
// observed before it changes what an operator is told to do.

// deferSignal projects a deferred row onto a choicetriage.Signal. The disposition
// (STOPPED_AUTH / STOPPED_LIMIT / …) is the surfaced question and BlockedBy is the
// "why"; both feed the authority test, so a row walled on auth/subscription
// triages to HUMAN_RESIDUAL while a throttle/limit row does not.
func deferSignal(r Row) choicetriage.Signal {
	return choicetriage.Signal{
		Source:   "resume",
		Question: string(r.Disp),
		Detail:   r.BlockedBy,
	}
}

// DeferNeedsHuman reports whether a deferred row genuinely needs a person to clear
// its wall (re-auth / re-subscribe), as opposed to a wall that clears on its own —
// an account or session throttle the fleet just waits behind, or a structural
// replay-safety block a clean-continuation reset owns. Only a HUMAN_RESIDUAL fold
// (a named auth/credential/permission wall) waits on a person.
func DeferNeedsHuman(r Row) bool {
	return choicetriage.Triage(deferSignal(r)).NeedsHuman
}

// PartitionDefer splits a Decisions' DEFER bucket into the rows that genuinely
// need a person (an auth/subscription wall) and the rows that clear on their own
// (a throttle/limit/structural wall the fleet waits behind). Pure and total; the
// input Decisions is not mutated, and order within each group is preserved.
func PartitionDefer(d Decisions) (needHuman, fleetWait []Row) {
	for _, r := range d.Defer {
		if DeferNeedsHuman(r) {
			needHuman = append(needHuman, r)
		} else {
			fleetWait = append(fleetWait, r)
		}
	}
	return needHuman, fleetWait
}

// TriageEnforced reports whether the decenter-the-human split is active for the
// given mode string. Only "enforce" flips the render; "warn", "" and anything else
// leave the single DEFER section unchanged so the change can soak. Mirrors
// operatorbrief's enforce/warn soak switch.
func TriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// TriageSelfcheck is the deterministic, no-I/O proof of the resume-seam fold: an
// auth/subscription wall needs a person, an account throttle and a session limit
// clear on their own, and PartitionDefer keeps only the auth wall on the human
// side. It mirrors operatorbrief.TriageSelfcheck at this seam.
func TriageSelfcheck() error {
	auth := Row{Disp: DispStoppedAuth, BlockedBy: "account auth/subscription disabled"}
	throttle := Row{Disp: DispStoppedMidtool, BlockedBy: "account throttled, resets 2026-07-09T12:00:00Z"}
	limit := Row{Disp: DispStoppedLimit, BlockedBy: "session limit, resets 2026-07-09T12:00:00Z"}

	if !DeferNeedsHuman(auth) {
		return fmt.Errorf("an auth/subscription wall must wait on a person, got a fleet-wait")
	}
	if DeferNeedsHuman(throttle) {
		return fmt.Errorf("an account throttle clears on its own; it must not wait on a person")
	}
	if DeferNeedsHuman(limit) {
		return fmt.Errorf("a session limit clears on its own; it must not wait on a person")
	}

	need, wait := PartitionDefer(Decisions{Defer: []Row{auth, throttle, limit}})
	if len(need) != 1 || need[0].Disp != DispStoppedAuth {
		return fmt.Errorf("want only the auth wall on the human side, got %+v", need)
	}
	if len(wait) != 2 {
		return fmt.Errorf("want the throttle and the limit as fleet-waits, got %d", len(wait))
	}
	if !TriageEnforced("enforce") || TriageEnforced("") || TriageEnforced("warn") {
		return fmt.Errorf("TriageEnforced must flip only on \"enforce\"")
	}
	return nil
}
