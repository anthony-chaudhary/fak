package dispatchtick

// preflight_velocity.go -- the rev-velocity collision-prior term: fold a lane's MEASURED
// revision velocity (revs/week, re-derived by internal/modver from the fak-module-versions/1
// ledger's DeltaRows) into COLLISION_RISK arbitration so a HOT lane -- one whose modules are
// moving fast -- advertises a RAISED collision prior (a hotter surface for a concurrent worker
// to land on) WITHOUT minting a new refusal. It matches the repo's warn-first advisory idiom
// (cf. the focus WIP-breadth term in preflight_focus.go and the host_churn cap term): the live
// arbitration output stays byte-identical until a lane actually clears the hot threshold, and
// even then the term only ADVISES -- it never holds, never refuses.
//
// WHY. COLLISION_RISK arbitration today is LEASE-based (`dos arbitrate`): it sees who holds a
// lane's file tree right now, not how fast that tree is moving. But a lane whose modules churn
// ~10 revs/week is a hotter collision surface than a dormant one even when no lease is held --
// two workers who both touch a fast-moving lane are likelier to actually conflict on the next
// merge. The fak-module-versions/1 ledger already records each module's rev count and its
// recent delta; this term reads that velocity per lane and RAISES the collision prior for the
// hot ones, as an advisory the arbiter can surface, never a hold. It reuses the existing
// COLLISION_RISK closed-vocabulary class as both the advisory token and the prior it raises,
// mirroring how the focus term reuses FOCUS_WIP_SATURATED -- so `dos man wedge COLLISION_RISK --explain`
// still binds it and no NEW refusal token enters the vocabulary.
//
// Pure: state in, decision out; no I/O, no git, no clock. The impure shell (cmd/fak) folds the
// modver ledger's per-module DeltaRows into a per-lane rev delta over a window and feeds
// RevDelta/Weeks/Threshold/Present in -- exactly the seam preflight_focus.go documents for the
// focusscore fold. The zero value (Present false) is a no-op, so a caller that wires nothing
// keeps today's arbitration byte-for-byte.

import "fmt"

// CollisionRisk is the closed-vocabulary class the velocity prior raises. It MUST stay
// byte-identical to the dos.toml [reasons.COLLISION_RISK] declaration so the token this term
// emits is the one `dos man wedge <TOKEN> --explain` verifies -- this term ADVISES on that class (raises its
// prior for a hot lane), it does not mint a new refusal.
const CollisionRisk = "COLLISION_RISK"

// DefaultHotRevsPerWeek is the velocity floor at/above which a lane is HOT: the issue's own
// calibration ("a module moving 10 revs/week is a hotter collision surface than a dormant one").
// Below it, ordinary movement is noise and the term abstains. The impure shell overlays
// FAK_HOT_REVS_PER_WEEK.
const DefaultHotRevsPerWeek = 10.0

// DefaultVelocityWindowWeeks is the trailing window a bare RevDelta is a rate over when the
// caller does not name one: one week, so a RevDelta summed over the last 7 days reads directly
// as revs/week. The shell passes a wider Weeks when it folds a longer ledger window.
const DefaultVelocityWindowWeeks = 1.0

// VelocityCheck carries the MEASURED per-lane rev movement the collision-prior term folds.
// RevDelta and Weeks come straight from a modver DeltaRows fold over the fak-module-versions/1
// ledger (never a worker self-report); Threshold is the tick's hot-floor override. The zero
// value (Present false) is a no-op.
type VelocityCheck struct {
	// Lane is the lane being arbitrated (e.g. "gateway", "modver").
	Lane string
	// RevDelta is the number of non-merge revs the lane's modules landed across the window --
	// the modver ledger's summed DeltaRows for the lane's module tree.
	RevDelta int
	// Weeks is the window width RevDelta is a rate over; <= 0 means DefaultVelocityWindowWeeks.
	Weeks float64
	// Threshold is the hot floor in revs/week; <= 0 means DefaultHotRevsPerWeek.
	Threshold float64
	// Present is whether the ledger folded >= 1 rev for this lane (a real signal to grade).
	// With no ledger row the term abstains -- no slander of a lane the ledger never saw.
	Present bool
}

// VelocityPrior is the closed advisory verdict of the rev-velocity collision-prior term.
// It NEVER carries a refusal verdict or an OK=false: a hot lane RAISES Prior and attaches the
// COLLISION_RISK token as an advisory the arbiter surfaces, it does not hold the launch.
type VelocityPrior struct {
	Lane        string  `json:"lane"`          // echoed lane
	RevsPerWeek float64 `json:"revs_per_week"` // derived velocity: RevDelta / Weeks (the signal)
	Threshold   float64 `json:"threshold"`     // hot floor graded against (revs/week)
	Hot         bool    `json:"hot"`           // Present && RevsPerWeek >= Threshold
	Prior       float64 `json:"prior"`         // raised collision prior: RevsPerWeek/Threshold when Hot, else 0
	Token       string  `json:"token"`         // COLLISION_RISK when Hot, else ""
	Reason      string  `json:"reason"`        // dos_check_reason-legible citation when Hot, else ""
}

// EvaluateVelocityPrior grades the rev-velocity collision-prior term. Pure: state in, decision
// out. It NEVER fires with no ledger signal (Present false) and NEVER fires below the hot floor
// (RevsPerWeek < Threshold) -- so a dormant, no-ledger, or sub-threshold lane is byte-identical
// to today (Prior 0, no token, no reason). A hot lane raises Prior in proportion to how far past
// the floor it moved (exactly at the floor -> 1.0, twice the floor -> 2.0), as an ADVISORY the
// arbiter adds to its lease-based collision prior -- this term never returns a hold.
func EvaluateVelocityPrior(v VelocityCheck) VelocityPrior {
	weeks := v.Weeks
	if weeks <= 0 {
		weeks = DefaultVelocityWindowWeeks
	}
	threshold := v.Threshold
	if threshold <= 0 {
		threshold = DefaultHotRevsPerWeek
	}
	p := VelocityPrior{Lane: v.Lane, Threshold: threshold}
	if v.Present && v.RevDelta > 0 {
		p.RevsPerWeek = float64(v.RevDelta) / weeks
	}
	p.Hot = v.Present && p.RevsPerWeek >= threshold
	if p.Hot {
		p.Prior = p.RevsPerWeek / threshold
		p.Token = CollisionRisk
		p.Reason = velocityHotReason(p)
	}
	return p
}

// velocityHotReason names the closed COLLISION_RISK class and cites the measured velocity (revs
// per week vs the hot floor) plus the raised prior, so a reader -- and `dos man wedge <TOKEN> --explain` -- can
// bind both the class and its evidence. It is explicit that this is advisory, not a hold.
func velocityHotReason(p VelocityPrior) string {
	return fmt.Sprintf("%s: lane %q is moving %.1f revs/week (>= hot floor %.1f) -- a hotter collision surface than a dormant lane; raising the advisory collision prior to %.2f for concurrent arbitration. Advisory only: still safe to launch, but prefer a disjoint lane or wait for the lease if a peer is already here.",
		CollisionRisk, p.Lane, p.RevsPerWeek, p.Threshold, p.Prior)
}

// Map renders the velocity prior as the arbitration/tick-payload block. It is attached to the
// arbitration output ONLY when Hot is true (Token non-empty), so a dormant / no-ledger lane
// stays byte-identical; when attached it lets `dos arbitrate` / `fak dispatch status` surface
// the rev-velocity signal distinctly from the lease-based collision hold and the focus/rate
// holds.
func (p VelocityPrior) Map() map[string]any {
	return map[string]any{
		"lane":          p.Lane,
		"token":         p.Token,
		"hot":           p.Hot,
		"revs_per_week": p.RevsPerWeek,
		"threshold":     p.Threshold,
		"prior":         p.Prior,
		"reason":        p.Reason,
	}
}
