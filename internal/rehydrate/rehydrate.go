// Package rehydrate is the horizon-gated re-entry orchestrator: the staged gate a resumed
// agent passes through BEFORE its first post-wake action, running strictly more
// revalidation the longer it slept. It is the CRaC `afterRestore` analog for fak's session
// image (epic #1178, Phase-2 spine, #1181) — the one place that turns the dormancy band
// (internal/dormancy, #1179) into "which rungs must clear before I am allowed to act."
//
// # The gap it closes
//
// internal/sessionimage.Rehydrate restores the drive/recall/trajectory and consults a
// keep-bit witness before re-firing an irreversible effect (the ACRFence distinction), but
// it runs the SAME path for a 5-minute and a 3-month restore: there is no staged
// revalidation keyed to how long the image was dormant. A credential that expired, a lease
// that was reaped and re-granted, a recalled memory whose SHA history rewrote, a prompt
// cache aged past its TTL, a plan the trunk moved past — none is re-checked as a function
// of the gap. This leaf is that staged gate.
//
// # What it does and does NOT do (it COMPOSES rungs, it never implements them)
//
// This is the SPINE. It owns two things: the staging POLICY (which rung runs at which
// dormancy band) and the COMPOSITION (run the applicable rungs in band order, refuse
// admission at the first that does not clear, with a closed reason). It implements NONE of
// the rungs' actual checks — the credential refresh (#1183), the lease fence (#1182), the
// recall revalidation (#1184), the cold-cache projection (#1186), and the plan-freshness
// check are injected as Rungs. A Rung supplies its CHECK; the orchestrator owns WHEN it
// fires. So this keystone lands first and the four children wire their real checks into it
// without re-deriving the ladder.
//
// # The staging ladder (strictly more rungs as the gap grows)
//
// Each reason fires at the dormancy band at which the thing it guards first decays — the
// bands internal/dormancy already documents and anchors to the resume cache TTLs:
//
//	warm    < ~5m   nothing decayed — resume verbatim ......................... 0 rungs
//	cool    < ~1h   prompt cache cold, KV gone ............................... COLD_CACHE
//	cold    < ~24h  + credentials expiring, recalled memory aging ........... + STALE_CRED, STALE_RECALL
//	frozen  < ~30d  + lease re-granted, plan stale .......................... + STALE_LEASE, STALE_PLAN
//	ancient >= ~30d + model/world changed ................................... (all of the above)
//
// so the number of rungs that must clear before the first action is monotonic in the gap
// (0, 1, 3, 5, 5): a longer sleep is strictly more revalidation, never less. The closed
// vocabulary stops at five today; the ancient-only "model/world changed" rung is a future
// increment, so ancient currently runs the same five as frozen (a full revalidation).
//
// # Honest fences
//
//   - Closed refusal vocabulary. A rung refuses with exactly one of {COLD_CACHE, STALE_CRED,
//     STALE_RECALL, STALE_LEASE, STALE_PLAN} — routed like the governor vocabulary, never
//     free-text. A rung whose reason is outside the set is dropped at gate construction
//     (the orchestrator never runs an unrecognized rung).
//   - Fail-closed staging. CanonicalFiresAt of an unknown reason is Ancient (run it at
//     every non-warm wake, never assume it is harmless), and Admit short-circuits on the
//     first refusal — admission is a single non-forgeable verdict, not a best-effort
//     "mostly cleared."
//   - Pure orchestration. The framework reads no clock and does no I/O; the dormancy band
//     is supplied by the caller (from internal/dormancy), and all I/O lives inside the
//     injected rungs' Check. A Gate holds no per-call state, so Admit is safe to call
//     concurrently.
//
// This is a tier-1 foundation leaf: it imports only dormancy(1) + stdlib, registers
// nothing, and is off the hot path. sessionimage.Rehydrate composes a Gate at its boundary.
package rehydrate

import (
	"context"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
)

// Reason is the closed refusal vocabulary a rehydration rung refuses with — one token per
// rung, routed like the governor vocabulary (never free-text). The empty Reason is "no
// refusal": a cleared rung, or an admitted gate.
type Reason string

const (
	// ReasonOK is the empty reason — a rung cleared, or a gate admitted.
	ReasonOK Reason = ""
	// ColdCache: the prompt cache aged past its TTL while dormant, so the planner must stop
	// assuming warm-cache latency/price and plan a re-warm (#1186, wires internal/resume's
	// cold projection + cachemeta's COLD_TTL). Fires from the Cool band.
	ColdCache Reason = "COLD_CACHE"
	// StaleCred: an OAuth/token credential may have expired during the gap — refresh before
	// the first upstream request rather than failing mid-request (#1183). Fires from Cold.
	StaleCred Reason = "STALE_CRED"
	// StaleRecall: a recalled memory's named artifact (a SHA, an import, a path) may have
	// been invalidated by history moving on — re-verify via dos_recall before injecting it
	// as confirmed, never hand a stale memory through as fact (#1184). Fires from Cold.
	StaleRecall Reason = "STALE_RECALL"
	// StaleLease: the lease may have been reaped and re-granted while dormant — fence the
	// generation and halt-and-reacquire before any write (#1182, on leaseref's fencing
	// token). Fires from Frozen.
	StaleLease Reason = "STALE_LEASE"
	// StalePlan: the plan/trunk may have moved on during a long sleep — re-check freshness
	// before resuming the planned work rather than acting on a stale plan. Fires from Frozen.
	StalePlan Reason = "STALE_PLAN"
)

// canonicalFiresAt is the staging policy: the minimum dormancy band at which each rung must
// run before admission. It is the orchestrator's load-bearing contribution — the children
// supply the checks, this map decides when each fires. Keyed only by the closed vocabulary;
// membership here IS the definition of a known reason.
var canonicalFiresAt = map[Reason]dormancy.Horizon{
	ColdCache:   dormancy.Cool,
	StaleCred:   dormancy.Cold,
	StaleRecall: dormancy.Cold,
	StaleLease:  dormancy.Frozen,
	StalePlan:   dormancy.Frozen,
}

// Known reports whether r is in the closed rung vocabulary. ReasonOK is the cleared/admitted
// marker, not a rung reason, so Known(ReasonOK) is false.
func (r Reason) Known() bool { _, ok := canonicalFiresAt[r]; return ok }

// CanonicalFiresAt is the minimum dormancy band at which a rung with this reason runs in the
// staged ladder. An unknown reason fails closed to Ancient (run it at every non-warm wake) —
// never silently treated as harmless. A child builds its rung with NewRung and inherits this
// band so the staging policy stays in one place.
func CanonicalFiresAt(r Reason) dormancy.Horizon {
	if h, ok := canonicalFiresAt[r]; ok {
		return h
	}
	return dormancy.Ancient
}

// Verdict is one rung's outcome. The zero Verdict (ReasonOK) is "cleared"; a non-empty
// Reason is a refusal carrying that closed token plus an optional human Detail.
type Verdict struct {
	Reason Reason
	Detail string
}

// Cleared reports whether the rung passed (returned no refusal).
func (v Verdict) Cleared() bool { return v.Reason == ReasonOK }

// Clear is the verdict a rung returns when its revalidation passed.
func Clear() Verdict { return Verdict{} }

// Refuse is the verdict a rung returns when its revalidation failed, carrying the rung's
// closed reason and a human-readable detail (which the gate routes verbatim).
func Refuse(reason Reason, detail string) Verdict { return Verdict{Reason: reason, Detail: detail} }

// Rung is one independently-witnessed revalidation check in the staged gate. Reason is its
// identity (the closed token it refuses with); FiresAt is the minimum dormancy band at which
// the orchestrator runs it; Check performs the revalidation and returns a Verdict. The four
// children of #1181 each supply a Rung — this package never implements Check.
type Rung interface {
	Reason() Reason
	FiresAt() dormancy.Horizon
	Check(ctx context.Context) Verdict
}

// rung is the default Rung built from a closure by NewRung / NewRungAt.
type rung struct {
	reason  Reason
	firesAt dormancy.Horizon
	check   func(context.Context) Verdict
}

func (r rung) Reason() Reason            { return r.reason }
func (r rung) FiresAt() dormancy.Horizon { return r.firesAt }

func (r rung) Check(ctx context.Context) Verdict {
	if r.check == nil {
		return Clear() // a rung with no check is a no-op that always clears
	}
	return r.check(ctx)
}

// NewRung builds a Rung that fires at the CANONICAL band for its reason, so a child supplies
// only its reason and its check and the orchestrator owns when the rung runs. This is the
// ordinary way the four children compose into the gate.
func NewRung(reason Reason, check func(context.Context) Verdict) Rung {
	return rung{reason: reason, firesAt: CanonicalFiresAt(reason), check: check}
}

// NewRungAt builds a Rung that fires at an explicit band, overriding the canonical staging
// policy — for a rung whose decay scale genuinely differs, or for a deterministic test.
func NewRungAt(reason Reason, firesAt dormancy.Horizon, check func(context.Context) Verdict) Rung {
	return rung{reason: reason, firesAt: firesAt, check: check}
}

// RungRecord is the per-rung witness of one Admit pass: which rung ran, the band it fired
// at, whether it cleared, and any detail. The orchestrator records every rung it ran (the
// "each rung is independently witnessed" requirement), so a refused admission names not only
// the blocker but the full ladder attempted up to it.
type RungRecord struct {
	Reason  Reason
	FiredAt dormancy.Horizon
	Cleared bool
	Detail  string
}

// Admission is the orchestrator's verdict for one wake. Horizon is the dormancy band the
// gate ran for; Ran is the ordered per-rung witness (every applicable rung up to and
// including any refuser); Admitted is true iff every applicable rung cleared; RefusedBy is
// the refusing rung's reason (ReasonOK when admitted); Detail is that rung's detail.
type Admission struct {
	Horizon   dormancy.Horizon
	Ran       []RungRecord
	Admitted  bool
	RefusedBy Reason
	Detail    string
}

// RanReasons returns the reasons of the rungs that ran, in ladder order — the witness of
// "which rungs this wake had to clear."
func (a Admission) RanReasons() []Reason {
	out := make([]Reason, 0, len(a.Ran))
	for _, r := range a.Ran {
		out = append(out, r.Reason)
	}
	return out
}

// Gate is the staged re-entry orchestrator: an ordered, immutable set of Rungs. Admit runs
// exactly the rungs whose FiresAt band the wake's dormancy band reaches, in band-ascending
// order, refusing admission at the first that does not clear. A Gate holds no per-call
// state, so Admit is safe to call concurrently.
type Gate struct {
	rungs []Rung
}

// NewGate builds a Gate from the given rungs, sorted into ladder (band-ascending) order with
// a stable tie-break by reason so Admit is deterministic. A nil rung, or a rung whose reason
// is outside the closed vocabulary, is dropped (fail-closed: the orchestrator never runs an
// unrecognized rung). Reasons are expected to be unique; a duplicate simply runs twice.
func NewGate(rungs ...Rung) *Gate {
	kept := make([]Rung, 0, len(rungs))
	for _, r := range rungs {
		if r == nil || !r.Reason().Known() {
			continue
		}
		kept = append(kept, r)
	}
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].FiresAt() != kept[j].FiresAt() {
			return kept[i].FiresAt() < kept[j].FiresAt()
		}
		return kept[i].Reason() < kept[j].Reason()
	})
	return &Gate{rungs: kept}
}

// Admit runs the staged gate for a wake whose dormancy band is h: every rung whose FiresAt
// band h reaches runs, in ladder order, and the first that does not clear refuses admission
// with its closed reason (a short-circuit — a later rung may assume an earlier one cleared,
// so a stale lease halts before a recall check). A warm wake (h below every rung's band)
// runs zero rungs and is admitted unconditionally — today's verbatim resume.
func (g *Gate) Admit(ctx context.Context, h dormancy.Horizon) Admission {
	adm := Admission{Horizon: h, Admitted: true}
	if g == nil {
		return adm
	}
	for _, r := range g.rungs {
		if !h.AtLeast(r.FiresAt()) {
			continue // this band has not decayed enough for this rung to be required
		}
		v := r.Check(ctx)
		adm.Ran = append(adm.Ran, RungRecord{
			Reason:  r.Reason(),
			FiredAt: r.FiresAt(),
			Cleared: v.Cleared(),
			Detail:  v.Detail,
		})
		if !v.Cleared() {
			adm.Admitted = false
			adm.RefusedBy = r.Reason() // authoritative: the rung's declared closed token
			adm.Detail = v.Detail
			return adm
		}
	}
	return adm
}
