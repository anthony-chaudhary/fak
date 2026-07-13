package supervisoragent

import (
	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// WorkerCensus is one raw per-worker row as the fleetmon census emits it: the
// worker's run/issue/lane identity plus the deterministic health CLASS token
// (fleetmon.Classification — a closed string enum). It is the closed OUTPUT of the
// health rung; the supervisor consumes the token, it never re-derives health from
// a transcript. By construction the row carries no transcript, last-message, or
// other payload field — only the identity fields and the class token.
type WorkerCensus struct {
	RunID string
	Issue string
	Lane  string
	Class fleetmon.Classification
}

// Sources is the set of upstream witness surfaces the assembler projects from,
// each wrapped in a Witnessed so a surface that could not be read is carried as
// absent (Present == false) rather than as a silently-empty present value — the
// fence-#1 "green absence is not a green witness" distinction.
//
// The two surfaces with a real upstream Go type are taken in that type and
// PROJECTED to the closed contract: the fleetmon census row (WorkerCensus, whose
// class token maps to a WorkerState) and the persisted lease table's own
// payload-free view (leaseref.ArbiterLease → Lease). The two whose canonical form
// is an external/wire digest — the dos_status liveness digest and the
// fak.escalation.v1 packet — are taken as their already-typed, payload-free heads
// at the source boundary (the wake-time caller builds those heads from the MCP
// digest / packet; this package does not re-open a transcript to derive them).
type Sources struct {
	Liveness    Witnessed[Liveness]
	Workers     Witnessed[[]WorkerCensus]
	Escalations Witnessed[[]Escalation]
	Leases      Witnessed[[]leaseref.ArbiterLease]
}

// Assemble projects the upstream witness surfaces into the closed SupervisorInput a
// supervisor agent consumes. It performs NO I/O and reads NO transcript: it maps
// each PRESENT surface to its payload-free projection and preserves an ABSENT
// surface as an absent witness (the fence-#1 "escalate, don't infer" marker). The
// mapping is total and lossy-by-design — it copies only the closed-contract fields,
// so a payload field on a source can never reach the agent even if one were added
// upstream.
func Assemble(src Sources) SupervisorInput {
	return SupervisorInput{
		Liveness:    src.Liveness, // already the payload-free digest head
		Workers:     projectSlice(src.Workers, projectWorker),
		Escalations: src.Escalations, // already the typed packet heads
		Leases:      projectSlice(src.Leases, projectLease),
	}
}

// projectSlice lifts a per-element projection over a Witnessed slice, preserving
// absence: an absent source stays absent (never a silently-empty present slice),
// and a present source (even an empty one) stays present.
func projectSlice[A, B any](w Witnessed[[]A], f func(A) B) Witnessed[[]B] {
	if !w.Present {
		return Absent[[]B]()
	}
	out := make([]B, 0, len(w.Value))
	for _, a := range w.Value {
		out = append(out, f(a))
	}
	return Seen(out)
}

// projectLease maps the persisted lease table's payload-free view onto the closed
// contract row: lane, lane_kind, and the leased tree globs. No holder, description,
// TTL, or generation field is copied — leaseref.ArbiterLease does not carry them,
// and the contract has no field for them.
func projectLease(l leaseref.ArbiterLease) Lease {
	return Lease{Lane: l.Lane, Kind: l.LaneKind, Tree: l.Tree}
}

// projectWorker maps one fleetmon census row onto the closed verdict: the identity
// fields copy through and the deterministic health class token is folded to the
// contract's WorkerState. No transcript-derived field exists on either side.
func projectWorker(w WorkerCensus) WorkerVerdict {
	return WorkerVerdict{RunID: w.RunID, Issue: w.Issue, Lane: w.Lane, State: workerStateFromClass(w.Class)}
}

// workerStateFromClass folds fleetmon's seven-way health classification onto the
// contract's five-token WorkerState. The fold is intentionally fail-safe: any
// class that is not a clean healthy/done/dead/stale maps to WorkerBlocked
// ("needs attention"), so ambiguous or future-added evidence is never silently
// read as healthy. The supervisor consumes this token; it never re-adjudicates it.
func workerStateFromClass(c fleetmon.Classification) WorkerState {
	switch c {
	case fleetmon.ClassHealthy:
		return WorkerHealthy
	case fleetmon.ClassCompletedFinal:
		return WorkerDone
	case fleetmon.ClassDead:
		return WorkerDead
	case fleetmon.ClassStaleTranscript, fleetmon.ClassStaleChild:
		return WorkerStale
	case fleetmon.ClassAuthRateBlocked, fleetmon.ClassAttention:
		return WorkerBlocked
	default:
		return WorkerBlocked // unknown/ambiguous evidence => needs attention, never healthy
	}
}
