package supervisoragent

import "time"

// Witnessed wraps a projected witness with an explicit presence bit. Present ==
// false is the "absent" marker: the witness could not be obtained at assembly
// time, and the decision layer must ESCALATE on it rather than infer around a
// missing signal (design-note fence #1: green absence is not a green witness).
// Value is only meaningful when Present.
type Witnessed[T any] struct {
	Present bool
	Value   T
}

// Seen wraps a value as a present witness.
func Seen[T any](v T) Witnessed[T] { return Witnessed[T]{Present: true, Value: v} }

// Absent returns a withheld witness of type T — the explicit missing marker a
// source surface leaves when it could not be read.
func Absent[T any]() Witnessed[T] { return Witnessed[T]{} }

// WorkerState is the supervisor-facing per-worker health class: the closed 5-token
// contract (healthy / done / dead / stale / blocked). It is a total projection of
// fleetmon.Classification (7 tokens) — the supervisor CONSUMES this token, it never
// re-adjudicates health from a transcript. The fold lives in workerStateFromClass
// (assemble.go) and is proven total by TestWorkerStateFold.
type WorkerState string

const (
	WorkerHealthy WorkerState = "healthy"
	WorkerDone    WorkerState = "done"
	WorkerDead    WorkerState = "dead"
	WorkerStale   WorkerState = "stale"
	WorkerBlocked WorkerState = "blocked"
)

// LaneRef is a payload-free reference to a lane: its name and concurrency kind. It
// carries no file content — only the structural identity of the lane.
type LaneRef struct {
	Lane string // the lane name
	Kind string // lane_kind: cluster / keyword / global
}

// Liveness is the supervisor's dos_status run digest, projected payload-free. It is
// the already-typed head the wake-time caller builds from the external dos_status
// A2A digest (there is no in-repo Go struct for the digest — it is the MCP kernel's
// shape): which run, its liveness-class token, its verified forward progress as a
// COMMIT COUNT (never a self-reported percentage), how long it has been silent, and
// the lease region it holds. It deliberately holds no free-text status line.
type Liveness struct {
	RunID     string        // the run / loop this digest keys on
	Class     string        // liveness-class token (e.g. "moving" / "stalled")
	Commits   int           // verified forward progress: commits since start (dos_status delta)
	SilentFor time.Duration // wall-clock since the last observed advance
	Region    []LaneRef     // the run's held-lease region
}

// WorkerVerdict is one worker's fleetmon health verdict, projected payload-free: its
// run/issue/lane identity and its health class. The classifier's prose —
// WorkerSample.Reasons / Blocker / ChildSummary — is EXCLUDED by construction. No
// transcript, no last message.
type WorkerVerdict struct {
	RunID string      // the run the worker serves
	Issue string      // the issue it was dispatched for
	Lane  string      // the lane it holds
	State WorkerState // the deterministic health class (folded from fleetmon)
}

// Escalation is one open escalation, projected to its typed head only. It is the
// already-typed payload-free head the wake-time caller builds from the escalation
// packet (the unified fak.escalation.v1 packet is unbuilt — issue #2271 — so today
// the caller adapts the current ledger/notify carrier into this head). It carries
// the routing fields a supervisor needs — which run/issue, the escalation class and
// severity, and the closed refusal-reason token — and NO free-text body: the
// supervisor routes on the typed head, never on prose.
type Escalation struct {
	ID         string // packet id
	RunID      string // the run it concerns
	Issue      string // the issue it concerns
	Class      string // escalation class token
	Severity   string // severity token: status / operator
	ReasonCode string // closed refusal-reason token (never prose)
}

// Lease is one row of the live lease table, projected payload-free: the lane, its
// concurrency kind, and the file-tree globs it covers. It mirrors the persisted
// arbiter view leaseref.ArbiterLease (lane / lane_kind / tree) exactly — no holder
// identity, description, or TTL rides along, and the tree is path patterns
// (structural), never file content.
type Lease struct {
	Lane string   // the lane name
	Kind string   // lane_kind (always "cluster" for a tree-scoped arbiter lease today)
	Tree []string // repo-relative globs the lease covers
}

// SupervisorInput is the closed projection a supervisor agent is handed. Every field
// is a typed, payload-free witness; there is no transcript, log body, or free-text
// field by construction. Each surface is Witnessed, so a source that could not be
// read is marked absent rather than silently empty — the difference between "no
// workers are running" (a present, empty list) and "the census could not be read"
// (an absent witness that must escalate).
type SupervisorInput struct {
	Liveness    Witnessed[Liveness]        // dos_status liveness/progress digest
	Workers     Witnessed[[]WorkerVerdict] // fleetmon per-worker health verdicts
	Escalations Witnessed[[]Escalation]    // open escalation packets (typed heads)
	Leases      Witnessed[[]Lease]         // the live lease table
}

// AbsentWitnesses names the surfaces whose witness could not be obtained, in a
// stable order. A non-empty result is the fence-#1 signal: the decision layer must
// escalate on it, never infer around a missing witness.
func (s SupervisorInput) AbsentWitnesses() []string {
	var out []string
	if !s.Liveness.Present {
		out = append(out, "liveness")
	}
	if !s.Workers.Present {
		out = append(out, "workers")
	}
	if !s.Escalations.Present {
		out = append(out, "escalations")
	}
	if !s.Leases.Present {
		out = append(out, "leases")
	}
	return out
}

// AnyAbsent reports whether any witness surface is absent — the one-bit "must
// escalate before deciding" gate for the decision layer.
func (s SupervisorInput) AnyAbsent() bool { return len(s.AbsentWitnesses()) > 0 }
