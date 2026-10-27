package supervisoragent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"
)

// This file is build-plan step S1 of the Chief-of-Staff runtime note
// (fak-private docs/notes/CONCEPT-CHIEF-OF-STAFF-AGENT-RUNTIME-2026-09-21.md,
// sections 6.1 and 6.2 D1; issue #13464): the addressed decision journal —
// the seat loop's durability contract. One payload-free DecisionRecord per
// wake makes a restart a replay rather than a re-decision, gives the keep-bit
// soak its durable time series, and is the observability source of truth.
//
// The record carries the input's DIGEST, never its contents: the assembled
// SupervisorInput stays out of the journal by construction, so the journal
// cannot become a transcript store. What it must NOT carry: a transcript, a
// log body, a free-text rationale, or a payload. TestDecisionRecordClosedJSONKeySet
// and TestDecisionRecordClosedFieldSet pin the closed field set exactly as
// input_test.go pins SupervisorInput.

// Outcome is the closed result token of one decided wake: what the lower
// stage did with the proposed verb. lowered: an admission verb ran and left
// its witnessed artifact. refused: the envelope held the verb below earned
// width under enforce — it was proposed for confirmation and nothing ran.
// escalated: the fail-closed path emitted the escalation packet. hold: the
// deliberate no-op — nothing ran, nothing was left.
type Outcome string

const (
	OutcomeLowered   Outcome = "lowered"
	OutcomeRefused   Outcome = "refused"
	OutcomeEscalated Outcome = "escalated"
	OutcomeHold      Outcome = "hold"
)

// DecisionRecord is one wake's addressed, payload-free decision. Every field
// is a closed token, a reference, or a value passed in by the caller: there
// is no transcript, log body, free-text rationale, or payload field by
// construction. Seq is assigned by the owning Journal (monotonic from 1);
// InputDigest digests the assembled SupervisorInput, never its contents;
// ArtifactRef names the witnessed artifact the lowered verb left (lease row /
// admit receipt / packet head) and is empty when none exists; DecidedAt is
// the wake-time stamp the caller passes in — this type never reads a clock.
type DecisionRecord struct {
	Seq             uint64          // monotonic sequence number, assigned by the Journal
	Topic           string          // the wake address this record was written for
	InputDigest     string          // hex digest of the assembled SupervisorInput (not its contents)
	ProposedVerb    ActionKind      // the SupervisorAction verb Decide proposed
	ReasonToken     Reason          // the closed reason token Decide returned
	EnvelopeVerdict EnvelopeVerdict // the envelope verdict on the proposed verb
	Outcome         Outcome         // lowered | refused | escalated | hold
	ArtifactRef     string          // witness ref of the lowered artifact; empty when none
	DecidedAt       time.Time       // the wake-time stamp the caller passed in
}

// Journal is the durable decision log the seat appends to: one addressed
// DecisionRecord per wake. Implementations own the sequence: Append stamps
// the record with the next monotonic Seq, so two writers can never mint the
// same number and a crash leaves a prefix that Replay resumes from a cursor.
type Journal interface {
	Append(DecisionRecord) error
	Replay(fromSeq uint64) ([]DecisionRecord, error)
	Last() (DecisionRecord, bool)
}

// MemoryJournal is the in-memory Journal: the golden behavior every durable
// implementation must match — seq monotonic from 1, replay stable (same
// cursor, same records, in order), last after append. It performs no I/O;
// the wake-time driver binds a durable backend behind the same interface.
type MemoryJournal struct {
	mu      sync.Mutex
	records []DecisionRecord
	nextSeq uint64
}

// NewMemoryJournal returns an empty journal whose first Append stamps Seq 1.
func NewMemoryJournal() *MemoryJournal { return &MemoryJournal{nextSeq: 1} }

// Append stamps r with the next monotonic Seq — the journal owns the
// sequence, so any Seq the caller set is overwritten — and stores a copy.
// The in-memory implementation always succeeds; the error slot exists so a
// durable backend can report a real failure without changing the interface.
func (j *MemoryJournal) Append(r DecisionRecord) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	r.Seq = j.nextSeq
	j.nextSeq++
	j.records = append(j.records, r)
	return nil
}

// Replay returns every record with Seq >= fromSeq in sequence order, as
// copies: mutating a returned record cannot reach the journal. A cursor past
// the tail yields an empty (non-nil) slice — the restart-resume shape.
func (j *MemoryJournal) Replay(fromSeq uint64) ([]DecisionRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]DecisionRecord, 0)
	for _, r := range j.records {
		if r.Seq >= fromSeq {
			out = append(out, r)
		}
	}
	return out, nil
}

// Last returns the most recently appended record and true, or the zero
// record and false for an empty journal.
func (j *MemoryJournal) Last() (DecisionRecord, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if len(j.records) == 0 {
		return DecisionRecord{}, false
	}
	return j.records[len(j.records)-1], true
}

// inputDigestFrame is the canonical encoding DigestInput hashes: one fixed
// field-order frame over the closed projection. Struct marshaling preserves
// declaration order, so the encoding is deterministic without sorting.
type inputDigestFrame struct {
	LivenessPresent    bool
	Liveness           Liveness
	WorkersPresent     bool
	Workers            []WorkerVerdict
	EscalationsPresent bool
	Escalations        []Escalation
	LeasesPresent      bool
	Leases             []Lease
}

// DigestInput renders the assembled SupervisorInput to the hex SHA-256 digest
// a DecisionRecord journals instead of the input itself. It is pure and
// deterministic: the same input always digests the same way, and flipping any
// surface between present and absent — or changing any projected value —
// changes the digest. Absence is part of the digest: a green absence and a
// green presence must not collide.
func DigestInput(in SupervisorInput) string {
	frame := inputDigestFrame{
		LivenessPresent:    in.Liveness.Present,
		Liveness:           in.Liveness.Value,
		WorkersPresent:     in.Workers.Present,
		Workers:            in.Workers.Value,
		EscalationsPresent: in.Escalations.Present,
		Escalations:        in.Escalations.Value,
		LeasesPresent:      in.Leases.Present,
		Leases:             in.Leases.Value,
	}
	b, err := json.Marshal(frame)
	if err != nil {
		// Unreachable: the frame is plain data. Fail closed anyway — a digest
		// that cannot be computed must not be guessed.
		panic("supervisoragent: canonical input frame will not marshal: " + err.Error())
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
