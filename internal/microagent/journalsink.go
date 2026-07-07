package microagent

import "github.com/anthony-chaudhary/fak/internal/journal"

// JournalSink is the host's ONE shared audit sink backed by the durable,
// hash-chained decision journal (internal/journal) — the in-process answer to
// #2011 (epic #2000 M11). Instead of one per-process JSONL per guarded agent,
// the whole host holds ONE mutex-guarded, tamper-evident chain and every hosted
// agent appends its lifecycle rows to it (O(1) files per host). Each row is
// tagged with the agent id (its session TraceID). Because the SAME journal is
// typically the kernel's registered ABI emitter, a QUARANTINE the shared gateway
// raises lands in the SAME chain with its witness + call_seq intact
// (journal.Row) — mixing lifecycle rows in drops no adjudication forensics.
//
// Record is called under the host locks for per-agent ordering; it delegates to
// the journal, which serializes its own appends under its own mutex (a distinct
// lock, always taken after the host's — no inversion), and never calls back into
// the Host. A nil sink or nil journal makes Record a no-op, so a host that was
// not configured with a durable journal degrades to silence rather than panics.
type JournalSink struct{ J *journal.Journal }

// NewJournalSink wraps a host-scoped journal as an AuditSink. The caller owns the
// journal's lifecycle (Open/Close); the sink only appends.
func NewJournalSink(j *journal.Journal) *JournalSink { return &JournalSink{J: j} }

// Record appends one lifecycle Event to the shared chain as an agent-audit row,
// mapping the host EventKind to the journal's closed KindAgent* vocabulary. An
// unmapped kind is dropped rather than written under an unknown Kind.
func (s *JournalSink) Record(ev Event) {
	if s == nil || s.J == nil {
		return
	}
	kind := journalKind(ev.Kind)
	if kind == "" {
		return
	}
	s.J.AppendAgentEvent(kind, ev.Agent, ev.Err)
}

// journalKind maps a host lifecycle EventKind to the journal's closed agent-audit
// Kind vocabulary, returning "" for a kind with no durable row.
func journalKind(k EventKind) string {
	switch k {
	case EventSpawn:
		return journal.KindAgentSpawn
	case EventReject:
		return journal.KindAgentReject
	case EventDone:
		return journal.KindAgentDone
	case EventCancel:
		return journal.KindAgentCancel
	case EventError:
		return journal.KindAgentError
	}
	return ""
}

// JournalSink implements AuditSink.
var _ AuditSink = (*JournalSink)(nil)
