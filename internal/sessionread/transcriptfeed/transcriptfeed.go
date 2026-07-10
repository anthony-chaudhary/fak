// Package transcriptfeed is the THIRD cursor-CDC feed of the session read/query/observe
// plane (epic #4176, child C5 / #4196) — the transcript-event tail a peer or monitor
// subscribes to and re-attaches to by cursor.
//
// Two CDC feeds already ship on the SAME discipline:
//
//   - GET /v1/fak/changes — the vDSO coherence bus: typed write Mutations + Revocations,
//     "what an OTHER agent changed or refuted" (internal/gateway/coherence.go).
//   - GET /v1/fak/events — the durable, hash-chained audit journal: the attested tail of
//     adjudicated syscalls (#3171).
//
// Both carry COHERENCE + AUDIT rows — not transcript CONTENT. There is no way for a peer
// to *tail* a session's turn / tool-call / decision stream and be woken as it happens.
// This feed closes that gap: it is the A2A `tasks/resubscribe` / AG-UI `STATE_DELTA`
// analog over transcript events — `turn-open` / `turn-close` / `tool-terminal` / `decision`
// — filtered, taint-safe, and cursor-resumable.
//
// # Why re-implement the coherence discipline instead of importing it
//
// The coherence feed lives in internal/gateway, which this package MUST NOT import (it is
// a leaf of the read plane, session- and gateway-blind by construction — its only edges
// are stdlib, internal/resume/transcript, and internal/sessionread). So the cursor-CDC
// discipline that internal/gateway/coherence.go and internal/gateway/session_changes.go
// share is REIMPLEMENTED here, deliberately, to the same three invariants:
//
//  1. Feed-local monotone Seq minted at append (THE CURSOR). Like the sibling sessionFeed,
//     which mints its own Seq because a per-source revision is not globally monotone, a
//     transcript record's UUID/timestamp is not a monotone cross-session cursor — so the
//     feed assigns its own Seq under a mutex at Append.
//  2. A bounded ring (oldest-at-front, cap ~1024). A never-draining consumer cannot grow
//     it without limit; the oldest events fall off and a lapsed consumer sees a Seq gap
//     and re-syncs to head.
//  3. drain(principal, sinceSeq) returns every retained ev.Seq > sinceSeq that is
//     visibleTo(principal), PLUS the highest retained Seq as the next cursor.
//
// # The cursor-over-ALL-retained invariant (why principal-scoped re-attach is gap-free)
//
// Drain advances the returned cursor over EVERY retained event — not just the visible /
// emitted ones. This is the load-bearing invariant: a principal-scoped consumer's cursor
// stays monotone and never re-scans another principal's already-elapsed Seqs. Without it,
// a tenant that filters out a peer's events would keep re-reading the retained window
// looking for Seqs it will never be shown. With it, "no gap / no duplicate" holds even
// under principal scoping — the consumer's next `?since=<cursor>` starts strictly after
// the whole retained tail it has now observed.
//
// # The taint screen (why this feed is an exfiltration-safe surface)
//
// An event derived from a quarantined (sealed / tombstoned) transcript span must NEVER
// carry that span's raw bytes onto the feed — otherwise the observe plane becomes an
// exfiltration channel around the read-plane taint gate (C1). The screen is applied at
// INGRESS, in EventsFromRecords: a quarantined record is emitted as a bounded STUB —
// the Kind (structural metadata) is preserved, but the content (tool name + text summary)
// is withheld behind the read-plane taint marker sessionread.ReasonReadTaintWithheld.
//
// Withholding rather than DROPPING the event is deliberate: the stub still mints a Seq,
// so the cursor space stays continuous and a consumer learns "a span existed here but was
// withheld" without ever seeing its bytes — the cursor-over-all-retained invariant and
// the taint screen reinforce each other.
//
// # The five-habit consumer contract
//
// Consumers follow the same contract documented for the change feed
// (docs/integrations/consuming-the-fak-changes-feed.md): resume-from-cursor
// (since=0 = everything retained), at-least-once dedupe by Seq, retention-gap => re-sync
// (a first returned Seq > since+1 means the window lapsed), fail-before-advance, and
// principal-scoped visibility. This package is a PRODUCER of transcript events, not the
// coherence bus — but a peer drains it identically.
package transcriptfeed

import (
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// defaultFeedCap bounds the retained transcript-event window, matching the coherence and
// session-change feeds so all three CDC surfaces share one retention footprint.
const defaultFeedCap = 1024

// maxSummary bounds a redacted event descriptor. An event NEVER carries full transcript
// text — only a short, whitespace-collapsed prefix — so even a non-quarantined event is a
// bounded descriptor, not a payload dump.
const maxSummary = 120

// Event kinds. A record maps to exactly one: a user turn OPENS, an assistant plain turn
// CLOSES, an assistant tool choice with no paired result yet is a DECISION, and a paired
// tool_use+tool_result is a TERMINAL tool observation.
const (
	KindTurnOpen     = "turn-open"
	KindTurnClose    = "turn-close"
	KindToolTerminal = "tool-terminal"
	KindDecision     = "decision"
)

// TranscriptEvent is one wire-facing transcript-feed entry: a redacted descriptor of one
// transcript record, ordered by the feed's monotone Seq (the drain cursor).
//
// It carries a bounded descriptor only — Kind, the record UUID, an optional tool name, and
// a short text Summary — never the record's full text. A quarantined record is emitted with
// Withheld=true and Reason set, and carries NO Tool/Summary bytes at all (the taint stub).
type TranscriptEvent struct {
	Kind     string `json:"kind"`               // one of the Kind* constants
	Seq      uint64 `json:"seq"`                // feed-local monotone sequence (the drain cursor)
	UUID     string `json:"uuid,omitempty"`     // the source record's uuid (an id, not span content)
	Tool     string `json:"tool,omitempty"`     // tool name for a decision / tool-terminal (withheld if tainted)
	Summary  string `json:"summary,omitempty"`  // bounded, redacted text prefix (withheld if tainted)
	Withheld bool   `json:"withheld,omitempty"` // taint stub: the span's content was screened out
	Reason   string `json:"reason,omitempty"`   // sessionread taint marker when Withheld
	// principal is the isolation principal that owns this event (unexported: an internal
	// routing key, NOT wire-facing — emitting it would re-leak the very tenant identity the
	// scoped drain exists to hide). "" for a global / principal-less event. Drain uses it to
	// scope a tenant's feed to its own events; see Drain and visibleTo.
	principal string
}

// Feed is a bounded ring of TranscriptEvents — the observer's sliding window onto a
// session's transcript stream. A consumer drains it by cursor. Bounded so a never-draining
// consumer cannot grow it without limit: the oldest events fall off, and a consumer that
// lapses past the retained window sees a Seq gap and re-syncs to head.
type Feed struct {
	mu   sync.Mutex
	ring []TranscriptEvent // chronological; oldest at front
	cap  int
	seq  uint64 // monotone; assigned to each event at Append
}

// NewFeed builds a transcript-event ring. capacity<=0 uses defaultFeedCap (shared with the
// coherence and session-change feeds).
func NewFeed(capacity int) *Feed {
	if capacity <= 0 {
		capacity = defaultFeedCap
	}
	return &Feed{cap: capacity}
}

// Append records one transcript event, MINTING the next monotone feed Seq under the mutex
// and trimming the oldest event once the ring is full. The caller's ev.Seq is ignored and
// overwritten — the feed is the sole authority for the cursor, exactly like the coherence
// bus Seq — so no producer can forge a non-monotone cursor.
func (f *Feed) Append(ev TranscriptEvent) {
	f.mu.Lock()
	f.seq++
	ev.Seq = f.seq
	f.ring = appendRingCapped(f.ring, ev, f.cap)
	f.mu.Unlock()
}

// Drain returns every retained event with Seq > sinceSeq (sinceSeq==0 => all retained)
// that is VISIBLE to the requesting principal, plus the highest retained Seq now known
// (the consumer's next cursor).
//
// Drain is a READ: it never mutates the ring or the Seq counter, so observing a session
// can never advance its loop (the done-condition's side-effect-free requirement). Two
// drains from the same cursor return identical results.
//
// The cursor advances over ALL retained events — not just the visible ones — so a
// principal-scoped consumer's next cursor stays monotone and it never re-scans another
// principal's already-elapsed Seqs. This is what makes principal-scoped re-attach gap-free
// AND duplicate-free: the next `?since=<cursor>` starts strictly after the whole retained
// tail already observed.
func (f *Feed) Drain(principal string, sinceSeq uint64) ([]TranscriptEvent, uint64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]TranscriptEvent, 0, len(f.ring))
	cursor := sinceSeq
	for _, ev := range f.ring {
		if ev.Seq > sinceSeq && visibleTo(ev, principal) {
			out = append(out, ev)
		}
		if ev.Seq > cursor {
			cursor = ev.Seq // advance over EVERY retained event, visible or not
		}
	}
	return out, cursor
}

// visibleTo reports whether a draining principal may see an event. An empty drainer
// principal (single-tenant / admin / observer) sees everything; a tenant sees principal-less
// events (global) and its own events, never a peer tenant's. Mirrors the coherence feed's
// visibleTo exactly.
func visibleTo(ev TranscriptEvent, principal string) bool {
	return principal == "" || ev.principal == "" || ev.principal == principal
}

// appendRingCapped appends ev to a bounded ring buffer, dropping the oldest events so the
// slice never exceeds capacity. A non-positive capacity leaves the ring unbounded. This is
// a local re-implementation of the identically named helper in internal/gateway/coherence.go
// (which this package cannot import) — the same trim policy the coherence and session-change
// feeds keep.
func appendRingCapped[T any](ring []T, ev T, capacity int) []T {
	ring = append(ring, ev)
	if capacity > 0 && len(ring) > capacity {
		ring = ring[len(ring)-capacity:] // drop the oldest
	}
	return ring
}

// EventsFromRecords maps transcript records to redacted feed events for a principal, with
// no taint screening. It is the convenience form of EventsFromRecordsScreened for a source
// known to carry no quarantined spans.
func EventsFromRecords(recs []transcript.Record, principal string) []TranscriptEvent {
	return EventsFromRecordsScreened(recs, principal, nil)
}

// EventsFromRecordsScreened maps transcript records to redacted feed events for a principal,
// applying the taint screen at INGRESS: for any record the quarantined predicate reports
// true, the event is emitted as a bounded STUB — Kind preserved (structural metadata), but
// Tool and Summary withheld, Withheld set, and Reason set to the read-plane taint marker
// sessionread.ReasonReadTaintWithheld. A nil predicate screens nothing.
//
// Records that map to no event kind (control / summary / metadata rows that are neither a
// user/assistant turn nor a tool terminal) are skipped.
//
// The screen runs HERE, before the event ever reaches a Feed, so a quarantined span's raw
// bytes can never cross the stream boundary regardless of who later drains the feed — the
// observe plane cannot become an exfiltration channel around the C1 read-plane taint gate.
func EventsFromRecordsScreened(recs []transcript.Record, principal string, quarantined func(transcript.Record) bool) []TranscriptEvent {
	out := make([]TranscriptEvent, 0, len(recs))
	for _, r := range recs {
		kind, tool, ok := classify(r)
		if !ok {
			continue
		}
		ev := TranscriptEvent{Kind: kind, UUID: r.UUID, principal: principal}
		if quarantined != nil && quarantined(r) {
			// Taint stub: the record derives from a sealed/tombstoned span. Kind is
			// structural (safe); the content — tool name + text summary — is withheld
			// behind the read-plane taint marker. No payload bytes are attached.
			ev.Withheld = true
			ev.Reason = sessionread.ReasonReadTaintWithheld
		} else {
			ev.Tool = tool
			ev.Summary = summarize(r)
		}
		out = append(out, ev)
	}
	return out
}

// classify maps one record to its event kind (and tool name, where a tool is named). The
// mapping is deterministic and structural — it never reads span text:
//
//   - a paired tool_use+tool_result (LastToolUseName!="" && HasToolResult()) => tool-terminal;
//   - an assistant tool choice with no paired result yet => decision;
//   - a user turn => turn-open;
//   - an assistant plain turn => turn-close;
//   - anything else (control / summary / metadata) => no event.
func classify(r transcript.Record) (kind, tool string, ok bool) {
	if name := r.LastToolUseName(); name != "" {
		if r.HasToolResult() {
			return KindToolTerminal, name, true
		}
		return KindDecision, name, true
	}
	switch r.Role() {
	case "user":
		return KindTurnOpen, "", true
	case "assistant":
		return KindTurnClose, "", true
	}
	return "", "", false
}

// summarize is a bounded, whitespace-collapsed prefix of a record's text — a redacted
// descriptor, never the full transcript text. It is emitted only for a NON-quarantined
// event (a quarantined event withholds it entirely).
func summarize(r transcript.Record) string {
	s := strings.Join(strings.Fields(r.Text()), " ")
	if len(s) > maxSummary {
		s = s[:maxSummary]
	}
	return s
}
