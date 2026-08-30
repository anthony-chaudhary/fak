package gateway

import "sync"

// admissionRecord is the recorded outcome of screening ONE unique inbound tool
// result, keyed by (trace, content-digest). It is written once, at the result's
// FIRST arrival, and consulted on every subsequent client replay of the same
// content instead of re-screening — the admit-once contract (#2417).
//
// It supersedes the notedResults / notedToolFailures replay-dedup maps: those
// suppressed the human-facing SYMPTOM (the repeated banner) while admitInboundResults
// still re-ran the full result-side stack over the whole client-replayed transcript
// every turn. Keying admission to the entry deletes the re-screening itself — "was
// this result screened?" is now a ledger query, not a probe of a side map.
//
// When the content-addressed hash-chained session ledger (#2416) lands, this per-trace
// record migrates onto the ledger node for the entry; the contract this type already
// enforces — screen once, record the verdict, consult it on replay — is the one that
// wire needs. Until then this is a self-contained gateway-local ledger that owns the
// admit-once behavior without waiting on the heavier event-log epic.
type admissionRecord struct {
	// screened is set once the kernel has actually admitted this content; verdict/content
	// are authoritative only then. A record created solely to dedup a tool-failure note
	// (failNoteFirst) starts unscreened, so admit still screens it on first real arrival.
	screened bool
	verdict  WireVerdict // the recorded admission verdict (ALLOW/TRANSFORM/QUARANTINE/…)
	content  string      // the model-facing content to forward on replay (paged-out on a hold)
	rewrote  bool        // the screen paged the bytes out (QUARANTINE/TRANSFORM) — apply content on replay
	failNote bool        // the exit-143 recovery note has already been surfaced for this result
}

// admissionLedger keys result admission to content, per trace. A zero value is ready
// to use; it carries its own mutex so it can be a plain Server field.
type admissionKey struct {
	callID string
	digest string
}

type admissionLedger struct {
	mu      sync.Mutex
	byTrace map[string]map[admissionKey]*admissionRecord
}

// traceLocked returns the per-trace record map, creating it and bounding the ledger to
// maxResetHealthSessions traces (the same reaper convention as turnSafety/resetHealth).
// The caller must hold l.mu. The trace being fetched is never the one evicted.
func (l *admissionLedger) traceLocked(trace string) map[admissionKey]*admissionRecord {
	if l.byTrace == nil {
		l.byTrace = map[string]map[admissionKey]*admissionRecord{}
	}
	if _, ok := l.byTrace[trace]; !ok && len(l.byTrace) >= maxResetHealthSessions {
		for k := range l.byTrace {
			if k == trace {
				continue
			}
			delete(l.byTrace, k)
			break
		}
	}
	seen := l.byTrace[trace]
	if seen == nil {
		seen = map[admissionKey]*admissionRecord{}
		l.byTrace[trace] = seen
	}
	return seen
}

// admit returns the admission outcome for (trace, digest), screening EXACTLY ONCE. On
// the first arrival it runs screen() — the real kernel result-side admission — records
// the verdict, and reports fresh=true. On every later replay of the same content it
// returns the recorded verdict WITHOUT re-screening and reports fresh=false. An empty
// trace has no session to key on and always screens fresh, matching the pre-ledger
// un-deduped fallback.
//
// screen() runs OUTSIDE the lock so kernel admission never serializes the whole ledger.
// A single request is single-threaded over its transcript, so the dominant replay path
// never races; a rare concurrent same-trace turn may screen twice, which is idempotent
// (same content → same verdict) and no worse than the pre-ledger re-screen-every-turn.
func (l *admissionLedger) admit(trace, callID, digest string, screen func() (WireVerdict, string, bool)) (*admissionRecord, bool) {
	key := admissionKey{callID: callID, digest: digest}
	if l == nil || trace == "" {
		v, c, rw := screen()
		return &admissionRecord{screened: true, verdict: v, content: c, rewrote: rw}, true
	}
	l.mu.Lock()
	if rec := l.traceLocked(trace)[key]; rec != nil && rec.screened {
		l.mu.Unlock()
		return rec, false
	}
	l.mu.Unlock()

	v, c, rw := screen()

	l.mu.Lock()
	defer l.mu.Unlock()
	seen := l.traceLocked(trace)
	if rec := seen[key]; rec != nil && rec.screened {
		// A peer screened the same call result while we were unlocked. Keep the first
		// record and treat ours as a replay so we do not double the eviction/reset.
		return rec, false
	}
	rec := seen[key]
	if rec == nil {
		rec = &admissionRecord{}
		if note := seen[admissionKey{digest: digest}]; note != nil {
			rec.failNote = note.failNote
		}
		seen[key] = rec
	}
	rec.screened, rec.verdict, rec.content, rec.rewrote = true, v, c, rw
	return rec, true
}

// failNoteFirst reports whether the exit-143 recovery note for (trace, digest) has not
// yet been surfaced, marking it surfaced. Tool-failure dedup is a ledger query too, so
// the separate notedToolFailures map is gone. An empty trace always reports true (the
// un-deduped fallback). A digest with no admission record yet gets a bare record whose
// only live field is failNote — a later admit() still screens it (screened is false).
func (l *admissionLedger) failNoteFirst(trace, digest string) bool {
	if l == nil || trace == "" {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	seen := l.traceLocked(trace)
	key := admissionKey{digest: digest}
	rec := seen[key]
	if rec == nil {
		rec = &admissionRecord{}
		seen[key] = rec
	}
	if rec.failNote {
		return false
	}
	rec.failNote = true
	return true
}

// records counts the unique screened results recorded for a trace — the admit-once
// witness: it equals the number of distinct tool-result contents admitted, not the
// number of turns the client replayed them over.
func (l *admissionLedger) records(trace string) int {
	if l == nil {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, rec := range l.byTrace[trace] {
		if rec.screened {
			n++
		}
	}
	return n
}
