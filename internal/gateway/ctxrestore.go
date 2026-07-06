package gateway

import (
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxrestore.go — the restore-by-ID arm of the guard's context API: the recovery edge that turns
// the compaction tombstone from an orientation-only string into a CALLABLE handle. When the
// `fak guard -- claude` Anthropic passthrough compacts away a session's ORIGINATING task (the first
// user turn), agent.CompactAnthropicHistory embeds a content-address (id=<sha256hex>) in the stub
// AND hands the gateway the full dropped bytes (CompactOutcome.RestoreID / RestoreBytes). This file
// stashes digest→bytes per session trace so a model resuming PAST the compaction — reading history,
// wanting its next step, seeing "[fak] originating task (compacted, id=…)" — can call
// fak_context_restore(id) to page the whole task back in.
//
// The design axis (Law A2 — every value carries its owner): the stash is a WITNESSED record of what
// fak's OWN transform dropped, content-addressed with the SAME sha256-hex scheme as
// ctxplan.Digest / recall / blob / memq, so a compaction handle and a ctxplan handle are
// interchangeable recovery addresses. Restore is READ-ONLY: it returns bytes fak already had in
// hand and dropped from the wire; it never fabricates context and never re-enters the request path.
//
// Trust gate: restore is the demand-page counterpart of ctxplan.Store.Materialize, and it honors the
// same refusal contract. A span the operator later SEALS (trust quarantine) or TOMBSTONES (context
// control) must NOT be resurrectable through the handle — otherwise restore defeats the very
// suppression that dropped it. A refused entry returns ErrRestoreRefused wrapping the reason; a
// miss (unknown id, evicted, or a fresh session that never tombstoned) returns ErrRestoreMiss. The
// handle is a recovery convenience, never a trust bypass.

// maxCtxRestoreSessions bounds the per-session stash exactly like maxCtxValueSessions: on overflow
// the whole table takes a generational reset, so a gateway minting a fresh trace per session cannot
// grow it without limit. A tombstone is rare (it fires only when compaction drops the FIRST user
// turn), so one entry per trace is the common case and this ceiling is generous.
const maxCtxRestoreSessions = 8192

// maxCtxRestoreEntriesPerSession bounds how many distinct originating-task handles one trace retains.
// A session tombstones its originating task at most once per window era; a handful covers re-anchored
// eras (a head-anchored burst can re-tombstone after a window rewrite). Oldest-out on overflow so the
// MOST RECENT task a resuming model is likeliest to ask for is the one kept.
const maxCtxRestoreEntriesPerSession = 8

// ErrRestoreMiss is returned when no entry addresses the requested id (unknown, evicted, or a
// session that never tombstoned). Distinct from a refusal so the caller can tell "never had it" from
// "had it, the gate held".
var ErrRestoreMiss = errors.New("gateway: no restorable context for that id")

// ErrRestoreRefused wraps a trust-gate refusal — a sealed or tombstoned span the handle must not
// resurrect. It carries the ctxplan sentinel (ErrSealed / ErrTombstoned) so a caller can branch on
// WHICH gate held, keeping restore's refusal vocabulary identical to Store.Materialize's.
var ErrRestoreRefused = errors.New("gateway: context restore refused by the trust gate")

// restoreEntry is one content-addressed dropped span: the id (sha256 hex, the ctxplan.Digest
// scheme), the excerpt fak also embedded in the stub (so a restore reply can echo WHAT it is
// alongside the bytes), and the verbatim JSON of the dropped turn. sealed/tombstoned mirror the
// ctxplan.Span gates: an operator context-control action flips them and Materialize then refuses,
// so a suppressed span cannot be paged back in through the handle.
type restoreEntry struct {
	id         string
	excerpt    string
	bytes      []byte
	sealed     bool
	tombstoned bool
}

// sessionCtxRestore is one trace's ordered stash of restore entries (oldest first, so overflow drops
// the oldest). Mutated only under Server.ctxRestoreMu.
type sessionCtxRestore struct {
	entries []restoreEntry
}

// stashRestore records a fired-tombstone's dropped originating task under its content-address, so
// fak_context_restore(id) can later page it back in. id is the sha256-hex handle agent.Compact
// embedded in the stub; excerpt is the bounded orientation line; taskBytes is the verbatim dropped
// turn. A nil server, empty trace, empty id, or empty bytes is a safe no-op (nothing callable to
// stash). Re-stashing the same id refreshes it in place (the digest is a pure function of the bytes,
// so a repeat is the same task). Called from the compaction path (compactAnthropicRawWithReason) on
// a fired tombstone.
func (s *Server) stashRestore(trace, id, excerpt string, taskBytes []byte) {
	if s == nil || strings.TrimSpace(trace) == "" || strings.TrimSpace(id) == "" || len(taskBytes) == 0 {
		return
	}
	s.ctxRestoreMu.Lock()
	defer s.ctxRestoreMu.Unlock()
	if s.ctxRestore == nil {
		s.ctxRestore = make(map[string]*sessionCtxRestore)
	}
	if _, ok := s.ctxRestore[trace]; !ok && len(s.ctxRestore) >= maxCtxRestoreSessions {
		s.ctxRestore = make(map[string]*sessionCtxRestore) // generational reset, like ctxValue
	}
	sess := s.ctxRestore[trace]
	if sess == nil {
		sess = &sessionCtxRestore{}
		s.ctxRestore[trace] = sess
	}
	// Refresh in place if we already hold this id (same bytes ⇒ same digest); preserve any gate flags
	// an operator already set on it, so a re-tombstone cannot silently un-seal a quarantined span.
	for i := range sess.entries {
		if sess.entries[i].id == id {
			sess.entries[i].excerpt = excerpt
			sess.entries[i].bytes = append([]byte(nil), taskBytes...)
			return
		}
	}
	sess.entries = append(sess.entries, restoreEntry{
		id:      id,
		excerpt: excerpt,
		bytes:   append([]byte(nil), taskBytes...),
	})
	if len(sess.entries) > maxCtxRestoreEntriesPerSession {
		sess.entries = sess.entries[len(sess.entries)-maxCtxRestoreEntriesPerSession:]
	}
}

// CtxRestoreResult is the fak_context_restore reply. Bytes is the verbatim dropped turn (the full
// task, not the excerpt); Excerpt echoes the orientation line the stub carried, so a model gets both
// "what it is" and "all of it" in one answer. Provenance is WITNESSED — fak is returning bytes it
// authored the drop of, never a guess or a fabrication.
type CtxRestoreResult struct {
	Schema     string `json:"schema"`
	TraceID    string `json:"trace_id"`
	ID         string `json:"id"`
	Excerpt    string `json:"excerpt,omitempty"`
	Bytes      string `json:"bytes"` // the verbatim JSON of the dropped originating-task turn
	Provenance string `json:"provenance"`
}

const ctxRestoreSchema = "fak-ctxrestore-result/1"

// ContextRestoreRequest is the fak_context_restore MCP argument shape: the content-address handle a
// tombstone embedded, and an optional trace (omitted resolves to the gateway default trace — the
// wrapped session itself under `fak guard`, so a model restoring its OWN dropped task needs no
// out-of-band identity).
type ContextRestoreRequest struct {
	ID      string `json:"id"`
	TraceID string `json:"trace_id"`
}

// restoreContext resolves a fak_context_restore call: page the dropped originating-task bytes back in
// by their content-address, honoring the trust gate. A sealed/tombstoned entry is REFUSED
// (ErrRestoreRefused) — the handle recovers a dropped span, never a suppressed one — and an unknown
// id is a miss (ErrRestoreMiss). The lookup is exact on the digest; there is no fuzzy match, because
// a content-address either names bytes we hold or it does not.
func (s *Server) restoreContext(req ContextRestoreRequest) (CtxRestoreResult, error) {
	id := strings.TrimSpace(req.ID)
	if id == "" {
		return CtxRestoreResult{}, errors.New("fak_context_restore id is required")
	}
	trace := s.traceFor(req.TraceID)
	s.ctxRestoreMu.Lock()
	defer s.ctxRestoreMu.Unlock()
	sess := s.ctxRestore[trace]
	if sess == nil {
		return CtxRestoreResult{}, ErrRestoreMiss
	}
	for _, e := range sess.entries {
		if e.id != id {
			continue
		}
		switch {
		case e.sealed:
			return CtxRestoreResult{}, errWrap(ErrRestoreRefused, ctxplan.ErrSealed)
		case e.tombstoned:
			return CtxRestoreResult{}, errWrap(ErrRestoreRefused, ctxplan.ErrTombstoned)
		}
		return CtxRestoreResult{
			Schema:     ctxRestoreSchema,
			TraceID:    trace,
			ID:         id,
			Excerpt:    e.excerpt,
			Bytes:      string(e.bytes),
			Provenance: "WITNESSED",
		}, nil
	}
	return CtxRestoreResult{}, ErrRestoreMiss
}

// errWrap joins a surface error with its underlying cause so callers can errors.Is on EITHER —
// ErrRestoreRefused for the branch and the ctxplan sentinel for the specific gate.
func errWrap(surface, cause error) error {
	return &wrappedErr{surface: surface, cause: cause}
}

type wrappedErr struct {
	surface, cause error
}

func (w *wrappedErr) Error() string { return w.surface.Error() + ": " + w.cause.Error() }
func (w *wrappedErr) Is(target error) bool {
	return errors.Is(w.surface, target) || errors.Is(w.cause, target)
}
