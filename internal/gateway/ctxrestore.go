package gateway

import (
	"context"
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/screen"
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
	// cluster and kind are the evidence-cluster and evidence-kind edges the enumeration tool
	// (fak_context_spans) surfaces alongside the handle. They are EMPTY for the current
	// compaction-tombstone source — a dropped originating task carries no cluster membership — and
	// exist so a richer span source (issue #3062) can populate the decisive→cluster edges without a
	// second stash shape. stashRestore leaves them zero; contextSpans reads them (empty in, empty out).
	cluster string
	kind    string
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

// bindTraceOwner records, first-writer-wins, the principal that owns a session trace — the C1
// read-scope floor (#4192). Called from the served-request boundary (handleAnthropicMessages) the
// first time a trace serves a turn, so the principal that drove the session is the one a later
// read-self op must match. A nil server or empty trace is a safe no-op. The owner may legitimately
// be "" (the single-tenant no-RequireKey loopback, where every caller shares the "" principal); an
// already-bound trace keeps its first owner, so a mid-session principal change cannot re-home a
// trace's dropped task under a new owner. Bounded by the same generational reset as ctxRestore.
func (s *Server) bindTraceOwner(trace, principal string) {
	if s == nil || strings.TrimSpace(trace) == "" {
		return
	}
	s.traceOwnerMu.Lock()
	defer s.traceOwnerMu.Unlock()
	if s.traceOwner == nil {
		s.traceOwner = make(map[string]string)
	}
	if _, ok := s.traceOwner[trace]; ok {
		return // first-writer-wins
	}
	if len(s.traceOwner) >= maxCtxRestoreSessions {
		s.traceOwner = make(map[string]string) // generational reset, like ctxRestore/ctxValue
	}
	s.traceOwner[trace] = principal
}

// traceOwnerOf returns the principal bound to a trace and whether any owner was recorded. An
// unbound trace reads as ("", false) — the read-scope floor treats an unbound owner as "" (the
// single-tenant default), so a caller with no principal still self-reads while a named principal is
// refused.
func (s *Server) traceOwnerOf(trace string) (string, bool) {
	s.traceOwnerMu.RLock()
	defer s.traceOwnerMu.RUnlock()
	owner, ok := s.traceOwner[trace]
	return owner, ok
}

// scopeReadSelf is the C1 read-scope floor for the two read-self context ops (fak_context_restore /
// fak_context_spans, #4192): a caller may address a trace's dropped originating task ONLY when its
// principal matches the trace's owner. When caller and owner are the SAME principal — including both
// "" on the no-RequireKey loopback, where there is no per-principal boundary to cross — it is a
// self-read and admitted. Otherwise the refusal is routed through the kernel floor
// (screen.Authorize) so the closed READ_SCOPE_DENIED token and its byte-free detail come from one
// place; for a read-self op with mismatched (or empty) principals Authorize always refuses. Returns
// nil when the read is in-scope, else a *screen.Refusal carrying READ_SCOPE_DENIED.
func (s *Server) scopeReadSelf(caller string, op sessionread.ReadOp, trace string) error {
	owner, _ := s.traceOwnerOf(trace)
	if caller == owner {
		return nil
	}
	return screen.Authorize(screen.ScopeRequest{Op: op, Caller: caller, TargetOwner: owner})
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
// tombstone embedded, an optional trace (omitted resolves to the gateway default trace — the wrapped
// session itself under `fak guard`, so a model restoring its OWN dropped task needs no out-of-band
// identity), and an optional recall image dir (issue #3062). ImageDir generalizes restore beyond the
// compaction-tombstone stash: when the per-trace stash does not hold the digest, a named recall core
// image is consulted so a recall PAGE addressed by its recall digest pages back in under the image's
// own trust gate — the SAME content-address, one restore call. Omitted, restore is stash-only, exactly
// as Slice 1 behaved (backward-compatible).
type ContextRestoreRequest struct {
	ID       string `json:"id"`
	TraceID  string `json:"trace_id"`
	ImageDir string `json:"image_dir,omitempty"`
}

// restoreContext resolves a fak_context_restore call: page dropped-span bytes back in by their
// content-address, honoring the trust gate. Resolution is layered over ONE unified sha256-hex id space
// (issue #3062): the per-trace compaction-tombstone stash is consulted FIRST (a hit — bytes OR a
// sealed/tombstoned refusal — is authoritative), and only a genuine stash MISS falls through to a
// content-addressed ctxplan.Store. A sealed/tombstoned span is REFUSED (ErrRestoreRefused wrapping the
// ctxplan sentinel) whichever source held it — the handle recovers a dropped span, never a suppressed
// one — and a digest no source holds is a miss (ErrRestoreMiss). The lookup is exact on the digest;
// there is no fuzzy match, because a content-address either names bytes we hold or it does not.
//
// The store fall-through is source-agnostic (see ctxrestore_store.go): it wires the ctxview-elision
// source (the trace's retained SessionPlanner, now a ctxplan.Store) and the recall-image source
// (ImageDir → recall.CtxStore), both reachable in the gateway lane. A memq-cell store plugs into the
// same restoreFromStore adapter behind a Store handle, with no new routing here.
func (s *Server) restoreContext(caller string, req ContextRestoreRequest) (CtxRestoreResult, error) {
	id := strings.TrimSpace(req.ID)
	if id == "" {
		return CtxRestoreResult{}, errors.New("fak_context_restore id is required")
	}
	trace := s.traceFor(req.TraceID)

	// 0) The C1 read-scope floor (#4192): a read-self op may page a trace's dropped originating
	//    task back in ONLY when the caller's principal owns the trace. The scope check runs BEFORE
	//    any source lookup, so a cross-principal caller cannot even learn whether the trace holds a
	//    stash — it is refused READ_SCOPE_DENIED with no existence leak, and defense-in-depth with
	//    the outbound taint screen restoreFromStash applies to the bytes it does disclose.
	if err := s.scopeReadSelf(caller, sessionread.OpContextRestore, trace); err != nil {
		return CtxRestoreResult{}, err
	}

	// 1) The per-trace compaction-tombstone stash — the default source. A hit here (bytes or a
	//    trust-gate refusal) is authoritative; only a genuine miss falls through to a Store.
	res, err := s.restoreFromStash(trace, id)
	if err == nil || !errors.Is(err, ErrRestoreMiss) {
		return res, err
	}

	// 2) Stash miss — the ctxview-elision source (#3062). The trace's retained view planner holds a
	//    LOSSLESS, content-addressed store of every span the planned-view rewrite ELIDED from the
	//    passthrough; route the SAME digest through it (as a ctxplan.Store) so a ctxview elision pages
	//    back in under the store's own trust gate. Needs no request field — it is the trace's OWN
	//    dropped context, the closest analogue to the compaction tombstone. A hit (bytes, or a
	//    sealed/tombstoned refusal) is authoritative; a trace that never planned a turn has no retained
	//    planner and is a plain miss that falls through to the caller-named recall image below.
	if planner := s.existingSessionPlanner(trace); planner != nil {
		sp, body, ierr := restoreFromStore(context.Background(), planner, id)
		if ierr == nil {
			return CtxRestoreResult{
				Schema:     ctxRestoreSchema,
				TraceID:    trace,
				ID:         id,
				Excerpt:    sp.Descriptor,
				Bytes:      string(body),
				Provenance: "WITNESSED",
			}, nil
		}
		if !errors.Is(ierr, ErrRestoreMiss) {
			return CtxRestoreResult{}, ierr // a sealed/tombstoned refusal is authoritative
		}
		// genuine miss — fall through to the caller-named recall-image source
	}

	// 3) Still a miss — generalize beyond the compaction tombstone (#3062). When the caller names a
	//    persisted recall core image, route the SAME content-address through its ctxplan.Store so a
	//    recall page pages back in under the image's own trust gate.
	if dir := strings.TrimSpace(req.ImageDir); dir != "" {
		sp, body, ierr := s.restoreFromImage(context.Background(), dir, id)
		if ierr != nil {
			return CtxRestoreResult{}, ierr
		}
		return CtxRestoreResult{
			Schema:     ctxRestoreSchema,
			TraceID:    trace,
			ID:         id,
			Excerpt:    sp.Descriptor,
			Bytes:      string(body),
			Provenance: "WITNESSED",
		}, nil
	}

	return CtxRestoreResult{}, ErrRestoreMiss
}

// restoreFromStash resolves a restore-by-digest call against the per-trace compaction-tombstone stash:
// the Slice 1 source, unchanged in behavior. A matching entry returns its bytes, or a refusal wrapping
// the ctxplan sentinel when an operator sealed/tombstoned it; an id the stash does not hold (unknown
// trace, unknown id, evicted) is ErrRestoreMiss — the "never had it" signal restoreContext branches on
// to decide whether to fall through to a content-addressed Store. Holds ctxRestoreMu only for the
// in-memory scan (no I/O), so the fall-through's recall-image load never runs under the stash lock.
func (s *Server) restoreFromStash(trace, id string) (CtxRestoreResult, error) {
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
		// Outbound taint screen (#4192): the SAME kernel screen the read plane declares
		// (screen.ScreenOutbound) decides whether these bytes may cross the boundary. A sealed or
		// tombstoned span refuses with READ_TAINT_WITHHELD and its bytes never leave; a clean span
		// discloses byte-exact. The refusal is wrapped to ALSO satisfy the historical
		// ErrRestoreRefused + ctxplan-sentinel contract, so a caller can still branch on WHICH gate
		// held while the closed READ_TAINT_WITHHELD token is recoverable via screen.RefusalReason.
		body, serr := screen.ScreenOutbound(screen.Span{Bytes: e.bytes, Sealed: e.sealed, Tombstoned: e.tombstoned})
		if serr != nil {
			return CtxRestoreResult{}, restoreTaintRefusal(serr, e.sealed)
		}
		return CtxRestoreResult{
			Schema:     ctxRestoreSchema,
			TraceID:    trace,
			ID:         id,
			Excerpt:    e.excerpt,
			Bytes:      string(body),
			Provenance: "WITNESSED",
		}, nil
	}
	return CtxRestoreResult{}, ErrRestoreMiss
}

// restoreTaintRefusal wraps the outbound taint screen's refusal (a *screen.Refusal carrying
// READ_TAINT_WITHHELD) so the returned error simultaneously satisfies every reader of the restore
// refusal contract: errors.Is(err, ErrRestoreRefused) for the branch, errors.Is(err, ctxplan.Err*)
// for the specific gate, and screen.RefusalReason(err) == READ_TAINT_WITHHELD for the closed wire
// token. sealed selects the ctxplan sentinel (Sealed checked first, mirroring ScreenOutbound's own
// order); the screen refusal is carried verbatim so its byte-free detail reaches the caller.
func restoreTaintRefusal(screenErr error, sealed bool) error {
	sentinel := ctxplan.ErrTombstoned
	if sealed {
		sentinel = ctxplan.ErrSealed
	}
	return &restoreRefusal{parts: []error{ErrRestoreRefused, sentinel, screenErr}}
}

// restoreRefusal joins the several errors a restore taint refusal must answer to (the
// ErrRestoreRefused surface, the ctxplan gate sentinel, and the closed screen.Refusal) behind a
// single value. Its Unwrap() []error lets errors.Is and errors.As traverse all three, so no reader
// of the historical or the new refusal vocabulary is lost.
type restoreRefusal struct{ parts []error }

func (e *restoreRefusal) Error() string {
	msgs := make([]string, 0, len(e.parts))
	for _, p := range e.parts {
		msgs = append(msgs, p.Error())
	}
	return strings.Join(msgs, ": ")
}
func (e *restoreRefusal) Unwrap() []error { return e.parts }

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
