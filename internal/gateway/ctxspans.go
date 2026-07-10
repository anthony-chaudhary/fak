package gateway

import (
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// ctxspans.go — the enumeration arm of the guard's context API: the DISCOVERY edge that lets a model
// SEE which content-addressed dropped spans a session trace currently holds, so it can then call
// fak_context_restore(id) on one. It is the read-only counterpart to ctxrestore.go's restore-by-id:
// restore pages a handle back into bytes; this lists the handles first. Where a resuming model reading
// "[fak] originating task (compacted, id=…)" already knows the one id its stub carried,
// fak_context_spans answers the broader question — "what ELSE is restorable here?" — without the model
// having to have seen every tombstone stub go by.
//
// The design axis (Law A2 — every value carries its owner): each row is the gateway analogue of a
// ctxplan.Span — SAFE metadata only. It surfaces the id (the sha256-hex recovery handle, the same
// ctxplan.Digest address space restore keys on), the orientation Descriptor, the byte COST as a size,
// and the evidence-cluster / evidence-kind edges — but NEVER the stashed bytes themselves. Bytes cross
// the gate only through restore's trust-checked page-in, never through enumeration; listing what is
// recoverable must not itself be a recovery. That is why CtxSpan has no bytes field to leak.
//
// Trust gate: enumeration LISTS a sealed or tombstoned span (so the operator's suppression is legible —
// the span is still on the record) but marks Restorable=false, mirroring how ctxplan.Store.Spans reports
// a span's Sealed/Tombstoned flags while Materialize refuses its bytes. A model reading this list learns
// the span exists and is suppressed, and that fak_context_restore would refuse it — the same refusal
// contract, surfaced one step earlier.
//
// Read-only in the strong sense: contextSpans takes the same Server.ctxRestoreMu the stash mutates, but
// only to read a consistent snapshot; it mutates nothing and returns nothing that could page bytes back
// in out of band. Unlike restore-by-id — where an unknown id is a genuine MISS (ErrRestoreMiss) — an
// empty or unknown trace is a valid EMPTY answer here: enumerating nothing is a fact, not a failure.

// CtxSpan is one enumerated restorable handle: the SAFE, byte-free projection of a stashed
// restoreEntry that fak_context_spans surfaces so a model can decide what to pass to
// fak_context_restore. It carries the id (the sha256-hex content-address, the recovery handle), the
// Descriptor (the orientation excerpt the stub also embedded — never the full bytes), and Bytes (the
// LENGTH of the stashed span, the token-cost proxy, exactly as ctxplan.Span.Bytes is a size and not the
// content). EvidenceCluster/EvidenceKind expose the decisive→cluster edges a richer span source carries
// (empty for the current compaction-tombstone source). Sealed/Tombstoned mirror the trust gates;
// Restorable is their fold (= !Sealed && !Tombstoned) — the one bit that says whether
// fak_context_restore would page this handle back in or refuse it.
type CtxSpan struct {
	ID              string `json:"id"`                         // sha256-hex content-address — the handle to pass to fak_context_restore
	Descriptor      string `json:"descriptor,omitempty"`       // SAFE orientation excerpt; NEVER the full stashed bytes
	Bytes           int64  `json:"bytes"`                      // size of the stashed span (the cost proxy), not its content
	EvidenceCluster string `json:"evidence_cluster,omitempty"` // the cluster this span belongs to, when a source carries it
	EvidenceKind    string `json:"evidence_kind,omitempty"`    // decisive|support|current|optional, when a source carries it
	Sealed          bool   `json:"sealed,omitempty"`           // quarantined by the trust gate — listed, not restorable
	Tombstoned      bool   `json:"tombstoned,omitempty"`       // suppressed by context control — listed, not restorable
	Restorable      bool   `json:"restorable"`                 // = !Sealed && !Tombstoned: whether restore would page it in
}

// CtxSpansResult is the fak_context_spans reply: the ordered list of restorable handles a trace holds,
// oldest first (the stash's own order, so overflow-drop and re-anchor eras read in the order they
// happened). Count is len(Spans) so a caller need not re-count. There is no bytes field ANYWHERE in this
// shape — enumeration surfaces what is recoverable, never the recoverable content itself.
type CtxSpansResult struct {
	Schema  string    `json:"schema"`
	TraceID string    `json:"trace_id"`
	Spans   []CtxSpan `json:"spans"`
	Count   int       `json:"count"`
}

const ctxSpansSchema = "fak-ctxspans-result/1"

// ContextSpansRequest is the fak_context_spans MCP argument shape: only the optional trace. An omitted
// trace resolves to the gateway default trace (the wrapped session itself under `fak guard`) via
// s.traceFor — exactly like ContextRestoreRequest — so a model enumerating its OWN restorable handles
// needs no out-of-band identity.
type ContextSpansRequest struct {
	TraceID string `json:"trace_id"`
}

// contextSpans resolves a fak_context_spans call: enumerate the content-addressed dropped spans a trace
// currently holds as SAFE, byte-free rows. It locks Server.ctxRestoreMu only to read a consistent
// snapshot, maps each stashed restoreEntry to a CtxSpan in stash order, and mutates nothing. A missing
// or empty trace is NOT an error (unlike restore-by-id's miss): it returns an empty, non-nil Spans slice
// with Count 0, because enumerating a trace that has dropped nothing is a valid empty answer. Sealed and
// tombstoned entries are LISTED (their suppression is legible) with Restorable=false, and their bytes are
// never offered here — only restore, trust-gated, can page bytes back in.
func (s *Server) contextSpans(caller string, req ContextSpansRequest) (CtxSpansResult, error) {
	trace := s.traceFor(req.TraceID)

	// C1 read-scope floor (#4192): enumeration surfaces the byte-free orientation descriptors of a
	// trace's dropped tasks, which are themselves the other principal's task text — so a
	// cross-principal enumeration is refused READ_SCOPE_DENIED, mirroring the restore path. A
	// self-read (caller principal owns the trace, including both "" on the loopback) proceeds.
	if err := s.scopeReadSelf(caller, sessionread.OpContextSpans, trace); err != nil {
		return CtxSpansResult{}, err
	}

	// Non-nil empty slice by default: an empty enumeration is an answer, not a null.
	spans := []CtxSpan{}

	s.ctxRestoreMu.Lock()
	if sess := s.ctxRestore[trace]; sess != nil {
		spans = make([]CtxSpan, 0, len(sess.entries))
		for i := range sess.entries {
			e := sess.entries[i]
			row := CtxSpan{
				ID:              e.id,
				Descriptor:      e.excerpt,
				Bytes:           int64(len(e.bytes)),
				EvidenceCluster: e.cluster,
				Sealed:          e.sealed,
				Tombstoned:      e.tombstoned,
				Restorable:      !e.sealed && !e.tombstoned,
			}
			// Only assert an evidence-kind edge when the entry actually carries one: empty in, empty out,
			// so enumeration never fabricates an "optional" edge onto a source (today's compaction
			// tombstone) that declared no kind. NormEvidenceKind is used only to canonicalize a real kind.
			if e.kind != "" {
				row.EvidenceKind = ctxplan.NormEvidenceKind(e.kind)
			}
			spans = append(spans, row)
		}
	}
	s.ctxRestoreMu.Unlock()

	return CtxSpansResult{
		Schema:  ctxSpansSchema,
		TraceID: trace,
		Spans:   spans,
		Count:   len(spans),
	}, nil
}
