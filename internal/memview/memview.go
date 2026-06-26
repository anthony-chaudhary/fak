package memview

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ViewKind enumerates the first-class derived views (#904 "Which views should be
// first-class?"). The type is OPEN — a producer may name its own kind — but the
// closed set below covers the issue's minimum. Only snippet has a defined
// auto-materialization here; summary/qa/fact declare the contract surface for
// later children (a lossy summary must never be materialized as a fact without a
// quality witness, the selection-integrity question #904 names).
type ViewKind string

const (
	// KindSnippet is a verbatim extractive slice of the raw source (LOSSLESS: the
	// view's bytes ARE a sub-slice of the source, so the content is fully
	// accountable to the source digest). The first implemented view.
	KindSnippet ViewKind = "snippet"
	// KindSummary is a LOSSY semantic summary produced by a model/skill. Its body
	// is not a sub-slice of the source; it is admissible only as a derived view,
	// never as a canonical fact, and must re-enter adjudication before any effect.
	KindSummary ViewKind = "summary"
	// KindQA is a (question, answer) pair extracted from the source by a model.
	KindQA ViewKind = "qa"
	// KindFact is a structured fact. Not auto-materialized: an unsupported summary
	// claim must never become a fact without a quality witness (#904).
	KindFact ViewKind = "fact"
)

// InvalidationRule names how a view's staleness is decided. The default and only
// shipped rule is digest-change. The closed set leaves room for a future
// freshness/validity rule (the bitemporal Zep/Graphiti spine, tracked separately)
// without repurposing the field.
type InvalidationRule string

const (
	// InvalidateOnDigestChange is the default: a view is valid iff its source
	// span's digest equals the page's CURRENT digest. Changed source bytes =>
	// changed digest => the view is stale, whatever the view's own body says.
	InvalidateOnDigestChange InvalidationRule = "digest"
)

// RawPage is the immutable CANONICAL source a view is derived from — the "raw
// memory cell." It is an interface so the typed view contract stays decoupled
// from any one store: a recall.Page (tier 3) adapts to it without memview
// importing recall, and a ctxmmu-paged abi.Ref adapts to it the same way. The
// digest binds the view to EXACT source bytes.
type RawPage interface {
	// Digest is the content address (sha256 hex) of the bytes Bytes returns. Two
	// pages with the same digest hold the same bytes; a changed source changes the
	// digest and invalidates every view bound to the old one.
	Digest() string
	// Bytes returns the raw canonical bytes. A sealed/quarantined upstream page
	// refuses here (returning ErrSealed) so a tainted source can never back an
	// admissible view; materialization is the only consumer of Bytes.
	Bytes() ([]byte, error)
	// Role names the producing tool/skill (provenance of the SOURCE).
	Role() string
	// Taint is the source's abi.TaintLabel; a view INHERITS it.
	Taint() abi.TaintLabel
}

// SourceSpan is the provenance of a view: a byte window [Offset, Offset+Length)
// into a raw page, bound to that page's content digest. A view whose span falls
// outside the page is refused at materialization. Digest-binding is what makes a
// changed source invalidate the view: the span's Digest is the digest the view
// was derived FROM, so once the page's current digest differs, the view is stale.
type SourceSpan struct {
	Digest string // the raw page's content address the view was derived FROM
	Offset int    // byte offset into the raw page (>= 0)
	Length int    // byte length of the source window (> 0)
}

// MemoryViewRecord is the typed virtual-view contract (#904 "minimum local
// MemoryViewRecord"). A view is a DERIVED projection of canonical raw bytes; it
// is never canonical itself. The record carries the provenance a reviewer — or a
// re-adjudication — needs to decide whether the view may enter context or back an
// effect:
//
//   - Source      : the digest-bound byte span the view came from (provenance)
//   - Kind        : the view type (snippet / summary / qa / fact / ...)
//   - Producer    : what generated the view ("summarizer@v1", "qa-extract@v2",
//     a named graph selector). A different producer OR a different
//     span is a DIFFERENT record, so a selector/graph mutation is
//     visible as a producer/source change, not a silent rewrite
//     (#904 selection-integrity).
//   - Taint       : inherited from the source page
//   - Freshness   : monotonic epoch the view was materialized under
//   - Witness     : optional external trust witness the view was generated under
//   - Invalidation: how staleness is decided (default: digest change)
//   - Body        : the materialized derived content; nil if not materialized or
//     if the source was refused (Quarantine)
//
// A view executes NO tool effects. A materialized view carries an abi.Verdict; a
// Quarantine verdict means the body may not enter context, and any effect must
// come from re-entering adjudication over the raw source — never from the view
// directly (#904 "Which views can execute effects? None directly.").
type MemoryViewRecord struct {
	ID           string           // stable view id (caller-assigned)
	Kind         ViewKind         // snippet | summary | qa | fact | ...
	Producer     string           // generator + version (selector provenance)
	Source       SourceSpan       // digest-bound provenance
	Taint        abi.TaintLabel   // inherited from the source page
	Freshness    uint64           // materialization epoch (monotonic)
	Witness      string           // external trust witness, optional
	Invalidation InvalidationRule // how staleness is decided
	Body         []byte           // materialized derived content; nil if not materialized
}

// IsValid reports whether the view is still valid against the page's CURRENT
// source digest. Under the default digest rule a view bound to digest D is stale
// the moment the source's current digest differs from D — so editing the raw
// bytes invalidates EVERY derived view, exactly the cache-line-invalidation
// semantics #904 asks for. An unknown rule, or an empty currentDigest (unknown
// source), is fail-closed STALE: an unprovenanced view is never trusted.
func (r MemoryViewRecord) IsValid(currentSourceDigest string) bool {
	switch r.Invalidation {
	case InvalidateOnDigestChange, "":
		return currentSourceDigest != "" && r.Source.Digest == currentSourceDigest
	}
	return false
}

// Errors returned by materialization. A view whose provenance does not land
// inside its source (ErrSpanOutOfRange), or names no window (ErrEmptySpan),
// cannot be accountable to bytes it does not span and is refused.
var (
	ErrSpanOutOfRange = errors.New("memview: source span out of range")
	ErrEmptySpan      = errors.New("memview: source span is empty")
)

// VerdictFor returns the admission verdict for a view derived from the given raw
// page, WITHOUT materializing: a TRUSTED source is Admit; a TAINTED or QUARANTINED
// source taints the view to Quarantine. This is the gate a cached/materialized
// view re-enters before any effect (#904 "no view executes effects directly").
func VerdictFor(page RawPage) abi.Verdict {
	if page.Taint() == abi.TaintTrusted {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "memview"}
	}
	return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonTrustViolation,
		By: "memview", Payload: abi.QuarantinePayload{}}
}

// Materializer derives views from a raw page under a fixed producer identity and
// a monotonic freshness epoch. It implements #904's ladder step 3: the one
// low-risk view — a source-linked SNIPPET over an immutable raw page, digest-
// bound, with a materialization verdict.
type Materializer struct {
	Producer string // generator identity stamped on every record (selector provenance)
	Epoch    uint64 // monotonic freshness epoch (caller bumps it on re-generation)
}

// MaterializeSnippet derives a verbatim snippet view of the raw page over the
// window [offset, offset+length). Because a snippet is lossless (a sub-slice of
// the source), the view's Body is fully accountable to the source digest, and a
// later IsValid(currentDigest) catches any source edit.
//
// The returned abi.Verdict is the view's admission gate:
//   - Allow      : clean (TRUSTED) source, in-range span; the record is
//     materialized with a Body equal to the source sub-slice.
//   - Quarantine : the source page is TAINTED/QUARANTINED (a tainted source can
//     never back an admissible view); the record is still returned
//     (it carries the provenance + the refusal verdict, so a caller
//     can audit the refusal) but its Body is nil.
//   - error      : span out of range / empty, or the source refused its bytes
//     (ErrSpanOutOfRange / ErrEmptySpan / the page's own error).
//
// The record's Producer binds the selector: a different producer (or a different
// span) is a different record, so a graph-selector mutation surfaces as a
// producer/source change rather than a silent rewrite (#904 selection-integrity).
func (m Materializer) MaterializeSnippet(id string, page RawPage, offset, length int) (MemoryViewRecord, abi.Verdict, error) {
	if length <= 0 {
		return MemoryViewRecord{}, abi.Verdict{}, ErrEmptySpan
	}
	d := page.Digest()
	body, err := page.Bytes()
	if err != nil {
		return MemoryViewRecord{}, abi.Verdict{}, err
	}
	if offset < 0 || offset+length > len(body) {
		return MemoryViewRecord{}, abi.Verdict{}, ErrSpanOutOfRange
	}
	rec := MemoryViewRecord{
		ID:           id,
		Kind:         KindSnippet,
		Producer:     m.Producer,
		Source:       SourceSpan{Digest: d, Offset: offset, Length: length},
		Taint:        page.Taint(),
		Freshness:    m.Epoch,
		Invalidation: InvalidateOnDigestChange,
	}
	v := VerdictFor(page)
	if v.Kind == abi.VerdictQuarantine {
		// Tainted source: return the provenance + refusal verdict, NO body — the
		// view may not enter context. A caller that stores the record keeps the
		// audit trail but can never serve the bytes.
		return rec, v, nil
	}
	rec.Body = append([]byte(nil), body[offset:offset+length]...)
	return rec, v, nil
}

// Digest is the canonical content address (sha256 hex) — the SAME scheme
// internal/recall.Digest and the blob store use, so a memview digest, a recall
// digest, and a blob digest are interchangeable for the same bytes. memview
// carries its own copy so the typed view contract stays a tier-2 mechanism
// (imports only the frozen abi) rather than a tier-3 composer over recall.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
