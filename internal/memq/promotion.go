package memq

import "fmt"

// Consent classes for a context->memory promotion — was the durable write something
// the user (or an equivalent authoritative producer) explicitly asked for, or did the
// system infer it was durable-worthy without being told? This is a closed, small
// vocabulary (mirroring the closed Durability* vocabulary above) so an explain surface
// can render it without free text.
const (
	ConsentExplicit = "explicit" // the user/operator directly asked for this to be remembered
	ConsentInferred = "inferred" // the producer inferred durability (e.g. tense/lexical prior, a driver default)
	ConsentUnknown  = "unknown"  // no consent signal was recorded; fails closed to the weakest claim
)

// NormConsent maps any consent string to the canonical vocabulary, failing closed to
// ConsentUnknown for a missing/unrecognized value — same posture as NormDurability.
func NormConsent(s string) string {
	switch s {
	case ConsentExplicit, ConsentInferred, ConsentUnknown:
		return s
	default:
		return ConsentUnknown
	}
}

// PromotionRecord is the audit record CONTEXT-IS-NOT-MEMORY.md calls for: it captures
// WHY a fact crossed from live context into durable memq storage, so a caller can
// explain that promotion later from structure alone — never from asking a model to
// narrate/summarize the cell's history. Every field is either copied verbatim off the
// Cell at promotion time or supplied explicitly by the caller; nothing here is
// synthesized after the fact.
//
// A "promotion" in memq's model is any MemStore.Add whose resulting cell durability is
// NOT DurabilityTurn — turn-class cells are pure context (they are never eligible for
// memory per CONTEXT-IS-NOT-MEMORY.md §4 Move 1) and so never mint a record. This
// mirrors the doc's decision tree: the record only exists on the branch that actually
// reaches "durable memory."
type PromotionRecord struct {
	// CellID is the address of the promoted cell (memq.Cell.ID) — the join key back to
	// the page table.
	CellID string `json:"cell_id"`
	// SourceSpan identifies WHERE the fact came from in the live turn: the ordinal
	// step, the producing role, and a safe extractive descriptor (never sealed bytes —
	// same posture as Cell.Descriptor). This is the "source span/reference" the issue
	// asks for.
	SourceSpan SourceSpan `json:"source_span"`
	// Durability is the class earned at promotion time (turn|session|bounded|durable —
	// reuses the existing memq/ctxmmu/recall vocabulary; NormDurability applied).
	Durability string `json:"durability"`
	// Consent records whether a human/operator explicitly asked for this promotion, or
	// whether it was inferred by a producer/classifier (ConsentExplicit|ConsentInferred
	// |ConsentUnknown; NormConsent applied).
	Consent string `json:"consent"`
	// Producer names WHAT wrote this cell: a tool name, "user", a driver
	// ("memq/consolidate"), or a backend ("codex:<path>"). Free text but always
	// populated — NormProducer defaults an empty value to "unknown" so the field is
	// never silently blank.
	Producer string `json:"producer"`
	// Expiry is the optional validity bound for a `bounded` promotion (an opaque,
	// caller-defined tick/step/timestamp string — memq does not interpret it, mirroring
	// recall.Page.ValidTo's "auditable, re-checked elsewhere" posture). Empty means no
	// stated expiry: a `durable` record is expected to leave this empty; a `bounded`
	// record without one is a data-quality fact Explain surfaces, not an error.
	Expiry string `json:"expiry,omitempty"`
	// Reason is the free-text justification the producer supplied (e.g. a driver's
	// tombstone/promotion reason, or "user said: remember this"). Never used as the
	// ONLY explanation — Explain renders it alongside the structured fields, not
	// instead of them.
	Reason string `json:"reason,omitempty"`
	// Seq is the ledger-assigned insertion order (0-based) — deterministic, so replay
	// over the same inputs produces byte-identical records (no wall-clock dependence).
	Seq int `json:"seq"`
}

// SourceSpan is the safe, minimal pointer back to the live-context span a promotion
// came from — never the sealed bytes themselves.
type SourceSpan struct {
	Step       int    `json:"step"`
	Role       string `json:"role,omitempty"`
	Descriptor string `json:"descriptor,omitempty"`
	Digest     string `json:"digest,omitempty"`
}

// NormProducer defaults an empty producer to "unknown" so PromotionRecord.Producer is
// never silently blank — same fail-closed-to-a-named-value posture as NormDurability
// and NormConsent.
func NormProducer(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

// PromotionLedger is an append-only, deterministic record of every context->memory
// promotion. It is NOT a Backend — it is the audit trail a Backend (MemStore today)
// attaches records to as it promotes cells, and the thing `fak memory explain-promotion`
// reads back. Append-only mirrors memq's own no-hard-delete posture (doc.go): a
// promotion record is never edited or removed, only ever added.
type PromotionLedger struct {
	records []PromotionRecord
	byCell  map[string][]int // cell ID -> indices into records, insertion order
}

// NewPromotionLedger returns an empty ledger.
func NewPromotionLedger() *PromotionLedger {
	return &PromotionLedger{byCell: map[string][]int{}}
}

// Record appends a promotion record for a cell that is NOT DurabilityTurn. A
// DurabilityTurn cell is context-only by construction (CONTEXT-IS-NOT-MEMORY.md) and
// never earns a record — Record is a no-op for it, so a caller can call it
// unconditionally on every Add without special-casing turn-class writes itself.
// Durability/Consent are normalized to the closed vocabularies before storage, so a
// stored record can never carry an unrecognized class.
func (l *PromotionLedger) Record(rec PromotionRecord) {
	durability := NormDurability(rec.Durability)
	if durability == DurabilityTurn {
		return // context-only; never promoted, never audited as a promotion
	}
	rec.Durability = durability
	rec.Consent = NormConsent(rec.Consent)
	rec.Producer = NormProducer(rec.Producer)
	rec.Seq = len(l.records)
	l.byCell[rec.CellID] = append(l.byCell[rec.CellID], len(l.records))
	l.records = append(l.records, rec)
}

// For returns every promotion record for a cell ID, oldest first (a cell may be
// re-promoted/reclassified over its life; the ledger keeps the whole history, it never
// overwrites). ok is false when the cell has no promotion record at all — either it
// was never promoted (turn-class) or the ID is unknown to this ledger.
func (l *PromotionLedger) For(cellID string) (recs []PromotionRecord, ok bool) {
	idx, found := l.byCell[cellID]
	if !found || len(idx) == 0 {
		return nil, false
	}
	out := make([]PromotionRecord, len(idx))
	for i, ix := range idx {
		out[i] = l.records[ix]
	}
	return out, true
}

// Latest returns the most recent promotion record for a cell ID (ok=false if none).
func (l *PromotionLedger) Latest(cellID string) (PromotionRecord, bool) {
	recs, ok := l.For(cellID)
	if !ok {
		return PromotionRecord{}, false
	}
	return recs[len(recs)-1], true
}

// All returns every record in insertion order (a snapshot copy — the caller cannot
// mutate the ledger through it).
func (l *PromotionLedger) All() []PromotionRecord {
	out := make([]PromotionRecord, len(l.records))
	copy(out, l.records)
	return out
}

// Explanation is the structured, ledger-only account of why a cell is in durable
// memory — the done-condition artifact for #1595. Every field traces to a
// PromotionRecord field; Explain performs no model call and synthesizes no prose
// beyond formatting the record, so a caller can render it verbatim as the answer to
// "why do you remember this."
type Explanation struct {
	CellID     string     `json:"cell_id"`
	Found      bool       `json:"found"`
	SourceSpan SourceSpan `json:"source_span,omitempty"`
	Durability string     `json:"durability,omitempty"`
	Consent    string     `json:"consent,omitempty"`
	Producer   string     `json:"producer,omitempty"`
	Expiry     string     `json:"expiry,omitempty"`
	Reason     string     `json:"reason,omitempty"`
	// Narrative is a fixed-template sentence assembled ONLY from the fields above —
	// string concatenation, not a model summarization call. It exists so a CLI/agent
	// has one human-readable line, without reopening the model-narration door the
	// issue explicitly closes.
	Narrative string `json:"narrative"`
}

// Explain answers "why is this cell in durable memory" using ONLY the ledger's
// structured record for cellID — never a model narration. A cell with no promotion
// record (never promoted, or unknown) yields Found=false and a narrative that says so
// plainly; it does not guess or fall back to summarizing the cell body.
func (l *PromotionLedger) Explain(cellID string) Explanation {
	rec, ok := l.Latest(cellID)
	if !ok {
		return Explanation{
			CellID:    cellID,
			Found:     false,
			Narrative: fmt.Sprintf("cell %q has no promotion record: it was never promoted past turn-class context, or is unknown to this ledger", cellID),
		}
	}
	expiry := rec.Expiry
	if expiry == "" {
		expiry = "none stated"
	}
	reason := rec.Reason
	if reason == "" {
		reason = "none recorded"
	}
	narrative := fmt.Sprintf(
		"cell %q was promoted to %s memory from step %d (role=%q, descriptor=%q) by producer %q with %s consent; expiry: %s; reason: %s",
		cellID, rec.Durability, rec.SourceSpan.Step, rec.SourceSpan.Role, rec.SourceSpan.Descriptor,
		rec.Producer, rec.Consent, expiry, reason,
	)
	return Explanation{
		CellID:     cellID,
		Found:      true,
		SourceSpan: rec.SourceSpan,
		Durability: rec.Durability,
		Consent:    rec.Consent,
		Producer:   rec.Producer,
		Expiry:     rec.Expiry,
		Reason:     rec.Reason,
		Narrative:  narrative,
	}
}
