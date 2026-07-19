package dojo

import "fmt"

// claim_context_restore.go is the context-restore/restore_recall KPI cell (#4486):
// fak_context_restore's hit rate on dropped context spans — the fraction of the
// spans fak drops (compaction/trim) that a later turn pages back in via
// fak_context_restore. It is the KPI for fak's context-continuity concept, scored
// against fak's OWN context-span ledger's recorded drops/restores instead of an
// assumed number.
//
// The claim registers through the additive RegisterClaim seam, so this cell never
// edits — and never conflicts on — the central Registry literal.
//
// Layering (why the fold lives here but the lever does not): internal/dojo is the
// gym's pure core (architest pins it tier 1 — the corpus/ledger-scanning levers
// live in the cmd/fak shell). So the SCORING is a pure, total fold over the drop /
// restore counts the shell reduces from fak's durable context-span ledger; this
// file reads no ledger and does no I/O, which keeps the fold unit-testable without
// a ledger on disk and keeps the tier legal.

// The one anchored literal for this cell — the single target the RSI loop's
// RECALIBRATE arm rewrites, carried in the cell's own file rather than the shared
// map. It is an ESTIMATE (not a floor): a best-guess central tendency the loop may
// re-point at the measured fraction, never a guarantee reality must not breach.
var _ = RegisterClaim("context-restore", "restore_recall", claim(0.5,
	"seed theory (#4486): about half of the context spans fak drops (compaction/trim) are later paged back in by fak_context_restore — the restore-recall KPI for fak's context-continuity concept, restored spans over dropped spans folded from fak's OWN context-span ledger. Restore recall is WITNESSED (fak authors both the drop and the restore, and controls the mechanism), a genuine estimate the RSI loop recalibrates toward the measured fraction. A ledger that records drops but relays NO restore field scores UNMEASURED rather than a fabricated 0.0 — it cannot tell 'nothing was restored' from 'restores are not recorded' (#4490's honesty rule)"))

// ContextSpanLedger is the reduced view of fak's durable context-span drop/restore
// ledger that the fold needs — the three facts, nothing about the ledger's on-disk
// shape. The cmd/fak shell adapts the real ledger (the gateway-usage ledger, which
// records each compaction's dropped turns) into this; the dojo core never learns
// the ledger's schema.
type ContextSpanLedger struct {
	// DroppedSpans is how many context spans the ledger recorded as dropped
	// (shed/trimmed) — the recall denominator.
	DroppedSpans int
	// RestoredSpans is how many of those dropped spans a later turn paged back in
	// via fak_context_restore — the recall numerator. Meaningful only when
	// RestoreRecorded is true.
	RestoredSpans int
	// RestoreRecorded reports whether the ledger relays a restore count at all. It
	// is the honesty bit that separates a genuine "0 restored" from "the ledger
	// does not record restores": false forces UNMEASURED so a schema that logs
	// drops but not restores can never be read as a fabricated 0.0 recall.
	RestoreRecorded bool
}

// ContextRestoreEpisodes folds the context-span ledger into the dojo's
// (prediction, outcome) pair for context-restore/restore_recall — ONE episode
// scored against the registered claim. It is pure and total so the fold is
// unit-testable without a ledger on disk.
//
// Honesty rules (the confusion risk #4486 names): a ledger with no dropped spans
// yields an UNMEASURED episode (nothing to fold a recall from), and a ledger that
// records drops but relays no restore field ALSO scores UNMEASURED rather than a
// fabricated 0.0 — the restore count is not observable, not observed-as-zero. Only
// a ledger that records BOTH drops and restores produces a measured recall, clamped
// into [0,1] because a span can be restored at most once (a restored count above
// the dropped count is an adapter bug, not a recall > 1).
func ContextRestoreEpisodes(led ContextSpanLedger) []ScoredInput {
	pred := Registry.MustPredict("context-restore", "restore_recall", "fraction")
	if led.DroppedSpans <= 0 {
		return []ScoredInput{{
			Prediction: pred,
			Outcome: Outcome{
				Measured: false,
				Source:   "no dropped spans in the context-span ledger — nothing to fold a restore recall from",
			},
		}}
	}
	if !led.RestoreRecorded {
		return []ScoredInput{{
			Prediction: pred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.DroppedSpans,
				Source: fmt.Sprintf("%d dropped span(s) recorded but the ledger relays no restore field — restore recall is UNMEASURED rather than a fabricated 0.0 (the ledger cannot tell 'nothing restored' from 'restores not recorded')",
					led.DroppedSpans),
			},
		}}
	}
	restored := led.RestoredSpans
	if restored < 0 {
		restored = 0
	}
	if restored > led.DroppedSpans {
		restored = led.DroppedSpans // a span is restored at most once; keep recall in [0,1]
	}
	return []ScoredInput{{
		Prediction: pred,
		Outcome: Outcome{
			Realized:   float64(restored) / float64(led.DroppedSpans),
			Provenance: Witnessed,
			Measured:   true,
			Sample:     led.DroppedSpans,
			Source: fmt.Sprintf("%d of %d dropped spans later restored by fak_context_restore, folded from fak's own context-span ledger (WITNESSED)",
				restored, led.DroppedSpans),
		},
	}}
}
