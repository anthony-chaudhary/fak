package dojo

import "testing"

// Tests for the context-restore/restore_recall KPI cell (#4486). They pin the
// registered claim (the RSI rewrite anchor) and the pure fold's classes: the
// measured recall, the [0,1] clamp, and the two honest UNMEASURED paths (no drops,
// and drops-without-a-restore-field).

// TestContextRestoreClaimIsRegisteredEstimate pins the cell's one anchored literal.
// It is an ESTIMATE the RSI loop may recalibrate, never a floor: marking it
// IntentionalFloor would tell the loop this is a guarantee to defend rather than a
// theory to re-point at the measured fraction (#4486's confusion risk).
func TestContextRestoreClaimIsRegisteredEstimate(t *testing.T) {
	c, ok := Registry.Lookup("context-restore", "restore_recall")
	if !ok {
		t.Fatal("context-restore/restore_recall did not resolve through the composed Lookup")
	}
	if c.Claimed != 0.5 {
		t.Fatalf("pinned claim drifted: Claimed = %v, want 0.5", c.Claimed)
	}
	if c.IntentionalFloor {
		t.Fatal("the cell is an estimate the RSI loop recalibrates, but it is marked IntentionalFloor")
	}
	if c.LowerIsBetter {
		t.Fatal("a higher restore recall is the good outcome, but the cell is marked LowerIsBetter")
	}
	if c.Basis == "" {
		t.Fatal("the cell carries no prose basis")
	}
	// The cell registers additively, so the central literal must not have grown.
	if _, central := Registry[claimKey{"context-restore", "restore_recall"}]; central {
		t.Fatal("the cell landed in the central Registry literal; it must register through the additive seam")
	}
}

// TestContextRestoreEpisodeMeasuresRecall is the cell's core witness: a ledger that
// records BOTH drops and restores folds one episode carrying the registered claim
// and the realized restored/dropped fraction, WITNESSED (fak's own mechanism).
func TestContextRestoreEpisodeMeasuresRecall(t *testing.T) {
	got := ContextRestoreEpisodes(ContextSpanLedger{DroppedSpans: 10, RestoredSpans: 7, RestoreRecorded: true})
	if len(got) != 1 {
		t.Fatalf("want exactly one episode, got %d: %+v", len(got), got)
	}
	in := got[0]
	if in.Prediction.Lever != "context-restore" || in.Prediction.Metric != "restore_recall" {
		t.Fatalf("episode addresses the wrong cell: %+v", in.Prediction)
	}
	if in.Prediction.Claimed != 0.5 {
		t.Fatalf("episode must carry the ONE registered claim, got %v", in.Prediction.Claimed)
	}
	if in.Prediction.Unit != "fraction" {
		t.Fatalf("episode unit = %q, want fraction", in.Prediction.Unit)
	}
	if !in.Outcome.Measured {
		t.Fatalf("a ledger with drops AND restores must measure, got %+v", in.Outcome)
	}
	if in.Outcome.Realized != 0.7 {
		t.Fatalf("realized = %v, want 0.7 (7 of 10 dropped spans restored)", in.Outcome.Realized)
	}
	if in.Outcome.Sample != 10 {
		t.Fatalf("sample = %d, want 10 (the dropped-span denominator)", in.Outcome.Sample)
	}
	if in.Outcome.Provenance != Witnessed {
		t.Fatalf("provenance = %v, want Witnessed (fak authors both the drop and the restore)", in.Outcome.Provenance)
	}
}

// TestContextRestoreEpisodeClampsRecall pins the [0,1] clamp: a restored count above
// the dropped count is an adapter bug, so the fold reports a full 1.0 recall over
// the dropped denominator rather than a nonsensical recall > 1.
func TestContextRestoreEpisodeClampsRecall(t *testing.T) {
	got := ContextRestoreEpisodes(ContextSpanLedger{DroppedSpans: 4, RestoredSpans: 6, RestoreRecorded: true})
	if len(got) != 1 || !got[0].Outcome.Measured {
		t.Fatalf("want one measured episode, got %+v", got)
	}
	if got[0].Outcome.Realized != 1.0 {
		t.Fatalf("realized = %v, want 1.0 (restored clamped to the dropped denominator)", got[0].Outcome.Realized)
	}
}

// TestContextRestoreUnmeasured pins #4486's honesty rule: a ledger with no dropped
// spans, and a ledger that records drops but relays no restore field, BOTH score
// UNMEASURED — never a fabricated 0.0 recall that would slander the restore
// mechanism as never landing.
func TestContextRestoreUnmeasured(t *testing.T) {
	for _, tc := range []struct {
		name string
		led  ContextSpanLedger
	}{
		{"no dropped spans at all", ContextSpanLedger{}},
		{"drops recorded, no restore field", ContextSpanLedger{DroppedSpans: 12, RestoreRecorded: false}},
		{"drops recorded, restore field present but no drops", ContextSpanLedger{DroppedSpans: 0, RestoreRecorded: true}},
	} {
		got := ContextRestoreEpisodes(tc.led)
		if len(got) != 1 {
			t.Fatalf("%s: want exactly one UNMEASURED episode, got %d: %+v", tc.name, len(got), got)
		}
		if got[0].Outcome.Measured {
			t.Fatalf("%s: must score UNMEASURED, got %+v", tc.name, got[0].Outcome)
		}
		if got[0].Outcome.Realized != 0 || got[0].Outcome.Source == "" {
			t.Fatalf("%s: an UNMEASURED episode must carry no number and an honest source, got %+v", tc.name, got[0].Outcome)
		}
		if got[0].Prediction.Claimed != 0.5 {
			t.Fatalf("%s: the UNMEASURED episode must still carry the registered claim, got %+v", tc.name, got[0].Prediction)
		}
	}
}

// TestContextRestoreEpisodeScoresAsUnmeasured proves the no-restore-field episode
// survives the real scorer as an UNMEASURED verdict — the honesty rule holds
// end-to-end through Score, not just in the fold's own struct. This is the shape
// `fak dojo run` scores today, since fak's ledger records drops but no restores yet.
func TestContextRestoreEpisodeScoresAsUnmeasured(t *testing.T) {
	in := ContextRestoreEpisodes(ContextSpanLedger{DroppedSpans: 12, RestoreRecorded: false})[0]
	e := Score("context-span-ledger", in.Prediction, in.Outcome, DefaultCalibBand())
	if e.Verdict != VerdictUnmeasured {
		t.Fatalf("a drops-without-restores ledger must score %s, got %s (%+v)", VerdictUnmeasured, e.Verdict, e)
	}
}

// TestContextRestoreMeasuredScoresThroughScorer proves the measured path survives
// the real scorer as a real (non-UNMEASURED) verdict, so the cell lights up the
// moment the ledger begins recording restores.
func TestContextRestoreMeasuredScoresThroughScorer(t *testing.T) {
	in := ContextRestoreEpisodes(ContextSpanLedger{DroppedSpans: 10, RestoredSpans: 5, RestoreRecorded: true})[0]
	e := Score("context-span-ledger", in.Prediction, in.Outcome, DefaultCalibBand())
	if e.Verdict == VerdictUnmeasured {
		t.Fatalf("a drops-AND-restores ledger must produce a scored verdict, got UNMEASURED (%+v)", e)
	}
	if e.Realized != 0.5 {
		t.Fatalf("scored realized = %v, want 0.5 (5 of 10 restored)", e.Realized)
	}
}
