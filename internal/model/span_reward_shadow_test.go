package model

import (
	"math"
	"testing"
)

func spanRewardShadowModel(t *testing.T) *Model {
	t.Helper()
	cfg := Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1, TieWordEmbeddings: true,
	}
	return NewSynthetic(cfg)
}

func TestShadowSpanRewardLeaveOneOutRecordsRewardAndEvicts(t *testing.T) {
	m := spanRewardShadowModel(t)
	rec := RecordedSpanRewardSession{
		ContextIDs: []int{1, 5, 2, 7, 3, 9, 4, 8},
		ProbeIDs:   []int{6, 10},
		Spans: []SpanRewardSegment{
			{ID: "span:a", From: 0, Len: 2, Age: 3},
			{ID: "span:b", From: 2, Len: 2, Age: 2},
			{ID: "span:c", From: 4, Len: 2, Age: 1},
			{ID: "span:d", From: 6, Len: 2, Age: 0},
		},
		SuccessGate:  0.75,
		RecencyGamma: 0.9,
	}

	report, err := m.ShadowSpanRewardLeaveOneOut(rec, SpanRewardOptions{CorrelationThreshold: 0.5})
	if err != nil {
		t.Fatalf("ShadowSpanRewardLeaveOneOut: %v", err)
	}
	if m.AttnObserverSet() {
		t.Fatalf("shadow scorer did not restore the model attention observer")
	}
	if len(report.Rows) != len(rec.Spans) {
		t.Fatalf("rows = %d, want %d", len(report.Rows), len(rec.Spans))
	}
	if report.CheckedRows != len(rec.Spans) {
		t.Fatalf("checked rows = %d, want all %d", report.CheckedRows, len(rec.Spans))
	}
	if report.Verdict != SpanRewardCorrelate && report.Verdict != SpanRewardRefute {
		t.Fatalf("verdict = %q, want correlate or refute", report.Verdict)
	}
	if !report.SpearmanDefined {
		t.Fatalf("reward-vs-LOO Spearman should be defined for this recorded session: %+v", report)
	}

	seenRank := map[int]bool{}
	var tail *SpanRewardRow
	for i := range report.Rows {
		row := &report.Rows[i]
		if row.RawAttentionMass < 0 || row.RawAttentionMass > 1 {
			t.Fatalf("%s raw mass = %.6f outside [0,1]", row.ID, row.RawAttentionMass)
		}
		if row.SuccessGate != 0.75 {
			t.Fatalf("%s success gate = %.3f, want 0.75", row.ID, row.SuccessGate)
		}
		if row.Rank < 1 || row.Rank > len(rec.Spans) || seenRank[row.Rank] {
			t.Fatalf("%s invalid/non-unique rank %d in %+v", row.ID, row.Rank, report.Rows)
		}
		seenRank[row.Rank] = true
		if !row.LeaveOneOutChecked || row.EvictRemoved != row.Len {
			t.Fatalf("%s LOO not checked via exact eviction: %+v", row.ID, row)
		}
		if row.EvictRepositionMaxDiff > fmaCrossPathTol {
			t.Fatalf("%s eviction reposition residual %.3e > %.0e", row.ID, row.EvictRepositionMaxDiff, fmaCrossPathTol)
		}
		if row.LeaveOneOutDelta < 0 || math.IsNaN(row.LeaveOneOutDelta) || math.IsInf(row.LeaveOneOutDelta, 0) {
			t.Fatalf("%s invalid LOO delta %.6f", row.ID, row.LeaveOneOutDelta)
		}
		if row.ID == "span:d" {
			tail = row
		}
	}
	if tail == nil {
		t.Fatalf("missing tail span row")
	}
	if tail.EvictVsRecomputeMaxDiff != 0 {
		t.Fatalf("tail-span eviction should match a full no-span recompute at max|delta|=0, got %.3e", tail.EvictVsRecomputeMaxDiff)
	}
}

func TestSpanRewardSummaryRequiresNormalizedCorrelation(t *testing.T) {
	rows := []SpanRewardRow{
		{ID: "sink", RawAttentionMass: 0.9, NormalizedReward: 0.1, LeaveOneOutChecked: true, LeaveOneOutDelta: 0.1},
		{ID: "mid", RawAttentionMass: 0.6, NormalizedReward: 0.5, LeaveOneOutChecked: true, LeaveOneOutDelta: 0.5},
		{ID: "useful", RawAttentionMass: 0.1, NormalizedReward: 0.9, LeaveOneOutChecked: true, LeaveOneOutDelta: 0.9},
	}
	report := summarizeSpanRewardReport(rows, 0.8)

	if report.Verdict != SpanRewardCorrelate {
		t.Fatalf("verdict = %q, want CORRELATE: %+v", report.Verdict, report)
	}
	if !report.ConfoundNormalizationImproved {
		t.Fatalf("normalized reward should improve over raw attention: %+v", report)
	}
	if math.Abs(report.SpearmanRewardDelta-1) > 1e-9 {
		t.Fatalf("normalized Spearman = %.6f, want 1", report.SpearmanRewardDelta)
	}
	if math.Abs(report.SpearmanRawDelta+1) > 1e-9 {
		t.Fatalf("raw Spearman = %.6f, want -1", report.SpearmanRawDelta)
	}
}

func TestShadowSpanRewardTopBottomKChecksOnlyExtremes(t *testing.T) {
	m := spanRewardShadowModel(t)
	rec := RecordedSpanRewardSession{
		ContextIDs:  []int{1, 5, 2, 7, 3, 9, 4, 8},
		ProbeIDs:    []int{6, 10},
		SuccessGate: 1,
		Spans: []SpanRewardSegment{
			{ID: "span:a", From: 0, Len: 2},
			{ID: "span:b", From: 2, Len: 2},
			{ID: "span:c", From: 4, Len: 2},
			{ID: "span:d", From: 6, Len: 2},
		},
	}
	report, err := m.ShadowSpanRewardLeaveOneOut(rec, SpanRewardOptions{TopBottomK: 1})
	if err != nil {
		t.Fatalf("ShadowSpanRewardLeaveOneOut: %v", err)
	}
	if report.CheckedRows != 2 {
		t.Fatalf("TopBottomK=1 checked %d rows, want 2", report.CheckedRows)
	}
	for _, row := range report.Rows {
		if row.Rank == 1 || row.Rank == len(report.Rows) {
			if !row.LeaveOneOutChecked {
				t.Fatalf("extreme rank row not checked: %+v", row)
			}
			continue
		}
		if row.LeaveOneOutChecked {
			t.Fatalf("non-extreme rank row was checked under TopBottomK=1: %+v", row)
		}
	}
}
