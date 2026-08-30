package abi

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type batchTestBackend struct {
	stageResults   []KVResidency
	restoreResults []KVResidency
	stageErr       error
	restoreErr     error
	stageCalls     []KVResidencyRequest
	restoreCalls   []KVResidencyRequest
	cancelAfter    int
	cancel         context.CancelFunc
}

func (*batchTestBackend) Len() int                { return 0 }
func (*batchTestBackend) Prefill([]int) []float32 { return nil }
func (*batchTestBackend) Evict(int, int) int      { return 0 }
func (*batchTestBackend) ModelID() string         { return "batch-test" }
func (b *batchTestBackend) StageSpan(_ context.Context, digest string, from, n int) (KVResidency, error) {
	b.stageCalls = append(b.stageCalls, KVResidencyRequest{Digest: digest, From: from, Positions: n})
	i := len(b.stageCalls) - 1
	if b.cancelAfter > 0 && len(b.stageCalls) == b.cancelAfter && b.cancel != nil {
		b.cancel()
	}
	if b.stageErr != nil {
		return KVResidency{}, b.stageErr
	}
	return b.stageResults[i], nil
}
func (b *batchTestBackend) RestoreSpan(_ context.Context, digest string) (KVResidency, error) {
	b.restoreCalls = append(b.restoreCalls, KVResidencyRequest{Digest: digest})
	i := len(b.restoreCalls) - 1
	if b.restoreErr != nil {
		return KVResidency{}, b.restoreErr
	}
	return b.restoreResults[i], nil
}

type nativeBatchBackend struct {
	batchTestBackend
	stageBatchCalls   int
	restoreBatchCalls int
}

func (b *nativeBatchBackend) StageSpans(_ context.Context, reqs []KVResidencyRequest) []KVResidency {
	b.stageBatchCalls++
	b.stageCalls = append(b.stageCalls, reqs...)
	return b.stageResults
}
func (b *nativeBatchBackend) RestoreSpans(_ context.Context, reqs []KVResidencyRequest) []KVResidency {
	b.restoreBatchCalls++
	b.restoreCalls = append(b.restoreCalls, reqs...)
	return b.restoreResults
}

var batchRequests = []KVResidencyRequest{
	{Digest: "a", From: 0, Positions: 2},
	{Digest: "b", From: 2, Positions: 3},
	{Digest: "c", From: 5, Positions: 4},
}

func receipt(req KVResidencyRequest, outcome KVResidencyOutcome) KVResidency {
	return KVResidency{Outcome: outcome, Digest: req.Digest, Positions: req.Positions}
}

func TestStageSpansSerialFallbackPreservesOrderAndOutcomes(t *testing.T) {
	b := &batchTestBackend{stageResults: []KVResidency{
		receipt(batchRequests[0], KVResidencyOK), receipt(batchRequests[1], KVResidencyMiss), receipt(batchRequests[2], KVResidencyFault),
	}}
	got := StageSpans(context.Background(), b, batchRequests)
	if len(got) != len(batchRequests) || len(b.stageCalls) != len(batchRequests) {
		t.Fatalf("receipts/calls = %d/%d, want %d/%d", len(got), len(b.stageCalls), len(batchRequests), len(batchRequests))
	}
	for i, want := range []KVResidencyOutcome{KVResidencyOK, KVResidencyMiss, KVResidencyFault} {
		if got[i].Digest != batchRequests[i].Digest || got[i].Outcome != want {
			t.Fatalf("receipt[%d] = %+v, want digest %q outcome %v", i, got[i], batchRequests[i].Digest, want)
		}
	}
}

func TestBatchExtensionsAreSelected(t *testing.T) {
	b := &nativeBatchBackend{batchTestBackend: batchTestBackend{
		stageResults:   []KVResidency{receipt(batchRequests[0], KVResidencyOK)},
		restoreResults: []KVResidency{receipt(batchRequests[0], KVResidencyMiss)},
	}}
	StageSpans(context.Background(), b, batchRequests[:1])
	RestoreSpans(context.Background(), b, batchRequests[:1])
	if b.stageBatchCalls != 1 || b.restoreBatchCalls != 1 {
		t.Fatalf("native batch calls stage/restore = %d/%d, want 1/1", b.stageBatchCalls, b.restoreBatchCalls)
	}
	// Native methods record requests themselves; legacy methods would add a second call.
	if len(b.stageCalls) != 1 || len(b.restoreCalls) != 1 {
		t.Fatalf("legacy fallback unexpectedly called: stage/restore records = %d/%d", len(b.stageCalls), len(b.restoreCalls))
	}
}

func TestBatchNormalizationFailsClosed(t *testing.T) {
	tests := []struct {
		name       string
		results    []KVResidency
		wantFaults []bool
		prefix     string
	}{
		{"short", []KVResidency{receipt(batchRequests[0], KVResidencyOK)}, []bool{false, true, true}, "abi: kv batch cardinality mismatch"},
		{"extra", append([]KVResidency{receipt(batchRequests[0], KVResidencyOK), receipt(batchRequests[1], KVResidencyOK), receipt(batchRequests[2], KVResidencyOK)}, KVResidency{}), []bool{true, true, true}, "abi: kv batch cardinality mismatch"},
		{"digest", []KVResidency{{Outcome: KVResidencyOK, Digest: "wrong", Positions: 2}, receipt(batchRequests[1], KVResidencyOK), receipt(batchRequests[2], KVResidencyOK)}, []bool{true, false, false}, "abi: kv batch digest mismatch"},
		{"positions", []KVResidency{{Outcome: KVResidencyOK, Digest: "a", Positions: 99}, receipt(batchRequests[1], KVResidencyOK), receipt(batchRequests[2], KVResidencyOK)}, []bool{true, false, false}, "abi: kv batch position mismatch"},
		{"unknown", []KVResidency{{Outcome: KVResidencyUnknown, Digest: "a", Positions: 2}, receipt(batchRequests[1], KVResidencyOK), receipt(batchRequests[2], KVResidencyOK)}, []bool{true, false, false}, "abi: kv batch unknown outcome"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := &nativeBatchBackend{batchTestBackend: batchTestBackend{stageResults: tt.results}}
			got := StageSpans(context.Background(), b, batchRequests)
			if len(got) != len(batchRequests) {
				t.Fatalf("len = %d, want %d", len(got), len(batchRequests))
			}
			for i, wantFault := range tt.wantFaults {
				if (got[i].Outcome == KVResidencyFault) != wantFault {
					t.Fatalf("receipt[%d] = %+v, want fault=%v", i, got[i], wantFault)
				}
				if wantFault && i == 0 && !strings.HasPrefix(got[i].Reason, tt.prefix) {
					t.Fatalf("reason = %q, want prefix %q", got[i].Reason, tt.prefix)
				}
			}
		})
	}
}

func TestRestoreSpansFillsLegacyMissingPositionsOnNonOK(t *testing.T) {
	b := &batchTestBackend{restoreResults: []KVResidency{
		{Outcome: KVResidencyMiss, Digest: "a"}, {Outcome: KVResidencyFault, Digest: "b"}, receipt(batchRequests[2], KVResidencyOK),
	}}
	got := RestoreSpans(context.Background(), b, batchRequests)
	for i := range got {
		if got[i].Positions != batchRequests[i].Positions {
			t.Fatalf("receipt[%d].Positions = %d, want %d", i, got[i].Positions, batchRequests[i].Positions)
		}
	}
}

func TestBatchCancellation(t *testing.T) {
	t.Run("before start", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		b := &batchTestBackend{}
		got := StageSpans(ctx, b, batchRequests)
		if len(b.stageCalls) != 0 {
			t.Fatalf("backend called %d times", len(b.stageCalls))
		}
		assertFaultPrefix(t, got, "abi: kv batch canceled")
	})
	t.Run("during serial fallback", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		b := &batchTestBackend{stageResults: []KVResidency{receipt(batchRequests[0], KVResidencyOK)}, cancelAfter: 1, cancel: cancel}
		got := StageSpans(ctx, b, batchRequests)
		if len(b.stageCalls) != 1 || got[0].Outcome != KVResidencyOK {
			t.Fatalf("completed receipt/calls = %+v/%d", got[0], len(b.stageCalls))
		}
		assertFaultPrefix(t, got[1:], "abi: kv batch canceled")
	})
}

func TestBatchLegacyErrorsNilBackendAndEmpty(t *testing.T) {
	b := &batchTestBackend{stageErr: errors.New("stage broke"), restoreErr: errors.New("restore broke")}
	assertFaultPrefix(t, StageSpans(context.Background(), b, batchRequests[:1]), "abi: kv batch stage: stage broke")
	assertFaultPrefix(t, RestoreSpans(context.Background(), b, batchRequests[:1]), "abi: kv batch restore: restore broke")
	assertFaultPrefix(t, StageSpans(context.Background(), nil, batchRequests[:1]), "abi: kv batch nil backend")
	var typedNil *batchTestBackend
	assertFaultPrefix(t, RestoreSpans(context.Background(), typedNil, batchRequests[:1]), "abi: kv batch nil backend")
	if got := StageSpans(context.Background(), b, nil); len(got) != 0 || len(b.stageCalls) != 1 {
		t.Fatalf("empty stage returned %d receipts or called backend again", len(got))
	}
	if got := RestoreSpans(nil, b, nil); len(got) != 0 || len(b.restoreCalls) != 1 {
		t.Fatalf("empty restore returned %d receipts or called backend again", len(got))
	}
	assertFaultPrefix(t, RestoreSpans(nil, b, batchRequests[:1]), "abi: kv batch canceled")
}

func assertFaultPrefix(t *testing.T, got []KVResidency, prefix string) {
	t.Helper()
	for i, res := range got {
		if res.Outcome != KVResidencyFault || !strings.HasPrefix(res.Reason, prefix) {
			t.Fatalf("receipt[%d] = %+v, want fault prefix %q", i, res, prefix)
		}
	}
}
