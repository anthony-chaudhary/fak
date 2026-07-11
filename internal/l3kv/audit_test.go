package l3kv

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// recordingKV is a fake abi.KVBackend that records the args each method was called
// with and returns caller-configured sentinels — so a test can prove AuditKV
// forwards the right args and passes the inner return through unchanged.
type recordingKV struct {
	// recorded call args
	prefillIDs    []int
	evictFrom     int
	evictN        int
	stageCtx      context.Context
	stageDigest   string
	stageFrom     int
	stageN        int
	restoreCtx    context.Context
	restoreDigest string

	// per-op call counts
	calls map[string]int

	// configured returns
	lenRet     int
	prefillRet []float32
	evictRet   int
	modelIDRet string
	stageRet   abi.KVResidency
	stageErr   error
	restoreRet abi.KVResidency
	restoreErr error
}

func newRecordingKV() *recordingKV { return &recordingKV{calls: map[string]int{}} }

func (m *recordingKV) Len() int {
	m.calls[OpLen]++
	return m.lenRet
}

func (m *recordingKV) Prefill(ids []int) []float32 {
	m.calls[OpPrefill]++
	m.prefillIDs = ids
	return m.prefillRet
}

func (m *recordingKV) Evict(from, n int) int {
	m.calls[OpEvict]++
	m.evictFrom, m.evictN = from, n
	return m.evictRet
}

func (m *recordingKV) ModelID() string {
	m.calls[OpModelID]++
	return m.modelIDRet
}

func (m *recordingKV) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	m.calls[OpStageSpan]++
	m.stageCtx, m.stageDigest, m.stageFrom, m.stageN = ctx, digest, from, n
	return m.stageRet, m.stageErr
}

func (m *recordingKV) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	m.calls[OpRestoreSpan]++
	m.restoreCtx, m.restoreDigest = ctx, digest
	return m.restoreRet, m.restoreErr
}

// TestAuditKVForwardsAndRecords is the #3388 witness: for every method, AuditKV
// (1) calls the inner backend exactly once with the right args, (2) passes the
// inner return through unchanged, and (3) records exactly one entry with the
// correct op name and the documented bytes estimate.
func TestAuditKVForwardsAndRecords(t *testing.T) {
	ctx := context.Background()
	logits := []float32{0.1, 0.2, 0.3, 0.4, 0.5} // len 5 → 20 bytes estimate
	stageRes := abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: "d", Positions: 20, BytesMoved: 4096}
	restoreRes := abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: "d", BytesMoved: 8192}
	stageErr := errors.New("stage boom")
	restoreErr := errors.New("restore boom")

	inner := newRecordingKV()
	inner.lenRet = 42
	inner.prefillRet = logits
	inner.evictRet = 7
	inner.modelIDRet = "mock-model" // len 10
	inner.stageRet, inner.stageErr = stageRes, stageErr
	inner.restoreRet, inner.restoreErr = restoreRes, restoreErr

	rec := NewMemRecorder()
	a := NewAuditKV(inner, rec)

	cases := []struct {
		name      string
		wantOp    string
		wantBytes int64
		// invoke exercises the op and asserts forwarding + return passthrough.
		invoke func(t *testing.T)
	}{
		{
			name: "Len", wantOp: OpLen, wantBytes: 0,
			invoke: func(t *testing.T) {
				if got := a.Len(); got != 42 {
					t.Fatalf("Len passthrough = %d, want 42", got)
				}
			},
		},
		{
			name: "Prefill", wantOp: OpPrefill, wantBytes: int64(len(logits)) * 4,
			invoke: func(t *testing.T) {
				ids := []int{1, 2, 3}
				got := a.Prefill(ids)
				if !reflect.DeepEqual(got, logits) {
					t.Fatalf("Prefill passthrough = %v, want %v", got, logits)
				}
				if !reflect.DeepEqual(inner.prefillIDs, ids) {
					t.Fatalf("Prefill forwarded ids = %v, want %v", inner.prefillIDs, ids)
				}
			},
		},
		{
			name: "Evict", wantOp: OpEvict, wantBytes: 7,
			invoke: func(t *testing.T) {
				if got := a.Evict(3, 9); got != 7 {
					t.Fatalf("Evict passthrough = %d, want 7", got)
				}
				if inner.evictFrom != 3 || inner.evictN != 9 {
					t.Fatalf("Evict forwarded (from,n) = (%d,%d), want (3,9)", inner.evictFrom, inner.evictN)
				}
			},
		},
		{
			name: "ModelID", wantOp: OpModelID, wantBytes: int64(len("mock-model")),
			invoke: func(t *testing.T) {
				if got := a.ModelID(); got != "mock-model" {
					t.Fatalf("ModelID passthrough = %q, want mock-model", got)
				}
			},
		},
		{
			name: "StageSpan", wantOp: OpStageSpan, wantBytes: stageRes.BytesMoved,
			invoke: func(t *testing.T) {
				res, err := a.StageSpan(ctx, "digestX", 10, 20)
				if res != stageRes || !errors.Is(err, stageErr) {
					t.Fatalf("StageSpan passthrough = (%+v, %v), want (%+v, %v)", res, err, stageRes, stageErr)
				}
				if inner.stageDigest != "digestX" || inner.stageFrom != 10 || inner.stageN != 20 {
					t.Fatalf("StageSpan forwarded (digest,from,n) = (%q,%d,%d), want (digestX,10,20)",
						inner.stageDigest, inner.stageFrom, inner.stageN)
				}
				if inner.stageCtx != ctx {
					t.Fatalf("StageSpan did not forward ctx")
				}
			},
		},
		{
			name: "RestoreSpan", wantOp: OpRestoreSpan, wantBytes: restoreRes.BytesMoved,
			invoke: func(t *testing.T) {
				res, err := a.RestoreSpan(ctx, "digestY")
				if res != restoreRes || !errors.Is(err, restoreErr) {
					t.Fatalf("RestoreSpan passthrough = (%+v, %v), want (%+v, %v)", res, err, restoreRes, restoreErr)
				}
				if inner.restoreDigest != "digestY" {
					t.Fatalf("RestoreSpan forwarded digest = %q, want digestY", inner.restoreDigest)
				}
				if inner.restoreCtx != ctx {
					t.Fatalf("RestoreSpan did not forward ctx")
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := rec.Len()
			tc.invoke(t)

			// The inner backend was called exactly once for this op.
			if inner.calls[tc.wantOp] != 1 {
				t.Fatalf("inner.%s call count = %d, want 1", tc.wantOp, inner.calls[tc.wantOp])
			}
			// Exactly one entry was recorded for this call.
			entries := rec.Entries()
			if len(entries) != before+1 {
				t.Fatalf("recorded %d entries, want exactly one more than %d", len(entries), before)
			}
			got := entries[len(entries)-1]
			if got.Op != tc.wantOp {
				t.Fatalf("recorded op = %q, want %q", got.Op, tc.wantOp)
			}
			if got.Bytes != tc.wantBytes {
				t.Fatalf("recorded bytes = %d, want %d", got.Bytes, tc.wantBytes)
			}
			if got.Dur < 0 {
				t.Fatalf("recorded negative duration %v", got.Dur)
			}
		})
	}

	// One call per op, six ops → exactly six recorded entries total.
	if n := rec.Len(); n != len(cases) {
		t.Fatalf("total recorded entries = %d, want %d", n, len(cases))
	}
}

// TestNilRecorderDefaultsToNop proves the decorator is safe to construct with a nil
// recorder (zero-overhead default) and still forwards results unchanged.
func TestNilRecorderDefaultsToNop(t *testing.T) {
	inner := newRecordingKV()
	inner.lenRet = 5
	a := NewAuditKV(inner, nil) // nil recorder → NopRecorder, must not panic
	if got := a.Len(); got != 5 {
		t.Fatalf("Len with nil recorder = %d, want 5", got)
	}
	if inner.calls[OpLen] != 1 {
		t.Fatalf("inner.Len call count = %d, want 1", inner.calls[OpLen])
	}
}

// TestAuditKVIsInterfaceIdentical proves an *AuditKV is a drop-in abi.KVBackend and
// composes over another abi.KVBackend (e.g. an l3kv-wrapped one) transparently.
func TestAuditKVIsInterfaceIdentical(t *testing.T) {
	var kv abi.KVBackend = NewAuditKV(newRecordingKV(), NewMemRecorder())
	// Wrapping an AuditKV in the durable l3kv backend (itself an abi.KVBackend)
	// must type-check — the decorator is interface-identical at both seams.
	_ = New(kv, newMemStore())
}
