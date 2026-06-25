package engine_test

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// fakeCapacityKV is a stand-in abi.KVBackend for the capacity adapter. It records the
// Evict it received and answers StageSpan from a configurable outcome so the adapter's
// fail-safe ordering (stage before evict) and the typed-fault path can be exercised
// with NO live model or GPU — the same offline discipline the rest of the engine tests
// use. RestoreSpan answers a typed MISS, like the in-process default.
type fakeCapacityKV struct {
	len        int
	modelID    string
	stageOut   abi.KVResidencyOutcome
	stageErr   error
	stageBytes int64
	stageCalls int
	evicts     []struct{ from, n int }
}

func (f *fakeCapacityKV) Len() int                    { return f.len }
func (f *fakeCapacityKV) Prefill(ids []int) []float32 { return nil }
func (f *fakeCapacityKV) ModelID() string             { return f.modelID }
func (f *fakeCapacityKV) Evict(from, n int) int {
	f.evicts = append(f.evicts, struct{ from, n int }{from, n})
	return n
}
func (f *fakeCapacityKV) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	f.stageCalls++
	if f.stageErr != nil {
		return abi.KVResidency{}, f.stageErr
	}
	return abi.KVResidency{Outcome: f.stageOut, Digest: digest, Positions: n, BytesMoved: f.stageBytes}, nil
}
func (f *fakeCapacityKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

// A demote and a spill both STAGE to the colder tier then EVICT the live span, landing a
// typed HIT offload in the cache-entry stream. This is the load-bearing Plank-4 control
// path: a PlanPlacement decision turned into a real Evict + stage.
func TestCapacityAdapterExecutesDemoteAndSpill(t *testing.T) {
	for _, tc := range []struct {
		name   string
		action cachemeta.PlacementAction
		to     cachemeta.ResidencyTier
	}{
		{"demote_to_dram", cachemeta.ActionDemote, cachemeta.TierDRAM},
		{"spill_to_disk", cachemeta.ActionSpill, cachemeta.TierDisk},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kv := &fakeCapacityKV{len: 4096, modelID: "m", stageOut: abi.KVResidencyOK}
			rec := engine.NewCacheEventRecorder()
			adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}

			d := cachemeta.PlacementDecision{
				Action:       tc.action,
				FromTier:     cachemeta.TierHBM,
				ToTier:       tc.to,
				Directive:    cachemeta.KVOffload,
				EstMoveBytes: 1 << 20,
				Reason:       "beats_recompute",
			}
			res, err := adp.Execute(context.Background(), engine.PlacementMove{
				Decision: d, SpanDigest: "span-A", From: 100, N: 2048,
				ModelID: "m", PositionMode: cachemeta.PositionPrefixAligned, Owner: "kvmmu",
			})
			if err != nil {
				t.Fatalf("Execute: unexpected err %v", err)
			}
			if !res.Applied || res.Evicted != 2048 {
				t.Fatalf("demote/spill must evict the live span: Applied=%v Evicted=%d", res.Applied, res.Evicted)
			}
			if kv.stageCalls != 1 {
				t.Fatalf("expected one stage to the colder tier, got %d", kv.stageCalls)
			}
			if len(kv.evicts) != 1 || kv.evicts[0].from != 100 || kv.evicts[0].n != 2048 {
				t.Fatalf("evict not recorded as [100,+2048): %+v", kv.evicts)
			}
			// The move lands a typed HIT offload on the kv_transfer plane, to the decision's tier.
			if res.Recorded.Verdict.Kind != cachemeta.LookupHit {
				t.Fatalf("a successful demote/spill is a serveable HIT, got %s", res.Recorded.Verdict.Kind)
			}
			if res.Recorded.Entry.Residency.Tier != tc.to {
				t.Fatalf("offload residency tier = %s, want %s", res.Recorded.Entry.Residency.Tier, tc.to)
			}
			if res.Recorded.Entry.Labels["direction"] != "offload" {
				t.Fatalf("not recorded as an offload: %+v", res.Recorded.Entry.Labels)
			}
			// The in-process default StageSpan reports no bytes moved, so the decision's
			// byte ESTIMATE carries through (a real backend's measured bytes would win).
			if res.Recorded.Entry.Metrics.BytesTransferred != 1<<20 {
				t.Fatalf("bytes moved = %d, want the estimate %d", res.Recorded.Entry.Metrics.BytesTransferred, 1<<20)
			}
			if got := rec.Metrics().Snapshot().Events; got != 1 {
				t.Fatalf("expected the move folded into the metric stream (1 event), got %d", got)
			}
		})
	}
}

// A real (measured) byte count from the backend wins over the decision's estimate — the
// adapter trusts the physical plane's number once it has one.
func TestCapacityAdapterStagedBytesWinOverEstimate(t *testing.T) {
	kv := &fakeCapacityKV{len: 4096, stageOut: abi.KVResidencyOK, stageBytes: 3 << 20}
	adp := &engine.CapacityAdapter{KV: kv, Recorder: engine.NewCacheEventRecorder()}
	res, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision: cachemeta.PlacementDecision{
			Action: cachemeta.ActionDemote, FromTier: cachemeta.TierHBM, ToTier: cachemeta.TierDRAM,
			Directive: cachemeta.KVOffload, EstMoveBytes: 1 << 20,
		},
		SpanDigest: "span-B", From: 0, N: 10,
	})
	if err != nil {
		t.Fatalf("Execute: unexpected err %v", err)
	}
	if res.Recorded.Entry.Metrics.BytesTransferred != 3<<20 {
		t.Fatalf("measured stage bytes must win: got %d want %d", res.Recorded.Entry.Metrics.BytesTransferred, 3<<20)
	}
}

// Fail-safe + never-silent: a staging FAULT (outcome, or a transport error) MUST NOT
// evict the live span — the move is retained, not lost — and the fault is recorded as a
// typed FAULT(residency_fault) so a caller cannot fold it into a silent recompute.
func TestCapacityAdapterStageFaultRetainsAndRecordsFault(t *testing.T) {
	cases := []struct {
		name string
		out  abi.KVResidencyOutcome
		err  error
	}{
		{"stage_fault_outcome", abi.KVResidencyFault, nil},
		{"stage_transport_error", abi.KVResidencyOK, errors.New("rdma timeout")},
		{"stage_miss_fail_closed", abi.KVResidencyMiss, nil},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			kv := &fakeCapacityKV{len: 4096, stageOut: tc.out, stageErr: tc.err}
			rec := engine.NewCacheEventRecorder()
			adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}

			res, err := adp.Execute(context.Background(), engine.PlacementMove{
				Decision: cachemeta.PlacementDecision{
					Action: cachemeta.ActionDemote, FromTier: cachemeta.TierHBM, ToTier: cachemeta.TierCXL,
					Directive: cachemeta.KVOffload, EstMoveBytes: 1 << 20,
				},
				SpanDigest: "span-C", From: 50, N: 64,
			})
			if err != nil {
				t.Fatalf("Execute: a staging fault is a typed outcome, not a Go err: %v", err)
			}
			if res.Applied || res.Evicted != 0 {
				t.Fatalf("a failed stage must retain the live span: Applied=%v Evicted=%d", res.Applied, res.Evicted)
			}
			if len(kv.evicts) != 0 {
				t.Fatalf("a failed stage must NOT evict, got %+v", kv.evicts)
			}
			if res.Recorded.Verdict.Kind != cachemeta.LookupFault ||
				res.Recorded.Verdict.Reason != cachemeta.ReasonResidencyFault {
				t.Fatalf("staging fault must be FAULT(residency_fault), got %+v", res.Recorded.Verdict)
			}
			if !res.Recorded.SilentRecompute() {
				t.Fatal("a staging fault must be flagged non-serveable (cannot be silently recomputed)")
			}
			if got := rec.Metrics().Snapshot().Faults; got != 1 {
				t.Fatalf("expected the fault folded into metrics, got %d", got)
			}
		})
	}
}

// An evict (no colder tier had room) is the recompute-on-demand path: it skips staging
// and drops the span outright, still recording a typed offload.
func TestCapacityAdapterEvictSkipsStaging(t *testing.T) {
	kv := &fakeCapacityKV{len: 4096, stageOut: abi.KVResidencyOK}
	rec := engine.NewCacheEventRecorder()
	adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}

	res, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision: cachemeta.PlacementDecision{
			Action: cachemeta.ActionEvict, FromTier: cachemeta.TierHBM, ToTier: cachemeta.TierRecompute,
			Directive: cachemeta.KVOffload, Reason: "no_colder_tier_with_room",
		},
		SpanDigest: "span-D", From: 7, N: 9,
	})
	if err != nil {
		t.Fatalf("Execute: unexpected err %v", err)
	}
	if !res.Applied || res.Evicted != 9 {
		t.Fatalf("evict must drop the live span: Applied=%v Evicted=%d", res.Applied, res.Evicted)
	}
	if kv.stageCalls != 0 {
		t.Fatalf("an evict must not stage (recompute on demand), got %d stage calls", kv.stageCalls)
	}
	if len(kv.evicts) != 1 {
		t.Fatalf("evict must drop the span, got %+v", kv.evicts)
	}
	if res.Recorded.Verdict.Kind != cachemeta.LookupHit {
		t.Fatalf("a completed evict is a serveable offload, got %s", res.Recorded.Verdict.Kind)
	}
}

// A promote (KVRestore) is the reverse direction and a keep is a no-op: neither is this
// adapter's control path, so neither touches the live cache.
func TestCapacityAdapterPromoteAndKeepNotApplied(t *testing.T) {
	for _, action := range []cachemeta.PlacementAction{cachemeta.ActionPromote, cachemeta.ActionKeep} {
		kv := &fakeCapacityKV{len: 4096, stageOut: abi.KVResidencyOK}
		adp := &engine.CapacityAdapter{KV: kv}
		res, err := adp.Execute(context.Background(), engine.PlacementMove{
			Decision:   cachemeta.PlacementDecision{Action: action, FromTier: cachemeta.TierDRAM, ToTier: cachemeta.TierHBM},
			SpanDigest: "span-E", From: 0, N: 4,
		})
		if err != nil {
			t.Fatalf("%s: unexpected err %v", action, err)
		}
		if res.Applied {
			t.Fatalf("%s must not be applied by this adapter", action)
		}
		if kv.stageCalls != 0 || len(kv.evicts) != 0 {
			t.Fatalf("%s must not touch the live cache: stage=%d evicts=%v", action, kv.stageCalls, kv.evicts)
		}
	}
}

// A nil KV backend is a typed error, not a nil-deref — the adapter cannot execute
// against a cache it does not hold.
func TestCapacityAdapterNilKVIsTypedError(t *testing.T) {
	adp := &engine.CapacityAdapter{Recorder: engine.NewCacheEventRecorder()}
	if _, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision: cachemeta.PlacementDecision{Action: cachemeta.ActionDemote},
	}); err == nil {
		t.Fatal("expected a typed error for a nil KV backend")
	}
}

// A CapacityAdapter satisfies the compile-time shape the kernel calls: it holds the live
// KVBackend and the recorder, and Execute is the control-path entry point.
var _ = (*engine.CapacityAdapter)(nil)
