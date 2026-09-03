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
	len              int
	modelID          string
	stageOut         abi.KVResidencyOutcome
	stageErr         error
	stageBytes       int64
	stageCalls       int
	restoreOut       abi.KVResidencyOutcome
	restoreErr       error
	restoreBytes     int64
	restorePositions int
	restoreCalls     int
	evicts           []struct{ from, n int }
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
	f.restoreCalls++
	if f.restoreErr != nil {
		return abi.KVResidency{}, f.restoreErr
	}
	out := f.restoreOut
	if out == abi.KVResidencyUnknown {
		out = abi.KVResidencyMiss
	}
	pos := f.restorePositions
	if pos == 0 && out == abi.KVResidencyOK {
		pos = 4
	}
	return abi.KVResidency{Outcome: out, Digest: digest, Positions: pos, BytesMoved: f.restoreBytes}, nil
}

type placementAwareKV struct {
	*fakeCapacityKV
	tier        string
	evictDigest string
	evicted     int
}

func (f *placementAwareKV) StageSpanTo(ctx context.Context, digest string, from, n int, tier string) (abi.KVResidency, error) {
	f.tier = tier
	return f.fakeCapacityKV.StageSpan(ctx, digest, from, n)
}

func (f *placementAwareKV) EvictDigest(digest string, _, n int) int {
	f.evictDigest = digest
	if f.evicted >= 0 {
		return f.evicted
	}
	return n
}

// A demote, spill, and compress-demote all STAGE to the colder tier then EVICT the live
// span, landing a typed HIT offload in the cache-entry stream. This is the load-bearing
// Plank-4 control path: a PlanPlacement decision turned into a real Evict + stage.
func TestCapacityAdapterExecutesDemoteAndSpill(t *testing.T) {
	for _, tc := range []struct {
		name   string
		action cachemeta.PlacementAction
		to     cachemeta.ResidencyTier
	}{
		{"demote_to_dram", cachemeta.ActionDemote, cachemeta.TierDRAM},
		{"spill_to_disk", cachemeta.ActionSpill, cachemeta.TierDisk},
		{"compress_demote_to_disk", cachemeta.ActionCompressDemote, cachemeta.TierDisk},
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

func TestCapacityAdapterPassesTargetTierAndDigestToPhysicalOwner(t *testing.T) {
	kv := &placementAwareKV{
		fakeCapacityKV: &fakeCapacityKV{stageOut: abi.KVResidencyOK, stageBytes: 4096},
		evicted:        -1,
	}
	adp := &engine.CapacityAdapter{KV: kv}
	res, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision: cachemeta.PlacementDecision{
			Action: cachemeta.ActionDemote, FromTier: cachemeta.TierHBM, ToTier: cachemeta.TierDRAM,
			Directive: cachemeta.KVOffload, EstMoveBytes: 2048,
		},
		SpanDigest: "complete-prefix", From: 0, N: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Applied || res.Evicted != 32 {
		t.Fatalf("result=%+v", res)
	}
	if kv.tier != string(cachemeta.TierDRAM) {
		t.Fatalf("stage target tier=%q, want %q", kv.tier, cachemeta.TierDRAM)
	}
	if kv.evictDigest != "complete-prefix" {
		t.Fatalf("digest eviction=%q, want complete-prefix", kv.evictDigest)
	}
	if len(kv.evicts) != 0 {
		t.Fatalf("range eviction ran instead of digest eviction: %+v", kv.evicts)
	}
}

func TestCapacityAdapterDoesNotClaimDigestReclaimWhenOwnerRefusesEviction(t *testing.T) {
	kv := &placementAwareKV{
		fakeCapacityKV: &fakeCapacityKV{stageOut: abi.KVResidencyOK, stageBytes: 4096},
		evicted:        0,
	}
	rec := engine.NewCacheEventRecorder()
	adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}
	res, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision: cachemeta.PlacementDecision{
			Action: cachemeta.ActionDemote, FromTier: cachemeta.TierHBM, ToTier: cachemeta.TierDRAM,
			Directive: cachemeta.KVOffload, EstMoveBytes: 2048,
		},
		SpanDigest: "complete-prefix", From: 0, N: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Applied || res.Evicted != 0 {
		t.Fatalf("refused digest eviction claimed reclaim: %+v", res)
	}
	if res.Recorded.Verdict.Kind != cachemeta.LookupFault {
		t.Fatalf("refused digest eviction verdict=%+v, want fault", res.Recorded.Verdict)
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

// A keep is a no-op: it is not executed and does not touch the live cache.
func TestCapacityAdapterKeepNotApplied(t *testing.T) {
	kv := &fakeCapacityKV{len: 4096, stageOut: abi.KVResidencyOK}
	adp := &engine.CapacityAdapter{KV: kv}
	res, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision:   cachemeta.PlacementDecision{Action: cachemeta.ActionKeep, FromTier: cachemeta.TierDRAM, ToTier: cachemeta.TierHBM},
		SpanDigest: "span-E", From: 0, N: 4,
	})
	if err != nil {
		t.Fatalf("Keep: unexpected err %v", err)
	}
	if res.Applied {
		t.Fatalf("Keep must not be applied by this adapter")
	}
	if kv.stageCalls != 0 || len(kv.evicts) != 0 || kv.restoreCalls != 0 {
		t.Fatalf("Keep must not touch the live cache: stage=%d evicts=%v restore=%d", kv.stageCalls, kv.evicts, kv.restoreCalls)
	}
}

// A promote (KVRestore) executes RestoreSpan via RestoreAdapter, recording a typed KVRestore event (#1469).
func TestCapacityAdapterPromoteExecutesRestore(t *testing.T) {
	t.Run("promote_success", func(t *testing.T) {
		kv := &fakeCapacityKV{len: 4096, restoreOut: abi.KVResidencyOK, restoreBytes: 1024, restorePositions: 4}
		rec := engine.NewCacheEventRecorder()
		adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}
		res, err := adp.Execute(context.Background(), engine.PlacementMove{
			Decision:   cachemeta.PlacementDecision{Action: cachemeta.ActionPromote, FromTier: cachemeta.TierDRAM, ToTier: cachemeta.TierHBM},
			SpanDigest: "span-promote", From: 0, N: 4, ModelID: "m",
		})
		if err != nil {
			t.Fatalf("Promote: unexpected err %v", err)
		}
		if !res.Applied {
			t.Fatalf("Promote with OK outcome must be applied")
		}
		if kv.restoreCalls != 1 {
			t.Fatalf("expected 1 restore call, got %d", kv.restoreCalls)
		}
		if res.Recorded.Verdict.Kind != cachemeta.LookupHit {
			t.Fatalf("expected LookupHit verdict, got %s", res.Recorded.Verdict.Kind)
		}
	})

	t.Run("promote_miss", func(t *testing.T) {
		kv := &fakeCapacityKV{len: 4096, restoreOut: abi.KVResidencyMiss}
		rec := engine.NewCacheEventRecorder()
		adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}
		res, err := adp.Execute(context.Background(), engine.PlacementMove{
			Decision:   cachemeta.PlacementDecision{Action: cachemeta.ActionPromote, FromTier: cachemeta.TierDRAM, ToTier: cachemeta.TierHBM},
			SpanDigest: "span-miss", From: 0, N: 4, ModelID: "m",
		})
		if err != nil {
			t.Fatalf("Promote: unexpected err %v", err)
		}
		if res.Applied {
			t.Fatalf("Promote with Miss outcome must not be applied")
		}
		if res.Recorded.Verdict.Reason != cachemeta.ReasonRestoreMiss {
			t.Fatalf("expected ReasonRestoreMiss, got %s", res.Recorded.Verdict.Reason)
		}
	})

	t.Run("promote_fault", func(t *testing.T) {
		kv := &fakeCapacityKV{len: 4096, restoreErr: errors.New("io error")}
		rec := engine.NewCacheEventRecorder()
		adp := &engine.CapacityAdapter{KV: kv, Recorder: rec}
		res, err := adp.Execute(context.Background(), engine.PlacementMove{
			Decision:   cachemeta.PlacementDecision{Action: cachemeta.ActionPromote, FromTier: cachemeta.TierDRAM, ToTier: cachemeta.TierHBM},
			SpanDigest: "span-fault", From: 0, N: 4, ModelID: "m",
		})
		if err != nil {
			t.Fatalf("Promote: unexpected err %v", err)
		}
		if res.Applied {
			t.Fatalf("Promote with Fault outcome must not be applied")
		}
		if res.Recorded.Verdict.Reason != cachemeta.ReasonResidencyFault {
			t.Fatalf("expected ReasonResidencyFault, got %s", res.Recorded.Verdict.Reason)
		}
	})
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
