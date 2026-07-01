package engine_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// A routing/offload/restore sequence must be normalized into the SAME cache-entry
// stream (plane kv_transfer) as tool/context entries, with residency tier + owner
// recorded separately from the payload.
func TestCacheEventRecorderNormalizesIntoCacheEntryStream(t *testing.T) {
	rec := engine.NewCacheEventRecorder()

	off := rec.Record(engine.CacheEvent{
		Direction:  cachemeta.KVOffload,
		SpanDigest: "span-1",
		Tokens:     2048,
		ModelID:    "m",
		FromTier:   cachemeta.TierHBM,
		ToTier:     cachemeta.TierDRAM,
		Owner:      "kvbm",
		BytesMoved: 1 << 20,
		Outcome:    cachemeta.KVTransferOK,
	})
	if off.Entry.Plane != cachemeta.PlaneKVTransfer || off.Entry.ID.MediaType != cachemeta.MediaKVSpan {
		t.Fatalf("offload not on the kv_transfer plane: %+v", off.Entry)
	}
	// Residency tier + owner separate from payload.
	if off.Entry.Residency.Tier != cachemeta.TierDRAM || off.Entry.Residency.Owner != "kvbm" {
		t.Fatalf("residency not recorded separately: %+v", off.Entry.Residency)
	}
	if off.Entry.Labels["direction"] != "offload" || off.Entry.Labels["to_tier"] != "dram" {
		t.Fatalf("transition labels missing: %+v", off.Entry.Labels)
	}
	if off.Verdict.Kind != cachemeta.LookupHit {
		t.Fatalf("ok offload should HIT, got %s", off.Verdict.Kind)
	}

	rt := rec.Record(engine.CacheEvent{
		Direction: cachemeta.KVRoute, SpanDigest: "span-1", ToTier: cachemeta.TierRemote,
		Owner: "router", Outcome: cachemeta.KVTransferOK,
	})
	if rt.Entry.Labels["direction"] != "route" || rt.Verdict.Kind != cachemeta.LookupHit {
		t.Fatalf("route not normalized: %+v / %s", rt.Entry.Labels, rt.Verdict.Kind)
	}

	if got := rec.Metrics().Snapshot().Events; got != 2 {
		t.Fatalf("expected 2 normalized events, got %d", got)
	}
}

// §2.2 acceptance: a failure to restore/load KV is a typed MISS or FAULT, never a
// silent recompute. SilentRecompute() flags any non-Hit so the caller cannot fold a
// fault away.
func TestCacheEventRestoreFaultIsTypedNeverSilent(t *testing.T) {
	rec := engine.NewCacheEventRecorder()

	fault := rec.Record(engine.CacheEvent{
		Direction: cachemeta.KVRestore, Outcome: cachemeta.KVTransferFault, FaultReason: "page-in EIO",
	})
	if fault.Verdict.Kind != cachemeta.LookupFault || fault.Verdict.Reason != cachemeta.ReasonResidencyFault {
		t.Fatalf("restore fault must be FAULT(residency_fault), got %+v", fault.Verdict)
	}
	if !fault.SilentRecompute() {
		t.Fatal("a fault must be flagged as non-serveable (cannot be silently recomputed)")
	}

	miss := rec.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, Outcome: cachemeta.KVTransferMissed})
	if miss.Verdict.Kind != cachemeta.LookupMiss || miss.Verdict.Reason != cachemeta.ReasonRestoreMiss {
		t.Fatalf("restore miss must be MISS(restore_miss), got %+v", miss.Verdict)
	}
	if !miss.SilentRecompute() {
		t.Fatal("a miss must be flagged as non-serveable")
	}

	snap := rec.Metrics().Snapshot()
	if snap.RestoreFault != 1 || snap.RestoreMiss != 1 || snap.Faults != 1 || snap.Misses != 1 {
		t.Fatalf("typed miss/fault not counted in metrics: %+v", snap)
	}
}

func TestCacheTierMemoryClassProjection(t *testing.T) {
	for _, tc := range []struct {
		tier cachemeta.ResidencyTier
		want compute.MemoryClass
	}{
		{cachemeta.TierHBM, compute.MemoryKVCache},
		{cachemeta.TierDRAM, compute.MemoryDDRCache},
		{cachemeta.TierNUMAFar, compute.MemoryDDRCache},
		{cachemeta.TierCXL, compute.MemoryDDRCache},
		{cachemeta.TierDisk, compute.MemoryOffload},
		{cachemeta.TierRemote, compute.MemoryOffload},
		{cachemeta.TierProvider, compute.MemoryOffload},
		{cachemeta.TierUnknown, compute.MemoryUnknown},
	} {
		if got := engine.CacheTierMemoryClass(tc.tier); got != tc.want {
			t.Fatalf("CacheTierMemoryClass(%s) = %s, want %s", tc.tier, got, tc.want)
		}
	}
}

// Cache events must be exposable as metrics (not only internal engine counters):
// the snapshot keys by direction x outcome x to_tier x memory_class and renders
// Prometheus text, including byte/token breakdowns by class.
func TestCacheEventMetricsExposedAsPrometheus(t *testing.T) {
	rec := engine.NewCacheEventRecorder()
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVOffload, ToTier: cachemeta.TierDRAM, BytesMoved: 200, Tokens: 20, Outcome: cachemeta.KVTransferOK})
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVOffload, ToTier: cachemeta.TierDisk, BytesMoved: 100, Tokens: 10, Outcome: cachemeta.KVTransferOK})
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVOffload, ToTier: cachemeta.TierDisk, BytesMoved: 50, Tokens: 5, Outcome: cachemeta.KVTransferOK})
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, ToTier: cachemeta.TierHBM, Outcome: cachemeta.KVTransferFault, FaultReason: "x"})

	snap := rec.Metrics().Snapshot()
	if snap.Events != 4 || snap.BytesMoved != 350 || snap.TokensMoved != 35 {
		t.Fatalf("aggregate totals wrong: %+v", snap)
	}
	// The two offload-ok-disk events collapse into one keyed row with count 2.
	var foundDisk, foundDRAM bool
	for _, r := range snap.Rows {
		if r.Direction == "offload" && r.Outcome == "ok" && r.ToTier == "disk" {
			foundDisk = true
			if r.MemoryClass != string(compute.MemoryOffload) || r.Count != 2 || r.BytesMoved != 150 || r.TokensMoved != 15 {
				t.Fatalf("offload/ok/disk row wrong: %+v", r)
			}
		}
		if r.Direction == "offload" && r.Outcome == "ok" && r.ToTier == "dram" {
			foundDRAM = true
			if r.MemoryClass != string(compute.MemoryDDRCache) || r.Count != 1 || r.BytesMoved != 200 || r.TokensMoved != 20 {
				t.Fatalf("offload/ok/dram row wrong: %+v", r)
			}
		}
	}
	if !foundDisk || !foundDRAM {
		t.Fatalf("missing expected classed rows in %+v", snap.Rows)
	}

	prom := snap.Prometheus()
	for _, want := range []string{
		"fak_engine_cache_events_total 4",
		"fak_engine_cache_restore_fault_total 1",
		"fak_engine_cache_bytes_moved_total 350",
		`fak_engine_cache_event_breakdown_total{direction="offload",outcome="ok",to_tier="disk",memory_class="offload"} 2`,
		`fak_engine_cache_event_breakdown_total{direction="offload",outcome="ok",to_tier="dram",memory_class="ddr_cache"} 1`,
		`fak_engine_cache_bytes_moved_breakdown_total{direction="offload",outcome="ok",to_tier="dram",memory_class="ddr_cache"} 200`,
		`fak_engine_cache_tokens_moved_breakdown_total{direction="offload",outcome="ok",to_tier="dram",memory_class="ddr_cache"} 20`,
		"# TYPE fak_engine_cache_faults_total counter",
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
}

// #1945: CacheEvent.Direction (and Outcome/ToTier) are plain strings, not
// compiler-enforced enums, so a buggy or adversarial engine adapter can feed an
// unbounded stream of distinct (direction, outcome, to_tier, memory_class)
// combinations. byKey must stay bounded — events beyond the cap fold into an
// observable overflow bucket instead of growing the map forever.
func TestCacheEventMetricsCapsByKeyCardinality(t *testing.T) {
	rec := engine.NewCacheEventRecorder()

	const nUnique = 2000
	for i := 0; i < nUnique; i++ {
		rec.Record(engine.CacheEvent{
			Direction: cachemeta.KVTransferDirection(fmt.Sprintf("synthetic-direction-%d", i)),
			ToTier:    cachemeta.TierDRAM,
			Outcome:   cachemeta.KVTransferOK,
			Tokens:    1,
		})
	}

	snap := rec.Metrics().Snapshot()
	if snap.Events != nUnique {
		t.Fatalf("Events = %d, want %d (overflow must not drop the event count)", snap.Events, nUnique)
	}
	if len(snap.Rows) > 256 {
		t.Fatalf("byKey grew unbounded: %d rows, want <= 256", len(snap.Rows))
	}
	if !snap.KeysCapped {
		t.Fatal("KeysCapped = false, want true once the key bound is hit")
	}
	wantOverflow := uint64(nUnique - len(snap.Rows))
	if snap.OverflowEvents != wantOverflow {
		t.Fatalf("OverflowEvents = %d, want %d (nUnique - distinct rows kept)", snap.OverflowEvents, wantOverflow)
	}
	if snap.OverflowEvents == 0 {
		t.Fatal("expected some events to have overflowed for this test to be meaningful")
	}

	prom := snap.Prometheus()
	if !strings.Contains(prom, "fak_engine_cache_keys_capped 1") {
		t.Fatalf("Prometheus output missing capped gauge=1:\n%s", prom)
	}
	if !strings.Contains(prom, fmt.Sprintf("fak_engine_cache_event_overflow_total %d", wantOverflow)) {
		t.Fatalf("Prometheus output missing overflow counter %d:\n%s", wantOverflow, prom)
	}
}

// The recorder fans every normalized (entry, verdict) out to an installed sink so a
// structured logger / tracer can observe the same stream.
func TestCacheEventRecorderSinkFanout(t *testing.T) {
	rec := engine.NewCacheEventRecorder()
	var seen []cachemeta.LookupKind
	rec.SetSink(func(_ cachemeta.Entry, v cachemeta.LookupVerdict) {
		seen = append(seen, v.Kind)
	})
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVOffload, Outcome: cachemeta.KVTransferOK})
	rec.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, Outcome: cachemeta.KVTransferMissed})
	if len(seen) != 2 || seen[0] != cachemeta.LookupHit || seen[1] != cachemeta.LookupMiss {
		t.Fatalf("sink did not observe the stream: %+v", seen)
	}
}
