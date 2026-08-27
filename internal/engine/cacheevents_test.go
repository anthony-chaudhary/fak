package engine_test

import (
	"encoding/json"
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
	if !off.Published {
		t.Fatalf("legacy identity-less event was suppressed: %+v", off)
	}
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
		{cachemeta.TierRemoteDRAM, compute.MemoryOffload},
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

func TestCacheEventRecorderConsolidatesNativeLogicalVisibility(t *testing.T) {
	rec := engine.NewCacheEventRecorderWithConsolidator(cachemeta.NewCacheEventConsolidator(4))
	receipt := engine.NativeCacheEventRoutingReceipt()
	if receipt.Engine != "fak_native" || receipt.Model != "Qwen3.8" || receipt.Fallback != "none" {
		t.Fatalf("native routing receipt = %+v, want fak_native/Qwen3.8/no fallback", receipt)
	}

	block := cachemeta.NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-native")
	event := func(sourceID, eventID string, sequence uint64, action cachemeta.CacheVisibilityAction) engine.CacheEvent {
		direction := cachemeta.KVRestore
		fromTier, toTier := cachemeta.TierDRAM, cachemeta.TierHBM
		if action == cachemeta.CacheVisibilityRemove {
			direction = cachemeta.KVOffload
			fromTier, toTier = cachemeta.TierHBM, cachemeta.TierUnknown
		}
		return engine.CacheEvent{
			Direction:        direction,
			SpanDigest:       block.Digest,
			ModelID:          block.ModelID,
			TokenizerID:      block.TokenizerID,
			FromTier:         fromTier,
			ToTier:           toTier,
			Outcome:          cachemeta.KVTransferOK,
			VisibilityAction: action,
			SourceID:         sourceID,
			EventID:          eventID,
			EventSequence:    sequence,
			LogicalBlock:     block,
			RoutingReceipt:   receipt,
		}
	}

	var published int
	rec.SetSink(func(_ cachemeta.Entry, _ cachemeta.LookupVerdict) { published++ })

	firstStore := rec.Record(event("source-a", "a-store", 1, cachemeta.CacheVisibilityStore))
	if !firstStore.Published || firstStore.Receipt.Engine != "fak_native" || firstStore.Receipt.Model != "Qwen3.8" {
		t.Fatalf("first native STORE result = %+v, want published fak_native/Qwen3.8 receipt", firstStore)
	}
	for label, want := range map[string]string{
		"routing_engine":            "fak_native",
		"routing_model":             "Qwen3.8",
		"routing_fallback":          "none",
		"source_id":                 "source-a",
		"event_id":                  "a-store",
		"event_sequence":            "1",
		"logical_block_key_version": cachemeta.CacheLogicalBlockKeyVersion,
		"logical_block_digest":      "block-native",
	} {
		if got := firstStore.Entry.Labels[label]; got != want {
			t.Fatalf("first native STORE label %q = %q, want %q", label, got, want)
		}
	}

	secondStore := rec.Record(event("source-b", "b-store", 1, cachemeta.CacheVisibilityStore))
	if secondStore.Published || secondStore.Suppression != cachemeta.CacheEventDuplicateProducer {
		t.Fatalf("second native STORE result = %+v, want duplicate-producer suppression", secondStore)
	}
	firstRemove := rec.Record(event("source-a", "a-remove", 2, cachemeta.CacheVisibilityRemove))
	if firstRemove.Published || firstRemove.Suppression != cachemeta.CacheEventSourceStillResident {
		t.Fatalf("first native REMOVE result = %+v, want source-still-resident suppression", firstRemove)
	}
	finalRemove := rec.Record(event("source-b", "b-remove", 2, cachemeta.CacheVisibilityRemove))
	if !finalRemove.Published {
		t.Fatalf("final native REMOVE result = %+v, want published", finalRemove)
	}

	snapshot := rec.Metrics().Snapshot()
	if snapshot.Events != 2 || snapshot.SuppressedProducer != 1 || snapshot.SuppressedRemove != 1 {
		t.Fatalf("consolidated recorder metrics = %+v, want 2 published and two typed suppressions", snapshot)
	}
	if published != 2 {
		t.Fatalf("sink publications = %d, want first STORE and final REMOVE only", published)
	}
}

func nativeVisibilityEvent(source, event string, sequence uint64, action cachemeta.CacheVisibilityAction) engine.CacheEvent {
	ev := engine.CacheEvent{
		Direction:        cachemeta.KVRestore,
		SpanDigest:       "block-1",
		ModelID:          "Qwen3.8",
		TokenizerID:      "qwen3",
		ToTier:           cachemeta.TierHBM,
		Owner:            "native-router",
		Outcome:          cachemeta.KVTransferOK,
		VisibilityAction: action,
		SourceID:         source,
		EventID:          event,
		EventSequence:    sequence,
		LogicalBlock:     cachemeta.NewCacheLogicalBlockKey("Qwen3.8", "qwen3", "block-1"),
		RoutingReceipt:   engine.NativeCacheEventRoutingReceipt(),
	}
	if action == cachemeta.CacheVisibilityRemove {
		ev.Direction = cachemeta.KVOffload
		ev.FromTier = cachemeta.TierHBM
		ev.ToTier = cachemeta.TierUnknown
	}
	return ev
}

func TestCacheEventRecorderNativeVisibilityThroughFinalRemove(t *testing.T) {
	rec := engine.NewCacheEventRecorder()
	var visible []cachemeta.Entry
	rec.SetSink(func(entry cachemeta.Entry, _ cachemeta.LookupVerdict) {
		visible = append(visible, entry)
	})

	storeA := rec.Record(nativeVisibilityEvent("rank-0", "a/store", 1, cachemeta.CacheVisibilityStore))
	storeB := rec.Record(nativeVisibilityEvent("rank-1", "b/store", 1, cachemeta.CacheVisibilityStore))
	removeA := rec.Record(nativeVisibilityEvent("rank-0", "a/remove", 2, cachemeta.CacheVisibilityRemove))
	removeB := rec.Record(nativeVisibilityEvent("rank-1", "b/remove", 2, cachemeta.CacheVisibilityRemove))
	if !storeA.Published || storeB.Published || removeA.Published || !removeB.Published {
		t.Fatalf("logical visibility decisions = storeA:%+v storeB:%+v removeA:%+v removeB:%+v", storeA, storeB, removeA, removeB)
	}
	if len(visible) != 2 {
		t.Fatalf("sink saw %d events, want first STORE and final REMOVE only: %+v", len(visible), visible)
	}
	if visible[0].Labels["source_id"] != "rank-0" || visible[1].Labels["source_id"] != "rank-1" {
		t.Fatalf("published source identities missing: %+v", visible)
	}
	if visible[0].Labels["logical_block_key_version"] != cachemeta.CacheLogicalBlockKeyVersion {
		t.Fatalf("versioned logical-block key missing: %+v", visible[0].Labels)
	}
	if visible[0].Labels["visibility_action"] != "store" || visible[1].Labels["visibility_action"] != "remove" {
		t.Fatalf("published logical edges are not explicit: %+v", visible)
	}

	receiptJSON, err := json.Marshal(storeA.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	receipt := string(receiptJSON)
	for _, want := range []string{`"engine":"fak_native"`, `"model":"Qwen3.8"`, `"fallback":"none"`} {
		if !strings.Contains(receipt, want) {
			t.Fatalf("native-routing receipt missing %q: %s", want, receipt)
		}
	}
	if strings.Contains(strings.ToLower(receipt), "llama") {
		t.Fatalf("native-routing receipt contains a forbidden llama fallback: %s", receipt)
	}

	snap := rec.Metrics().Snapshot()
	if snap.Events != 2 || snap.SuppressedProducer != 1 || snap.SuppressedRemove != 1 {
		t.Fatalf("published/suppressed metrics = %+v", snap)
	}
	prom := snap.Prometheus()
	for _, want := range []string{
		"fak_engine_cache_events_total 2",
		"fak_engine_cache_suppressed_producer_total 1",
		"fak_engine_cache_suppressed_remove_total 1",
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("Prometheus output missing %q:\n%s", want, prom)
		}
	}
}

func TestCacheEventRecorderCompatibilityAndUnknownCounters(t *testing.T) {
	rec := engine.NewCacheEventRecorder()
	legacy := rec.Record(engine.CacheEvent{Direction: cachemeta.KVRestore, Outcome: cachemeta.KVTransferOK})
	if !legacy.Published {
		t.Fatalf("identity-less legacy event must remain pass-through: %+v", legacy)
	}
	unknown := rec.Record(engine.CacheEvent{
		Direction:        cachemeta.KVRestore,
		Outcome:          cachemeta.KVTransferOK,
		VisibilityAction: cachemeta.CacheVisibilityStore,
	})
	if unknown.Published || unknown.Suppression != cachemeta.CacheEventUnknown {
		t.Fatalf("unidentified visibility event = %+v, want unknown suppression", unknown)
	}
	if got := rec.Metrics().Snapshot().UnknownEvents; got != 1 {
		t.Fatalf("UnknownEvents = %d, want 1", got)
	}
}
