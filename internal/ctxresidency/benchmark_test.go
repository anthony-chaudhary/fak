package ctxresidency_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	benchSnapshotSink   ctxresidency.Snapshot
	benchBandsSink      ctxresidency.Bands
	benchTierSink       ctxresidency.Tier
	benchCapSnapSink    ctxresidency.CapSnapshot
	benchBlastSink      ctxresidency.BlastRadius
	benchCapKeySink     ctxresidency.CapKey
	benchLoaderSnapSink ctxresidency.LoaderSnapshot
	benchBoolSink       bool
)

func buildBenchContext(numSegments, numDeps int) (*kvmmu.Context, *ctxmmu.MMU) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	c := kvmmu.NewWithGate(model.NewSynthetic(synthCfg()).NewSession(), mmu)

	for i := 0; i < numSegments; i++ {
		segID := fmt.Sprintf("seg-%d", i)
		tokens := []int{(i * 2) % 48, (i*2 + 1) % 48}
		if i%5 == 4 {
			_, _, _ = c.AdmitResult(ctx, segID, "quarantine_tool", tokens, []byte(poisonBody))
		} else {
			c.Append(segID, "tool", tokens)
		}
	}

	segments := c.Segments()
	for i := 0; i < numDeps && i < len(segments); i++ {
		kv := segments[i].KV
		if kv.Valid() {
			idx := cachemeta.FromAttentionIndex(cachemeta.AttentionIndex{
				Tokens:           []int{1, 2},
				ModelID:          "llama",
				TokenizerID:      "tok",
				IndexerID:        "idx:v1",
				LayerGroup:       "0-1",
				Layers:           []int{0, 1},
				DecisionDigest:   cachemeta.DigestBytes([]byte(fmt.Sprintf("dep-%d", i))),
				ParentKV:         kv,
				Owner:            "bench",
				Causal:           true,
				CausalityWitness: "bench:query",
			})
			c.TrackEntry(idx)
		}
	}
	return c, mmu
}

// TestBenchmarkOperationsSanity verifies that every benchmarked execution path runs
// correctly and passes expected invariant checks before performance loops run.
func TestBenchmarkOperationsSanity(t *testing.T) {
	ctx := context.Background()

	// 1. Query path sanity
	c, mmu := buildBenchContext(10, 2)
	snap := ctxresidency.Query(c, mmu)
	if len(snap.Spans) != 10 {
		t.Fatalf("expected 10 spans, got %d", len(snap.Spans))
	}
	if snap.CommittedTokens == 0 || snap.ReclaimableTokens == 0 {
		t.Fatalf("expected both committed and reclaimable tokens in mixed context: %+v", snap)
	}

	// 2. Pressure classification sanity
	bands := snap.Pressure(1000, 50, 80)
	if bands.Class == ctxresidency.PressureUnknown {
		t.Fatalf("expected resolved pressure class, got %v", bands.Class)
	}

	// 3. Tier classification sanity
	tier := ctxresidency.ClassifyTier(ctxresidency.BlockProfile{
		UsedEveryTurn:       true,
		Small:               true,
		IdentityLoadBearing: true,
	})
	if tier != ctxresidency.TierSpine {
		t.Fatalf("expected TierSpine, got %v", tier)
	}

	// 4. CapResidency lifecycle sanity
	cr := ctxresidency.NewCapResidency(mmu)
	key := skillKey("sanity-cap", "v1")
	cr.Fault(key, "sha256:sanity", []byte("sanity body"), nil)
	cr.Touch(key)
	cr.Pin(key)
	cr.Unpin(key)
	if radius := cr.MeasureBlastRadius(key); radius.Tokens != len("sanity body") {
		t.Fatalf("unexpected blast radius: %+v", radius)
	}
	evicted, _, ok := cr.EvictColdest(ctx)
	if !ok || evicted != key {
		t.Fatalf("expected evicted key %v, got %v (ok=%v)", key, evicted, ok)
	}
	capSnap := cr.Snapshot()
	if capSnap.Held != 1 {
		t.Fatalf("expected 1 held capability in snapshot, got %d", capSnap.Held)
	}
}

// BenchmarkQuery_Small measures the latency and allocations of context residency
// queries on typical compact multi-turn contexts (5 segments, 1 cachemeta dependency).
func BenchmarkQuery_Small(b *testing.B) {
	c, mmu := buildBenchContext(5, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSnapshotSink = ctxresidency.Query(c, mmu)
	}
}

// BenchmarkQuery_Large measures context residency queries over deep >100k token
// windows with 50 segments, 10 cachemeta dependents, and active quarantine states.
func BenchmarkQuery_Large(b *testing.B) {
	c, mmu := buildBenchContext(50, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSnapshotSink = ctxresidency.Query(c, mmu)
	}
}

// BenchmarkQuery_KVOnly measures the pure KV-level residency projection when the
// byte-level MMU is omitted.
func BenchmarkQuery_KVOnly(b *testing.B) {
	c, _ := buildBenchContext(20, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSnapshotSink = ctxresidency.Query(c, nil)
	}
}

// BenchmarkPressureBands measures the pure arithmetic budget partition and
// threshold classification across varying context pressure conditions.
func BenchmarkPressureBands(b *testing.B) {
	cases := [][5]int{
		{100, 200, 1000, 50, 80}, // Any
		{550, 100, 1000, 50, 80}, // Bounded
		{850, 50, 1000, 50, 80},  // Checkpoint
		{0, 0, 0, 50, 80},        // Unknown
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		benchBandsSink = ctxresidency.PressureBands(tc[0], tc[1], tc[2], tc[3], tc[4])
	}
}

// BenchmarkSnapshot_Pressure measures the method invocation converting a live
// point-in-time Snapshot into three-band pressure allocations.
func BenchmarkSnapshot_Pressure(b *testing.B) {
	snap := ctxresidency.Snapshot{
		CommittedTokens:   600,
		ReclaimableTokens: 200,
		ResidentTokens:    800,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBandsSink = snap.Pressure(1000, 50, 80)
	}
}

// BenchmarkClassifyTier measures layout-tier resolution (Rung 4, issue #1262)
// across spine, policy floor, and overlay block profiles.
func BenchmarkClassifyTier(b *testing.B) {
	profiles := []ctxresidency.BlockProfile{
		{UsedEveryTurn: true, Small: true, IdentityLoadBearing: true, SafetyLoadBearing: false},
		{UsedEveryTurn: false, Small: false, IdentityLoadBearing: false, SafetyLoadBearing: true},
		{UsedEveryTurn: false, Small: false, IdentityLoadBearing: false, SafetyLoadBearing: false},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := profiles[i%len(profiles)]
		benchTierSink = ctxresidency.ClassifyTier(p)
	}
}

// BenchmarkCapResidency_Fault measures capability admission throughput and
// dependency map registration within the working set tracker.
func BenchmarkCapResidency_Fault(b *testing.B) {
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	body := []byte("capability body text for benchmark admission")
	keys := make([]ctxresidency.CapKey, 256)
	for i := range keys {
		keys[i] = skillKey(fmt.Sprintf("skill-%d", i), "v1")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cr.Fault(keys[i%len(keys)], "sha256:digest", body, nil)
	}
}

// BenchmarkCapResidency_Touch measures fast-path LRU warming for procedural-cache
// hits without full re-admit allocation.
func BenchmarkCapResidency_Touch(b *testing.B) {
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	body := []byte("capability body text")
	keys := make([]ctxresidency.CapKey, 128)
	for i := range keys {
		keys[i] = skillKey(fmt.Sprintf("skill-%d", i), "v1")
		cr.Fault(keys[i], "sha256:digest", body, nil)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cr.Touch(keys[i%len(keys)])
	}
}

// BenchmarkCapResidency_PinUnpin measures the latency of bracketing an in-flight
// tool execution with CAS capability pins.
func BenchmarkCapResidency_PinUnpin(b *testing.B) {
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	key := skillKey("active-tool", "v1")
	cr.Fault(key, "sha256:digest", []byte("active body"), nil)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cr.Pin(key)
		cr.Unpin(key)
	}
}

// BenchmarkCapResidency_MeasureBlastRadius measures non-destructive blast radius
// reads over the live capability dependency graph.
func BenchmarkCapResidency_MeasureBlastRadius(b *testing.B) {
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	deps := []ctxresidency.CapKey{
		skillKey("child-1", "v1"),
		skillKey("child-2", "v1"),
		skillKey("child-3", "v1"),
	}
	for _, d := range deps {
		cr.Fault(d, "sha256:child", []byte("child body"), nil)
	}
	parent := skillKey("parent-skill", "v1")
	cr.Fault(parent, "sha256:parent", []byte("parent body"), deps)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBlastSink = cr.MeasureBlastRadius(parent)
	}
}

// BenchmarkCapResidency_RefuseIfCrossesSpine measures the Rung-4 invariant 3
// layout-tier eviction guard check.
func BenchmarkCapResidency_RefuseIfCrossesSpine(b *testing.B) {
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	spineKey := skillKey("spine-core", "v1")
	cr.Fault(spineKey, "sha256:spine", []byte("spine body"), nil)
	cr.SetTier(spineKey, ctxresidency.TierSpine)

	overlayKey := skillKey("overlay-tool", "v1")
	cr.Fault(overlayKey, "sha256:overlay", []byte("overlay body"), nil)
	cr.SetTier(overlayKey, ctxresidency.TierOverlay)

	keys := []ctxresidency.CapKey{spineKey, overlayKey}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		radius, refused := cr.RefuseIfCrossesSpine(keys[i%2])
		benchBlastSink = radius
		benchBoolSink = refused
	}
}

// BenchmarkCapResidency_EvictColdest measures deterministic LRU eviction of the
// coldest evictable capability under pressure with witnessed page-out.
func BenchmarkCapResidency_EvictColdest(b *testing.B) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	body := []byte("payload body for eviction")
	key := skillKey("cold-victim", "v1")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cr.Fault(key, "sha256:digest", body, nil)
		evicted, radius, ok := cr.EvictColdest(ctx)
		if !ok {
			b.Fatal("expected eviction")
		}
		benchCapKeySink = evicted
		benchBlastSink = radius
	}
}

// BenchmarkCapResidency_WorkingSetChurn measures steady-state working-set churn
// where new capabilities fault in and coldest capabilities page out under pressure.
func BenchmarkCapResidency_WorkingSetChurn(b *testing.B) {
	ctx := context.Background()
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	body := []byte("payload body 32 bytes")
	for i := 0; i < 32; i++ {
		k := skillKey(fmt.Sprintf("cap-%02d", i), "v1")
		cr.Fault(k, fmt.Sprintf("sha256:%d", i), body, nil)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := skillKey(fmt.Sprintf("churn-%04d", i%64), "v1")
		cr.Fault(k, "sha256:churn", body, nil)
		evicted, radius, _ := cr.EvictColdest(ctx)
		benchCapKeySink = evicted
		benchBlastSink = radius
	}
}

// BenchmarkCapResidency_Snapshot measures full point-in-time capability residency
// reads across a 64-item working set with key sorting and state tallying.
func BenchmarkCapResidency_Snapshot(b *testing.B) {
	mmu := ctxmmu.New()
	cr := ctxresidency.NewCapResidency(mmu)
	for i := 0; i < 64; i++ {
		k := skillKey(fmt.Sprintf("cap-%03d", i), "v1")
		cr.Fault(k, fmt.Sprintf("sha256:cap-%d", i), []byte("body"), nil)
		if i%4 == 0 {
			cr.SetTier(k, ctxresidency.TierSpine)
		} else if i%4 == 1 {
			cr.Pin(k)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCapSnapSink = cr.Snapshot()
	}
}

// BenchmarkLoaderJournal_Reconcile measures reading and folding capability
// audit events from a durable journal and reconciling against kernel counters.
func BenchmarkLoaderJournal_Reconcile(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "audit_journal.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		b.Fatalf("open journal: %v", err)
	}
	const (
		numFaults = 30
		numEvicts = 15
		numBinds  = 5
	)

	for i := 0; i < numFaults; i++ {
		j.Emit(abi.Event{
			Kind:   abi.EvCapFault,
			Fields: map[string]any{"cap_kind": "skill", "cap_name": fmt.Sprintf("skill-%d", i), "cap_digest": "d"},
		})
	}
	for i := 0; i < numEvicts; i++ {
		j.Emit(abi.Event{
			Kind:   abi.EvCapEvict,
			Fields: map[string]any{"cap_kind": "skill", "cap_name": fmt.Sprintf("skill-%d", i), "cap_digest": "d"},
		})
	}
	for i := 0; i < numBinds; i++ {
		j.Emit(abi.Event{
			Kind:   abi.EvCapVersionBind,
			Fields: map[string]any{"cap_kind": "skill", "cap_name": fmt.Sprintf("skill-%d", i), "cap_from": "v1", "cap_to": "v2"},
		})
	}
	if err := j.Close(); err != nil {
		b.Fatalf("close journal: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := ctxresidency.LoaderJournal(path, numFaults, numEvicts, numBinds)
		if err != nil {
			b.Fatalf("LoaderJournal failed: %v", err)
		}
		benchLoaderSnapSink = snap
	}
}
