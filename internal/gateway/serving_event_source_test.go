package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// serving_event_source_test.go — the #903 source-tag regression. The "learn P/D
// disaggregation + KV routing" readout
// (docs/serving/pd-disaggregation-kv-routing-sota.md) names four cache/serving
// event SOURCES a governed serve must keep separable:
//
//	provider     — a remote engine's prompt-prefix cache (cost/latency telemetry, never trust)
//	ride-engine  — a ridden vLLM/SGLang/LMCache/Dynamo KV routing/offload event
//	native-fak   — fak's own radix/KV-prefix reuse
//	vcache/vDSO  — fak's tier-2/3 tool-result (value) cache
//
// Each lowers onto a DISTINCT cachemeta plane, so the unified cache-entry stream
// is already source-tagged by plane; and the provider source must NEVER fold into
// a local-reuse total (the benchmark-honesty gate: a provider saving is not a fak
// win unless the event source says so). This pins both properties as one
// regression in the serve lane — GPU-free, no live engine, no network.
func TestServingEventSourcesAreTagged(t *testing.T) {
	// One event per source, each built the way its real adapter builds it.
	provider := cachemeta.FromProviderCache(cachemeta.ProviderCache{
		Provider: "anthropic", ModelID: "claude-opus-4-8",
		CachedTokens: 4096, PromptTokens: 8192, SerializerID: "serializer-v1",
	})
	// ride-engine: exercise the real live-engine recorder (internal/engine), the
	// adapter a ridden vLLM/SGLang/Dynamo KV routing/offload event flows through.
	rideRes := engine.NewCacheEventRecorder().Record(engine.CacheEvent{
		Direction: cachemeta.KVRoute, SpanDigest: "span-ride-1", Tokens: 2048,
		ModelID: "qwen3.6", ToTier: cachemeta.TierDRAM, Owner: "vllm",
		Outcome: cachemeta.KVTransferOK, BytesMoved: 1 << 20,
	})
	ride := rideRes.Entry
	native := cachemeta.FromKVPrefix(cachemeta.KVPrefix{
		Tokens: []int{1, 2, 3, 4, 5}, ModelID: "qwen3.6", Owner: "radixkv",
	})
	vcache := cachemeta.FromStaticTool("clock_now",
		abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"now":"t"}`)})

	// 1. Each source lands on its own plane — the source tag is present and the
	//    four are pairwise distinct (no source is silently aliased onto another).
	bySource := map[string]cachemeta.Plane{
		"provider":    provider.Plane,
		"ride-engine": ride.Plane,
		"native-fak":  native.Plane,
		"vcache/vdso": vcache.Plane,
	}
	want := map[string]cachemeta.Plane{
		"provider":    cachemeta.PlaneProvider,
		"ride-engine": cachemeta.PlaneKVTransfer,
		"native-fak":  cachemeta.PlaneKVPrefix,
		"vcache/vdso": cachemeta.PlaneToolResult,
	}
	seen := map[cachemeta.Plane]bool{}
	for src, plane := range bySource {
		if plane != want[src] {
			t.Errorf("source %q on plane %q, want %q", src, plane, want[src])
		}
		if seen[plane] {
			t.Errorf("plane %q is shared by two sources — sources not separable", plane)
		}
		seen[plane] = true
	}
	if len(seen) != 4 {
		t.Fatalf("want 4 distinct source planes, got %d: %v", len(seen), seen)
	}

	// 2. Folded into the unified cache stream, each source stays its own row, so a
	//    scrape reads the per-source mix instead of one blended hit rate.
	stream := cachemeta.NewStreamMetrics()
	for _, e := range []cachemeta.Entry{provider, ride, native, vcache} {
		stream.Observe("hit", e)
	}
	planes := map[string]bool{}
	for _, r := range stream.Snapshot().Rows {
		planes[r.Plane] = true
	}
	for _, p := range []cachemeta.Plane{
		cachemeta.PlaneProvider, cachemeta.PlaneKVTransfer,
		cachemeta.PlaneKVPrefix, cachemeta.PlaneToolResult,
	} {
		if !planes[string(p)] {
			t.Errorf("unified stream dropped source plane %q: have %v", p, planes)
		}
	}

	// 3. The benchmark-honesty gate: ONLY the provider source folds into the
	//    provider counter; the three local sources never inflate a local-reuse
	//    total. A provider saving is not a fak win unless the source says so.
	var arm metrics.Arm
	if !arm.FoldCacheEntry(provider) {
		t.Fatal("provider event must fold as provider telemetry (FoldCacheEntry=false)")
	}
	for _, e := range []cachemeta.Entry{ride, native, vcache} {
		if arm.FoldCacheEntry(e) {
			t.Errorf("local source on plane %q was folded as PROVIDER telemetry", e.Plane)
		}
	}
	if arm.ProviderCacheReadTokens != 4096 || arm.ProviderCacheHits != 1 {
		t.Fatalf("provider counters wrong: read=%d hits=%d, want 4096/1",
			arm.ProviderCacheReadTokens, arm.ProviderCacheHits)
	}
	if arm.InTokens != 0 || arm.VDSOHits != 0 {
		t.Fatalf("provider saving leaked into local counters: InTokens=%d VDSOHits=%d",
			arm.InTokens, arm.VDSOHits)
	}
}
