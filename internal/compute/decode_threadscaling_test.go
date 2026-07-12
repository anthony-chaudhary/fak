package compute

import "testing"

// These tests reuse smallDecodeGeom() (decode_throughput_test.go, same package) whose
// WeightBytesPerToken is exactly 1312, and decodeFloatNear() from the same file, so every golden
// number below falls out by hand arithmetic and pins the thread-scaling model against drift.

func TestDecodeThreadScalingLinearThenBusBound(t *testing.T) {
	g := smallDecodeGeom() // WeightBytesPerToken = 1312
	// per-core = 1312 B/s ⇒ exactly 1 tok/s on one core; aggregate = 8×that ⇒ bus saturates at 8 cores.
	const perCore = 1312.0
	const aggregate = perCore * 8

	// 4 cores: below saturation, scaling is linear ⇒ 4 tok/s, 4× speedup, not yet bus-bound.
	four := DecodeThreadScalingProfile(g, perCore, aggregate, 4)
	if !decodeFloatNear(four.SingleThreadTokPerSec, 1.0) {
		t.Fatalf("SingleThreadTokPerSec = %v, want 1.0", four.SingleThreadTokPerSec)
	}
	if !decodeFloatNear(four.MultiThreadTokPerSec, 4.0) {
		t.Fatalf("MultiThreadTokPerSec@4 = %v, want 4.0", four.MultiThreadTokPerSec)
	}
	if four.SaturatingThreads != 8 {
		t.Fatalf("SaturatingThreads = %d, want 8", four.SaturatingThreads)
	}
	if !decodeFloatNear(four.Speedup, 4.0) {
		t.Fatalf("Speedup@4 = %v, want 4.0", four.Speedup)
	}
	if four.BusBound {
		t.Fatal("4 < 8 cores must NOT be bus-bound")
	}

	// 16 cores: past saturation, the bus caps the stream at aggregate ⇒ 8 tok/s, 8× speedup, bus-bound.
	sixteen := DecodeThreadScalingProfile(g, perCore, aggregate, 16)
	if !decodeFloatNear(sixteen.MultiThreadTokPerSec, 8.0) {
		t.Fatalf("MultiThreadTokPerSec@16 = %v, want 8.0 (bus cap)", sixteen.MultiThreadTokPerSec)
	}
	if !decodeFloatNear(sixteen.Speedup, 8.0) {
		t.Fatalf("Speedup@16 = %v, want 8.0", sixteen.Speedup)
	}
	if !sixteen.BusBound {
		t.Fatal("16 ≥ 8 cores must be bus-bound")
	}
}

func TestDecodeThreadScalingGuards(t *testing.T) {
	g := smallDecodeGeom()
	// Empty geometry (no weight bytes) ⇒ zero model, no scaling claim.
	if got := DecodeThreadScalingProfile(PrefillGeometry{}, 1e9, 1e11, 8); got.MultiThreadTokPerSec != 0 || got.Speedup != 0 {
		t.Fatalf("empty geometry = %+v, want zero model", got)
	}
	// Non-positive per-core bandwidth or thread count ⇒ zero model.
	if got := DecodeThreadScalingProfile(g, 0, 1e11, 8); got.MultiThreadTokPerSec != 0 {
		t.Fatalf("zero per-core bandwidth = %+v, want zero model", got)
	}
	if got := DecodeThreadScalingProfile(g, 1e9, 1e11, 0); got.MultiThreadTokPerSec != 0 {
		t.Fatalf("zero threads = %+v, want zero model", got)
	}
	// Aggregate unset (0) ⇒ clamped to per-core: one core's worth, saturates at 1, no headroom known.
	deg := DecodeThreadScalingProfile(g, 1312.0, 0, 4)
	if deg.SaturatingThreads != 1 {
		t.Fatalf("SaturatingThreads with unset aggregate = %d, want 1", deg.SaturatingThreads)
	}
	if !decodeFloatNear(deg.MultiThreadTokPerSec, deg.SingleThreadTokPerSec) {
		t.Fatalf("unset aggregate: multi %v must equal single %v", deg.MultiThreadTokPerSec, deg.SingleThreadTokPerSec)
	}
	if !decodeFloatNear(deg.Speedup, 1.0) || !deg.BusBound {
		t.Fatalf("unset aggregate: want Speedup 1.0 and bus-bound, got Speedup=%v BusBound=%v", deg.Speedup, deg.BusBound)
	}
}

func TestGradeDecodeParallelismIssueScenario(t *testing.T) {
	g := smallDecodeGeom() // WeightBytesPerToken = 1312
	// Model the issue's 256-thread EPYC: one core = 1 tok/s, the socket bus = 100 tok/s of aggregate
	// (so the bus saturates at 100 cores — dozens of cores' worth of real parallel headroom).
	const perCore = 1312.0
	const aggregate = perCore * 100
	// The issue's measurement: ~500 tokens over 600 s ≈ 0.83 tok/s — consistent with ONE streaming
	// core (EffectiveThreads < 1), the quantitative signature of an effectively single-threaded decode.
	v := GradeDecodeParallelism(500, 600, g, perCore, aggregate, 256)
	if v.EffectiveThreads >= 1.0 {
		t.Fatalf("EffectiveThreads = %v, want < 1 (≈one core)", v.EffectiveThreads)
	}
	if !v.SingleThreaded {
		t.Fatalf("0.83 tok/s on a 100-core-bus box must grade SingleThreaded (verdict=%+v)", v)
	}
	// Utilization of the engaged-core ceiling is a tiny sliver — decode is nowhere near the cores it has.
	if v.Utilization >= 0.02 {
		t.Fatalf("Utilization = %v, want a tiny fraction of the multi-thread ceiling", v.Utilization)
	}
}

func TestGradeDecodeParallelismThreadedNotFlagged(t *testing.T) {
	g := smallDecodeGeom()
	const perCore = 1312.0
	const aggregate = perCore * 100
	// A decode genuinely engaging its cores: 80 tok/s of the 100 tok/s bus ceiling ⇒ ~80 effective
	// cores' worth of bandwidth, 0.8 utilization — must NOT be flagged single-threaded.
	v := GradeDecodeParallelism(80, 1, g, perCore, aggregate, 256)
	if v.SingleThreaded {
		t.Fatalf("80 tok/s (of 100) must NOT be single-threaded (verdict=%+v)", v)
	}
	if !decodeFloatNear(v.Utilization, 0.8) {
		t.Fatalf("Utilization = %v, want 0.8", v.Utilization)
	}
	if v.EffectiveThreads < singleThreadedThreshold {
		t.Fatalf("EffectiveThreads = %v, want ≥ threshold", v.EffectiveThreads)
	}
}

func TestGradeDecodeParallelismSingleCoreBoxNotFlagged(t *testing.T) {
	g := smallDecodeGeom()
	// A genuinely single-core box: aggregate == per-core, so there is NO parallel headroom to forgo.
	// Even a throughput at the one-core ceiling must not be labeled "effectively single-threaded" —
	// the verdict fires only when multi-core headroom (Speedup ≥ threshold) actually exists.
	const perCore = 1312.0
	v := GradeDecodeParallelism(500, 600, g, perCore, perCore, 1)
	if v.SingleThreaded {
		t.Fatalf("a true single-core box must NOT be flagged single-threaded (verdict=%+v)", v)
	}
}
