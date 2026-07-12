package compute

import "math"

// decode_threadscaling.go — the host-tractable witness for issue #3176's SECOND question:
// "Does decode parallelize across cores here, or is it effectively single-threaded? A 1.5B Q8
// should be much faster than ~<1 tok/s on this hardware." decode_throughput.go already answers
// the FIRST question (is the SIMD lane engaged, or did it fall to the scalar reference?) and
// pins the memory-bound tok/s CEILING at a device's peak bandwidth. But that ceiling is the
// AGGREGATE-bandwidth ceiling — the throughput only a kernel that streams weights on many cores
// at once can reach. This file adds the missing rung: how that ceiling is BUILT UP from cores,
// so an operator can tell a decode that is scalar-and-single-threaded (the ~<1 tok/s the issue
// reports on a 256-core box) from one that is genuinely engaging the socket's cores.
//
// The load-bearing physics, continuing decode_throughput.go's roofline: decode is memory-bound
// on weight streaming, so tok/s is (bytes streamed per second) ÷ WeightBytesPerToken. A SINGLE
// core streams weights at some per-core memory bandwidth (a hardware ceiling one thread cannot
// beat no matter how wide its SIMD, because the bottleneck is the load ports / L2-miss stream,
// not the ALU). T cores stream at T× that — until they collectively saturate the socket's
// AGGREGATE memory bandwidth (the memory controllers), past which more cores buy nothing. So
// decode tok/s scales LINEARLY with cores up to SaturatingThreads = ⌈aggregate ÷ per-core⌉, then
// is bus-bound flat. On a 256-core EPYC SaturatingThreads is typically a few dozen, so a decode
// pinned to one core leaves a ~SaturatingThreads× multiple of throughput on the table — which is
// the quantitative shape of "effectively single-threaded" the issue is asking us to confirm.
//
// Counting is exact (DecodeWeightBytes); the two bandwidths (per-core and aggregate) are the
// caller's, measured on the target box — the same caller-supplies-the-device-peaks discipline as
// decode_throughput.go's ceiling and prefill.go's ridge. No hardware constant is baked in here.
// The invalidating assumption, stated plainly: this models the memory-bandwidth ceiling of a
// PERFECTLY parallel weight stream. A real kernel also pays thread-launch, NUMA-remote, and
// reduction overheads, so measured multi-core tok/s sits at or below MultiThreadTokPerSec; the
// model is an upper bound on parallel scaling, not a promise a kernel will reach it.

// DecodeThreadScaling is the memory-bandwidth thread-scaling model for decode at a model
// geometry: the single-core streaming ceiling, the T-core ceiling, and where the socket's
// aggregate memory bus caps further scaling. It is the analytic answer to #3176's "does decode
// parallelize across cores?" — the ceiling grows linearly with cores until BusBound, and the
// gap between SingleThreadTokPerSec and MultiThreadTokPerSec is the parallel headroom a
// single-threaded kernel forgoes.
type DecodeThreadScaling struct {
	Threads               int     // cores the kernel engages to stream weights (T)
	SingleThreadTokPerSec float64 // ceiling with ONE streaming core: perCoreBytesPerSec ÷ WeightBytesPerToken
	MultiThreadTokPerSec  float64 // ceiling with Threads cores: min(T·perCore, aggregate) ÷ WeightBytesPerToken
	SaturatingThreads     int     // cores that saturate the aggregate bus: ⌈aggregate ÷ perCore⌉ (≥1)
	Speedup               float64 // MultiThreadTokPerSec ÷ SingleThreadTokPerSec — the parallel headroom (≥1)
	BusBound              bool    // Threads ≥ SaturatingThreads — the memory bus, not the core count, is the wall
}

// DecodeThreadScalingProfile builds the thread-scaling model for a geometry from the two device
// bandwidths and the core count the kernel engages. perCoreBytesPerSec is the weight-stream
// bandwidth a single core sustains; aggregateBytesPerSec is the socket's peak memory bandwidth
// across all controllers. Both are the caller's measurements. Guards: a non-positive per-core
// bandwidth, non-positive threads, or an empty geometry yields the zero model (no scaling claim
// and no divide-by-zero). An aggregate below a single core's bandwidth is clamped up to it —
// a caller who knows only the per-core figure (aggregate 0/unset) then degrades to "one core's
// worth, no bus headroom known" (SaturatingThreads 1, Speedup 1) rather than a bogus sub-core bus.
func DecodeThreadScalingProfile(g PrefillGeometry, perCoreBytesPerSec, aggregateBytesPerSec float64, threads int) DecodeThreadScaling {
	wb := DecodeWeightBytes(g)
	ts := DecodeThreadScaling{Threads: threads}
	if wb <= 0 || perCoreBytesPerSec <= 0 || threads <= 0 {
		return ts
	}
	agg := aggregateBytesPerSec
	if agg < perCoreBytesPerSec {
		agg = perCoreBytesPerSec // one core already reaches at least its own bandwidth; the bus is never below that
	}
	fwb := float64(wb)
	ts.SingleThreadTokPerSec = perCoreBytesPerSec / fwb
	engaged := perCoreBytesPerSec * float64(threads)
	if engaged > agg {
		engaged = agg // more cores than the bus can feed: bandwidth caps the stream
	}
	ts.MultiThreadTokPerSec = engaged / fwb
	ts.SaturatingThreads = int(math.Ceil(agg / perCoreBytesPerSec))
	if ts.SaturatingThreads < 1 {
		ts.SaturatingThreads = 1
	}
	if ts.SingleThreadTokPerSec > 0 {
		ts.Speedup = ts.MultiThreadTokPerSec / ts.SingleThreadTokPerSec
	}
	ts.BusBound = threads >= ts.SaturatingThreads
	return ts
}

// singleThreadedThreshold is the EffectiveThreads value below which a decode is graded
// "effectively single-threaded": a throughput implying fewer than this many streaming cores,
// on a box whose bus could feed at least this many, is the single-core signature. 2.0 means
// "the observed throughput is consistent with ~one core, not two" — robust to the exact per-core
// bandwidth figure, and it also serves as the minimum Speedup that must EXIST for the verdict to
// fire (so a genuinely single-core box, where there is no parallelism to forgo, is never flagged).
const singleThreadedThreshold = 2.0

// DecodeParallelismVerdict grades a measured decode throughput against the thread-scaling model,
// turning #3176's "single-threaded?" question into a number. EffectiveThreads is how many cores'
// worth of memory bandwidth the observed tok/s represents (observed ÷ single-thread ceiling);
// Utilization is the observed share of the ENGAGED-core ceiling (observed ÷ multi-thread ceiling).
// SingleThreaded is the one-bit answer: the throughput implies ~one streaming core while the box
// has real multi-core headroom — the state the issue reports (~0.8 tok/s where the socket's bus
// could serve dozens of cores' worth).
type DecodeParallelismVerdict struct {
	ObservedTokPerSec     float64
	SingleThreadTokPerSec float64 // one-core ceiling from the model
	MultiThreadTokPerSec  float64 // engaged-core ceiling from the model
	EffectiveThreads      float64 // observed ÷ SingleThreadTokPerSec — cores the throughput implies are streaming
	Utilization           float64 // observed ÷ MultiThreadTokPerSec — share of the engaged-core ceiling reached
	SingleThreaded        bool    // EffectiveThreads < threshold while multi-core headroom exists — "effectively single-threaded"
}

// GradeDecodeParallelism folds one wall-clock decode observation (tokens over seconds) and the
// device's per-core/aggregate bandwidths + engaged core count into the parallelism verdict. It
// is the single call a /metrics parallelism gauge or a bench harness makes to answer "is this
// box's decode using its cores, or is it pinned to one?" — no timer of its own, just the caller's
// one measurement against the exact host-computed thread-scaling ceilings. The SingleThreaded
// verdict fires only when there is genuine parallel headroom to forgo (Speedup ≥ threshold), so a
// truly single-core host is never mislabeled: the issue's 256-core box is, its bus feeding dozens.
func GradeDecodeParallelism(tokens int, seconds float64, g PrefillGeometry, perCoreBytesPerSec, aggregateBytesPerSec float64, threads int) DecodeParallelismVerdict {
	ts := DecodeThreadScalingProfile(g, perCoreBytesPerSec, aggregateBytesPerSec, threads)
	obs := ObservedTokPerSec(tokens, seconds)
	v := DecodeParallelismVerdict{
		ObservedTokPerSec:     obs,
		SingleThreadTokPerSec: ts.SingleThreadTokPerSec,
		MultiThreadTokPerSec:  ts.MultiThreadTokPerSec,
	}
	if ts.SingleThreadTokPerSec > 0 {
		v.EffectiveThreads = obs / ts.SingleThreadTokPerSec
	}
	if ts.MultiThreadTokPerSec > 0 {
		v.Utilization = obs / ts.MultiThreadTokPerSec
	}
	v.SingleThreaded = obs > 0 &&
		v.EffectiveThreads < singleThreadedThreshold && // observed implies ~one streaming core
		ts.Speedup >= singleThreadedThreshold // and there was real multi-core headroom to forgo
	return v
}
