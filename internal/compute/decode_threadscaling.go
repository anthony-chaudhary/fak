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

// DecodeNUMALocality is the NUMA-node-aware refinement of the thread-scaling model, added for
// #3176's newest datapoint: an external report on the issue observed that on the reporter's
// multi-die EPYC, decode workers spread across NUMA nodes fetch single-node-resident weights over
// the inter-die fabric (Infinity Fabric) instead of the local memory channels — the "threads
// roaming across nodes" collapse that drags a 256-thread box to sub-1 tok/s. The flat
// DecodeThreadScalingProfile above treats the socket as ONE aggregate bandwidth pool; that ceiling
// is honest only if every streaming core reaches the weights at full bandwidth. For batch-1 decode
// the weights are resident on a SINGLE node (first-touch on the loading core, or an explicit
// `numactl -m`), so the real ceiling is that node's LOCAL bandwidth with that node's cores, and
// workers placed on other nodes stream the same bytes across the fabric — adding latency, not
// bandwidth. This rung computes the local-node saturating thread count (the "how many threads
// within one node" the report asks for) and flags when a requested worker count spills off the
// node. It is a witness, not a scheduler: it does not place threads, it quantifies the lever
// (`numactl -N k -m k` + a within-node worker count) so an operator or the budget heuristic can.
type DecodeNUMALocality struct {
	Nodes                  int     // NUMA nodes across the socket
	CoresPerNode           int     // cores homed on each node
	RequestedThreads       int     // decode workers the kernel would dispatch (e.g. the budget's cap)
	LocalSaturatingThreads int     // workers that saturate ONE node's local bandwidth: min(⌈perNodeBW÷perCore⌉, CoresPerNode) — the recommended within-node decode width
	LocalCeilingTokPerSec  float64 // tok/s a single-node-resident decode reaches: min(LocalSaturatingThreads·perCore, perNodeBW) ÷ WeightBytesPerToken
	Roams                  bool    // RequestedThreads > CoresPerNode — workers spill onto remote nodes and fetch weights over the fabric
	RoamingThreads         int     // how many requested workers land OFF the local node (0 if none) — the fabric-bound remainder
}

// DecodeNUMALocalityProfile builds the NUMA-locality model for a geometry from the single-core
// weight-stream bandwidth, ONE node's local memory bandwidth, and the socket's node/core topology.
// perCoreBytesPerSec is the same per-core figure DecodeThreadScalingProfile takes; perNodeBytesPerSec
// is a SINGLE node's local memory bandwidth (across that node's controllers only), not the socket
// aggregate; coresPerNode and nodes describe the die topology; requestedThreads is the worker count
// whose placement we are judging (e.g. the 16 the many-core cap picks, or an unpinned 256). Guards:
// a non-positive per-core bandwidth, cores-per-node, node count, thread count, or empty geometry
// yields the zero model (no locality claim, no divide-by-zero). A per-node bandwidth below a single
// core's is clamped up to it — a node always reaches at least one of its cores' bandwidth.
func DecodeNUMALocalityProfile(g PrefillGeometry, perCoreBytesPerSec, perNodeBytesPerSec float64, coresPerNode, nodes, requestedThreads int) DecodeNUMALocality {
	wb := DecodeWeightBytes(g)
	m := DecodeNUMALocality{Nodes: nodes, CoresPerNode: coresPerNode, RequestedThreads: requestedThreads}
	if wb <= 0 || perCoreBytesPerSec <= 0 || coresPerNode <= 0 || nodes <= 0 || requestedThreads <= 0 {
		return DecodeNUMALocality{} // zero model — no topology or geometry to reason about
	}
	perNode := perNodeBytesPerSec
	if perNode < perCoreBytesPerSec {
		perNode = perCoreBytesPerSec // a node reaches at least one of its cores' bandwidth
	}
	fwb := float64(wb)
	// Threads that saturate the node's LOCAL bandwidth, but never more cores than the node has.
	satByBW := int(math.Ceil(perNode / perCoreBytesPerSec))
	if satByBW < 1 {
		satByBW = 1
	}
	local := satByBW
	if local > coresPerNode {
		local = coresPerNode // a node has only this many cores to stream locally
	}
	m.LocalSaturatingThreads = local
	engaged := perCoreBytesPerSec * float64(local)
	if engaged > perNode {
		engaged = perNode // the node's local bus caps its own cores
	}
	m.LocalCeilingTokPerSec = engaged / fwb
	// A worker count beyond one node's cores MUST place threads on other nodes, where they fetch
	// the single-node-resident weights across the fabric — that is the roaming regime.
	if requestedThreads > coresPerNode {
		m.Roams = true
		m.RoamingThreads = requestedThreads - coresPerNode
	}
	return m
}
