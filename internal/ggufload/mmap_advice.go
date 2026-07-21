package ggufload

// mmap_advice.go — the PURE choice core that picks the memory-map access hint for a
// GGUF weight map from the host NUMA topology and the access pattern, WITHOUT touching a
// syscall, a file, a clock, or a device (issue #5284; borrowed clean-room from a study of
// llama.cpp @571d0d54, src/llama-mmap.cpp:445-476, verdict inspire — cite only, no bytes
// vendored). The body's rule: on a multi-node NUMA box, mark the weight map RANDOM so
// pages fault local to the thread that first touches them (first-touch) instead of piling
// onto one node and defeating the interleave story fak already invests in
// (internal/compute/decode_interleave.go, numactl --interleave); on a single-node /
// non-NUMA box, hint SEQUENTIAL for a fast bulk streaming load. An unknown / undetectable
// topology fails CLOSED to the neutral NORMAL hint so we never mis-advise a box we could
// not read.
//
// This layer returns an abstract value only. An OS-specific shim (the follow-on) will
// translate ChooseMmapAdvice's result into the real madvise constant (MADV_RANDOM /
// MADV_SEQUENTIAL / MADV_NORMAL) and issue it after the map, degrading to a no-op on
// platforms without madvise (Windows). Keeping the rule here — total, deterministic,
// syscall-free — means it is fully unit-testable without a real NUMA host or a multi-GB
// checkpoint.

// MmapAdvice is the abstract memory-map access hint an OS shim later translates to the
// real madvise constant. The zero value is AdviceNormal, the safe fail-closed default.
type MmapAdvice uint8

const (
	// AdviceNormal is the neutral, no-preference hint (maps to MADV_NORMAL). It is the
	// fail-closed default for an unknown topology.
	AdviceNormal MmapAdvice = iota
	// AdviceRandom hints random access (maps to MADV_RANDOM): disable eager read-ahead so
	// pages fault local to the first-touching thread — the NUMA-multinode and random-expert
	// choice.
	AdviceRandom
	// AdviceSequential hints a forward streaming scan (maps to MADV_SEQUENTIAL): aggressive
	// read-ahead for a fast bulk load — the single-node / non-NUMA weight-stream choice.
	AdviceSequential
)

// String formats the hint as a stable lowercase token for logs and tests.
func (a MmapAdvice) String() string {
	switch a {
	case AdviceRandom:
		return "random"
	case AdviceSequential:
		return "sequential"
	default:
		return "normal"
	}
}

// AccessPattern names how the mapped weights will be read. The zero value is the ordinary
// forward weight stream.
type AccessPattern uint8

const (
	// AccessWeightStream is a forward, mostly-sequential read of the resident weight bytes —
	// the ordinary bulk load.
	AccessWeightStream AccessPattern = iota
	// AccessExpertRandom is scattered, on-demand access of routed expert shards (MoE
	// per-token expert selection): no sequential locality to exploit.
	AccessExpertRandom
)

// NumaTopology is the pure topology signal the hint reads. It carries only what the rule
// needs; the caller probes the host once (or hands in a test fixture) and passes it in.
type NumaTopology struct {
	// NodeCount is the number of online NUMA nodes. Zero or negative means UNKNOWN — the host
	// could not be probed — which fails closed to AdviceNormal.
	NodeCount int
	// IsNuma marks the host as genuinely NUMA (more than one node backing real cross-node
	// latency). A caller may set this false on a single-socket box even with NodeCount >= 1.
	IsNuma bool
	// Interleave records an explicit interleave intent (e.g. numactl --interleave, or fak's
	// own in-process interleave apply). It is a strong signal to treat the map as NUMA and
	// hint random, since striped pages have no single-node sequential locality.
	Interleave bool
}

// multinode reports whether the topology should be treated as multi-node NUMA for the hint.
// An explicit interleave intent forces it; otherwise it needs both the NUMA mark and more
// than one online node.
func (t NumaTopology) multinode() bool {
	if t.Interleave {
		return true
	}
	return t.IsNuma && t.NodeCount > 1
}

// ChooseMmapAdvice is the pure choice core: given the host NUMA topology and the access
// pattern, it returns the memory-map access hint an OS shim will later apply after the
// weight map. The order is deliberate:
//
//  1. Fail closed. An unknown / undetectable topology (NodeCount <= 0) gets AdviceNormal so
//     we never mis-hint a box we could not read.
//  2. Random access overrides topology. Scattered expert-shard reads have no sequential
//     locality on ANY box, so they always want AdviceRandom.
//  3. NUMA weight stream. On a multi-node NUMA box (or under an explicit interleave intent)
//     hint AdviceRandom so pages fault local to the first-touching thread instead of piling
//     onto one node.
//  4. Otherwise. A single-node / non-NUMA weight stream takes AdviceSequential for a fast
//     bulk read-ahead load.
//
// It is a total function of its two inputs — no syscall, no clock, no device — so the OS
// shim inherits a fully checked rule.
func ChooseMmapAdvice(topo NumaTopology, access AccessPattern) MmapAdvice {
	if topo.NodeCount <= 0 {
		return AdviceNormal
	}
	if access == AccessExpertRandom {
		return AdviceRandom
	}
	if topo.multinode() {
		return AdviceRandom
	}
	return AdviceSequential
}
