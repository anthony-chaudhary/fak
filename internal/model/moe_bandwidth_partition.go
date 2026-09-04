package model

import (
	"fmt"
	"math"
	"slices"
)

// moe_bandwidth_partition.go — intra-step bandwidth-adaptive miss partitioning
// (FreeToken q* co-execution solver), #11038.
//
// In an offloaded Mixture-of-Experts serve, expert weights exceed GPU VRAM and are
// backed by host RAM. During a decode step, expert ring cache misses incur latency:
// streaming all misses over PCIe stalls the GPU, while computing all misses on the
// host CPU leaves the GPU idle.
//
// Because consumer PCIe interconnects (e.g. PCIe Gen4 x16 ~24-28 GB/s or x8 ~12-14 GB/s)
// and host memory channels (e.g. DDR5 dual-channel ~60-80 GB/s) operate independently,
// this solver calculates the optimal integer partition m* of K missing experts between
// PCIe streaming to GPU and concurrent host CPU execution such that:
//
//	T_pcie(m) = m * (ExpertBytes / PCIe_BW + GPU_latency)
//	T_cpu(K - m) = (K - m) * (ExpertBytes / CPU_BW)
//	T_total(m) = max(T_pcie(m), T_cpu(K - m))
//
// The solver selects m* in [0, K] to minimize T_total(m) and skew |T_pcie - T_cpu|.
// The top m* missing experts by recency are assigned to PCIe streaming (where they
// enter the GPU's paged expert ring cache for future hits), while the remaining
// (K - m*) overflow experts execute concurrently on host CPU without polluting VRAM.

// MoEMissCandidate represents an activated expert that missed the device ring cache.
type MoEMissCandidate struct {
	// ExpertID is the model's global or layer-local expert ordinal.
	ExpertID int
	// Recency is a timestamp, logical sequence number, or cache access score.
	// Higher recency indicates the expert was used more recently or has higher priority
	// to become device-resident in the GPU LRU ring.
	Recency int64
}

// MoEBandwidthProfile defines the measured hardware transfer and execution rates.
type MoEBandwidthProfile struct {
	// PCIeBandwidthBytesPerSec is the measured DMA transfer throughput over PCIe (e.g. 24e9 for 24 GB/s).
	PCIeBandwidthBytesPerSec float64
	// CPUBandwidthBytesPerSec is the effective host memory bandwidth or CPU GEMV throughput (e.g. 60e9 for 60 GB/s).
	CPUBandwidthBytesPerSec float64
	// ExpertSizeBytes is the size in bytes of one routed expert's weights.
	ExpertSizeBytes int64
	// GPUComputeLatencySecPerExpert is the execution latency on GPU once weights arrive (default 0).
	GPUComputeLatencySecPerExpert float64
	// HostCPUEnabled indicates whether host CPU co-execution is supported/enabled.
	// If false, all misses are assigned to PCIe streaming.
	HostCPUEnabled bool
}

// Validate checks that profile parameters are sane and non-negative.
func (p MoEBandwidthProfile) Validate() error {
	if p.PCIeBandwidthBytesPerSec < 0 {
		return fmt.Errorf("model: PCIeBandwidthBytesPerSec = %g, want >= 0", p.PCIeBandwidthBytesPerSec)
	}
	if p.CPUBandwidthBytesPerSec < 0 {
		return fmt.Errorf("model: CPUBandwidthBytesPerSec = %g, want >= 0", p.CPUBandwidthBytesPerSec)
	}
	if p.ExpertSizeBytes < 0 {
		return fmt.Errorf("model: ExpertSizeBytes = %d, want >= 0", p.ExpertSizeBytes)
	}
	if p.GPUComputeLatencySecPerExpert < 0 {
		return fmt.Errorf("model: GPUComputeLatencySecPerExpert = %g, want >= 0", p.GPUComputeLatencySecPerExpert)
	}
	return nil
}

// MoEPartitionSplit represents the optimal integer division of missing experts.
type MoEPartitionSplit struct {
	// PCIeExperts are the experts assigned to PCIe streaming to device.
	PCIeExperts []MoEMissCandidate
	// CPUExperts are the experts assigned to concurrent host CPU execution.
	CPUExperts []MoEMissCandidate
	// EstimatedPCIeTimeSec is the estimated duration for PCIe transfer + compute.
	EstimatedPCIeTimeSec float64
	// EstimatedCPUTimeSec is the estimated duration for host CPU compute.
	EstimatedCPUTimeSec float64
	// CompletionTimeSec is max(EstimatedPCIeTimeSec, EstimatedCPUTimeSec).
	CompletionTimeSec float64
	// SkewSec is |EstimatedPCIeTimeSec - EstimatedCPUTimeSec|.
	SkewSec float64
	// QStar is the continuous optimal PCIe fraction: PCIe_BW / (PCIe_BW + CPU_BW).
	QStar float64
}

// OptimalMissPartitionRatio calculates the continuous optimal fraction q* of work
// that should be routed to PCIe streaming to balance completion times across channels.
// Returns a value in [0.0, 1.0].
func OptimalMissPartitionRatio(pcieBW, cpuBW float64) float64 {
	if pcieBW <= 0 && cpuBW <= 0 {
		return 1.0 // fallback to PCIe
	}
	if pcieBW <= 0 {
		return 0.0 // PCIe unavailable, route to CPU
	}
	if cpuBW <= 0 {
		return 1.0 // CPU unavailable, route to PCIe
	}
	return pcieBW / (pcieBW + cpuBW)
}

// EstimateChannelTimes computes the execution times for a given partition of counts.
func (p MoEBandwidthProfile) EstimateChannelTimes(pcieCount, cpuCount int) (pcieTime, cpuTime float64) {
	if pcieCount > 0 && p.PCIeBandwidthBytesPerSec > 0 && p.ExpertSizeBytes > 0 {
		transfer := (float64(pcieCount) * float64(p.ExpertSizeBytes)) / p.PCIeBandwidthBytesPerSec
		compute := float64(pcieCount) * p.GPUComputeLatencySecPerExpert
		pcieTime = transfer + compute
	} else if pcieCount > 0 {
		pcieTime = math.Inf(1)
	}

	if cpuCount > 0 && p.CPUBandwidthBytesPerSec > 0 && p.ExpertSizeBytes > 0 {
		cpuTime = (float64(cpuCount) * float64(p.ExpertSizeBytes)) / p.CPUBandwidthBytesPerSec
	} else if cpuCount > 0 {
		cpuTime = math.Inf(1)
	}

	return pcieTime, cpuTime
}

// SolveBandwidthPartition finds the integer split of missing experts that minimizes
// the bottleneck completion time max(T_pcie, T_cpu) and execution skew.
func SolveBandwidthPartition(profile MoEBandwidthProfile, misses []MoEMissCandidate) (MoEPartitionSplit, error) {
	if err := profile.Validate(); err != nil {
		return MoEPartitionSplit{}, err
	}

	n := len(misses)
	if n == 0 {
		return MoEPartitionSplit{
			PCIeExperts: []MoEMissCandidate{},
			CPUExperts:  []MoEMissCandidate{},
			QStar:       OptimalMissPartitionRatio(profile.PCIeBandwidthBytesPerSec, profile.CPUBandwidthBytesPerSec),
		}, nil
	}

	// Deterministically sort misses by Recency descending; tie-break on ExpertID ascending.
	sortedMisses := make([]MoEMissCandidate, n)
	copy(sortedMisses, misses)
	slices.SortFunc(sortedMisses, func(a, b MoEMissCandidate) int {
		if a.Recency != b.Recency {
			if a.Recency > b.Recency {
				return -1
			}
			return 1
		}
		if a.ExpertID < b.ExpertID {
			return -1
		} else if a.ExpertID > b.ExpertID {
			return 1
		}
		return 0
	})

	qStar := OptimalMissPartitionRatio(profile.PCIeBandwidthBytesPerSec, profile.CPUBandwidthBytesPerSec)

	// Graceful degradation: if CPU co-execution is disabled or CPU bandwidth is 0,
	// all misses go to PCIe.
	if !profile.HostCPUEnabled || profile.CPUBandwidthBytesPerSec <= 0 {
		pTime, cTime := profile.EstimateChannelTimes(n, 0)
		return MoEPartitionSplit{
			PCIeExperts:          sortedMisses,
			CPUExperts:           []MoEMissCandidate{},
			EstimatedPCIeTimeSec: pTime,
			EstimatedCPUTimeSec:  cTime,
			CompletionTimeSec:    pTime,
			SkewSec:              pTime,
			QStar:                qStar,
		}, nil
	}

	// If PCIe bandwidth is 0 or unavailable, all misses go to CPU.
	if profile.PCIeBandwidthBytesPerSec <= 0 {
		pTime, cTime := profile.EstimateChannelTimes(0, n)
		return MoEPartitionSplit{
			PCIeExperts:          []MoEMissCandidate{},
			CPUExperts:           sortedMisses,
			EstimatedPCIeTimeSec: pTime,
			EstimatedCPUTimeSec:  cTime,
			CompletionTimeSec:    cTime,
			SkewSec:              cTime,
			QStar:                qStar,
		}, nil
	}

	// Evaluate all integer partitions m in [0, n] to find the minimum max(T_pcie, T_cpu).
	bestM := 0
	bestMaxTime := math.Inf(1)
	bestSkew := math.Inf(1)
	bestPTime := 0.0
	bestCTime := 0.0

	for m := 0; m <= n; m++ {
		pTime, cTime := profile.EstimateChannelTimes(m, n-m)
		maxTime := math.Max(pTime, cTime)
		skew := math.Abs(pTime - cTime)

		if maxTime < bestMaxTime || (math.Abs(maxTime-bestMaxTime) < 1e-12 && skew < bestSkew) {
			bestM = m
			bestMaxTime = maxTime
			bestSkew = skew
			bestPTime = pTime
			bestCTime = cTime
		}
	}

	return MoEPartitionSplit{
		PCIeExperts:          sortedMisses[:bestM],
		CPUExperts:           sortedMisses[bestM:],
		EstimatedPCIeTimeSec: bestPTime,
		EstimatedCPUTimeSec:  bestCTime,
		CompletionTimeSec:    bestMaxTime,
		SkewSec:              bestSkew,
		QStar:                qStar,
	}, nil
}
