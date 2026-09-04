package model

import (
	"math"
	"testing"
)

func TestOptimalMissPartitionRatio(t *testing.T) {
	tests := []struct {
		name   string
		pcieBW float64
		cpuBW  float64
		want   float64
	}{
		{"equal_bandwidth", 50e9, 50e9, 0.5},
		{"pcie_one_third", 25e9, 50e9, 1.0 / 3.0},
		{"pcie_zero", 0, 50e9, 0.0},
		{"cpu_zero", 25e9, 0, 1.0},
		{"both_zero", 0, 0, 1.0},
		{"consumer_4070_profile", 24e9, 72e9, 24.0 / 96.0}, // 0.25
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := OptimalMissPartitionRatio(tt.pcieBW, tt.cpuBW)
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("OptimalMissPartitionRatio(%g, %g) = %g, want %g", tt.pcieBW, tt.cpuBW, got, tt.want)
			}
		})
	}
}

func TestSolveBandwidthPartition_BalancedSplit(t *testing.T) {
	// 25 GB/s PCIe vs 50 GB/s CPU -> 1/3 PCIe, 2/3 CPU.
	// 6 misses of 100 MB each.
	// Expected optimal split: 2 experts to PCIe, 4 experts to CPU.
	// PCIe time: 2 * 0.1 GB / 25 GB/s = 0.008s.
	// CPU time: 4 * 0.1 GB / 50 GB/s = 0.008s.
	// Skew: 0.0s.
	profile := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec: 25e9,
		CPUBandwidthBytesPerSec:  50e9,
		ExpertSizeBytes:          100 * 1024 * 1024, // 100 MiB
		HostCPUEnabled:           true,
	}

	misses := []MoEMissCandidate{
		{ExpertID: 10, Recency: 100},
		{ExpertID: 11, Recency: 200},
		{ExpertID: 12, Recency: 300},
		{ExpertID: 13, Recency: 400},
		{ExpertID: 14, Recency: 500},
		{ExpertID: 15, Recency: 600},
	}

	split, err := SolveBandwidthPartition(profile, misses)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(split.PCIeExperts) != 2 {
		t.Errorf("got %d PCIe experts, want 2", len(split.PCIeExperts))
	}
	if len(split.CPUExperts) != 4 {
		t.Errorf("got %d CPU experts, want 4", len(split.CPUExperts))
	}

	// Verify the top recencies (600, 500) went to PCIe
	if split.PCIeExperts[0].ExpertID != 15 || split.PCIeExperts[1].ExpertID != 14 {
		t.Errorf("PCIe experts = %v, want experts 15 and 14", split.PCIeExperts)
	}

	if math.Abs(split.EstimatedPCIeTimeSec-split.EstimatedCPUTimeSec) > 1e-9 {
		t.Errorf("completion times not balanced: pcie=%g, cpu=%g, skew=%g",
			split.EstimatedPCIeTimeSec, split.EstimatedCPUTimeSec, split.SkewSec)
	}
}

func TestSolveBandwidthPartition_RecencyOrderingAndTieBreak(t *testing.T) {
	profile := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec: 30e9,
		CPUBandwidthBytesPerSec:  30e9,
		ExpertSizeBytes:          50 * 1024 * 1024,
		HostCPUEnabled:           true,
	}

	// 4 misses: two with Recency=500 (Expert 3 and Expert 1), two with Recency=100.
	misses := []MoEMissCandidate{
		{ExpertID: 3, Recency: 500},
		{ExpertID: 4, Recency: 100},
		{ExpertID: 1, Recency: 500},
		{ExpertID: 2, Recency: 100},
	}

	split, err := SolveBandwidthPartition(profile, misses)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(split.PCIeExperts) != 2 || len(split.CPUExperts) != 2 {
		t.Fatalf("expected 2 PCIe and 2 CPU, got %d and %d", len(split.PCIeExperts), len(split.CPUExperts))
	}

	// Tie-break between ID 1 and ID 3 (both Recency 500) should order ID 1 before ID 3
	if split.PCIeExperts[0].ExpertID != 1 || split.PCIeExperts[1].ExpertID != 3 {
		t.Errorf("expected PCIe experts [1, 3], got %v", split.PCIeExperts)
	}
	// Tie-break between ID 2 and ID 4 (both Recency 100) should order ID 2 before ID 4
	if split.CPUExperts[0].ExpertID != 2 || split.CPUExperts[1].ExpertID != 4 {
		t.Errorf("expected CPU experts [2, 4], got %v", split.CPUExperts)
	}
}

func TestSolveBandwidthPartition_DegradationModes(t *testing.T) {
	baseProfile := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec: 25e9,
		CPUBandwidthBytesPerSec:  50e9,
		ExpertSizeBytes:          100 * 1024 * 1024,
		HostCPUEnabled:           true,
	}

	misses := []MoEMissCandidate{
		{ExpertID: 1, Recency: 10},
		{ExpertID: 2, Recency: 20},
		{ExpertID: 3, Recency: 30},
	}

	t.Run("empty_misses", func(t *testing.T) {
		split, err := SolveBandwidthPartition(baseProfile, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(split.PCIeExperts) != 0 || len(split.CPUExperts) != 0 {
			t.Errorf("expected empty split, got pcie=%d cpu=%d", len(split.PCIeExperts), len(split.CPUExperts))
		}
	})

	t.Run("cpu_disabled", func(t *testing.T) {
		p := baseProfile
		p.HostCPUEnabled = false
		split, err := SolveBandwidthPartition(p, misses)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(split.PCIeExperts) != 3 || len(split.CPUExperts) != 0 {
			t.Errorf("expected all 3 to PCIe, got pcie=%d cpu=%d", len(split.PCIeExperts), len(split.CPUExperts))
		}
	})

	t.Run("cpu_zero_bandwidth", func(t *testing.T) {
		p := baseProfile
		p.CPUBandwidthBytesPerSec = 0
		split, err := SolveBandwidthPartition(p, misses)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(split.PCIeExperts) != 3 || len(split.CPUExperts) != 0 {
			t.Errorf("expected all 3 to PCIe, got pcie=%d cpu=%d", len(split.PCIeExperts), len(split.CPUExperts))
		}
	})

	t.Run("pcie_zero_bandwidth", func(t *testing.T) {
		p := baseProfile
		p.PCIeBandwidthBytesPerSec = 0
		split, err := SolveBandwidthPartition(p, misses)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(split.PCIeExperts) != 0 || len(split.CPUExperts) != 3 {
			t.Errorf("expected all 3 to CPU, got pcie=%d cpu=%d", len(split.PCIeExperts), len(split.CPUExperts))
		}
	})
}

func TestSolveBandwidthPartition_Validation(t *testing.T) {
	tests := []struct {
		name    string
		profile MoEBandwidthProfile
	}{
		{"negative_pcie", MoEBandwidthProfile{PCIeBandwidthBytesPerSec: -1}},
		{"negative_cpu", MoEBandwidthProfile{CPUBandwidthBytesPerSec: -1}},
		{"negative_expert_bytes", MoEBandwidthProfile{ExpertSizeBytes: -1}},
		{"negative_gpu_latency", MoEBandwidthProfile{GPUComputeLatencySecPerExpert: -1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := SolveBandwidthPartition(tt.profile, []MoEMissCandidate{{ExpertID: 1}})
			if err == nil {
				t.Errorf("expected error for invalid profile, got nil")
			}
		})
	}
}

func TestSolveBandwidthPartition_OptimalityCheck(t *testing.T) {
	// For asymmetric bandwidths (e.g. 18 GB/s PCIe vs 65 GB/s CPU, 8 misses),
	// verify that the returned m* has max(T_pcie, T_cpu) <= any other choice of m in [0, 8].
	profile := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec:      18e9,
		CPUBandwidthBytesPerSec:       65e9,
		ExpertSizeBytes:               80 * 1024 * 1024,
		GPUComputeLatencySecPerExpert: 0.0001, // 0.1 ms
		HostCPUEnabled:                true,
	}

	misses := make([]MoEMissCandidate, 8)
	for i := range misses {
		misses[i] = MoEMissCandidate{ExpertID: i, Recency: int64(i * 10)}
	}

	split, err := SolveBandwidthPartition(profile, misses)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	chosenPcieCount := len(split.PCIeExperts)
	chosenCpuCount := len(split.CPUExperts)

	pTime, cTime := profile.EstimateChannelTimes(chosenPcieCount, chosenCpuCount)
	chosenMax := math.Max(pTime, cTime)

	for m := 0; m <= len(misses); m++ {
		pt, ct := profile.EstimateChannelTimes(m, len(misses)-m)
		candidateMax := math.Max(pt, ct)
		if candidateMax < chosenMax-1e-9 {
			t.Errorf("suboptimal choice: m=%d gave max=%g < chosen m=%d max=%g",
				m, candidateMax, chosenPcieCount, chosenMax)
		}
	}
}
