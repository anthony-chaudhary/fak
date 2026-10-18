//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"testing"
	"time"
)

func vulkanMemoryCapacityFixture() VulkanMemoryCapacitySnapshot {
	now := time.Unix(1, 0)
	return VulkanMemoryCapacitySnapshot{
		DeviceGeneration: 1, DeviceName: "synthetic", StartedAt: now, CompletedAt: now,
		Guard: VulkanMemoryGuardSnapshot{Scope: "all_vk_device_memory", LimitBytes: 4096,
			LiveBytes: 1024, PeakBytes: 2048, Configured: true, ConfigurationValid: true, AccountingValid: true},
		MaxBufferBytes: 2048, MaxStorageBufferRange: 2048,
		MaxMemoryAllocationSize: 4096, MaxMemoryAllocationCount: 16,
		Heaps: []VulkanMemoryCapacityHeap{{SizeBytes: 8192, BudgetBytes: 4096, UsageBytes: 2048, BudgetKnown: true}},
		Types: []VulkanMemoryCapacityType{{HeapIndex: 0}},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanMemoryCapacityUnknownAndPressure(t *testing.T) {
	s := vulkanMemoryCapacityFixture()
	if err := validateVulkanMemoryCapacitySnapshot(s); err != nil {
		t.Fatal(err)
	}
	s.Heaps[0].UsageBytes = s.Heaps[0].BudgetBytes + 1
	if err := validateVulkanMemoryCapacitySnapshot(s); err != nil {
		t.Fatalf("pressure must remain observable: %v", err)
	}
	s.Heaps[0].BudgetKnown = false
	s.Heaps[0].BudgetBytes, s.Heaps[0].UsageBytes = 0, 0
	s.Guard.Configured, s.Guard.LimitBytes = false, 0
	s.MaxMemoryAllocationSize = 0
	if err := validateVulkanMemoryCapacitySnapshot(s); err != nil {
		t.Fatalf("explicit unknowns must remain observable: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanMemoryCapacityInvalidMetadata(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*VulkanMemoryCapacitySnapshot)
	}{
		{"generation", func(s *VulkanMemoryCapacitySnapshot) { s.DeviceGeneration = 0 }},
		{"interval", func(s *VulkanMemoryCapacitySnapshot) { s.CompletedAt = s.StartedAt.Add(-time.Nanosecond) }},
		{"accounting", func(s *VulkanMemoryCapacitySnapshot) { s.Guard.AccountingValid = false }},
		{"configuration", func(s *VulkanMemoryCapacitySnapshot) { s.Guard.ConfigurationValid = false }},
		{"live-peak", func(s *VulkanMemoryCapacitySnapshot) { s.Guard.LiveBytes = s.Guard.PeakBytes + 1 }},
		{"limit", func(s *VulkanMemoryCapacitySnapshot) { s.Guard.LimitBytes = 1 }},
		{"unknown-limit", func(s *VulkanMemoryCapacitySnapshot) { s.Guard.Configured = false }},
		{"empty-heaps", func(s *VulkanMemoryCapacitySnapshot) { s.Heaps = nil }},
		{"heaps-overflow", func(s *VulkanMemoryCapacitySnapshot) { s.Heaps = make([]VulkanMemoryCapacityHeap, 17) }},
		{"types-overflow", func(s *VulkanMemoryCapacitySnapshot) { s.Types = make([]VulkanMemoryCapacityType, 33) }},
		{"heap-index", func(s *VulkanMemoryCapacitySnapshot) { s.Types[0].HeapIndex = ^uint32(0) }},
		{"heap-size", func(s *VulkanMemoryCapacitySnapshot) { s.Heaps[0].SizeBytes = 0 }},
		{"unknown-budget", func(s *VulkanMemoryCapacitySnapshot) { s.Heaps[0].BudgetKnown = false }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := vulkanMemoryCapacityFixture()
			tc.mutate(&s)
			if err := validateVulkanMemoryCapacitySnapshot(s); err == nil {
				t.Fatal("invalid observation accepted")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanMemoryCapacityFaultsStayVisible(t *testing.T) {
	s := vulkanMemoryCapacityFixture()
	s.Quarantined, s.BatchActive, s.InflightActive, s.RestoreActive = true, true, true, true
	s.SubmissionStatus = -4
	if err := validateVulkanMemoryCapacitySnapshot(s); err != nil {
		t.Fatalf("valid fault observation refused: %v", err)
	}
	if !s.Quarantined || !s.BatchActive || !s.InflightActive || !s.RestoreActive || s.SubmissionStatus != -4 {
		t.Fatal("observation cleared backend fault")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanMemoryCapacityNilBackend(t *testing.T) {
	var backend *vulkanBackend
	if _, err := backend.VulkanMemoryCapacitySnapshot(); err == nil {
		t.Fatal("nil backend supplied a snapshot")
	}
}
