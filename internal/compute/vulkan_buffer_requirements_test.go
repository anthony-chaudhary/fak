//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"strings"
	"testing"
	"time"
)

func vulkanBufferRequirementsFixture() ([]VulkanBufferRequirementsRequest, VulkanBufferRequirementsSnapshot) {
	requests := []VulkanBufferRequirementsRequest{
		{SizeBytes: 1000, Profile: VulkanBufferRequirementsStorage},
		{SizeBytes: 2048, Profile: VulkanBufferRequirementsTransfer},
	}
	now := time.Unix(1, 0)
	s := VulkanBufferRequirementsSnapshot{
		DeviceGeneration: 1, MemoryTypeCount: 2, StartedAt: now, CompletedAt: now,
		Rows: []VulkanBufferRequirementsRow{
			{Request: requests[0], RequiredBytes: 1024, Alignment: 256, MemoryTypeBits: 1},
			{Request: requests[1], RequiredBytes: 4096, Alignment: 4096, MemoryTypeBits: 3, RequiresDedicated: true, PrefersDedicated: true},
		},
	}
	return requests, s
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanBufferRequirementsBounds(t *testing.T) {
	requests, _ := vulkanBufferRequirementsFixture()
	if err := validateVulkanBufferRequirementsRequests(requests); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range [][]VulkanBufferRequirementsRequest{
		nil, make([]VulkanBufferRequirementsRequest, vulkanBufferRequirementsMaxProbes+1),
		{{SizeBytes: 0, Profile: VulkanBufferRequirementsStorage}},
		{{SizeBytes: 1 << 63, Profile: VulkanBufferRequirementsStorage}},
		{{SizeBytes: 1, Profile: 0}},
		{{SizeBytes: 1, Profile: 3}},
	} {
		if err := validateVulkanBufferRequirementsRequests(invalid); err == nil {
			t.Fatal("invalid request accepted")
		}
	}
	atLimit := make([]VulkanBufferRequirementsRequest, vulkanBufferRequirementsMaxProbes)
	for i := range atLimit {
		atLimit[i] = requests[i%len(requests)]
	}
	if err := validateVulkanBufferRequirementsRequests(atLimit); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanBufferRequirementsDriverFacts(t *testing.T) {
	requests, s := vulkanBufferRequirementsFixture()
	if err := validateVulkanBufferRequirementsSnapshot(requests, s); err != nil {
		t.Fatal(err)
	}
	if s.Rows[1].RequiredBytes == requests[1].SizeBytes || !s.Rows[1].RequiresDedicated ||
		!s.Rows[1].PrefersDedicated || s.Rows[1].MemoryTypeBits != 3 {
		t.Fatal("actual padding, type mask or dedicated facts lost")
	}
	s.MemoryTypeCount = 32
	s.Rows[0].MemoryTypeBits = 1 << 31
	if err := validateVulkanBufferRequirementsSnapshot(requests, s); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanBufferRequirementsMalformedSnapshot(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*VulkanBufferRequirementsSnapshot)
	}{
		{"generation", func(s *VulkanBufferRequirementsSnapshot) { s.DeviceGeneration = 0 }},
		{"type-zero", func(s *VulkanBufferRequirementsSnapshot) { s.MemoryTypeCount = 0 }},
		{"type-overflow", func(s *VulkanBufferRequirementsSnapshot) { s.MemoryTypeCount = 33 }},
		{"interval", func(s *VulkanBufferRequirementsSnapshot) { s.CompletedAt = s.StartedAt.Add(-time.Nanosecond) }},
		{"partial", func(s *VulkanBufferRequirementsSnapshot) { s.Rows = s.Rows[:1] }},
		{"request", func(s *VulkanBufferRequirementsSnapshot) { s.Rows[1].Request.SizeBytes++ }},
		{"undersized", func(s *VulkanBufferRequirementsSnapshot) { s.Rows[1].RequiredBytes = 1 }},
		{"alignment-zero", func(s *VulkanBufferRequirementsSnapshot) { s.Rows[1].Alignment = 0 }},
		{"alignment-nonpower", func(s *VulkanBufferRequirementsSnapshot) { s.Rows[1].Alignment = 3 }},
		{"mask-zero", func(s *VulkanBufferRequirementsSnapshot) { s.Rows[1].MemoryTypeBits = 0 }},
		{"mask-overflow", func(s *VulkanBufferRequirementsSnapshot) { s.Rows[1].MemoryTypeBits = 4 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			requests, s := vulkanBufferRequirementsFixture()
			tc.mutate(&s)
			if err := validateVulkanBufferRequirementsSnapshot(requests, s); err == nil {
				t.Fatal("malformed snapshot accepted")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanBufferRequirementsOwnerBusy(t *testing.T) {
	v := &vulkanBackend{}
	owner := v.ForwardExecutionOwner()
	owner.Lock()
	defer owner.Unlock()
	requests, _ := vulkanBufferRequirementsFixture()
	got, err := v.VulkanBufferMemoryRequirements(requests)
	if err == nil || !strings.Contains(err.Error(), "owner busy") || got.DeviceGeneration != 0 || got.Rows != nil {
		t.Fatalf("reentrant call did not refuse before native probe: %+v, %v", got, err)
	}
	var absent *vulkanBackend
	if got, err = absent.VulkanBufferMemoryRequirements(requests); err == nil || got.Rows != nil {
		t.Fatal("nil backend supplied requirements")
	}
}
