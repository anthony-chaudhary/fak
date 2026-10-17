package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type q2kGatherBackend struct {
	compute.Backend
	name      string
	available bool
}

func (b q2kGatherBackend) Name() string                         { return b.name }
func (b q2kGatherBackend) V41Q2KEmbeddingGatherAvailable() bool { return b.available }

func TestServeVulkanQ2KEmbeddingGatherRequiresVulkanDeviceGather(t *testing.T) {
	cpu := compute.Pick("cpu-ref")
	for _, tc := range []struct {
		name    string
		backend compute.Backend
		want    bool
	}{
		{"nil", nil, false},
		{"cpu-ref", cpu, false},
		{"vulkan without gather", q2kGatherBackend{Backend: cpu, name: "vulkan"}, false},
		{"other backend with gather", q2kGatherBackend{Backend: cpu, name: "cuda", available: true}, false},
		{"vulkan with gather", q2kGatherBackend{Backend: cpu, name: "vulkan", available: true}, true},
	} {
		if got := serveVulkanQ2KEmbeddingGather(tc.backend); got != tc.want {
			t.Fatalf("%s: serveVulkanQ2KEmbeddingGather=%t want %t", tc.name, got, tc.want)
		}
	}
	if serveQwen38Q2KEmbeddingResident(q2kGatherBackend{Backend: cpu, name: "vulkan", available: true}, "missing.gguf", true) {
		t.Fatal("packed embedding admitted for an artifact that is not the witnessed UD-Q2_K_XL file")
	}
}
