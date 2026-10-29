package agent

import (
	"context"
	"testing"
)

// cudaMTPPlannerBackend is the Vulkan MTP software witness re-identified as
// CUDA: it advertises the same resident sequence paths, so only the backend
// name differs between the two arms.
type cudaMTPPlannerBackend struct {
	*vulkanMTPPlannerBackend
}

func (cudaMTPPlannerBackend) Name() string { return "cuda" }

// TestInKernelPlannerResidentMTPEngineNamesAdmittingBackend pins the receipt
// engine label to the backend that admitted resident MTP: a CUDA run records
// "mtp-cuda" and a Vulkan run keeps the legacy "mtp-vulkan" value.
//
// fak-test:runtime fast est=50ms lane=default
func TestInKernelPlannerResidentMTPEngineNamesAdmittingBackend(t *testing.T) {
	for _, tc := range []struct {
		name, backend, wantEngine string
	}{
		{name: "cuda", backend: "cuda", wantEngine: "mtp-cuda"},
		{name: "vulkan", backend: "vulkan", wantEngine: "mtp-vulkan"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			vulkan := newVulkanMTPPlannerBackend(t)
			var p *InKernelPlanner
			var prompt []int
			if tc.backend == "cuda" {
				p, prompt = vulkanMTPPlanner(t, cudaMTPPlannerBackend{vulkanMTPPlannerBackend: vulkan})
			} else {
				p, prompt = vulkanMTPPlanner(t, vulkan)
			}
			if !p.VulkanMTPEnabled() {
				t.Fatal("resident MTP not enabled")
			}
			res, err := p.generateReusedRecovering(context.Background(), prompt, 2, 0, 0, 0, nil, 0, 0, nil, nil)
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			r := res.vulkanMTP
			if r == nil {
				t.Fatal("missing resident MTP execution receipt")
			}
			if r.Engine != tc.wantEngine || r.Backend != tc.backend {
				t.Fatalf("receipt engine/backend = %q/%q, want %q/%q", r.Engine, r.Backend, tc.wantEngine, tc.backend)
			}
		})
	}
}
