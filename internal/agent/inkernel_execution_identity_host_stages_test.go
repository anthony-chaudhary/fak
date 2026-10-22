package agent

import (
	"bytes"
	"log"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func vulkanQwen35IdentityPlanner() *InKernelPlanner {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	backend := &pathIdentityBackend{
		namedIdentityBackend: &namedIdentityBackend{Backend: compute.Default(), name: "vulkan"},
		path:                 model.Qwen35GDNVulkanPath,
	}
	return &InKernelPlanner{m: m, modelID: "Qwen3.8-27B-UD-Q2_K_XL", backend: backend, q4k: true}
}

func captureExecutionSummary(t *testing.T, p *InKernelPlanner, q8kern, q8fusedMark string) string {
	t.Helper()
	var buf bytes.Buffer
	old := log.Writer()
	log.SetOutput(&buf)
	defer log.SetOutput(old)
	p.logExecutionSummary(q8kern, q8fusedMark, 20, 0, 0, 20, 0.1, 200, 9, 1.0, 8)
	return strings.TrimSpace(buf.String())
}

// summaryKey returns the value of one space-separated key=value token.
func summaryKey(line, key string) (string, bool) {
	for _, field := range strings.Fields(line) {
		if v, ok := strings.CutPrefix(field, key+"="); ok {
			return v, true
		}
	}
	return "", false
}

// TestInKernelExecutionIdentityVulkanSummaryOmitsQ8Dec pins the silent-degradation
// TICKET-01 rule: q8dec is the static CPUID SIMD tier, so a Vulkan-served turn must
// not print it as if the CPU decode path ran.
// fak-test:runtime fast est=10ms lane=default
func TestInKernelExecutionIdentityVulkanSummaryOmitsQ8Dec(t *testing.T) {
	got := captureExecutionSummary(t, vulkanQwen35IdentityPlanner(), "avx512", "+fused")
	if _, ok := summaryKey(got, "q8dec"); ok {
		t.Fatalf("vulkan execution summary %q carries a q8dec clause", got)
	}
	if v, _ := summaryKey(got, "forward_path"); v != model.Qwen35GDNVulkanPath {
		t.Fatalf("vulkan execution summary %q forward_path=%q, want %q", got, v, model.Qwen35GDNVulkanPath)
	}
}

type deviceMemoryIdentityBackend struct{ compute.Backend }

func (b deviceMemoryIdentityBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

// TestInKernelExecutionIdentityVulkanSummaryNamesHostStages asserts host= names
// exactly the host stages that advanced inside the turn's window.
// fak-test:runtime fast est=10ms lane=default
func TestInKernelExecutionIdentityVulkanSummaryNamesHostStages(t *testing.T) {
	p := vulkanQwen35IdentityPlanner()
	if got, _ := summaryKey(captureExecutionSummary(t, p, "avx512", "+fused"), "host"); got != inKernelHostStagesUnobserved {
		t.Fatalf("host= before any window = %q, want %q", got, inKernelHostStagesUnobserved)
	}

	compute.CountHostFallback("embedding_row_upload")
	p.hostStages.begin()
	if got, _ := summaryKey(captureExecutionSummary(t, p, "avx512", "+fused"), "host"); got != inKernelHostStagesNone {
		t.Fatalf("host= for a turn with no host work = %q, want %q (work before begin leaked in)", got, inKernelHostStagesNone)
	}

	p.hostStages.begin()
	observeHostSampledToken(deviceMemoryIdentityBackend{Backend: compute.Default()})
	compute.CountHostFallback("embedding_row_upload")
	got := captureExecutionSummary(t, p, "avx512", "+fused")
	if v, _ := summaryKey(got, "host"); v != inKernelHostStageSample+",embedding_row_upload" {
		t.Fatalf("host= = %q in %q, want sample,embedding_row_upload", v, got)
	}
	if _, ok := summaryKey(got, "q8dec"); ok {
		t.Fatalf("vulkan execution summary %q carries a q8dec clause", got)
	}
}

// TestInKernelExecutionIdentityHostSampleCountsDeviceMemoryOnly keeps a CPU
// session's own token selection out of the host-fallback stages.
// fak-test:runtime fast est=10ms lane=default
func TestInKernelExecutionIdentityHostSampleCountsDeviceMemoryOnly(t *testing.T) {
	var w inKernelHostStageWindow
	w.begin()
	observeHostSampledToken(nil)
	observeHostSampledToken(compute.Default())
	if stages, ok := w.stages(); !ok || len(stages) != 0 {
		t.Fatalf("host stages after CPU-session sampling = %v (ok=%v), want none", stages, ok)
	}
	observeHostSampledToken(deviceMemoryIdentityBackend{Backend: compute.Default()})
	if stages, _ := w.stages(); len(stages) != 1 || stages[0] != inKernelHostStageSample {
		t.Fatalf("host stages after device-memory sampling = %v, want [%s]", stages, inKernelHostStageSample)
	}
}

// TestInKernelExecutionIdentityCPUSessionSummaryKeepsQ8Dec keeps q8dec where the
// CPU session ran the decode, and no host= clause there.
// fak-test:runtime fast est=10ms lane=default
func TestInKernelExecutionIdentityCPUSessionSummaryKeepsQ8Dec(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	for _, p := range []*InKernelPlanner{
		{m: m, modelID: "Qwen3.8-27B-Q4_K_M"},
		{m: m, modelID: "Qwen3.8-27B-Q4_K_M", metal: true, q4k: true},
	} {
		got := captureExecutionSummary(t, p, "avx512", "+fused")
		if v, ok := summaryKey(got, "q8dec"); !ok || !strings.HasPrefix(v, "avx512+fused/") {
			t.Fatalf("CPU-session execution summary %q q8dec=%q, want avx512+fused/<n>w", got, v)
		}
		if _, ok := summaryKey(got, "host"); ok {
			t.Fatalf("CPU-session execution summary %q carries a host= clause", got)
		}
	}
}
