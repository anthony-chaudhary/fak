package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type deviceWeightMemoryPlanner struct {
	mem agent.DeviceWeightMemory
	ok  bool
}

func (p deviceWeightMemoryPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p deviceWeightMemoryPlanner) Model() string { return "device-weight-memory-test" }

func (p deviceWeightMemoryPlanner) DeviceWeightMemory() (agent.DeviceWeightMemory, bool) {
	return p.mem, p.ok
}

// A probe reads expert-ring page-in reuse and arena growth straight off /metrics; a planner or
// backend without the counters renders nothing rather than zeros.
// fak-test:runtime fast est=50ms lane=default
func TestDeviceWeightMemoryMetrics(t *testing.T) {
	srv := newTestServer(t)
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_device_") {
		t.Fatalf("a non-reporting planner emitted the device weight-memory family:\n%s", text)
	}
	srv.planner = deviceWeightMemoryPlanner{}
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_device_") {
		t.Fatalf("an unavailable report emitted the device weight-memory family:\n%s", text)
	}
	srv.planner = deviceWeightMemoryPlanner{ok: true, mem: agent.DeviceWeightMemory{
		Backend: "vulkan", Arena: true, ArenaReservedBytes: 268435456, ArenaLiveBytes: 66560,
		ArenaPeakReservedBytes: 268435456, ArenaMemoryAllocations: 1,
		PagedWeights: true, PagedWeightsReused: 21, PagedWeightsFreshly: 3,
	}}
	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_device_weight_arena_reserved_bytes{backend="vulkan"} 268435456`,
		`fak_gateway_device_weight_arena_live_bytes{backend="vulkan"} 66560`,
		`fak_gateway_device_weight_arena_peak_reserved_bytes{backend="vulkan"} 268435456`,
		`fak_gateway_device_weight_arena_memory_allocations_total{backend="vulkan"} 1`,
		`fak_gateway_device_paged_weight_page_ins_total{backend="vulkan",allocation="reused"} 21`,
		`fak_gateway_device_paged_weight_page_ins_total{backend="vulkan",allocation="fresh"} 3`,
	} {
		if !strings.Contains(text, want+"\n") {
			t.Fatalf("metrics missing %q", want)
		}
	}
	srv.planner = deviceWeightMemoryPlanner{ok: true, mem: agent.DeviceWeightMemory{Backend: "vulkan", Arena: true}}
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_device_paged_weight_page_ins_total") {
		t.Fatal("a backend without paged-weight counters emitted the page-in series")
	}
}
