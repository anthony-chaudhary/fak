package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type deviceWeightMemoryBackend struct {
	compute.Backend
}

func (b deviceWeightMemoryBackend) Name() string { return "vulkan" }

type deviceWeightMemoryArenaBackend struct{ deviceWeightMemoryBackend }

func (deviceWeightMemoryArenaBackend) VulkanWeightArenaCounters() (uint64, uint64, uint64, uint64, uint64) {
	return 2, 40, 1 << 28, 1 << 20, 1 << 29
}

type deviceWeightMemoryFullBackend struct{ deviceWeightMemoryArenaBackend }

func (deviceWeightMemoryFullBackend) VulkanOffloadWeightCounters() (uint64, uint64) { return 21, 3 }

// The planner projects exactly the counter contracts its backend offers; a backend with
// neither reports unavailable instead of zeros.
// fak-test:runtime fast est=10ms lane=default
func TestDeviceWeightMemoryFromBackend(t *testing.T) {
	if _, ok := DeviceWeightMemoryFromBackend(nil); ok {
		t.Fatal("nil backend reported device weight memory")
	}
	if _, ok := DeviceWeightMemoryFromBackend(deviceWeightMemoryBackend{}); ok {
		t.Fatal("backend without counters reported device weight memory")
	}
	arena, ok := DeviceWeightMemoryFromBackend(deviceWeightMemoryArenaBackend{})
	if !ok || !arena.Arena || arena.PagedWeights || arena.ArenaReservedBytes != 1<<28 || arena.ArenaPeakReservedBytes != 1<<29 || arena.ArenaMemoryAllocations != 2 {
		t.Fatalf("arena-only projection = %+v, %t", arena, ok)
	}
	full, ok := DeviceWeightMemoryFromBackend(deviceWeightMemoryFullBackend{})
	if !ok || full.Backend != "vulkan" || !full.PagedWeights || full.PagedWeightsReused != 21 || full.PagedWeightsFreshly != 3 {
		t.Fatalf("full projection = %+v, %t", full, ok)
	}
	if _, ok := (&InKernelPlanner{}).DeviceWeightMemory(); ok {
		t.Fatal("planner without a device backend reported device weight memory")
	}
}
