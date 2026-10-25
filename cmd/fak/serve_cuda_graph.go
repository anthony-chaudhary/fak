package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// serveCUDAGraphKV sizes graph-replay KV for every enable path (--cuda-graph or
// FAK_CUDA_GRAPH=1). It returns the fixed capacity for compute.SetCUDAGraphKVCapacity (at
// least the resolved context) and the startup message: the prealloc bytes per session, or
// why sessions run uncaptured. cfg is nil when no GGUF geometry was read at startup.
func serveCUDAGraphKV(flag bool, getenv func(string) string, be compute.Backend, cfg *model.Config, resolved int) (capacity int, msg string, requested bool) {
	envOn := getenv("FAK_CUDA_GRAPH") == "1"
	if !flag && !envOn {
		return 0, "", false
	}
	source := "--cuda-graph"
	if !flag {
		source = "FAK_CUDA_GRAPH=1"
	}
	capacity = max(resolved, compute.GraphKVDefaultPositions)
	if cfg == nil {
		return capacity, fmt.Sprintf("decode replay requested by %s; KV graph capacity=%d positions; model geometry not read at startup", source, capacity), true
	}
	_, free, _ := compute.DeviceMemoryInfo(be)
	sizing := model.CUDAGraphKVSizingFor(*cfg, be, true, capacity, free)
	switch {
	case !sizing.Eligible:
		msg = fmt.Sprintf("decode replay requested by %s; sessions run uncaptured: ineligible (%s)", source, sizing.Reason)
	case !sizing.Prealloc.Preallocate:
		msg = fmt.Sprintf("decode replay requested by %s; sessions run uncaptured: ineligible (%s): prealloc=%d bytes per session at %d positions, free=%d bytes", source, sizing.Prealloc.Reason, sizing.Prealloc.Bytes, capacity, free)
	default:
		msg = fmt.Sprintf("decode replay enabled by %s; KV graph capacity=%d positions; prealloc=%d bytes per session over %d KV rows", source, capacity, sizing.Prealloc.Bytes, sizing.KVRows)
	}
	return capacity, msg, true
}
