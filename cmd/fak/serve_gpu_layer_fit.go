package main

import (
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveDenseGPULayerHostFraction is the share of the artifact's on-disk weight bytes that a
// partial-offload serve (--gpu-layers N, 0 < N < NumLayers) places on the host: the tensors of
// decoder layers [N, NumLayers). Layers outside the decoder range (NextN/MTP blocks) and
// non-layer tensors (embeddings, output head, final norm) stay device-charged, so the fraction
// never under-charges the device for tensors whose placement the layer split does not decide.
// It returns 0 (no re-scope) when N is unset/out of range or the header cannot be sized.
func serveDenseGPULayerHostFraction(ws *ggufload.WeightSource, gpuLayers int) float64 {
	if ws == nil || ws.File == nil || gpuLayers <= 0 {
		return 0
	}
	cfg, err := ws.File.Config()
	if err != nil || gpuLayers >= cfg.NumLayers {
		return 0
	}
	var total, host int64
	for _, t := range ws.File.Tensors {
		b := t.OnDiskBytes()
		if b <= 0 {
			continue
		}
		total += b
		if layer, ok := ggufBlockLayer(t.Name); ok && layer >= gpuLayers && layer < cfg.NumLayers {
			host += b
		}
	}
	if total <= 0 || host <= 0 {
		return 0
	}
	return float64(host) / float64(total)
}

// ggufBlockLayer parses the decoder layer index from a GGUF "blk.<i>.<suffix>" tensor name.
func ggufBlockLayer(name string) (int, bool) {
	rest, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return 0, false
	}
	idx, _, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(idx)
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// applyDenseGPULayerSplit re-scopes the device weight demands of a partial-offload serve so
// admission honors --gpu-layers N: the host share (serveDenseGPULayerHostFraction) of every
// device-scoped weight demand is charged MemoryScopeHost instead of to the device. Before this the
// sizing and admission plans charged every layer to the device, so a 27B Q4_K_M that fits a
// 16 GB card at --gpu-layers 40 was refused as if fully offloaded (fak-private
// FAK-NATIVE-CUDA-BLOCKER-20261025 fix (a)). KV, scratch and runtime demands are unchanged, and
// gpuLayers <= 0 or >= NumLayers returns the plan untouched (full device placement).
func applyDenseGPULayerSplit(plan compute.MemoryPlan, ws *ggufload.WeightSource, gpuLayers int) compute.MemoryPlan {
	frac := serveDenseGPULayerHostFraction(ws, gpuLayers)
	if frac <= 0 || len(plan) == 0 {
		return plan
	}
	out := make(compute.MemoryPlan, 0, len(plan)+1)
	for _, demand := range plan {
		if demand.Class != compute.MemoryWeights || !demand.DeviceScoped() || demand.Bytes <= 0 {
			out = append(out, demand)
			continue
		}
		hostBytes := int64(float64(demand.Bytes) * frac)
		if hostBytes > demand.Bytes {
			hostBytes = demand.Bytes
		}
		if deviceBytes := demand.Bytes - hostBytes; deviceBytes > 0 {
			device := demand
			device.Bytes = deviceBytes
			device.Scope = compute.MemoryScopeDevice
			device.Detail += ":gpu-layers"
			out = append(out, device)
		}
		if hostBytes > 0 {
			host := demand
			host.Bytes = hostBytes
			host.Scope = compute.MemoryScopeHost
			host.Detail += ":host-layers"
			out = append(out, host)
		}
	}
	return out
}
