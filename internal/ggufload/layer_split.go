package ggufload

import (
	"errors"
	"math"
)

// Layer placement re-expressed from llama.cpp's load_tensors device mapping
// (ggml-org/llama.cpp src/llama-model.cpp:1496-1517 at
// 83078fec0db82d6b5a00d9599062c38c39145755, MIT, Copyright (c) 2023-2026 The
// ggml authors; license reproduced in THIRD_PARTY_NOTICES.md). The source
// normalizes per-device tensor-split fractions into a cumulative distribution,
// keeps the first n_layer-n_gpu_layers layers on the CPU, and places each
// offloaded layer on the device found by an upper-bound search over the
// cumulative shares. This file re-expresses that rule as a pure Go function
// over integers and floats; it performs no preflight estimation and no I/O.

// Sentinel outcomes of AssignLayers. Callers match these with errors.Is; the
// messages are part of the closed contract, never prose to parse.
var (
	// ErrAssignLayersArgs reports a negative layer or GPU-layer count.
	ErrAssignLayersArgs = errors.New("ggufload: assign-layers argument out of range")
	// ErrSplitNegative reports a negative tensor-split fraction.
	ErrSplitNegative = errors.New("ggufload: tensor split fraction is negative")
	// ErrSplitAllZero reports split fractions that sum to zero.
	ErrSplitAllZero = errors.New("ggufload: tensor split fractions sum to zero")
	// ErrSplitNonFinite reports a NaN or infinite tensor-split fraction.
	ErrSplitNonFinite = errors.New("ggufload: tensor split fraction is not finite")
)

// cpuDevice marks a layer kept on the host in an AssignLayers result.
const cpuDevice = -1

// AssignLayers maps each of nLayer layers to a device index from an
// nGPULayers offload budget and per-device tensor-split fractions. The result
// has one entry per layer: cpuDevice (-1) for a host-resident layer, otherwise
// the 0-based index into split.
//
// Semantics follow the cited llama.cpp placement: fractions are normalized to
// a cumulative distribution; layers below nLayer-nGPULayers stay on the CPU;
// offloaded layer i lands on the first device whose cumulative share exceeds
// i/nOffloaded (upper bound, strict). An empty split keeps every layer on the
// CPU, mirroring the source's no-devices arm. The final cumulative share is
// pinned to exactly 1 so the upper bound always yields a valid device index
// despite float64 summation drift.
func AssignLayers(nLayer, nGPULayers int, split []float64) ([]int, error) {
	if nLayer < 0 || nGPULayers < 0 {
		return nil, ErrAssignLayersArgs
	}
	for _, f := range split {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, ErrSplitNonFinite
		}
		if f < 0 {
			return nil, ErrSplitNegative
		}
	}
	out := make([]int, nLayer)
	devices := len(split)
	if devices == 0 {
		for i := range out {
			out[i] = cpuDevice
		}
		return out, nil
	}
	sum := 0.0
	for _, f := range split {
		sum += f
	}
	if sum <= 0 {
		return nil, ErrSplitAllZero
	}
	cum := make([]float64, devices)
	total := 0.0
	for i, f := range split {
		total += f / sum
		cum[i] = total
	}
	cum[devices-1] = 1.0

	gpuStart := nLayer - nGPULayers
	if gpuStart < 0 {
		gpuStart = 0
	}
	offloaded := nGPULayers
	if offloaded > nLayer {
		offloaded = nLayer
	}
	for il := 0; il < nLayer; il++ {
		if il < gpuStart || offloaded == 0 || il-gpuStart >= offloaded {
			out[il] = cpuDevice
			continue
		}
		out[il] = upperBoundCum(cum, float64(il-gpuStart)/float64(offloaded))
	}
	return out, nil
}

// upperBoundCum returns the first index i with cum[i] > v, or len(cum) when no
// entry exceeds v. cum must be non-decreasing.
func upperBoundCum(cum []float64, v float64) int {
	lo, hi := 0, len(cum)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if cum[mid] <= v {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}
