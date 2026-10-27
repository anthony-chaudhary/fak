// MIT License
//
// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package model

import (
	"sort"
	"sync/atomic"
)

// Full routed Expert.forward boundaries follow pinned DeepSeek model.py
// dba1be0a40aa45a94ad051997016db3960a90277:841-851,895-904. The
// callbacks remain F32; each publication below is owned. Quantized GEMM
// activation parity is separate from these explicit BF16 graph boundaries.
type v41RoutedTripleLoader func() (gate, up, down []float32, err error)
type v41RoutedDownLoader func() ([]float32, error)

const v41RoutedExpertHostHandled v41RoutedExpertOutcome = 3

func v41RoutedBF16(layer int, stage string, values []float32, width int) ([]float32, error) {
	return v41RoutedBF16Observed(layer, stage, values, width, nil)
}

func v41RoutedBF16Observed(layer int, stage string, values []float32, width int, observe func(string)) ([]float32, error) {
	if len(values) != width {
		return nil, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
	}
	out := make([]float32, width)
	return v41RoutedBF16Into(layer, stage, values, out, observe)
}

// Only fresh private projection/activation arrays may enter this publication.
// Borrowed inputs and callback results retain the ordinary defensive copy.
func v41RoutedBF16Owned(layer int, stage string, values []float32, width int) ([]float32, error) {
	if len(values) != width {
		return nil, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
	}
	return v41RoutedBF16Into(layer, stage, values, values, nil)
}

func v41RoutedBF16Into(layer int, stage string, values, out []float32, observe func(string)) ([]float32, error) {
	for i, v := range values {
		if !finite32(v) {
			return nil, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
		}
		if i == 0 && observe != nil {
			observe("bf16-boundary")
		}
		out[i] = v41RoundBF16(v)
		if !finite32(out[i]) {
			return nil, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
		}
	}
	return out, nil
}

func (m *Model) v41FullRoutedExpert(layer int, stem string, input []float32, weight float32, cfg Config, gateUp v41ExpertGateUpFunc, down v41ExpertDownFunc, triple v41RoutedTripleLoader, downWeight v41RoutedDownLoader, parallel bool) ([]float32, error) {
	return m.v41FullRoutedExpertWithDevice(layer, stem, input, weight, cfg, nil, gateUp, down, triple, downWeight, parallel)
}

// v41FullRoutedExpertWithDevice offers the pick to the fused device routed
// expert first. Handled keeps the clamp, SwiGLU and route weight on the device;
// a decline falls through to the gate/up and down seams, then the host triple.
// Device-only sessions bind a refusing wrapper, so their decline never does.
func (m *Model) v41FullRoutedExpertWithDevice(layer int, stem string, input []float32, weight float32, cfg Config, routed v41RoutedExpertFunc, gateUp v41ExpertGateUpFunc, down v41ExpertDownFunc, triple v41RoutedTripleLoader, downWeight v41RoutedDownLoader, parallel bool) ([]float32, error) {
	return m.v41FullRoutedExpertPublished(layer, stem, input, weight, cfg, routed, gateUp, down, triple, downWeight, parallel, nil)
}

// v41RoutedBF16Publish is v41RoutedBF16 routed through the session's ordinary
// device publication when one is bound.
func v41RoutedBF16Publish(layer int, stage string, values []float32, width int, publish v41BF16BoundaryFunc) ([]float32, error) {
	return v41RoutedBF16PublishInput(layer, stage, values, width, publish, false)
}

func v41RoutedBF16PublishInput(layer int, stage string, values []float32, width int, publish v41BF16BoundaryFunc, owned bool) ([]float32, error) {
	if len(values) != width {
		return nil, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
	}
	site := v41RoutedGateUpSite
	if stage == "down" {
		site = v41RoutedDownSite
	}
	// Selected publications keep their original ownership/dispatch boundary.
	owned = owned && publish == nil
	return v41PublishBF16Input(layer, site, values, publish, func(values []float32) ([]float32, error) {
		if owned {
			return v41RoutedBF16Owned(layer, stage, values, width)
		}
		return v41RoutedBF16(layer, stage, values, width)
	}, owned)
}

func (m *Model) v41FullRoutedExpertPublished(layer int, stem string, input []float32, weight float32, cfg Config, routed v41RoutedExpertFunc, gateUp v41ExpertGateUpFunc, down v41ExpertDownFunc, triple v41RoutedTripleLoader, downWeight v41RoutedDownLoader, parallel bool, publish v41BF16BoundaryFunc) ([]float32, error) {
	return m.v41FullRoutedExpertProjected(layer, stem, input, weight, cfg, routed, gateUp, down, triple, downWeight, parallel, publish, nil)
}

// A private pure-host cohort may supply its owned projection destinations. All
// publication, validation, activation and row error boundaries stay here.
func (m *Model) v41FullRoutedExpertProjected(layer int, stem string, input []float32, weight float32, cfg Config, routed v41RoutedExpertFunc, gateUp v41ExpertGateUpFunc, down v41ExpertDownFunc, triple v41RoutedTripleLoader, downWeight v41RoutedDownLoader, parallel bool, publish v41BF16BoundaryFunc, hostProjection v41HostExpertProjection) ([]float32, error) {
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	if H <= 0 || I <= 0 || !finite32(weight) {
		return nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
	}
	x, err := v41RoutedBF16Publish(layer, "gate/up", input, H, publish)
	if err != nil {
		return nil, err
	}
	if routed != nil {
		values, outcome, cause := routed(layer, stem, x, weight)
		switch outcome {
		case v41RoutedExpertHandled, v41RoutedExpertHostHandled:
			if cause != nil {
				return nil, v41ExpertOperationErr(layer, "down", cause)
			}
			output, err := v41RoutedBF16Publish(layer, "down", values, H, publish)
			if err != nil {
				return nil, err
			}
			if outcome == v41RoutedExpertHandled {
				m.v41NoteDeviceExpertContraction()
			}
			return output, nil
		case v41RoutedExpertError:
			if cause == nil {
				cause = v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
			}
			return nil, cause
		case v41RoutedExpertDeclined:
		default:
			return nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
		}
	}
	var fused, w2 []float32
	if gateUp != nil {
		h, outcome, cause := gateUp(layer, stem, x)
		switch outcome {
		case v41GateUpHandled:
			if cause != nil {
				return nil, v41ExpertOperationErr(layer, "gate/up", cause)
			}
			if len(h) != I {
				return nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
			}
			// Gate/up publishes F32 SwiGLU of BF16 projections. Do NOT narrow here:
			// the route weight is multiplied before the official down-input cast.
			fused = append([]float32(nil), h...)
		case v41GateUpError:
			return nil, v41ExpertOperationErr(layer, "gate/up", cause)
		case v41GateUpDeclined:
		default:
			return nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
		}
	}
	proj := matRows
	if parallel && I*H >= parThreshold {
		proj = parMatRows
	}
	if hostProjection != nil {
		proj = hostProjection
	}
	var opened int64
	if fused == nil {
		w1, w3, wd, cause := triple()
		if cause != nil {
			return nil, cause
		}
		if len(w1) != I*H || len(w3) != I*H || len(wd) != H*I {
			return nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
		}
		w2 = wd
		if parallel && I*H >= parThreshold && v41SwiGLUWitnessOn {
			atomic.AddInt64(&v41SwiGLUParallelCalls, 1)
		}
		opened = m.v41NowNanos()
		gate, cause := v41RoutedBF16PublishInput(layer, "gate/up", proj(w1, x, I, H), I, publish, true)
		if cause != nil {
			return nil, cause
		}
		up, cause := v41RoutedBF16PublishInput(layer, "gate/up", proj(w3, x, I, H), I, publish, true)
		if cause != nil {
			return nil, cause
		}
		clampSwiGLUProjections(gate, up, float32(cfg.SwigluLimit))
		if publish == nil {
			// The fresh host gate projection is dead after this elementwise
			// read. Reuse only this private buffer, never callback storage.
			fused = gate
		} else {
			fused = make([]float32, I)
		}
		for i := range fused {
			fused[i] = silu(gate[i]) * up[i]
		}
	}
	// Both paths own fused: callback output was copied and the host path
	// allocated it. No caller can observe the unweighted intermediate.
	weighted := fused
	for i, v := range fused {
		if !finite32(v) {
			return nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
		}
		weighted[i] = float32(v * weight)
	}
	operand, err := v41RoutedBF16PublishInput(layer, "down", weighted, I, publish, true)
	if err != nil {
		return nil, err
	}
	var output []float32
	// A host gate/up decline retains the historical host triple path. Only a
	// handled gate/up offers the existing down seam, preserving dispatch policy.
	if w2 == nil && down != nil {
		values, outcome, cause := down(layer, stem, operand)
		switch outcome {
		case v41DownHandled:
			if cause != nil {
				return nil, v41ExpertOperationErr(layer, "down", cause)
			}
			output, err = v41RoutedBF16Publish(layer, "down", values, H, publish)
			if err != nil {
				return nil, err
			}
		case v41DownError:
			return nil, v41ExpertOperationErr(layer, "down", cause)
		case v41DownDeclined:
		default:
			return nil, v41ExpertOperationErr(layer, "down", errV41ExpertResult)
		}
	}
	if output == nil {
		if w2 == nil {
			w2, err = downWeight()
			if err != nil {
				return nil, err
			}
			opened = m.v41NowNanos()
		}
		if len(w2) != H*I {
			return nil, v41ExpertOperationErr(layer, "down", errV41ExpertResult)
		}
		output, err = v41RoutedBF16PublishInput(layer, "down", proj(w2, operand, H, I), H, publish, true)
		if err != nil {
			return nil, err
		}
	}
	if opened != 0 {
		m.v41NoteExpertContractionNanos(m.v41NowNanos() - opened)
	} else {
		m.v41NoteExpertContraction()
	}
	return output, nil
}

// Sort contribution indices only. Routing picks and first-touch materialization
// remain unchanged. Real top-k picks are unique; stable slot order defines the
// deterministic extension for duplicate-expert synthetic fixtures.
func v41FullRoutedSum(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error) {
	if len(picks) != len(outputs) || width <= 0 {
		return nil, v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
	}
	order := make([]int, len(picks))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return picks[order[i]].expert < picks[order[j]].expert })
	sum := make([]float32, width)
	for _, slot := range order {
		if len(outputs[slot]) != width {
			return nil, v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
		}
		for i, v := range outputs[slot] {
			if !finite32(v) {
				return nil, v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
			}
			sum[i] = float32(sum[i] + v)
			if !finite32(sum[i]) {
				return nil, v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
			}
		}
	}
	return sum, nil
}
