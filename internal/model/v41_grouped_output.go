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
//
// Adapted from oMLX language.py:348-350 at 3f2d07e8dff257119329e0a2e9821df81182f05d
// and DeepSeek-V4.1-Flash at dba1be0a40aa45a94ad051997016db3960a90277 (MIT).
package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedOutputFunc func(layer int, o []float32, heads, headDim, groups, rank, dim int) ([]float32, v41DenseProjectionOutcome, error)

func (m *Model) v41GroupedOutputProjector(l, heads, headDim, groups, rank, dim int, scratch *v41ProjScratch) func([]float32) ([]float32, error) {
	var woA, woB []float32
	loaded := false
	return func(o []float32) ([]float32, error) {
		if scratch.groupedOutput != nil {
			y, outcome, err := scratch.groupedOutput(l, o, heads, headDim, groups, rank, dim)
			switch outcome {
			case v41ProjectionHandled:
				if err != nil || len(y) != dim {
					return nil, v41ProjectionOperationErr(l, "attn.wo_b.weight", err)
				}
				return y, nil
			case v41ProjectionError:
				return nil, v41ProjectionOperationErr(l, "attn.wo_b.weight", err)
			case v41ProjectionDeclined:
			default:
				return nil, v41ProjectionOperationErr(l, "attn.wo_b.weight", errV41ProjectionResult)
			}
		}
		opened := m.v41NowNanos()
		completed := 0
		var hostBytes int64
		defer func() { m.v41NoteGroupedOutput(0, 1, 0, completed, 0, 0, 0, hostBytes, opened) }()
		if !loaded {
			var err error
			woA, err = m.v41ProjF32Into(l, "attn.wo_a.weight", scratch.woA)
			hostBytes += int64(len(woA)) * 4
			if err != nil {
				return nil, err
			}
			scratch.woA = woA
			woB, err = m.v41ProjF32Into(l, "attn.wo_b.weight", scratch.woB)
			hostBytes += int64(len(woB)) * 4
			if err != nil {
				return nil, err
			}
			scratch.woB = woB
			loaded = true
		}
		y, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, heads, headDim, groups, rank, dim)
		if err != nil {
			return nil, v41StageErr(v41StageAttention, l, err)
		}
		completed = 1
		return y, nil
	}
}

type v41GroupedWeight struct {
	dtype compute.Dtype
	elems int
	host  func(offset, out, in int) (compute.Tensor, error)
}

func (s *Session) v41GroupedWeight(name string, logicalIn int) (v41GroupedWeight, bool, error) {
	m := s.M
	out, in, present := m.residentShape(name)
	elems, ok := checkedMulInt(out, in)
	if !present || !ok || out <= 0 || in <= 0 || logicalIn <= 0 {
		return v41GroupedWeight{}, false, nil
	}
	w := v41GroupedWeight{elems: elems}
	validated := false
	switch {
	case m.has(name):
		w.dtype = compute.F32
		w.host = func(offset, out, in int) (compute.Tensor, error) {
			values := m.tensor(name)
			if len(values) != elems {
				return compute.Tensor{}, errV41ProjectionResult
			}
			if !validated {
				for _, v := range values {
					if !finite32(v) {
						return compute.Tensor{}, errV41ProjectionResult
					}
				}
				validated = true
			}
			n, valid := checkedMulInt(out, in)
			if !valid || offset < 0 || n > len(values)-offset {
				return compute.Tensor{}, errV41ProjectionResult
			}
			return compute.NewF32(compute.Default(), []int{out, in}, values[offset:offset+n]), nil
		}
	case m.q8w[name] != nil:
		q := m.q8w[name]
		if in%qBlk != 0 || logicalIn%qBlk != 0 {
			return w, false, nil
		}
		w.dtype = compute.Q8_0
		w.host = func(offset, out, in int) (compute.Tensor, error) {
			if q.nblk != q.in/qBlk || len(q.q) != elems || len(q.d) != elems/qBlk {
				return compute.Tensor{}, errV41ProjectionResult
			}
			if !validated {
				for i, code := range q.q {
					if !finite32(float32(code) * q.d[i/qBlk]) {
						return compute.Tensor{}, errV41ProjectionResult
					}
				}
				validated = true
			}
			n, valid := checkedMulInt(out, in)
			if !valid || offset%qBlk != 0 || n%qBlk != 0 || offset < 0 || n > len(q.q)-offset {
				return compute.Tensor{}, errV41ProjectionResult
			}
			return compute.NewQ8(compute.Default(), []int{out, in}, q.q[offset:offset+n], q.d[offset/qBlk:(offset+n)/qBlk], qBlk), nil
		}
	case m.q4kw[name] != nil:
		q := m.q4kw[name]
		if in%qkK != 0 || logicalIn%qkK != 0 || q.nblk != in/qkK {
			return w, false, nil
		}
		expected, valid := checkedMulInt(elems/qkK, q4kBlockBytes)
		if !valid {
			return w, false, nil
		}
		w.dtype = compute.Q4_K
		var raw []byte
		w.host = func(offset, out, in int) (compute.Tensor, error) {
			if raw == nil {
				var err error
				raw, err = q.materializeRaw()
				if err != nil {
					return compute.Tensor{}, err
				}
			}
			if len(raw) != expected {
				return compute.Tensor{}, errV41ProjectionResult
			}
			if !validated {
				if err := v41GroupedPackedFinite(raw, q4kBlockBytes, compute.Q4_K, kindQ5K); err != nil {
					return compute.Tensor{}, err
				}
				validated = true
			}
			n, valid := checkedMulInt(out, in)
			if !valid || offset%qkK != 0 || n%qkK != 0 || offset < 0 || n > elems-offset {
				return compute.Tensor{}, errV41ProjectionResult
			}
			lo := offset / qkK * q4kBlockBytes
			hi := (offset + n) / qkK * q4kBlockBytes
			return compute.NewQ4K(compute.Default(), []int{out, in}, raw[lo:hi]), nil
		}
	case m.kqw[name] != nil:
		q := m.kqw[name]
		d, present := LookupQuantDescriptor(q.kind)
		if !present || !d.SupportsHAL() || d.BlockWeights() <= 0 || d.BlockBytes() <= 0 || in%d.BlockWeights() != 0 || logicalIn%d.BlockWeights() != 0 || q.nblk != in/d.BlockWeights() {
			return w, false, nil
		}
		expected, valid := checkedMulInt(elems/d.BlockWeights(), d.BlockBytes())
		if !valid {
			return w, false, nil
		}
		w.dtype = d.Dtype()
		var raw []byte
		w.host = func(offset, out, in int) (compute.Tensor, error) {
			if raw == nil {
				var err error
				raw, err = q.materializeRaw()
				if err != nil {
					return compute.Tensor{}, err
				}
			}
			if len(raw) != expected {
				return compute.Tensor{}, errV41ProjectionResult
			}
			if !validated {
				if err := v41GroupedPackedFinite(raw, d.BlockBytes(), d.Dtype(), q.kind); err != nil {
					return compute.Tensor{}, err
				}
				validated = true
			}
			n, valid := checkedMulInt(out, in)
			if !valid || offset%d.BlockWeights() != 0 || n%d.BlockWeights() != 0 || offset < 0 || n > elems-offset {
				return compute.Tensor{}, errV41ProjectionResult
			}
			lo := offset / d.BlockWeights() * d.BlockBytes()
			hi := (offset + n) / d.BlockWeights() * d.BlockBytes()
			return d.NewHostTensor(out, in, raw[lo:hi]), nil
		}
	default:
		return w, false, nil
	}
	if !compute.BackendSupportsDeviceWeightDtype(s.Backend, w.dtype) || (w.dtype != compute.F32 && !s.Backend.Caps().UploadDtype) {
		return w, false, nil
	}
	switch w.dtype {
	case compute.Q3_K:
		if b, ok := s.Backend.(interface{ SupportsQ3KMatMul() bool }); ok && !b.SupportsQ3KMatMul() {
			return w, false, nil
		}
	case compute.Q5_K:
		if b, ok := s.Backend.(interface{ SupportsQ5KMatMul() bool }); ok && !b.SupportsQ5KMatMul() {
			return w, false, nil
		}
	case compute.Q6_K:
		if b, ok := s.Backend.(interface{ SupportsQ6KMatMul() bool }); ok && !b.SupportsQ6KMatMul() {
			return w, false, nil
		}
	}
	return w, true, nil
}

func v41GroupedPackedFinite(raw []byte, blockBytes int, dtype compute.Dtype, kind kQuantKind) error {
	if blockBytes <= 0 || len(raw)%blockBytes != 0 {
		return errV41ProjectionResult
	}
	var scaleOffsets []int
	switch dtype {
	case compute.Q2_K:
		scaleOffsets = []int{blockBytes - 4, blockBytes - 2}
	case compute.Q3_K, compute.Q6_K:
		scaleOffsets = []int{blockBytes - 2}
	case compute.Q4_K, compute.Q5_K:
		scaleOffsets = []int{0, 2}
	}
	if scaleOffsets != nil {
		for _, off := range scaleOffsets {
			if off < 0 || off > blockBytes-2 {
				return errV41ProjectionResult
			}
		}
		for base := 0; base < len(raw); base += blockBytes {
			for _, off := range scaleOffsets {
				if binary.LittleEndian.Uint16(raw[base+off:])&0x7c00 == 0x7c00 {
					return errV41ProjectionResult
				}
			}
		}
		return nil
	}
	scratch := make([]float32, kind.blockWeights())
	for base := 0; base < len(raw); base += blockBytes {
		kQuantDequantSuperBlock(scratch, raw[base:base+blockBytes], kind)
		for _, v := range scratch {
			if !finite32(v) {
				return errV41ProjectionResult
			}
		}
	}
	return nil
}

func (s *Session) v41GroupedOutputFunc() v41GroupedOutputFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	return func(l int, o []float32, heads, headDim, groups, rank, dim int) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
		totalIn, inOK := checkedMulInt(heads, headDim)
		joinedWidth, joinOK := checkedMulInt(groups, rank)
		if heads <= 0 || headDim <= 0 || groups <= 0 || rank <= 0 || dim <= 0 || heads%groups != 0 || !inOK || !joinOK || len(o) != totalIn {
			return nil, v41ProjectionDeclined, nil
		}
		groupIn := totalIn / groups
		groupElems, groupOK := checkedMulInt(rank, groupIn)
		aElems, aOK := checkedMulInt(groups, groupElems)
		bElems, bOK := checkedMulInt(dim, joinedWidth)
		if !groupOK || !aOK || !bOK {
			return nil, v41ProjectionDeclined, nil
		}
		for _, v := range o {
			if !finite32(v) {
				return nil, v41ProjectionDeclined, nil
			}
		}
		aName, bName := layerName(l, "attn.wo_a.weight"), layerName(l, "attn.wo_b.weight")
		ar, ac, ap := s.M.residentShape(aName)
		br, bc, bp := s.M.residentShape(bName)
		if !ap || !bp || !((ar == joinedWidth && ac == groupIn) || (ar == rank && ac == totalIn)) || br != dim || bc != joinedWidth {
			return nil, v41ProjectionDeclined, nil
		}
		a, aSupported, _ := s.v41GroupedWeight(aName, groupIn)
		b, bSupported, _ := s.v41GroupedWeight(bName, joinedWidth)
		if !aSupported || !bSupported || a.elems != aElems || b.elems != bElems {
			return nil, v41ProjectionDeclined, nil
		}
		opened := s.M.v41NowNanos()
		completed, calls := 0, 0
		var upload, readback int64
		defer func() { s.M.v41NoteGroupedOutput(1, 0, completed, 0, calls, upload, readback, 0, opened) }()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-grouped-output", Layer: l, Stage: stage, Cause: err}
			s.halFailure = closed
			s.Close()
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, outcome, cause = nil, v41ProjectionError, closeFailure(err)
						return
					}
				}
				panic(r)
			}
		}()
		sourceA := make([]compute.Tensor, groups)
		aKeys := make([]string, groups)
		for g := 0; g < groups; g++ {
			aKeys[g] = fmt.Sprintf("v41-grouped:%s:%v:%d:%d:%d", aName, a.dtype, g, rank, groupIn)
			if _, present := s.halW[aKeys[g]]; present {
				continue
			}
			var err error
			sourceA[g], err = a.host(g*groupElems, rank, groupIn)
			if err != nil {
				return nil, v41ProjectionError, closeFailure(err)
			}
		}
		bKey := fmt.Sprintf("v41-grouped:%s:%v:%d:%d", bName, b.dtype, dim, joinedWidth)
		var sourceB compute.Tensor
		if _, present := s.halW[bKey]; !present {
			var err error
			sourceB, err = b.host(0, dim, joinedWidth)
			if err != nil {
				return nil, v41ProjectionError, closeFailure(err)
			}
		}
		run := func(weight compute.Tensor, input []float32, out int) ([]float32, error) {
			stage = "activation upload"
			x := s.uploadHostF32([]int{len(input)}, input, compute.MemoryActivation, "V4.1 grouped output activation")
			defer s.Backend.Free(x)
			upload += int64(len(input)) * 4
			stage = "matmul"
			calls++
			y := s.Backend.MatMul(weight, x)
			defer s.Backend.Free(y)
			stage = "readback"
			values := s.Backend.Read(y)
			readback += int64(len(values)) * 4
			if len(values) != out {
				return nil, errV41ProjectionResult
			}
			for _, v := range values {
				if !finite32(v) {
					return nil, errV41ProjectionResult
				}
			}
			return values, nil
		}
		joined := make([]float32, joinedWidth)
		for g := 0; g < groups; g++ {
			stage = "weight upload"
			key := aKeys[g]
			weight := s.cachedImmutableWeight(key, key, func() compute.Tensor { return s.Backend.Upload(sourceA[g], a.dtype) })
			values, err := run(weight, o[g*groupIn:(g+1)*groupIn], rank)
			if err != nil {
				return nil, v41ProjectionError, closeFailure(err)
			}
			copy(joined[g*rank:(g+1)*rank], values)
		}
		stage = "weight upload"
		key := bKey
		weight := s.cachedImmutableWeight(key, key, func() compute.Tensor { return s.Backend.Upload(sourceB, b.dtype) })
		var err error
		result, err = run(weight, joined, dim)
		if err != nil {
			return nil, v41ProjectionError, closeFailure(err)
		}
		completed = 1
		return result, v41ProjectionHandled, nil
	}
}
