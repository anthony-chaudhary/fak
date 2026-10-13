package model

import "github.com/anthony-chaudhary/fak/internal/compute"

type v41HeadPayloadError struct{ cause error }

// v41HeadProjectionDtype is the read-only admission used by serial projection
// and the shared-head cohort. It selects the same resident format and block
// layout without staging a weight, binding callbacks or opening a batch.
func (s *Session) v41HeadProjectionDtype(name string, out, in int) (compute.Dtype, bool) {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return compute.F32, false
	}
	wr, wc, present := s.M.v41HeadWeightShape(name)
	_, valid := checkedMulInt(out, in)
	if !valid || out <= 0 || in <= 0 || !present || wr != out || wc != in {
		return compute.F32, false
	}
	dtype := compute.F32
	switch {
	case s.M.has(name):
	case s.M.q8w[name] != nil:
		if in%qBlk != 0 {
			return dtype, false
		}
		dtype = compute.Q8_0
	case s.M.q4kw[name] != nil:
		if in%qkK != 0 || s.M.q4kw[name].nblk != in/qkK {
			return dtype, false
		}
		dtype = compute.Q4_K
	case s.M.kqw[name] != nil:
		q := s.M.kqw[name]
		desc, ok := LookupQuantDescriptor(q.kind)
		if !ok || !desc.SupportsHAL() || desc.BlockWeights() <= 0 || desc.BlockBytes() <= 0 || in%desc.BlockWeights() != 0 || q.nblk != in/desc.BlockWeights() {
			return dtype, false
		}
		dtype = desc.Dtype()
	case s.M.v41PackedHead(name) != nil:
		q := s.M.v41PackedHead(name)
		if q.prism != nil || in%qkK != 0 {
			return dtype, false
		}
		switch q.format {
		case packedEmbeddingQ2K:
			dtype = compute.Q2_K
		case packedEmbeddingQ4K:
			dtype = compute.Q4_K
		default:
			return dtype, false
		}
	default:
		return dtype, false
	}
	if !compute.BackendSupportsDeviceWeightDtype(s.Backend, dtype) || (dtype != compute.F32 && !s.Backend.Caps().UploadDtype) {
		return dtype, false
	}
	switch dtype {
	case compute.Q3_K:
		if native, ok := s.Backend.(interface{ SupportsQ3KMatMul() bool }); ok && !native.SupportsQ3KMatMul() {
			return dtype, false
		}
	case compute.Q5_K:
		if native, ok := s.Backend.(interface{ SupportsQ5KMatMul() bool }); ok && !native.SupportsQ5KMatMul() {
			return dtype, false
		}
	case compute.Q6_K:
		if native, ok := s.Backend.(interface{ SupportsQ6KMatMul() bool }); ok && !native.SupportsQ6KMatMul() {
			return dtype, false
		}
	}
	return dtype, true
}

func (s *Session) v41HeadWeightHAL(name string, dtype compute.Dtype, out, in int) compute.Tensor {
	fail := func(err error) { panic(v41HeadPayloadError{cause: err}) }
	elems, ok := checkedMulInt(out, in)
	if !ok {
		fail(errV41ProjectionResult)
	}

	if q := s.M.v41PackedHead(name); q != nil {
		key := "v41-head-packed:" + q.Format() + ":" + name
		return s.weightHALStagedBounded(key, name, func() compute.Tensor {
			expected, valid := checkedMulInt(elems/qkK, q.format.blockBytes())
			if !valid || q.vocab != out || q.hidden != in || in%qkK != 0 || len(q.raw) != expected {
				fail(errV41ProjectionResult)
			}
			switch q.format {
			case packedEmbeddingQ2K:
				if dtype != compute.Q2_K {
					fail(errV41ProjectionResult)
				}
				return compute.NewQ2K(compute.Default(), []int{out, in}, q.raw)
			case packedEmbeddingQ4K:
				if dtype != compute.Q4_K {
					fail(errV41ProjectionResult)
				}
				return compute.NewQ4K(compute.Default(), []int{out, in}, q.raw)
			default:
				fail(errV41ProjectionResult)
				return compute.Tensor{}
			}
		}, dtype, int64(len(q.raw)))
	}
	switch dtype {
	case compute.F32:
		if len(s.M.tensor(name)) != elems {
			fail(errV41ProjectionResult)
		}
		return s.weightHAL(name)
	case compute.Q8_0:
		q := s.M.q8w[name]
		if q == nil || q.nblk != in/qBlk || len(q.q) != elems || len(q.d) != elems/qBlk {
			fail(errV41ProjectionResult)
		}
		return s.weightHALQ8(name, q)
	case compute.Q4_K:
		q := s.M.q4kw[name]
		return s.weightHALStagedBounded("q4k:"+name, name, func() compute.Tensor {
			raw, err := q.materializeRaw()
			if err != nil {
				fail(err)
			}
			expected, valid := checkedMulInt(elems/qkK, q4kBlockBytes)
			if !valid || len(raw) != expected {
				fail(errV41ProjectionResult)
			}
			return compute.NewQ4K(compute.Default(), []int{out, in}, raw)
		}, dtype, q4kResidentBytes(q))
	default:
		q := s.M.kqw[name]
		desc, valid := LookupQuantDescriptor(q.kind)
		if !valid {
			fail(errV41ProjectionResult)
		}
		return s.weightHALStagedBounded(desc.KeyPrefix()+name, name, func() compute.Tensor {
			raw, err := q.materializeRaw()
			if err != nil {
				fail(err)
			}
			expected, valid := checkedMulInt(elems/desc.BlockWeights(), desc.BlockBytes())
			if !valid || len(raw) != expected {
				fail(errV41ProjectionResult)
			}
			return desc.NewHostTensor(out, in, raw)
		}, dtype, kQuantResidentBytes(q))
	}
}

func (s *Session) v41HeadCloseFailure(stage string, cause error) error {
	if s.halFailure == nil {
		s.halFailure = &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-head-projection", Layer: -1, Stage: stage, Cause: cause}
	}
	if !s.halClosed {
		// Teardown cannot replace the selected operation's primary failure or
		// leave its owner eligible for another operation.
		func() {
			defer func() { s.halClosed = true; _ = recover() }()
			s.Close()
		}()
	}
	return s.halFailure
}

func (m *Model) v41PackedHead(name string) *Q2KEmbedding {
	if name != "model.embed_tokens.weight" {
		return nil
	}
	if m.hasResidentWeight(name) {
		return nil
	}
	return m.Q2KEmbedding
}

func (m *Model) v41HeadWeightShape(name string) (out, in int, ok bool) {
	if out, in, ok = m.residentShape(name); ok {
		return
	}
	if q := m.v41PackedHead(name); q != nil {
		return q.vocab, q.hidden, true
	}
	return 0, 0, false
}

func (m *Model) v41PackedHeadRows(name string, panel []float32, out, in, rows int) ([]float32, error) {
	q := m.v41PackedHead(name)
	if q == nil || out != q.vocab || in != q.hidden || in <= 0 || in%q.format.blockWeights() != 0 {
		return nil, errV41ProjectionResult
	}
	blocks, ok := checkedMulInt(out, in/q.format.blockWeights())
	expected, bytesOK := checkedMulInt(blocks, q.format.blockBytes())
	n, outputOK := checkedMulInt(rows, out)
	if !ok || !bytesOK || !outputOK || len(q.raw) != expected {
		return nil, errV41ProjectionResult
	}
	y := make([]float32, n)
	weightRow := make([]float32, in)
	for row := 0; row < rows; row++ {
		original := panel[row*in : (row+1)*in]
		x := m.prismProjectInput(name, original)
		for token := 0; token < out; token++ {
			if err := q.GatherRow(token, weightRow, 1); err != nil {
				return nil, err
			}
			var sum float32
			for i, w := range weightRow {
				sum += w * x[i]
			}
			y[row*out+token] = sum
		}
		m.loraApply(name, original, y[row*out:(row+1)*out])
	}
	return y, nil
}

// v41ReleaseHeadTensors is deferred directly by the selected head operation.
// It attempts every transient free, preserving the operation's earlier panic
// or returned error. With no earlier failure, its first Free panic escapes so
// the selected callback still retires its owner rather than reporting success.
func v41ReleaseHeadTensors(backend compute.Backend, owned *[]compute.Tensor, err *error) {
	primary := recover()
	var cleanupPanic any
	for i := len(*owned) - 1; i >= 0; i-- {
		func() {
			defer func() {
				if r := recover(); r != nil && cleanupPanic == nil {
					cleanupPanic = r
				}
			}()
			backend.Free((*owned)[i])
		}()
	}
	*owned = nil
	if primary != nil {
		panic(primary)
	}
	if *err == nil && cleanupPanic != nil {
		panic(cleanupPanic)
	}
}
