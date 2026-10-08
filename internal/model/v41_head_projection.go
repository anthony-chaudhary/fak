package model

import "github.com/anthony-chaudhary/fak/internal/compute"

type v41HeadPayloadError struct{ cause error }

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
	closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-head-projection", Layer: -1, Stage: stage, Cause: cause}
	s.halFailure = closed
	s.Close()
	return closed
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
