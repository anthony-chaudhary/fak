package ggufload

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// A positive bound explicitly enables the mapped-host fallback. Zero/unset
// preserves the existing device/CPU placement without attempting pinned memory.
const inkling2MappedExpertBytesEnv = "FAK_INKLING2_MAPPED_EXPERT_BYTES"

func inkling2MappedExpertBound(lookup func(string) (string, bool)) (int64, error) {
	v, ok := lookup(inkling2MappedExpertBytesEnv)
	if !ok || strings.TrimSpace(v) == "" {
		return 0, nil
	}
	n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("inkling2: %s=%q is not a non-negative byte count", inkling2MappedExpertBytesEnv, v)
	}
	return n, nil
}

type inkling2MappedCandidate struct {
	dst          *model.Inkling2LoopExperts
	source       compute.Tensor
	experts      int
	rows, cols   int
	padK, nbytes int
}

// inkling2MappedExpertSource performs the same packed-layout preparation as
// inkling2DeviceWeight, but stops before allocation so the mapped-host uploader
// owns the destination placement.
func inkling2MappedExpertSource(be compute.Backend, q *inkling2QuantExperts) (compute.Tensor, int, int, error) {
	shape := []int{q.experts * q.rows, q.cols}
	switch q.typ {
	case TensorNVFP4:
		packed, scales, err := repackInkling2NVFP4(q.raw, shape[0], q.cols)
		if err != nil {
			return compute.Tensor{}, 0, 0, err
		}
		t, err := compute.NewNVFP4(be, shape, packed, scales, 1)
		return t, q.cols, len(packed) + len(scales), err
	case TensorQ2_K:
		return compute.NewQ2K(be, shape, q.raw), q.cols, len(q.raw), nil
	case TensorQ3_K:
		return compute.NewQ3K(be, shape, q.raw), q.cols, len(q.raw), nil
	case TensorQ4_K:
		return compute.NewQ4K(be, shape, q.raw), q.cols, len(q.raw), nil
	case TensorQ5_K:
		return compute.NewQ5K(be, shape, q.raw), q.cols, len(q.raw), nil
	case TensorQ6_K:
		return compute.NewQ6K(be, shape, q.raw), q.cols, len(q.raw), nil
	case TensorIQ2_XS, TensorIQ2_S, TensorIQ3_XXS, TensorIQ3_S, TensorIQ4_XS:
		dt := inkling2RawIQDtype[q.typ]
		return compute.NewRawIQ(be, dt, shape, q.raw), q.cols, len(q.raw), nil
	case TensorIQ4_NL:
		padded, padK := inkling2PadIQ4NL(q.raw, shape[0], q.cols)
		return compute.NewRawIQ(be, compute.IQ4_NL, []int{shape[0], padK}, padded), padK, len(padded), nil
	default:
		return compute.Tensor{}, 0, 0, &UnsupportedInkling2LoopTensorTypeError{Type: q.typ}
	}
}

// mapColdExperts is transactional: either every cold stack becomes a
// device-visible mapped-host Tensor before RowViews are made, or every original
// host expert remains installed. Partial pinned roots are freed on decline.
func (s *inkling2DeviceSet) mapColdExperts(m *Inkling2LoopModel, resident []bool, bound int64) (int, int64, error) {
	if bound == 0 {
		return 0, 0, nil
	}
	uploader, ok := s.be.(compute.MappedHostWeightUploader)
	if !ok {
		return 0, 0, fmt.Errorf("inkling2: %s requires mapped-host weight support from backend %s", inkling2MappedExpertBytesEnv, s.be.Name())
	}
	var candidates []inkling2MappedCandidate
	var total int64
	for _, i := range inkling2ExpertResidencyOrder(m.HP, len(m.Weights.Layers)) {
		if resident[i] {
			continue
		}
		lw := &m.Weights.Layers[i]
		for _, dst := range []*model.Inkling2LoopExperts{&lw.GateUp, &lw.Down} {
			if *dst == nil {
				continue
			}
			q, ok := (*dst).(*inkling2QuantExperts)
			if !ok {
				return 0, 0, fmt.Errorf("inkling2: mapped-host experts require a packed cold stack (got %T)", *dst)
			}
			source, padK, nbytes, err := inkling2MappedExpertSource(s.be, q)
			if err != nil {
				return 0, 0, err
			}
			total += int64(nbytes)
			candidates = append(candidates, inkling2MappedCandidate{dst: dst, source: source, experts: q.experts, rows: q.rows, cols: q.cols, padK: padK, nbytes: nbytes})
		}
	}
	if len(candidates) == 0 {
		return 0, 0, nil
	}
	if total > bound {
		return 0, 0, fmt.Errorf("inkling2: mapped-host cold experts require %d bytes, above %s=%d", total, inkling2MappedExpertBytesEnv, bound)
	}
	type publication struct {
		dst  *model.Inkling2LoopExperts
		root compute.Tensor
		out  *inkling2DeviceExperts
	}
	pubs := make([]publication, 0, len(candidates))
	rollback := func() {
		for _, p := range pubs {
			s.be.Free(p.root)
		}
	}
	remaining := bound
	for index, candidate := range candidates {
		root, mapped, err := uploader.UploadMappedHostWeight(candidate.source, candidate.source.Dtype, remaining)
		if err != nil {
			rollback()
			return 0, 0, err
		}
		if !mapped {
			rollback()
			return 0, 0, fmt.Errorf("inkling2: mapped-host allocation declined cold expert stack %d of %d", index+1, len(candidates))
		}
		out := &inkling2DeviceExperts{be: s.be, stack: root, w: make([]compute.Tensor, candidate.experts), rows: candidate.rows, cols: candidate.cols, padK: candidate.padK}
		for expert := range out.w {
			out.w[expert], err = compute.RowView(s.be, root, expert*candidate.rows, candidate.rows)
			if err != nil {
				s.be.Free(root)
				rollback()
				return 0, 0, fmt.Errorf("inkling2_loop: mapped expert %d of Q3_K stack: %w", expert, err)
			}
		}
		pubs = append(pubs, publication{dst: candidate.dst, root: root, out: out})
		remaining -= int64(candidate.nbytes)
	}
	for _, p := range pubs {
		*p.dst = p.out
		s.keep(p.root)
	}
	return len(pubs), total, nil
}
