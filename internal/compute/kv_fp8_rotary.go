package compute

import (
	"errors"
	"math"
)

var errKVFP8Rotary = errors.New("compute: FP8 KV rotary contract refused")

// This identity names the exact normalized-K producer and rotate-half kernel:
// k_qwen35_partial_rope_panel in cuda/transformer_ops.cuh at Fak 9b8ea46b.
// It is not interchangeable with full-head k_rope's 1/pow frequency expression,
// fused QK donors, scaled RoPE, or another arithmetic/build variant.
const kvFP8QwenPartialPanelV1 = 1

type kvFP8RotaryContract struct {
	producer           uint8
	mainLayer          int
	headDim, rotaryDim int
	thetaBits          uint64
}

// bindQwenRotary validates the entire mapping before changing any owner metadata.
// The currently bounded Qwen attention seam is 4 KV heads, hd256, rotary64.
// Generic appends do not call this method merely because they know a theta.
func (o *kvFP8Owner) bindQwenRotary(linear []bool, theta []float64, rotaryDim int) error {
	if err := o.usable(); err != nil {
		return err
	}
	if o.heads != 4 || o.headDim != 256 || rotaryDim != 64 ||
		len(linear)%4 != 0 || len(linear)/4 != o.layers || len(theta) != len(linear) {
		return errKVFP8Rotary
	}
	next := make([]kvFP8RotaryContract, o.layers)
	for main := range linear {
		if linear[main] != ((main+1)%4 != 0) ||
			!(theta[main] > 0) || math.IsNaN(theta[main]) || math.IsInf(theta[main], 0) {
			return errKVFP8Rotary
		}
		if !linear[main] {
			next[main/4] = kvFP8RotaryContract{
				producer: kvFP8QwenPartialPanelV1, mainLayer: main,
				headDim: o.headDim, rotaryDim: rotaryDim,
				thetaBits: math.Float64bits(theta[main]),
			}
		}
	}
	for l, want := range next {
		have := o.rotary[l]
		if have == want {
			continue
		}
		if have.producer != 0 || o.image.positions != 0 {
			return errKVFP8Rotary
		}
	}
	copy(o.rotary, next)
	return nil
}

type kvFP8EvictionPlan struct {
	owner                    *kvFP8Owner
	stamp                    uint64
	from, end, before, after int
	rotary                   []kvFP8RotaryContract
}

// planEvict assumes this owner's position contract is canonical: row i was
// produced at absolute position i. Importing arbitrary position lists is not
// implemented. The CUDA adapter must check that before installing an image.
func (o *kvFP8Owner) planEvict(from, count int) (*kvFP8EvictionPlan, error) {
	if err := o.usable(); err != nil {
		return nil, err
	}
	before := o.image.positions
	if from < 0 || count <= 0 || from >= before {
		return nil, nil
	}
	end := before
	if count < before-from {
		end = from + count
	}
	if end < before {
		for _, c := range o.rotary {
			if c.producer != kvFP8QwenPartialPanelV1 || c.headDim != o.headDim ||
				c.rotaryDim != 64 || o.headDim != 256 {
				return nil, errKVFP8Rotary
			}
		}
	}
	return &kvFP8EvictionPlan{
		owner: o, stamp: o.stamp.Load(), from: from, end: end,
		before: before, after: before - (end - from),
		rotary: append([]kvFP8RotaryContract(nil), o.rotary...),
	}, nil
}

// sourceRow maps destination rows to old rows. Only suffix rows are rerotated;
// prefix K/code/scale bytes, all retained Kraw and all retained V/code/scales
// are copied bit-for-bit. Every layer's rerotated K is numerically validated in
// DISJOINT staging before replace is allowed to publish any layer.
func (p *kvFP8EvictionPlan) sourceRow(row int) (source int, rerotate bool, err error) {
	if p == nil || row < 0 || row >= p.after {
		return 0, false, ErrKVFP8Geometry
	}
	if row < p.from {
		return row, false, nil
	}
	return row + p.end - p.from, true, nil
}

func (p *kvFP8EvictionPlan) apply(workspaceSizes []int, peakBudget int64, ops kvFP8OwnerOps,
	stage func(dst, src *kvFP8Image, plan *kvFP8EvictionPlan) kvFP8Completion) error {
	if p == nil || p.owner == nil || stage == nil || p.owner.stamp.Load() != p.stamp ||
		p.owner.image.positions != p.before {
		return errKVFP8Owner
	}
	return p.owner.replace(p.after, workspaceSizes, peakBudget, ops,
		func(dst, src *kvFP8Image) kvFP8Completion { return stage(dst, src, p) }, false)
}
