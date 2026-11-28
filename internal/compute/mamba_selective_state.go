package compute

import (
	"errors"
	"math"
)

// ADAPT of state-spaces/mamba (Apache-2.0),
// mamba_ssm/ops/triton/selective_state_update.py:24 @e9594ce1c732:
// decode-step selective state update with fp32 math, z-gate, dstate block
// selection, and optional state_batch_indices slot indirection (negative index
// skips pad rows). Component and host oracle only: no hardware or speed claim.

// MambaBlockSizeForDState returns the tuned BLOCK_DSTATE tile size for dstate.
func MambaBlockSizeForDState(dstate int) int {
	switch {
	case dstate <= 16:
		return 16
	case dstate <= 32:
		return 32
	case dstate <= 64:
		return 64
	case dstate <= 128:
		return 128
	default:
		return 256
	}
}

// SelectiveStateUpdateDecode updates state[slot][dim][dstate] in place for each
// batch item: state = state*dA + dB*x, out = sum(state*C) (+ D*x), gated by silu(z).
// If stateBatchIndices is non-nil, negative entries are pad rows and are skipped.
func SelectiveStateUpdateDecode(
	state []float32,
	stateBatchIndices []int,
	x, dt, a, b, c, d, z []float32,
	batch, dim, dstate int,
) ([]float32, error) {
	if batch <= 0 || dim <= 0 || dstate <= 0 ||
		len(x) != batch*dim || len(dt) != batch*dim ||
		len(a) != dim*dstate || len(b) != batch*dstate || len(c) != batch*dstate {
		return nil, errors.New("compute: invalid selective state update dimensions")
	}
	if stateBatchIndices != nil && len(stateBatchIndices) != batch {
		return nil, errors.New("compute: stateBatchIndices length mismatch")
	}
	out := make([]float32, batch*dim)
	for bi := 0; bi < batch; bi++ {
		slot := bi
		if stateBatchIndices != nil {
			slot = stateBatchIndices[bi]
			if slot < 0 {
				continue // negative-index pad skip
			}
		}
		if (slot+1)*dim*dstate > len(state) {
			return nil, errors.New("compute: state slot out of bounds")
		}
		for di := 0; di < dim; di++ {
			xv := float64(x[bi*dim+di])
			dtv := float64(dt[bi*dim+di])
			var y float64
			sBase := (slot*dim + di) * dstate
			for s := 0; s < dstate; s++ {
				dA := math.Exp(float64(a[di*dstate+s]) * dtv)
				dB := dtv * float64(b[bi*dstate+s])
				st := float64(state[sBase+s])*dA + dB*xv
				state[sBase+s] = float32(st)
				y += st * float64(c[bi*dstate+s])
			}
			if d != nil {
				y += float64(d[di]) * xv
			}
			if z != nil {
				zv := float64(z[bi*dim+di])
				y *= zv / (1.0 + math.Exp(-zv)) // silu(z)
			}
			out[bi*dim+di] = float32(y)
		}
	}
	return out, nil
}
