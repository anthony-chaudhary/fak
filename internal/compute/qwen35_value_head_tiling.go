package compute

import (
	"errors"
	"fmt"
	"math"
)

// A Qwen3.5-family GGUF stores ssm_out (linear_attn.out_proj) input columns in the value-head
// "tiled" order c = r*numKeyHeads + k, while fak's GDN core emits the "grouped" order
// h = k*ratio + r (ratio = numValueHeads/numKeyHeads) that the recurrence's key-head sharing
// needs. A native i-quant/k-quant ssm_out cannot have its columns reordered (one 256-weight
// super-block spans several heads), so the loader keeps the GGUF bytes and the core is permuted
// into the tiled order instead, immediately before the output projection.

// ErrQwen35ValueHeadTiling is the closed refusal for a malformed tiling request.
var ErrQwen35ValueHeadTiling = errors.New("compute: invalid Qwen3.5 value-head tiling geometry")

// Qwen35TiledValueHeadSource returns src such that tiled head position c reads grouped core
// head src[c]. It is the single permutation source the host reference, the device kernel and
// the loader's column contract are checked against.
func Qwen35TiledValueHeadSource(numKeyHeads, numValueHeads int) ([]int, error) {
	if numKeyHeads <= 0 || numValueHeads <= 0 || numValueHeads%numKeyHeads != 0 {
		return nil, fmt.Errorf("%w: key heads %d, value heads %d", ErrQwen35ValueHeadTiling, numKeyHeads, numValueHeads)
	}
	ratio := numValueHeads / numKeyHeads
	src := make([]int, numValueHeads)
	for c := range src {
		src[c] = (c%numKeyHeads)*ratio + c/numKeyHeads
	}
	return src, nil
}

// TileQwen35ValueHeads is the host reference for the tiling gather over a [tokens, nV*vHd]
// grouped core panel. Every output element is one input element, so it is exact.
func TileQwen35ValueHeads(core []float32, tokens, numKeyHeads, numValueHeads, valueHeadDim int) ([]float32, error) {
	src, err := Qwen35TiledValueHeadSource(numKeyHeads, numValueHeads)
	if err != nil {
		return nil, err
	}
	if tokens <= 0 || valueHeadDim <= 0 || numValueHeads > math.MaxInt/valueHeadDim ||
		tokens > math.MaxInt/(numValueHeads*valueHeadDim) || len(core) != tokens*numValueHeads*valueHeadDim {
		return nil, fmt.Errorf("%w: %d values for %d tokens x %d heads x %d", ErrQwen35ValueHeadTiling, len(core), tokens, numValueHeads, valueHeadDim)
	}
	width := numValueHeads * valueHeadDim
	out := make([]float32, len(core))
	for t := 0; t < tokens; t++ {
		row := t * width
		for c, h := range src {
			copy(out[row+c*valueHeadDim:row+(c+1)*valueHeadDim], core[row+h*valueHeadDim:row+(h+1)*valueHeadDim])
		}
	}
	return out, nil
}

// Qwen35GDNTiledOutBackend is the optional whole-operation GDN decode for a layer whose out_proj
// keeps the GGUF tiled input-column order: it is Qwen35GDNDecode with the core tiled
// (Qwen35TiledValueHeadSource) before the output projection. A backend without it cannot serve
// such a layer, and the model refuses that pairing at admission.
type Qwen35GDNTiledOutBackend interface {
	Qwen35GDNDecodeTiledOut(
		normalizedInput,
		inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj,
		convState, recurrentState Tensor,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
		rmsNormEpsilon float32,
	) (output, nextConvState, nextRecurrentState Tensor, err error)
}
