package model

import (
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// qwen35_tiled_out_proj.go: native-quant residency for the Qwen3.5-family GDN out_proj.
//
// The GGUF stores ssm_out's input columns in the tiled value-head order, while fak's GDN core
// emits the grouped order. The f32 normalizer reorders the columns, which a native i-quant or
// k-quant weight cannot do in place (a 256-weight super-block spans several 128-wide heads), so
// that route requantizes the weight to Q8_0. A tiled out_proj keeps the GGUF bytes and column
// order instead, and every consumer tiles the core (compute.TileQwen35ValueHeads on the host,
// Qwen35GDNTiledOutBackend / Qwen35SequenceLayer.GDNOutTiled on a device) before the matmul.
// Because W_gguf[:, c*d+i] = W_canonical[:, src(c)*d+i], W_gguf . tile(x) = W_canonical . x.

// Qwen35TiledOutProjection reports whether canon (resolved through the qwen35 source chain) is a
// target-layer GDN out_proj whose value heads are grouped (nV > nK), the only case in which the
// GGUF and canonical input-column orders differ.
func Qwen35TiledOutProjection(cfg Config, canon string) bool {
	if !cfg.IsQwen35Hybrid() {
		return false
	}
	nK, nV := cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads
	if nK <= 0 || nV <= nK || nV%nK != 0 || cfg.LinearValueHeadDim <= 0 {
		return false
	}
	name, keep := quantSourceTensorName(cfg, canon)
	if !keep || !isQuantWeight(name) || strings.HasPrefix(name, "mtp.") {
		return false
	}
	return strings.HasSuffix(name, ".linear_attn.out_proj.weight")
}

// Qwen35OutProjTiled reports whether the named out_proj is held in the GGUF tiled column order.
func (m *Model) Qwen35OutProjTiled(name string) bool {
	if m == nil || len(m.qwen35TiledOutProj) == 0 {
		return false
	}
	_, ok := m.qwen35TiledOutProj[name]
	return ok
}

// HasQwen35TiledOutProj reports whether any out_proj of this model is held in tiled order.
func (m *Model) HasQwen35TiledOutProj() bool {
	return m != nil && len(m.qwen35TiledOutProj) > 0
}

// qwen35TiledOutProjInput returns a [rows, nV*vHd] activation tiled into the GGUF column order
// when name is a tiled out_proj, and x unchanged otherwise. It never mutates x.
func (m *Model) qwen35TiledOutProjInput(name string, x []float32, rows int) []float32 {
	if !m.Qwen35OutProjTiled(name) {
		return x
	}
	nK, nV, _, vHd, _, _, _ := m.Cfg.linearAttnDims()
	out, err := compute.TileQwen35ValueHeads(x, rows, nK, nV, vHd)
	if err != nil {
		panic(fmt.Sprintf("model: tiled out_proj %s input: %v", name, err))
	}
	return out
}

// AddQwen35TiledOutProj stores a GDN out_proj in its native GGUF bytes and tiled column order.
// quant is the GGUF type name: Q4_K, or a 256-weight k-quant/i-quant. Only a backend whose
// consumers tile the core may be handed such a model (ValidateBackendForwardPath enforces it).
func (b *QuantBuilder) AddQwen35TiledOutProj(canon string, shape []int, raw []byte, quant string) error {
	if b == nil || b.m == nil {
		return fmt.Errorf("model: nil QuantBuilder")
	}
	cfg := b.m.Cfg
	if !Qwen35TiledOutProjection(cfg, canon) {
		return fmt.Errorf("model: tiled out_proj tensor %q is not a Qwen3.5 GDN out_proj with grouped value heads", canon)
	}
	_, _, _, _, _, valDim, _ := cfg.linearAttnDims()
	if len(shape) != 2 || shape[0] != cfg.HiddenSize || shape[1] != valDim || valDim%qkK != 0 {
		return fmt.Errorf("model: tiled out_proj %s has shape %v, want [%d %d] with a %d-multiple reduction", canon, shape, cfg.HiddenSize, valDim, qkK)
	}
	blockBytes := q4kBlockBytes
	var kind kQuantKind
	if quant != "Q4_K" {
		desc, ok := LookupQuantDescriptorByName(quant)
		if !ok || desc.BlockWeights() != qkK {
			return fmt.Errorf("model: tiled out_proj %s has unsupported quant %s", canon, quant)
		}
		blockBytes, kind = desc.BlockBytes(), desc.Kind()
	}
	rowBytes := shape[1] / qkK * blockBytes
	if shape[0] > math.MaxInt/rowBytes || len(raw) != shape[0]*rowBytes {
		return fmt.Errorf("model: tiled out_proj %s has %d payload bytes for shape %v", canon, len(raw), shape)
	}
	name, ok, err := b.residentQuantTarget(canon, shape)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("model: tiled out_proj %s is not a resident quant target", canon)
	}
	if b.m.q4kw[name] != nil || b.m.q8w[name] != nil || b.m.kqw[name] != nil || b.m.q2w[name] != nil {
		return fmt.Errorf("model: tiled out_proj %s already has a resident representation", name)
	}
	if _, exists := b.m.manifest[name]; exists {
		return fmt.Errorf("model: tiled out_proj %s already has a decoded representation", name)
	}
	if quant == "Q4_K" {
		if b.m.q4kw == nil {
			b.m.q4kw = map[string]*q4kTensor{}
		}
		b.m.q4kw[name] = quantizeQ4KFromRaw(raw, shape[0], shape[1])
	} else {
		if b.m.kqw == nil {
			b.m.kqw = map[string]*kQuantTensor{}
		}
		b.m.kqw[name] = quantizeKQuantFromRaw(raw, shape[0], shape[1], kind)
	}
	if b.m.qwen35TiledOutProj == nil {
		b.m.qwen35TiledOutProj = map[string]struct{}{}
	}
	b.m.qwen35TiledOutProj[name] = struct{}{}
	return nil
}
