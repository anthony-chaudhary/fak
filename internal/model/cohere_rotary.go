package model

// cohere_rotary.go — the load-time rotary re-layout for HuggingFace-source Cohere
// (Command-R / Command-R+ / Cohere2) checkpoints.
//
// The divergence it fixes. fak has exactly ONE rotary kernel, applyRopeRow (rope.go),
// and it implements Llama's rotate_half convention: within a head, element j pairs with
// element j+head_dim/2 under angle pos*inv_freq[j]. Cohere does NOT use that convention.
// CohereRotaryEmbedding builds its table with
//
//	emb = torch.repeat_interleave(freqs, 2, dim=-1)   # "diff from Llama: we
//	                                                  #  interleave() instead of cat()"
//
// and Cohere ships its OWN rotate_half,
//
//	x1 = x[..., ::2]; x2 = x[..., 1::2]; stack([-x2, x1]).flatten(-2)  # "different from
//	                                                                   #  e.g. Llama"
//
// so in an HF Cohere checkpoint element 2j pairs with element 2j+1 — the interleaved
// (GPT-J) convention. Rotating those rows with rotate_half mixes the wrong component
// pairs at the wrong frequencies. Nothing catches it structurally: every shape matches,
// position 0 is exact (a length-1 causal softmax is 1.0 regardless of q and k), and the
// error only grows with position, which is precisely the silent-wrong-answer shape the
// family CPU oracles exist to find (family_cohere_cpu_oracle_test.go).
//
// Why a load-time row permutation is the whole fix. Attention consumes the rotated q and
// k ONLY through the per-head dot product q·k; v and o_proj never see a rotated value.
// A permutation applied identically to q and k therefore leaves every score unchanged.
// Let sigma send head-local component 2j -> j and 2j+1 -> j+half. For y = sigma(x),
//
//	rotate_half(y)[j]      = y[j]*cos_j - y[j+half]*sin_j = x[2j]*cos_j - x[2j+1]*sin_j
//	rotate_half(y)[j+half] = y[j+half]*cos_j + y[j]*sin_j = x[2j+1]*cos_j + x[2j]*sin_j
//
// which is exactly sigma(cohere_rotate_half(x)). So permuting the OUTPUT ROWS of q_proj
// and k_proj by sigma at load makes fak's single half-split kernel reproduce Cohere's
// interleaved rotation bit-for-bit, with zero cost in the forward and no new axis
// threaded through the ~20 ropeRowQKInto call sites.
//
// The same permutation must reach anything else indexed by that head-local component
// axis: the optional per-head q_norm/k_norm weights (qk-norm runs after projection and
// before rotary, and its LayerNorm reduction is permutation-invariant while its weight is
// per-element) and any q/k projection bias (Cohere sets attention_bias False, so this is
// defensive).
//
// Why this is NOT applied in newModel. newModel is also the GGUF entry point
// (NewFromF32Tensors), and the GGUF path already runs a rotary re-layout of its own:
// command-r is not on ggufArchStoresHFRotaryLayout's allow-list, so every GGUF Cohere
// q/k tensor goes through unpermuteRotaryTensor on the way in. Applying sigma in
// newModel would COMPOSE with that one rather than replace it, and sigma is not an
// involution for head_dim > 4, so the composition is not a no-op. Hence the split:
// HF-source f32 loaders call newHFCheckpointModel, the GGUF loader keeps calling
// newModel, and this change leaves the GGUF path byte-identical to what it was.
//
// Note this says only that the GGUF path is UNCHANGED, not that it is correct. Whether
// llama.cpp's converter emits Cohere q/k in the layout unpermuteRotaryTensor expects is
// a separate question that this commit does not settle: there is no GGUF-path CPU
// oracle, and the oracle added here drives the HF f32 loaders. If GGUF Cohere is also
// mis-rotated it will need its own witness.
//
// KNOWN RESIDUAL, deliberately not papered over: the quantized HF loaders (GPTQ, AWQ, the
// lean Q8_0/Q4_K safetensors paths) keep q_proj/k_proj in their own decoded stores rather
// than in the f32 `raw` blob this pass rewrites, so a QUANTIZED Cohere checkpoint still
// rotates with the wrong pairing. Fixing that means permuting inside each quant decoder,
// which needs its own witness; this file does not claim it.

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// newHFCheckpointModel builds a Model from a checkpoint that arrived in HUGGINGFACE
// tensor layout, normalizing any layout the kernel's single rotary convention cannot
// express before handing off to newModel. Every HF-source f32 loader funnels through it;
// the GGUF loader calls newModel directly because internal/ggufload has already
// normalized the layout on its side.
func newHFCheckpointModel(cfg Config, man map[string]tensorMeta, raw []byte) (*Model, error) {
	if err := relayoutCohereRotaryTensors(cfg, man, raw); err != nil {
		return nil, err
	}
	return newModel(cfg, man, raw)
}

// usesCohereInterleavedRope reports whether the checkpoint's family rotates q/k with
// Cohere's interleaved (GPT-J) pairing rather than Llama's rotate_half. Both cohere and
// cohere2 do; every other family fak serves uses rotate_half, so this is exactly false
// for them and the load path is byte-identical.
func (c Config) usesCohereInterleavedRope() bool {
	return strings.Contains(c.archFamilyKey(), "cohere")
}

// relayoutCohereRotaryTensors rewrites the head-local component order of every tensor on
// the rotary axis from Cohere's interleaved layout into the half-split layout
// applyRopeRow expects. In place on raw; a no-op for every non-Cohere config.
func relayoutCohereRotaryTensors(cfg Config, man map[string]tensorMeta, raw []byte) error {
	if !cfg.usesCohereInterleavedRope() {
		return nil
	}
	hd := cfg.HeadDim
	if hd <= 0 || hd%2 != 0 {
		return fmt.Errorf("model: cohere rotary re-layout needs an even head_dim, got %d", hd)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		// rowElems 0 means "derive the row width from shape[0]" (a [out,in] projection);
		// rowElems 1 means every element is its own row on the component axis (a bias, or
		// a per-head (heads, head_dim) qk-norm weight).
		for _, tn := range []struct {
			name     string
			rowElems int
		}{
			{layerName(l, "self_attn.q_proj.weight"), 0},
			{layerName(l, "self_attn.k_proj.weight"), 0},
			{layerName(l, "self_attn.q_proj.bias"), 1},
			{layerName(l, "self_attn.k_proj.bias"), 1},
			{layerName(l, "self_attn.q_norm.weight"), 1},
			{layerName(l, "self_attn.k_norm.weight"), 1},
		} {
			meta, ok := man[tn.name]
			if !ok {
				continue
			}
			rowElems := tn.rowElems
			if rowElems == 0 {
				var err error
				if rowElems, err = rotaryRowElems(tn.name, meta); err != nil {
					return err
				}
			}
			if err := interleavedToHalfSplitRows(tn.name, meta, raw, hd, rowElems); err != nil {
				return err
			}
		}
	}
	return nil
}

// rotaryRowElems is the element width of one output row of a [out, in] projection.
func rotaryRowElems(name string, meta tensorMeta) (int, error) {
	if len(meta.Shape) == 0 || meta.Shape[0] <= 0 {
		return 0, fmt.Errorf("model: %s has no output dimension for the cohere rotary re-layout", name)
	}
	total := meta.Nbytes / 4
	if total%meta.Shape[0] != 0 {
		return 0, fmt.Errorf("model: %s has %d f32 values, not a multiple of its %d output rows", name, total, meta.Shape[0])
	}
	return total / meta.Shape[0], nil
}

// interleavedToHalfSplitRows applies sigma (head-local component 2j -> j, 2j+1 -> j+half)
// to the rows of one tensor, in place on raw. rowElems is how many f32 values one row on
// that component axis spans (the input width for a projection weight, 1 for a bias or a
// per-head norm weight).
func interleavedToHalfSplitRows(name string, meta tensorMeta, raw []byte, hd, rowElems int) error {
	if meta.Nbytes%4 != 0 {
		return fmt.Errorf("model: %s is not an f32 tensor (%d bytes)", name, meta.Nbytes)
	}
	total := meta.Nbytes / 4
	if rowElems <= 0 || total%rowElems != 0 {
		return fmt.Errorf("model: %s cannot be cut into %d-element rows (%d values)", name, rowElems, total)
	}
	rows := total / rowElems
	if rows%hd != 0 {
		return fmt.Errorf("model: %s has %d rotary rows, not a multiple of head_dim %d", name, rows, hd)
	}
	src := make([]float32, total)
	for i := 0; i < total; i++ {
		src[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[meta.Offset+i*4:]))
	}
	half := hd / 2
	for h := 0; h < rows/hd; h++ {
		base := h * hd
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				dst := (base + p*half + j) * rowElems
				from := src[(base+2*j+p)*rowElems:]
				for e := 0; e < rowElems; e++ {
					binary.LittleEndian.PutUint32(raw[meta.Offset+(dst+e)*4:], math.Float32bits(from[e]))
				}
			}
		}
	}
	return nil
}
