package model

// kimik31_forward.go — the Kimi-K3.1 composite forward (ticket 1611-kimi-k31,
// leaf L8; issue #14363). This is the per-layer composition the classify branch
// in arch_support.go (ForwardKimiK31) dispatches to: under ONE layer_types
// switch it runs each of the 93 decoder layers through the landed L5 KDA oracle
// (ForwardKimiK31KDAStep) or the landed L6 Gated-DSA oracle
// (ForwardKimiK31DSAStep), applying the pre-attention RMSNorm and the residual
// add the oracles deliberately leave to the caller.
//
// WHAT THIS OWNS: the wiring from a recognized K3.1 config (IsKimiK31, whose
// LayerTypes carry the L1 admission's "kda"/"dsa" tokens) to the per-layer
// oracles, so a K3.1 checkpoint is no longer dead code behind the L8 gate. The
// dispatch switches on the SAME predicates the classifier reads
// (IsKimiK31KDALayer / IsKimiK31DSALayer), so the named path cannot drift from
// what the composite actually runs.
//
// WHAT THIS DOES NOT OWN (later leaves, no surface invented here): checkpoint
// tensor resolution (the safetensors/GGUF loader leaves build
// KimiK31ForwardParams from real weights; this file takes resolved params so the
// composite stays testable on synthetic tensors), the latent-MoE / SiTU /
// attention-residual / MLA-output-gate leaves (Wave 1), and the full assembled
// 93-layer forward with the Kimi chat template + generation loop (Wave 5).
// Until those land the composition carries attention + residual only, with the
// tiny-tensor smoke test pinning finite logits through this path on synthetic
// weights.

import "fmt"

// KimiK31ForwardParams holds the resolved per-layer oracle params and the
// per-layer input norm for the Kimi K3.1 text stack. The loader leaves build it
// from checkpoint tensors; the composite itself never reads a tensor store, so
// it is exercisable on synthetic weights.
//
// Exactly one of KDA[l] / DSA[l] must be set per layer, matching the config's
// layer kind (IsKimiK31KDALayer(l) => KDA[l] != nil). AttnNorm[l] is that
// layer's pre-attention RMSNorm weight ([Hidden]); FinalNorm is the optional
// output RMSNorm weight ([Hidden]).
type KimiK31ForwardParams struct {
	Hidden    int
	RMSEps    float32
	AttnNorm  [][]float32
	FinalNorm []float32
	KDA       []*KimiK31KDAParams
	DSA       []*KimiK31DSAParams
}

// KimiK31ForwardState holds the recurrent/cell state for every layer of the
// composite: one KDA state per KDA layer and one Gated-DSA cell cache per DSA
// layer. Decode advances the SAME state token-by-token, so a prefill and a
// stepwise decode of the same ids are the same computation.
type KimiK31ForwardState struct {
	KDA map[int]*GLM5NextKDALayerState
	DSA map[int]*KimiK31DSALayerState
}

// NewKimiK31ForwardState allocates per-layer state matching p's geometry for
// the layers cfg marks KDA/DSA. capacity is the Gated-DSA cell budget (>= the
// longest sequence that will be run through the layer). The KDA state geometry
// is read from each layer's params (NumHeads, HeadDim, ConvKernel).
func NewKimiK31ForwardState(cfg Config, p *KimiK31ForwardParams, capacity int) (*KimiK31ForwardState, error) {
	if p == nil {
		return nil, fmt.Errorf("model: kimi-k31 forward: nil params")
	}
	if cfg.NumLayers <= 0 || p.Hidden <= 0 {
		return nil, fmt.Errorf("model: kimi-k31 forward: bad geometry (layers=%d hidden=%d)", cfg.NumLayers, p.Hidden)
	}
	if len(p.AttnNorm) != cfg.NumLayers {
		return nil, fmt.Errorf("model: kimi-k31 forward: AttnNorm has %d layers, want %d", len(p.AttnNorm), cfg.NumLayers)
	}
	if capacity < 1 {
		capacity = 1
	}
	st := &KimiK31ForwardState{
		KDA: make(map[int]*GLM5NextKDALayerState),
		DSA: make(map[int]*KimiK31DSALayerState),
	}
	for l := 0; l < cfg.NumLayers; l++ {
		kda := l < len(p.KDA) && p.KDA[l] != nil
		dsa := l < len(p.DSA) && p.DSA[l] != nil
		switch {
		case cfg.IsKimiK31KDALayer(l):
			if !kda {
				return nil, fmt.Errorf("model: kimi-k31 forward: layer %d is KDA but has no KDA params", l)
			}
			st.KDA[l] = NewGLM5NextKDALayerState(p.KDA[l].NumHeads, p.KDA[l].HeadDim, p.KDA[l].ConvKernel)
		case cfg.IsKimiK31DSALayer(l):
			if !dsa {
				return nil, fmt.Errorf("model: kimi-k31 forward: layer %d is DSA but has no DSA params", l)
			}
			dp := p.DSA[l]
			st.DSA[l] = NewKimiK31DSALayerState(capacity, dp.KVLoRA+dp.QkRope, dp.IdxDim)
		default:
			return nil, fmt.Errorf("model: kimi-k31 forward: layer %d carries no kda/dsa layer_types token", l)
		}
	}
	return st, nil
}

// ForwardKimiK31Token runs one token through every layer of the composite and
// returns the final hidden vector ([Hidden], post final norm). It applies, per
// layer, the pre-attention RMSNorm, the KDA or DSA oracle step, and the
// residual add; the config's layer kind selects the oracle (the same predicate
// ClassifyForwardPath named). x must already carry the embedding for the token.
func ForwardKimiK31Token(cfg Config, p *KimiK31ForwardParams, st *KimiK31ForwardState, x []float32) []float32 {
	if p == nil || st == nil {
		panic("model: kimi-k31 forward: nil params or state")
	}
	if len(x) != p.Hidden {
		panic(fmt.Sprintf("model: kimi-k31 forward: input length %d != Hidden %d", len(x), p.Hidden))
	}
	cur := append([]float32(nil), x...)
	xn := make([]float32, p.Hidden)
	for l := 0; l < cfg.NumLayers; l++ {
		rmsnormInto(xn, cur, p.AttnNorm[l], p.RMSEps)
		var attn []float32
		switch {
		case cfg.IsKimiK31KDALayer(l):
			attn = ForwardKimiK31KDAStep(st.KDA[l], p.KDA[l], xn)
		case cfg.IsKimiK31DSALayer(l):
			attn = ForwardKimiK31DSAStep(st.DSA[l], p.DSA[l], xn)
		default:
			panic(fmt.Sprintf("model: kimi-k31 forward: layer %d carries no kda/dsa layer_types token", l))
		}
		for i := range cur {
			cur[i] += attn[i]
		}
	}
	if p.FinalNorm != nil {
		out := make([]float32, p.Hidden)
		rmsnormInto(out, cur, p.FinalNorm, p.RMSEps)
		return out
	}
	return cur
}

// RunKimiK31Forward runs T tokens causally through the composite and returns the
// final hidden vector for every position ([T*Hidden]). The oracle steps are
// token-by-token by construction, so the sequence scan reproduces a batched
// causal forward with the same state threading the decode loop uses. (The
// ForwardPathKind constant it dispatches under is ForwardKimiK31; this entry
// point is named Run* to avoid colliding with it.)
func RunKimiK31Forward(cfg Config, p *KimiK31ForwardParams, st *KimiK31ForwardState, xSeq []float32, T int) []float32 {
	if p == nil {
		panic("model: kimi-k31 forward: nil params")
	}
	if len(xSeq) != T*p.Hidden {
		panic(fmt.Sprintf("model: kimi-k31 forward: xSeq length %d != T*Hidden %d", len(xSeq), T*p.Hidden))
	}
	out := make([]float32, T*p.Hidden)
	for t := 0; t < T; t++ {
		o := ForwardKimiK31Token(cfg, p, st, xSeq[t*p.Hidden:(t+1)*p.Hidden])
		copy(out[t*p.Hidden:(t+1)*p.Hidden], o)
	}
	return out
}
