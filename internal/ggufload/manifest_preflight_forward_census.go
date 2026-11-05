package ggufload

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// manifest_preflight_forward_census.go — the layout-guard (lg-preflight-wire) extension of the
// header-only manifest preflight: the forward-request census and the MTP declaration check that
// turn the ledger's post-hoc incident classes into pre-dispatch refusals.
//
// WHY: LML-01..LML-03/LML-09/LML-10 were each discovered only after a physical serve attempt
// (an unmapped V4.1 name, a published-vs-converter shape drift, an admitted-but-unconsumable
// geometry, a declared MTP head with zero tensors). The header already carries the full tensor
// directory, so the same seams aggregate OFF THE HEADER here, in the shared
// fak.new-model-refusal/1 vocabulary, before any payload byte is read.
//
// SCOPE: the deepseek41 forward requests every non-attention operand the native V4.1 assembly
// admits (embeddings, final norm, lm_head, per-layer norms, mHC trios, router gate + bias,
// shared experts, batched routed-expert blobs); the attention operands stay with
// deepseek41AttentionDescriptorDiagnostics so no seam is reported twice. Engram, compressor,
// indexer and vision sidecars are OUT of scope (their reader paths are separate seams), as is
// the device (op,dtype) census, which needs a live arch this header-only plan cannot expose.

// ds41FullHeadDim mirrors internal/model's v41KVLoraRank (512): the published V4.1 KV latent
// rank is a checkpoint constant the flat Config does not carry, so the full/reduced
// discriminator reads the flat HeadDim against it exactly as v41ForwardGeometry does.
const ds41FullHeadDim = 512

// ds41MHCMixWidth mirrors v41MHCMixWidth: (2+hc)*hc = 24 at the published hc_mult=4.
const ds41MHCMixWidth = 24

// checkMTPDeclaration refuses a checkpoint whose OWN METADATA declares an MTP
// self-speculation head but whose tensor directory carries none of it (ledger LML-09: a
// converted artifact declaring num_nextn_predict_layers yet shipping zero nextn tensors — a
// sourcing problem the header alone can name). The declaration is read from the two sources
// the served artifact itself carries: the parsed config (NumMTPLayers, which the qwen35
// family folds from <arch>.nextn_predict_layers) and the raw <arch>.nextn_predict_layers key
// (arches whose Config does not fold it in, such as deepseek41). The companion HF config.json
// is deliberately NOT a declaration source: the preflight judges the artifact being served,
// and a declaration that lives only in the reference config (never mirrored into the GGUF
// metadata) leaves the text forward unchanged — the head is simply absent, not mis-declared.
// The raw-key read runs through the traced lookup so a declared key counts as READ for key
// coverage.
func checkMTPDeclaration(out *ManifestPreflight, f *File, cfg model.Config) {
	depth := cfg.NumMTPLayers()
	if n, ok := f.Uint64(out.Arch + ".nextn_predict_layers"); ok && n <= uint64(math.MaxInt) && int(n) > depth {
		depth = int(n)
	}
	if depth <= 0 || !archShipsMTPOrVisionSidecar(cfg.ModelType) {
		return
	}
	for _, t := range f.Tensors {
		if isGLMMoeDsaMTPTensor(t.Name) {
			return // the head ships tensors; whether a reader consumes them is a separate seam
		}
	}
	out.Tensors.Missing++
	out.refuse(newmodel.RefusalTensorMissing, "mtp", "declares %d MTP layer(s) but the header carries no nextn/MTP tensor", depth)
}

// deepseek41ForwardRequestCensus aggregates, off the header alone, every NON-ATTENTION tensor
// the native V4.1 forward requests at its admitted geometry: the global embedding / final norm
// / lm_head, and per layer the block norms, the mHC coefficient trios, the router gate + score
// correction bias, the shared experts, and the batched routed-expert blobs the loader splits
// 1->E. Each gap becomes a refusal in the shared vocabulary (missing -> TENSOR_MISSING,
// present-at-wrong-dims -> TENSOR_SHAPE_MISMATCH), so a checkpoint that admission accepted but
// the execution reader cannot consume (LML-03/LML-10) is named BEFORE dispatch instead of after
// a multi-minute load. Attention operands stay with deepseek41AttentionDescriptorDiagnostics;
// Engram/compressor/indexer/vision sidecars and the device (op,dtype) census are out of scope.
func deepseek41ForwardRequestCensus(out *ManifestPreflight, f *File, cfg model.Config) {
	if !archIsDeepSeek41(cfg.ModelType) {
		return
	}
	layers, H, vocab := cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize
	E, I := cfg.NumExperts, cfg.MoEIntermediateSize
	if layers <= 0 || H <= 0 || vocab <= 0 {
		// The geometry guards that make the requests meaningful are already enforced by
		// File.Config/applyDeepSeek41Config; without them there is nothing derivable to request.
		return
	}
	if cfg.DeepSeekV41 != nil && cfg.DeepSeekV41.NumericalProfile != nil {
		out.refuse(newmodel.RefusalHeaderConfigInvalid, "config", "explicit DeepSeek-V4.1 numerical semantics are not qualified; the forward refuses this config")
		return
	}
	full, consistent := ds41ForwardFullGeometry(cfg)
	if !consistent {
		out.refuse(newmodel.RefusalHeaderConfigInvalid, "config", "published V4.1 attention envelope disagrees with the flat decoder axes; the forward geometry refuses this config")
		return
	}
	dimsByName := make(map[string][]uint64, len(f.Tensors))
	for _, t := range f.Tensors {
		if _, dup := dimsByName[t.Name]; !dup {
			dimsByName[t.Name] = t.Dims
		}
	}
	require := func(name string, want []uint64) {
		got, ok := dimsByName[name]
		if !ok {
			out.Tensors.Missing++
			out.refuse(newmodel.RefusalTensorMissing, name, "the forward requests %s %v; the header has no such tensor", name, want)
			return
		}
		if !dimsEqual(got, want) {
			out.Tensors.Shape++
			out.refuse(newmodel.RefusalTensorShape, name, "dims %v, want %v", got, want)
		}
	}
	Hu, Eu, Iu := uint64(H), uint64(E), uint64(I)

	// Globals: the embedding is required; the lm_head may tie to it, so output.weight is
	// checked only when present (mirrors v41ForwardAdmitted's head stage).
	require("token_embd.weight", []uint64{Hu, uint64(vocab)})
	require("output_norm.weight", []uint64{Hu})
	if got, ok := dimsByName["output.weight"]; ok && !dimsEqual(got, []uint64{Hu, uint64(vocab)}) {
		out.Tensors.Shape++
		out.refuse(newmodel.RefusalTensorShape, "output.weight", "dims %v, want %v (or tied to the embedding)", got, []uint64{Hu, uint64(vocab)})
	}
	if E <= 0 || I <= 0 {
		// An undeclared MoE axis is refused earlier by the forward's router-envelope
		// admission; there is no expert geometry to request against.
		return
	}
	mixW := uint64(ds41MHCMixWidth)
	for l := 0; l < layers; l++ {
		pfx := fmt.Sprintf("blk.%d.", l)
		require(pfx+"attn_norm.weight", []uint64{Hu})
		require(pfx+"ffn_norm.weight", []uint64{Hu})
		ds41CheckMHCTrio(out, dimsByName, pfx, true, !full, Hu, mixW)
		if full {
			ds41CheckMHCTrio(out, dimsByName, pfx, false, false, Hu, mixW)
		}
		require(pfx+"ffn_gate_inp.weight", []uint64{Hu, Eu})
		require(pfx+"exp_probs_b.bias", []uint64{Eu})
		require(pfx+"ffn_gate_shexp.weight", []uint64{Hu, Iu})
		require(pfx+"ffn_up_shexp.weight", []uint64{Hu, Iu})
		require(pfx+"ffn_down_shexp.weight", []uint64{Iu, Hu})
		// Batched routed-expert blobs: the loader splits each 1->E along the OUTER expert
		// axis, so the GGUF stores [in, out, E] innermost-first ([E, out, in] model order).
		require(pfx+"ffn_gate_exps.weight", []uint64{Hu, Iu, Eu})
		require(pfx+"ffn_up_exps.weight", []uint64{Hu, Iu, Eu})
		require(pfx+"ffn_down_exps.weight", []uint64{Iu, Hu, Eu})
	}
}

// ds41CheckMHCTrio admits one hyper-connection coefficient block — the mix matrix in either
// flattened declaration (logical [24, 4H] or the artifact's stored [4H, 24] transpose; the
// reduced fixture's legacy [24, H] additionally when allowLegacy) plus its base/scale vectors —
// mirroring v41AdmitMHCNamed + v41MHCWeightLayoutNamed. Converter-dialect spellings come first,
// then the underscore/dotted aliases; all resolve onto the same forward leaf.
func ds41CheckMHCTrio(out *ManifestPreflight, dims map[string][]uint64, pfx string, attn bool, allowLegacy bool, H, mixW uint64) {
	tag, mixSpells, baseSpells, scaleSpells := "attn",
		[]string{pfx + "hc_attn_fn.weight", pfx + "hc_attn_fn", pfx + "mhc_mixes.weight", pfx + "mhc.mixes.weight", pfx + "mhc_mixes", pfx + "mhc.mixes"},
		[]string{pfx + "hc_attn_base.weight", pfx + "hc_attn_base", pfx + "mhc_base.weight", pfx + "mhc.base", pfx + "mhc_base"},
		[]string{pfx + "hc_attn_scale.weight", pfx + "hc_attn_scale", pfx + "mhc_scale.weight", pfx + "mhc.scale", pfx + "mhc_scale"}
	if !attn {
		tag = "ffn"
		mixSpells = []string{pfx + "hc_ffn_fn.weight", pfx + "hc_ffn_fn", pfx + "mhc.ffn_mixes.weight", pfx + "mhc.ffn_mixes"}
		baseSpells = []string{pfx + "hc_ffn_base.weight", pfx + "hc_ffn_base"}
		scaleSpells = []string{pfx + "hc_ffn_scale.weight", pfx + "hc_ffn_scale"}
	}
	name, got, ok := ds41FirstPresentDims(dims, mixSpells...)
	if !ok {
		out.Tensors.Missing++
		out.refuse(newmodel.RefusalTensorMissing, mixSpells[0], "the forward requests the layer's mHC %s mix matrix (%s); the header carries none of %s", tag, mhcMixWant(allowLegacy, H, mixW), mixSpells)
		return
	}
	want := [][]uint64{{mixW, 4 * H}, {4 * H, mixW}}
	if allowLegacy {
		want = append(want, []uint64{mixW, H})
	}
	if !ds41DimsInSet(got, want) {
		out.Tensors.Shape++
		out.refuse(newmodel.RefusalTensorShape, name, "dims %v admit no mHC geometry (%s)", got, mhcMixWant(allowLegacy, H, mixW))
	}
	checkVector := func(spells []string, leaf string, want []uint64) {
		vname, vgot, vok := ds41FirstPresentDims(dims, spells...)
		if !vok {
			out.Tensors.Missing++
			out.refuse(newmodel.RefusalTensorMissing, spells[0], "the forward requests the layer's mHC %s %s vector %v; the header carries none of %s", tag, leaf, want, spells)
			return
		}
		if !dimsEqual(vgot, want) {
			out.Tensors.Shape++
			out.refuse(newmodel.RefusalTensorShape, vname, "dims %v, want %v", vgot, want)
		}
	}
	checkVector(baseSpells, "base", []uint64{mixW})
	checkVector(scaleSpells, "scale", []uint64{3})
}

// mhcMixWant renders the admitted mHC mix geometries for a refusal detail.
func mhcMixWant(allowLegacy bool, H, mixW uint64) string {
	s := fmt.Sprintf("want [%d %d] logical or [%d %d] stored transpose", mixW, 4*H, 4*H, mixW)
	if allowLegacy {
		s += fmt.Sprintf(" or [%d %d] reduced legacy", mixW, H)
	}
	return s
}

// ds41FirstPresentDims returns the first spelling present in the directory.
func ds41FirstPresentDims(dims map[string][]uint64, spells ...string) (string, []uint64, bool) {
	for _, s := range spells {
		if d, ok := dims[s]; ok {
			return s, d, true
		}
	}
	return "", nil, false
}

// ds41DimsInSet reports whether got equals one of the admitted shapes.
func ds41DimsInSet(got []uint64, set [][]uint64) bool {
	for _, w := range set {
		if dimsEqual(got, w) {
			return true
		}
	}
	return false
}

// ds41ForwardFullGeometry mirrors v41ForwardGeometry's full/reduced decision off the parsed
// config: no DeepSeekV41 envelope (or an empty one) classifies by the flat HeadDim against the
// published rank; a populated envelope requires the full published geometry unless the whole
// decoder stack was coherently narrowed (the reduced test fixture). consistent=false marks the
// lone-axis drift the forward itself refuses rather than running the reduced stand-in.
func ds41ForwardFullGeometry(cfg model.Config) (full, consistent bool) {
	m := cfg.DeepSeekV41
	if m == nil || m.Attention.HeadDim == 0 {
		return cfg.HeadDim == ds41FullHeadDim, true
	}
	attn := m.Attention
	if attn.HeadDim != cfg.HeadDim {
		reducedFixture := cfg.HeadDim != ds41FullHeadDim && cfg.NumLayers < attn.NumLayers
		if !reducedFixture {
			return false, false
		}
		return cfg.HeadDim == ds41FullHeadDim, true
	}
	type axis struct{ got, want int }
	for _, a := range []axis{
		{cfg.HeadDim, ds41FullHeadDim},
		{cfg.NumLayers, attn.NumLayers},
		{cfg.HiddenSize, attn.HiddenSize},
		{cfg.NumHeads, attn.NumHeads},
		{cfg.NumKVHeads, attn.NumKVHeads},
	} {
		if a.got != a.want {
			return false, false
		}
	}
	return true, true
}
