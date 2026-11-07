package ggufload

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// kimik31_tensors.go — the Kimi-K3.1 ("kimi-k31") GGUF tensor map, the required-
// tensor-family census, and the map-level routed-expert shard census (ticket
// 1611-kimi-k31 leaf L2). It builds on the L1 admission in kimik31.go: this file
// maps every K3.1 tensor family from its GGUF spelling onto the canonical names
// the native K3.1 forward (L5–L8) will read, refuses fail-closed a kimi-k31 file
// that ships tensors but is missing a required family, and accounts the
// routed-expert band across shards at the map level. There is no serving-path
// behavior change here: execution stays gated until ForwardKimiK31 lands in
// internal/model, exactly as the deepseek41 admission slice does for V4.1.
//
// TENSOR SPELLINGS (pinned, not guessed). The community K3.1 converter
// (csabakecskemeti/llama.cpp@kimi-k31, fetched 2026-11-07) registers
// MODEL_ARCH.KIMI_K31 = "kimi-k31" with a tensor list of K3's plus the four DSA
// indexer entries, and its conversion/kimi_k3.py emits precisely the spellings
// below; the measured official-checkpoint vocabulary (fak-private L0 artifact
// census, docs/research/KIMI-K31-L0-ARTIFACT-CENSUS-2026-11-07.md) pins the
// canonical (HF-side) names the map targets:
//   - KDA layers: ssm_conv1d_{q,k,v} / ssm_f_a / ssm_f_b / ssm_beta / ssm_a /
//     ssm_dt.bias / ssm_g (the full-rank gate replacing K2's g_a/g_b pair) /
//     ssm_norm, beside attn_q/attn_k/attn_v and attn_output;
//   - Gated-DSA layers: the MLA latent projections (attn_q_a[_norm] / attn_q_b /
//     attn_kv_a_mqa / attn_kv_a_norm / attn_kv_b or the split attn_k_b+attn_v_b),
//     the sigmoid output gate attn_gate, and the learned indexer under the
//     GLM-5.2-validated "indexer." sub-namespace (attn_q_b / attn_k / proj /
//     k_norm weight+bias) — the fork reads exactly those;
//   - cross-layer residual scoring: attn_res_score / ffn_res_score per layer and
//     the global output_res_score. The converter FUSES each <x>_res_norm *
//     <x>_res_proj pair into one [n_embd] vector, so the separate _res_norm
//     tensors never appear in a K3.1 GGUF;
//   - latent MoE: ffn_routed_down / ffn_routed_up / ffn_routed_norm beside the
//     deepseek2-convention router (ffn_gate_inp / exp_probs_b), shared experts
//     (ffn_*_shexp) and batched routed slabs (ffn_*_exps);
//   - dense layer 0: plain ffn_gate / ffn_up / ffn_down.
//
// TWO EXPERT DIALECTS CONVERGE ON ONE CANONICAL LEAF SET. The community Q2_K /
// MXFP4_MOE GGUFs ship the routed band as three batched slabs per MoE layer
// (ffn_gate_exps / ffn_down_exps / ffn_up_exps, [latent|ffn, ..., E]); the
// loader's 1->E splitter (archUsesGGUFBatchedMoEExperts +
// batchedExpertCanonicalName) expands them onto
// model.layers.<L>.block_sparse_moe.experts.<e>.{w1,w2,w3}.weight with the
// converter's w1=gate / w2=down / w3=up projection order. The official
// checkpoint instead stores PER-EXPERT MXFP4-group-32 packed tensors
// (block_sparse_moe.experts.<e>.w{1,2,3}.{weight_packed,weight_scale}, 494,592
// occurrences = 896 experts x 92 MoE layers x 6); that dialect is accepted by
// this map (kimik31PerExpertMXFP4Suffix) so manifest-level routing and a future
// MXFP4-native load resolve onto the same leaves.
//
// MXFP4 FOLLOW-UP GATE (named, deliberately NOT implemented in this leaf): the
// eager dequant path already decodes TensorMXFP4 (gguf_dequant.go), but the
// RESIDENT raw-quant expert split (splitGLMMoeDsaExpertsRawQuant) knows only
// 256-weight super-block geometries; an MXFP4_MOE GGUF whose expert slabs are
// stored as TensorMXFP4 needs the raw split extended to MXFP4 block geometry
// (1-byte E8M0 scale + 16 bytes of packed 4-bit weights per 32-element group)
// before it can stream the band without an f32 round-trip. The rank-1 target
// artifact (community Q2_K, gate/up Q2_K + down Q3_K slabs) needs none of that:
// its slabs ride the existing raw-quant path unchanged.
//
// A_LOG DOMAIN NOTE: the converter folds -exp(A_log) into the stored ssm_a
// value (negative decay coefficient), the same domain qwen35's linear_attn.A_log
// carries. The map resolves the name only; undoing the fold (log(-x)) belongs to
// the K3.1 forward's load-time normalization (L5), mirroring
// TransformInvertNegExpDecay, and must not be baked into the name map.
//
// OUT OF SCOPE: the vision tower (MoonViT-V2) and mm_projector ship in a
// separate --mmproj file in the community artifacts and carry no text-forward
// slot; a combined file shipping v.*/mm.* sidecars refuses with "no canonical
// mapping", which is the correct fail-closed outcome for this text-stack leaf.

// ErrKimiK31MissingTensorFamily is the typed refusal sentinel for a kimi-k31
// file whose tensor directory is present but missing one or more required
// tensor families. Every concrete error wraps it and names both the offending
// family and the first missing tensor, so an operator sees exactly what to fix.
var ErrKimiK31MissingTensorFamily = errors.New("gguf: kimi-k31 checkpoint is missing a required tensor family")

// kimik31GlobalTensorNames is the arch-aware arm of CanonicalTensorNameArch for
// K3.1 model-global tensors the shared switch does not cover. token_embd /
// output_norm / output already resolve through the arch-blind global switch.
var kimik31GlobalTensorNames = map[string]string{
	// Fused output_attn_res_norm * output_attn_res_proj vector (see the file
	// header): the final cross-layer residual mix before the output norm.
	"output_res_score": "model.output_attn_res_proj",
}

// kimik31CanonicalSuffix maps a K3.1 per-layer GGUF tensor suffix (after
// "blk.<L>.") to the canonical HF-suffixed name (after "model.layers.<L>.") the
// native K3.1 forward will read, following the deepseek41CanonicalSuffix
// pattern: map ONLY the arch-specific families; everything generic (attn_norm,
// ffn_norm, attn_q/attn_k/attn_v on KDA layers, attn_output, the dense layer's
// ffn_gate/up/down) falls through to the shared base map, whose spellings
// already match the measured census (self_attn.q_proj / o_proj, mlp.gate_proj …).
//
// Deliberately NOT mapped here: the batched routed-expert slabs
// (ffn_gate_exps / ffn_down_exps / ffn_up_exps) — one GGUF name cannot become E
// per-expert canonical tensors, so the 1->E splitter handles them BEFORE this
// map (archUsesGGUFBatchedMoEExperts + batchedExpertCanonicalName), exactly as
// for glm_moe_dsa / deepseek2 / deepseek41.
func kimik31CanonicalSuffix(suffix string) (string, bool) {
	mapped, ok := map[string]string{
		// ---- Gated-DSA MLA latent projections (measured census names) --------
		"attn_q_a.weight":       "self_attn.q_a_proj.weight",
		"attn_q_a_norm.weight":  "self_attn.q_a_layernorm.weight",
		"attn_q_b.weight":       "self_attn.q_b_proj.weight",
		"attn_kv_a_mqa.weight":  "self_attn.kv_a_proj_with_mqa.weight",
		"attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
		// kv_b arrives EITHER combined (attn_kv_b) or pre-split into the two
		// halves the converter emits (attn_k_b transposed, attn_v_b not). Both
		// forms resolve onto distinct non-colliding leaves; the K3.1 forward
		// consumes whichever the artifact shipped (L6 owns the merge decision).
		"attn_kv_b.weight": "self_attn.kv_b_proj.weight",
		"attn_k_b.weight":  "self_attn.k_b_proj.weight",
		"attn_v_b.weight":  "self_attn.v_b_proj.weight",
		// K3's sigmoid output gate applied to the attention output before
		// o_proj. The HF side spells it g_proj like the KDA full-rank gate, but
		// the GGUF side disambiguates (attn_gate vs ssm_g), so both map cleanly.
		"attn_gate.weight": "self_attn.o_gate.weight",

		// ---- Gated-DSA learned indexer (GLM-5.2-validated sub-namespace) -----
		// The K3.1 fork loads precisely these five (kimi-k31.cpp
		// load_arch_tensors); the canonical leaves mirror the GLM-5.2 DSA
		// indexer contract so the paged top-2048 machinery L6 ports has one
		// vocabulary across both sparse-MLA models.
		"indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
		"indexer.attn_k.weight":   "self_attn.indexer.wk.weight",
		"indexer.proj.weight":     "self_attn.indexer.weights_proj.weight",
		"indexer.k_norm.weight":   "self_attn.indexer.k_norm.weight",
		"indexer.k_norm.bias":     "self_attn.indexer.k_norm.bias",

		// ---- KDA (Kimi Delta Attention) linear-attention axis ----------------
		// Three separate conv1d weights (Q/K/V thirds), the forget-gate low-rank
		// pair f_a/f_b, the beta projection b_proj, the folded A_log (ssm_a, NO
		// ".weight" suffix — see the A_LOG DOMAIN NOTE), dt_bias, the FULL-RANK
		// gate ssm_g (K3 replaces K2's factored g_a/g_b), and the per-head
		// output RMSNorm ssm_norm.
		"ssm_conv1d_q.weight": "self_attn.q_conv1d.weight",
		"ssm_conv1d_k.weight": "self_attn.k_conv1d.weight",
		"ssm_conv1d_v.weight": "self_attn.v_conv1d.weight",
		"ssm_f_a.weight":      "self_attn.f_a_proj.weight",
		"ssm_f_b.weight":      "self_attn.f_b_proj.weight",
		"ssm_beta.weight":     "self_attn.b_proj.weight",
		"ssm_a":               "self_attn.A_log",
		"ssm_dt.bias":         "self_attn.dt_bias",
		"ssm_g.weight":        "self_attn.g_proj.weight",
		"ssm_norm.weight":     "self_attn.o_norm",

		// ---- Cross-layer residual scoring (fused norm*proj vectors) ----------
		"attn_res_score": "self_attention_res_proj",
		"ffn_res_score":  "mlp_res_proj",

		// ---- Latent MoE: routed experts run in the 3584-wide latent space ----
		"ffn_routed_down.weight": "block_sparse_moe.routed_expert_down_proj.weight",
		"ffn_routed_up.weight":   "block_sparse_moe.routed_expert_up_proj.weight",
		"ffn_routed_norm.weight": "block_sparse_moe.routed_expert_norm.weight",

		// ---- MoE router + shared experts (deepseek2-convention GGUF side) ----
		// The router gate is resolved HERE, not by the shared batched-MoE branch
		// of CanonicalTensorNameArch: K3.1's forward reads the census spelling
		// block_sparse_moe.gate.weight, not the deepseek2/GLM mlp.gate.weight.
		"ffn_gate_inp.weight":   "block_sparse_moe.gate.weight",
		"exp_probs_b.bias":      "block_sparse_moe.gate.e_score_correction_bias",
		"ffn_gate_shexp.weight": "block_sparse_moe.shared_experts.gate_proj.weight",
		"ffn_up_shexp.weight":   "block_sparse_moe.shared_experts.up_proj.weight",
		"ffn_down_shexp.weight": "block_sparse_moe.shared_experts.down_proj.weight",
	}[suffix]
	return mapped, ok
}

// kimik31PerExpertMXFP4Suffix maps the OFFICIAL-CHECKPOINT per-expert dialect
// (block_sparse_moe.experts.<e>.w{1,2,3}.{weight_packed,weight_scale}) to its
// canonical leaf, reporting whether the suffix is such an entry. This is the
// second expert dialect documented in the file header: the community GGUFs ship
// the batched slabs instead, and both converge on the same
// block_sparse_moe.experts.<e>.wX leaves. The packed/scale components keep
// their explicit component names (a future MXFP4-native load reads the pair,
// never a merged .weight).
func kimik31PerExpertMXFP4Suffix(suffix string) (string, bool) {
	const prefix = "block_sparse_moe.experts."
	if !strings.HasPrefix(suffix, prefix) {
		return "", false
	}
	rest := suffix[len(prefix):]
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return "", false
	}
	e, err := strconv.Atoi(rest[:dot])
	if err != nil || e < 0 {
		return "", false
	}
	switch tail := rest[dot+1:]; tail {
	case "w1.weight_packed", "w2.weight_packed", "w3.weight_packed",
		"w1.weight_scale", "w2.weight_scale", "w3.weight_scale":
		return fmt.Sprintf("block_sparse_moe.experts.%d.%s", e, tail), true
	}
	return "", false
}

// kimiK31RequiredFamilies is the ordered required-tensor census for a kimi-k31
// file that ships a tensor directory. Families are walked in stream order
// (globals first, then per-layer in decoder order, within a layer: norms ->
// residual scoring -> attention -> indexer -> FFN) so the FIRST failure found
// is the earliest gap an operator should fix. Requiredness follows the
// converter's own TENSOR_NOT_REQUIRED flags: optional members (attn_q_a_norm,
// attn_gate, ffn_routed_norm, the shared experts, indexer.k_norm.bias) are
// mapped but never demanded. The KV-b up-projection is required as a GROUP:
// either the combined attn_kv_b or the split attn_k_b + attn_v_b pair.
type kimiK31RequiredFamilies struct {
	name  string
	names []string
}

func kimiK31LayerPrefix(l int) string { return "blk." + strconv.Itoa(l) + "." }

// validateKimiK31RequiredTensors is the L2 fail-closed guard: a kimi-k31 file
// carrying tensors must carry every arch-defining tensor family, else it is a
// mislabeled or stripped artifact and the load refuses at the header naming the
// first missing family and tensor. A header-only fixture (len(f.Tensors) == 0)
// is a valid no-op — the schedule guard in applyKimiK31Config already separates
// the two cases — so config estimation and arch-normalization goldens stay
// admitted exactly as at L1.
func validateKimiK31RequiredTensors(f *File, cfg *model.Config) error {
	if len(f.Tensors) == 0 {
		return nil
	}
	if len(cfg.LayerTypes) != cfg.NumLayers {
		return fmt.Errorf("%w: family %q, first missing tensor <unresolved per-layer KDA/DSA schedule>",
			ErrKimiK31MissingTensorFamily, "layer-schedule")
	}
	names := make(map[string]bool, len(f.Tensors))
	perExpertByLayer := make(map[int]int)
	for _, t := range f.Tensors {
		names[t.Name] = true
		if l, suffix, ok := parseGLMBlkLayerSuffix(t.Name); ok {
			if _, isEntry := kimik31PerExpertMXFP4Suffix(suffix); isEntry {
				perExpertByLayer[l]++
			}
		}
	}
	missing := func(family, name string) error {
		return fmt.Errorf("%w: family %q, first missing tensor %s",
			ErrKimiK31MissingTensorFamily, family, name)
	}
	requireAll := func(family string, want ...string) error {
		for _, w := range want {
			if !names[w] {
				return missing(family, w)
			}
		}
		return nil
	}

	// ---- Model globals ------------------------------------------------------
	if err := requireAll("embeddings", "token_embd.weight"); err != nil {
		return err
	}
	if err := requireAll("output-head", "output_norm.weight", "output.weight"); err != nil {
		return err
	}
	if err := requireAll("cross-layer-residual-scoring", "output_res_score"); err != nil {
		return err
	}

	// ---- Per-layer walk -----------------------------------------------------
	for l := 0; l < cfg.NumLayers; l++ {
		pfx := kimiK31LayerPrefix(l)
		if err := requireAll("layer-norms", pfx+"attn_norm.weight", pfx+"ffn_norm.weight"); err != nil {
			return err
		}
		if err := requireAll("residual-scoring", pfx+"attn_res_score", pfx+"ffn_res_score"); err != nil {
			return err
		}
		switch cfg.LayerTypes[l] {
		case kimiK31LayerKDA:
			if err := requireAll("kda-attention",
				pfx+"ssm_g.weight",
				pfx+"ssm_conv1d_q.weight", pfx+"ssm_conv1d_k.weight", pfx+"ssm_conv1d_v.weight",
				pfx+"ssm_f_a.weight", pfx+"ssm_f_b.weight", pfx+"ssm_beta.weight",
				pfx+"ssm_a", pfx+"ssm_dt.bias", pfx+"ssm_norm.weight",
				pfx+"attn_q.weight", pfx+"attn_k.weight", pfx+"attn_v.weight",
				pfx+"attn_output.weight",
			); err != nil {
				return err
			}
		case kimiK31LayerDSA:
			if err := requireAll("dsa-attention",
				pfx+"attn_q_a.weight", pfx+"attn_q_b.weight",
				pfx+"attn_kv_a_mqa.weight", pfx+"attn_kv_a_norm.weight",
				pfx+"attn_output.weight",
			); err != nil {
				return err
			}
			// KV-b up-projection as a group: combined OR split halves.
			if !names[pfx+"attn_kv_b.weight"] &&
				!(names[pfx+"attn_k_b.weight"] && names[pfx+"attn_v_b.weight"]) {
				return missing("dsa-attention", pfx+"attn_kv_b.weight (or the split attn_k_b + attn_v_b pair)")
			}
			if err := requireAll("dsa-indexer",
				pfx+"indexer.attn_q_b.weight", pfx+"indexer.attn_k.weight",
				pfx+"indexer.proj.weight", pfx+"indexer.k_norm.weight",
			); err != nil {
				return err
			}
		default:
			return fmt.Errorf("%w: family %q, first missing tensor <unknown layer type %q at layer %d>",
				ErrKimiK31MissingTensorFamily, "layer-schedule", cfg.LayerTypes[l], l)
		}
		if l < cfg.FirstKDenseReplace {
			if err := requireAll("dense-ffn",
				pfx+"ffn_gate.weight", pfx+"ffn_up.weight", pfx+"ffn_down.weight",
			); err != nil {
				return err
			}
			continue
		}
		if err := requireAll("moe-router", pfx+"ffn_gate_inp.weight", pfx+"exp_probs_b.bias"); err != nil {
			return err
		}
		// Routed experts: EITHER the three batched slabs (community GGUF
		// dialect) OR the full per-expert set (official-checkpoint dialect,
		// NumExperts x 3 projections). A partial band is a stripped artifact.
		hasSlabs := names[pfx+"ffn_gate_exps.weight"] &&
			names[pfx+"ffn_down_exps.weight"] && names[pfx+"ffn_up_exps.weight"]
		if !hasSlabs && perExpertByLayer[l] != cfg.NumExperts*3 {
			first := pfx + "ffn_gate_exps.weight"
			if perExpertByLayer[l] > 0 {
				first = pfx + "block_sparse_moe.experts.0.w1.weight_packed"
			}
			return missing("routed-experts", first)
		}
		if err := requireAll("latent-moe-projections",
			pfx+"ffn_routed_down.weight", pfx+"ffn_routed_up.weight",
		); err != nil {
			return err
		}
	}
	return nil
}

// KimiK31ExpertBandCensus is the map-level accounting of the routed-expert band
// across a (possibly multi-shard merged) kimi-k31 tensor directory. It answers
// the shard-routing question at the map level: how many expert-band tensor
// occurrences exist, in which layers, and which shard routes each one — the
// 96-shard-style organization of the official checkpoint (494,592 occurrences
// = 896 experts x 92 MoE layers x 6 packed/scale components) without touching a
// single payload byte.
type KimiK31ExpertBandCensus struct {
	// Layers is the number of decoder layers observed carrying an expert band.
	Layers int
	// Experts is the per-layer expert count declared by any batched slab seen
	// (0 when the directory carries only the per-expert dialect).
	Experts int
	// Occurrences is the total expert-band tensor entries routed across all
	// shards: per-expert dialect counts one entry per
	// block_sparse_moe.experts.<e>.wX.{weight_packed,weight_scale}; a batched
	// slab counts at its declared expert count (the E expert-weight slots it
	// carries).
	Occurrences int
	// PerLayer[i] is the occurrence count of decoder layer i (nil entries mark
	// layers without an expert band, e.g. the leading dense layer).
	PerLayer []int
	// PerShard[i] is the occurrence count routed to shard i.
	PerShard []int
}

// CensusKimiK31ExpertBand walks f.Tensors once, classifies every routed-expert
// band member (batched slab or per-expert entry), and tallies occurrences per
// decoder layer and per shard via shardOf (the caller supplies the per-tensor
// shard assignment, e.g. the readerFor index openWeightsSplitFromFirst builds;
// shardCount comes from the config shard's split.count metadata). Duplicate
// tensor names — which the real merge seam already refuses — are refused here
// too so the census stays sound on hand-built directories. Non-expert tensors
// are ignored, so the census composes with any surrounding directory.
func CensusKimiK31ExpertBand(f *File, numLayers, shardCount int, shardOf func(name string) int) (*KimiK31ExpertBandCensus, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("gguf: kimi-k31 expert-band census got numLayers=%d, want positive", numLayers)
	}
	if shardCount <= 0 {
		return nil, fmt.Errorf("gguf: kimi-k31 expert-band census got shardCount=%d, want positive (read split.count from the config shard)", shardCount)
	}
	out := &KimiK31ExpertBandCensus{
		PerLayer: make([]int, numLayers),
		PerShard: make([]int, shardCount),
	}
	seen := make(map[string]bool, len(f.Tensors))
	layersSeen := 0
	for _, t := range f.Tensors {
		if seen[t.Name] {
			return nil, fmt.Errorf("gguf: kimi-k31 expert-band census sees duplicate tensor %s across shards", t.Name)
		}
		seen[t.Name] = true

		var (
			layer int
			occ   int
		)
		if l, e, ok := kimik31BatchedExpertSlabDims(t); ok {
			layer, occ = l, e
			if out.Experts == 0 {
				out.Experts = e
			} else if out.Experts != e {
				return nil, fmt.Errorf("gguf: kimi-k31 expert-band tensor %s declares %d experts but the band already declares %d: the slabs disagree", t.Name, e, out.Experts)
			}
		} else if l, ok := kimik31PerExpertEntryLayer(t.Name); ok {
			layer, occ = l, 1
		} else {
			continue // not an expert-band member
		}
		if layer < 0 || layer >= numLayers {
			return nil, fmt.Errorf("gguf: kimi-k31 expert-band tensor %s names layer %d outside block_count=%d", t.Name, layer, numLayers)
		}
		s := shardOf(t.Name)
		if s < 0 || s >= shardCount {
			return nil, fmt.Errorf("gguf: kimi-k31 expert-band tensor %s routes to shard %d outside the %d-shard set", t.Name, s, shardCount)
		}
		out.Occurrences += occ
		out.PerLayer[layer] += occ
		out.PerShard[s] += occ
	}
	for _, n := range out.PerLayer {
		if n > 0 {
			layersSeen++
		}
	}
	out.Layers = layersSeen
	return out, nil
}

// kimik31BatchedExpertSlabDims reports whether t is one of the three batched
// routed-expert slabs (ffn_gate_exps / ffn_down_exps / ffn_up_exps) and, if so,
// returns its decoder layer and declared expert count (the trailing GGUF dim of
// the stored [in, out, E] shape).
func kimik31BatchedExpertSlabDims(t TensorInfo) (layer, experts int, ok bool) {
	l, _, isSlab := glmMoeDsaBatchedExpert(t.Name)
	if !isSlab || len(t.Dims) == 0 {
		return 0, 0, false
	}
	e := int(t.Dims[len(t.Dims)-1])
	if e <= 0 {
		return 0, 0, false
	}
	return l, e, true
}

// kimik31PerExpertEntryLayer reports whether t.Name is an official-checkpoint
// per-expert MXFP4 entry (block_sparse_moe.experts.<e>.wX.{weight_packed,
// weight_scale}) and, if so, returns its decoder layer.
func kimik31PerExpertEntryLayer(name string) (int, bool) {
	l, suffix, ok := parseGLMBlkLayerSuffix(name)
	if !ok {
		return 0, false
	}
	_, isEntry := kimik31PerExpertMXFP4Suffix(suffix)
	return l, isEntry
}
