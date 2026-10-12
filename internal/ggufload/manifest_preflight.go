package ggufload

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// ManifestPreflightSchema identifies the header-only manifest coverage report.
const ManifestPreflightSchema = "fak.model-manifest-preflight/1"

// Manifest preflight verdicts.
const (
	ManifestPreflightReady   = "READY"
	ManifestPreflightRefused = "REFUSED"
)

// Tensor coverage modes: an arch with a declared tensor schema is checked
// request-by-request; every other arch falls back to the loader's canonical map.
const (
	TensorCoverageArchSchema   = "arch-schema"
	TensorCoverageCanonicalMap = "canonical-map"
)

// Load tiers a dtype gap refuses.
const (
	TierHost                = "host"
	TierHostResidentExperts = "host-resident-experts"
)

// Kernel ops the dtype census checks.
const (
	OpHostDequant         = "host_dequant"
	OpExpertResidentGEMV  = "expert_resident_gemv"
	manifestRefusalDetail = 6
)

// ManifestPreflightInput is a parsed header (all shards merged) plus the optional
// HF config.json bytes it must agree with.
type ManifestPreflightInput struct {
	Path     string
	File     *File
	HFConfig []byte
}

// ManifestKeyCoverage counts the arch-scoped keys and how many a reader consumed.
type ManifestKeyCoverage struct {
	Total      int `json:"total"`
	ArchScoped int `json:"arch_scoped"`
	Read       int `json:"read"`
	Unread     int `json:"unread"`
}

// ManifestHFCoverage counts the key-to-HF bindings compared.
type ManifestHFCoverage struct {
	Provided   bool `json:"provided"`
	Bindings   int  `json:"bindings"`
	Matched    int  `json:"matched"`
	Mismatched int  `json:"mismatched"`
	Absent     int  `json:"absent"`
}

// ManifestTensorCoverage counts tensor requests, present tensors and the gaps.
type ManifestTensorCoverage struct {
	Mode      string `json:"mode"`
	Present   int    `json:"present"`
	Requested int    `json:"requested,omitempty"`
	Consumed  int    `json:"consumed"`
	Dropped   int    `json:"dropped"`
	Missing   int    `json:"missing"`
	Orphan    int    `json:"orphan"`
	Shape     int    `json:"shape_mismatch"`
}

// ManifestPreflight is the one-pass report: every refusal, in the shared
// fak.new-model-refusal/1 vocabulary, plus the coverage counts behind them.
type ManifestPreflight struct {
	Schema       string                 `json:"schema"`
	Path         string                 `json:"path,omitempty"`
	Verdict      string                 `json:"verdict"`
	Arch         string                 `json:"arch,omitempty"`
	ModelType    string                 `json:"model_type,omitempty"`
	Keys         ManifestKeyCoverage    `json:"keys"`
	HF           ManifestHFCoverage     `json:"hf"`
	Tensors      ManifestTensorCoverage `json:"tensors"`
	DTypes       map[string]int         `json:"dtypes"`
	RefusedTiers []string               `json:"refused_tiers,omitempty"`
	Counts       map[string]int         `json:"counts,omitempty"`
	Refusals     []newmodel.Refusal     `json:"refusals"`
}

// Refused reports whether any refusal was found.
func (p ManifestPreflight) Refused() bool { return len(p.Refusals) > 0 }

func (p *ManifestPreflight) refuse(reason newmodel.RefusalReason, axis, format string, args ...any) {
	p.Refusals = append(p.Refusals, newmodel.HeaderRefusal(reason, axis, fmt.Sprintf(format, args...)))
}

// BuildManifestPreflight checks a GGUF header in one pass, never reading a tensor
// byte: every <arch>.* key has a reader and agrees with the HF config, every
// tensor the arch requests is present at its shape, every present tensor has a
// consumer, and every tensor dtype has a host kernel. It aggregates instead of
// stopping at the first problem, so one run names every seam.
func BuildManifestPreflight(in ManifestPreflightInput) ManifestPreflight {
	out := ManifestPreflight{
		Schema:   ManifestPreflightSchema,
		Path:     strings.TrimSpace(in.Path),
		DTypes:   map[string]int{},
		Refusals: []newmodel.Refusal{},
	}
	if in.File == nil {
		out.refuse(newmodel.RefusalHeaderConfigInvalid, "header", "no parsed GGUF header")
		return out.finish()
	}
	// Trace a private copy so the caller's File is untouched by Config's side effects.
	f := *in.File
	f.DeepSeek41Engram = nil
	f.TraceKeyReads()
	out.Arch, _ = f.String("general.architecture")

	cfg, cfgErr := f.Config()
	if cfgErr != nil {
		out.refuse(newmodel.RefusalHeaderConfigInvalid, "config", "%v", cfgErr)
	}
	out.ModelType = cfg.ModelType
	f.GGMLTokenizer()
	_, _ = f.PrismHadamardMeta()
	if cfgErr == nil {
		runArchKeyReaders(&f, cfg)
	}

	if cfgErr == nil {
		// A failed Config stops reading partway, so every later key would read as unread.
		checkKeyCoverage(&out, &f)
	}
	checkHFParity(&out, &f, cfg, in.HFConfig)
	if cfgErr == nil {
		checkTensorCoverage(&out, &f, cfg)
	}
	checkDTypes(&out, &f, cfg)
	return out.finish()
}

func (p ManifestPreflight) finish() ManifestPreflight {
	sort.SliceStable(p.Refusals, func(i, j int) bool {
		a, b := p.Refusals[i], p.Refusals[j]
		if a.Reason != b.Reason {
			return a.Reason < b.Reason
		}
		return a.Axis < b.Axis
	})
	p.Counts = map[string]int{}
	for _, r := range p.Refusals {
		p.Counts[string(r.Reason)]++
	}
	sort.Strings(p.RefusedTiers)
	p.Verdict = ManifestPreflightReady
	if p.Refused() {
		p.Verdict = ManifestPreflightRefused
	}
	return p
}

// runArchKeyReaders runs the header readers an arch's forward depends on beyond
// Config, so their key reads are counted.
func runArchKeyReaders(f *File, cfg model.Config) {
	if r, ok := archKeyReaders[cfg.ModelType]; ok {
		r(f, cfg)
	}
}

// archKeyReaders holds per-arch header readers outside File.Config.
var archKeyReaders = map[string]func(*File, model.Config){}

func checkKeyCoverage(out *ManifestPreflight, f *File) {
	read := map[string]bool{}
	for _, k := range f.ReadKeys() {
		read[k] = true
	}
	prefix := out.Arch + "."
	var unread []string
	for k := range f.Metadata {
		out.Keys.Total++
		if out.Arch == "" || !strings.HasPrefix(k, prefix) {
			continue
		}
		out.Keys.ArchScoped++
		if read[k] {
			out.Keys.Read++
			continue
		}
		unread = append(unread, k)
	}
	sort.Strings(unread)
	out.Keys.Unread = len(unread)
	for _, k := range unread {
		out.refuse(newmodel.RefusalKeyUnread, k, "no fak reader consumes %s = %s", k, renderMetaValue(f.Metadata[k]))
	}
}

func renderMetaValue(v Value) string {
	if v.Type == TypeArray {
		if items, ok := v.Value.([]Value); ok {
			return fmt.Sprintf("array[%d]", len(items))
		}
	}
	s := fmt.Sprint(v.Value)
	if len(s) > 64 {
		s = s[:64] + "..."
	}
	return s
}

// hfBinding ties one GGUF key (the suffix after "<arch>.") to the HF config
// field that must carry the same value. HF names a dotted path tried under
// text_config first, then at the root. Derive computes the expected value when
// no single HF field holds it.
type hfBinding struct {
	Key    string
	HF     string
	Derive func(hf map[string]any) (any, bool)
	// MinusKey names a sibling GGUF key whose value is subtracted first: llama.cpp
	// folds the NextN/MTP blocks into block_count, HF num_hidden_layers does not.
	MinusKey string
}

// archHFBindings is the per-arch GGUF key to HF config table.
var archHFBindings = map[string][]hfBinding{
	"deepseek41": {
		{Key: "block_count", HF: "num_hidden_layers", MinusKey: "nextn_predict_layers"},
		{Key: "context_length", HF: "max_position_embeddings"},
		{Key: "embedding_length", HF: "hidden_size"},
		{Key: "attention.head_count", HF: "num_attention_heads"},
		{Key: "attention.head_count_kv", HF: "num_key_value_heads"},
		{Key: "attention.key_length", HF: "head_dim"},
		{Key: "attention.value_length", HF: "head_dim"},
		{Key: "attention.layer_norm_rms_epsilon", HF: "rms_norm_eps"},
		{Key: "attention.q_lora_rank", HF: "q_lora_rank"},
		{Key: "attention.sliding_window", HF: "sliding_window"},
		{Key: "attention.output_group_count", HF: "o_groups"},
		{Key: "attention.output_lora_rank", HF: "o_lora_rank"},
		{Key: "attention.compress_ratios", HF: "compress_ratios"},
		{Key: "attention.compress_rope_freq_base", HF: "compress_rope_theta"},
		{Key: "attention.indexer.head_count", HF: "index_n_heads"},
		{Key: "attention.indexer.key_length", HF: "index_head_dim"},
		{Key: "attention.indexer.top_k", HF: "index_topk"},
		{Key: "rope.freq_base", HF: "rope_theta"},
		{Key: "rope.dimension_count", HF: "qk_rope_head_dim"},
		{Key: "rope.scaling.type", HF: "rope_scaling.rope_type"},
		{Key: "rope.scaling.factor", HF: "rope_scaling.factor"},
		{Key: "rope.scaling.original_context_length", HF: "rope_scaling.original_max_position_embeddings"},
		{Key: "rope.scaling.yarn_beta_fast", HF: "rope_scaling.beta_fast"},
		{Key: "rope.scaling.yarn_beta_slow", HF: "rope_scaling.beta_slow"},
		{Key: "expert_count", HF: "n_routed_experts"},
		{Key: "expert_used_count", HF: "num_experts_per_tok"},
		{Key: "expert_feed_forward_length", HF: "moe_intermediate_size"},
		{Key: "expert_shared_count", HF: "n_shared_experts"},
		{Key: "expert_weights_scale", HF: "routed_scaling_factor"},
		{Key: "expert_weights_norm", HF: "norm_topk_prob"},
		{Key: "expert_gating_func", HF: "scoring_func", Derive: deriveGatingFunc},
		{Key: "swiglu_clamp_exp", HF: "swiglu_limit"},
		{Key: "swiglu_clamp_shexp", HF: "swiglu_limit"},
		{Key: "hyper_connection.count", HF: "hc_mult"},
		{Key: "hyper_connection.sinkhorn_iterations", HF: "hc_sinkhorn_iters"},
		{Key: "hyper_connection.epsilon", HF: "hc_eps"},
		{Key: "engram.layer_ids", HF: "engram_layer_ids"},
		{Key: "engram.head_count", HF: "engram_n_heads"},
		{Key: "engram.key_length", HF: "engram_head_dim"},
		{Key: "engram.max_ngram_size", HF: "engram_max_ngram_size"},
		{Key: "engram.pad_id", HF: "engram_pad_token_id"},
	},
}

// commonHFBindings is the fallback table for an arch with no declared table:
// the llama-family keys every converter writes with one fixed HF meaning.
var commonHFBindings = []hfBinding{
	{Key: "block_count", HF: "num_hidden_layers", MinusKey: "nextn_predict_layers"},
	{Key: "context_length", HF: "max_position_embeddings"},
	{Key: "embedding_length", HF: "hidden_size"},
	{Key: "feed_forward_length", HF: "intermediate_size"},
	{Key: "attention.head_count", HF: "num_attention_heads"},
	{Key: "attention.head_count_kv", HF: "num_key_value_heads"},
	{Key: "attention.layer_norm_rms_epsilon", HF: "rms_norm_eps"},
	{Key: "rope.freq_base", HF: "rope_theta"},
}

// ggmlExpertGatingFunc is llama.cpp's llama_expert_gating_func_type numbering.
var ggmlExpertGatingFunc = map[string]int{"softmax": 1, "sigmoid": 2, "softmax_weight": 3, "sqrtsoftplus": 4}

func deriveGatingFunc(hf map[string]any) (any, bool) {
	v, ok := hfLookup(hf, "scoring_func")
	if !ok {
		return nil, false
	}
	s, ok := v.(string)
	if !ok {
		return nil, false
	}
	n, ok := ggmlExpertGatingFunc[s]
	return float64(n), ok
}

func checkHFParity(out *ManifestPreflight, f *File, cfg model.Config, raw []byte) {
	if len(raw) == 0 {
		return
	}
	out.HF.Provided = true
	var hf map[string]any
	if err := json.Unmarshal(raw, &hf); err != nil {
		out.refuse(newmodel.RefusalHFMismatch, "hf_config", "HF config is not a JSON object: %v", err)
		return
	}
	bindings := archHFBindings[cfg.ModelType]
	if len(bindings) == 0 {
		bindings = archHFBindings[out.Arch]
	}
	if len(bindings) == 0 {
		bindings = commonHFBindings
	}
	for _, b := range bindings {
		key := out.Arch + "." + b.Key
		gv, ok := f.Metadata[key]
		if !ok {
			continue
		}
		if b.MinusKey != "" {
			if gv, ok = subtractMeta(gv, f.Metadata[out.Arch+"."+b.MinusKey]); !ok {
				out.refuse(newmodel.RefusalHFMismatch, key, "GGUF %s minus %s is not a non-negative integer", key, b.MinusKey)
				continue
			}
		}
		out.HF.Bindings++
		var want any
		if b.Derive != nil {
			want, ok = b.Derive(hf)
		} else {
			want, ok = hfLookup(hf, b.HF)
		}
		if !ok {
			out.HF.Absent++
			continue
		}
		if metaValueEqual(gv, want) {
			out.HF.Matched++
			continue
		}
		out.HF.Mismatched++
		out.refuse(newmodel.RefusalHFMismatch, key, "GGUF %s = %s, HF %s = %v", key, renderMetaValue(gv), b.HF, renderHF(want))
	}
}

func subtractMeta(v, minus Value) (Value, bool) {
	a, ok := valueUint64(v)
	if !ok {
		return v, false
	}
	if minus.Value == nil {
		return v, true
	}
	b, ok := valueUint64(minus)
	if !ok || b > a {
		return v, false
	}
	return Value{Type: TypeUint64, Value: a - b}, true
}

func renderHF(v any) string {
	if a, ok := v.([]any); ok {
		return fmt.Sprintf("array[%d]", len(a))
	}
	return fmt.Sprint(v)
}

func hfLookup(hf map[string]any, path string) (any, bool) {
	for _, root := range []string{"text_config", ""} {
		cur := any(hf)
		if root != "" {
			cur = hf[root]
		}
		if v, ok := hfWalk(cur, path); ok {
			return v, true
		}
	}
	return nil, false
}

func hfWalk(cur any, path string) (any, bool) {
	for _, part := range strings.Split(path, ".") {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		cur, ok = m[part]
		if !ok || cur == nil {
			return nil, false
		}
	}
	return cur, true
}

// metaValueEqual compares a GGUF value with a JSON-decoded HF value. Floats
// compare at f32 precision; a GGUF array against an HF scalar requires every
// element to equal it (a per-layer clamp written from one HF limit).
func metaValueEqual(v Value, want any) bool {
	if v.Type == TypeArray {
		items, ok := v.Value.([]Value)
		if !ok {
			return false
		}
		if wa, ok := want.([]any); ok {
			if len(wa) != len(items) {
				return false
			}
			for i := range items {
				if !metaValueEqual(items[i], wa[i]) {
					return false
				}
			}
			return true
		}
		for _, it := range items {
			if !metaValueEqual(it, want) {
				return false
			}
		}
		return len(items) > 0
	}
	switch w := want.(type) {
	case bool:
		b, ok := v.Value.(bool)
		return ok && b == w
	case string:
		s, ok := v.Value.(string)
		return ok && s == w
	case float64:
		g, ok := metaNumber(v)
		if !ok {
			return false
		}
		if g == w {
			return true
		}
		if v.Type != TypeFloat32 && v.Type != TypeFloat64 {
			return false
		}
		scale := math.Max(math.Abs(g), math.Abs(w))
		return math.Abs(g-w) <= 1e-6*scale
	}
	return false
}

func metaNumber(v Value) (float64, bool) {
	switch v.Type {
	case TypeFloat32:
		return float64(v.Value.(float32)), true
	case TypeFloat64:
		return v.Value.(float64), true
	case TypeInt8:
		return float64(v.Value.(int8)), true
	case TypeInt16:
		return float64(v.Value.(int16)), true
	case TypeInt32:
		return float64(v.Value.(int32)), true
	case TypeInt64:
		return float64(v.Value.(int64)), true
	case TypeBool:
		if v.Value.(bool) {
			return 1, true
		}
		return 0, true
	default:
		if u, ok := valueUint64(v); ok {
			return float64(u), true
		}
		return 0, false
	}
}

// TensorRequest is one tensor an arch's forward reads: its GGUF name and GGUF
// dims (innermost first, as the header stores them). Optional requests may be
// absent without a refusal.
type TensorRequest struct {
	Name     string
	Dims     []uint64
	Optional bool
}

// archTensorSchemas declares, per canonical arch, every tensor the forward
// requests. An arch with no schema falls back to the loader's canonical map.
var archTensorSchemas = map[string]func(cfg model.Config, f *File) []TensorRequest{}

func checkTensorCoverage(out *ManifestPreflight, f *File, cfg model.Config) {
	out.Tensors.Present = len(f.Tensors)
	if schema, ok := archTensorSchemas[cfg.ModelType]; ok {
		out.Tensors.Mode = TensorCoverageArchSchema
		checkTensorSchema(out, f, schema(cfg, f))
		return
	}
	out.Tensors.Mode = TensorCoverageCanonicalMap
	for _, info := range f.Tensors {
		switch tensorDisposition(cfg, info.Name) {
		case tensorConsumed:
			out.Tensors.Consumed++
		case tensorDropped:
			out.Tensors.Dropped++
		default:
			out.Tensors.Orphan++
			out.refuse(newmodel.RefusalTensorOrphan, info.Name, "no canonical mapping and no loader route; the forward never reads it")
		}
	}
	if archIsDeepSeek41(cfg.ModelType) {
		ws := &WeightSource{File: f}
		for _, d := range deepseek41AttentionDescriptorDiagnostics(ws, cfg) {
			if strings.HasPrefix(d.Reason, "missing") {
				out.Tensors.Missing++
				out.refuse(newmodel.RefusalTensorMissing, d.Canonical, "%s", d.Reason)
				continue
			}
			out.Tensors.Shape++
			out.refuse(newmodel.RefusalTensorShape, d.Canonical, "%s", d.Reason)
		}
	}
}

func checkTensorSchema(out *ManifestPreflight, f *File, reqs []TensorRequest) {
	present := make(map[string]TensorInfo, len(f.Tensors))
	for _, t := range f.Tensors {
		present[t.Name] = t
	}
	requested := make(map[string]bool, len(reqs))
	for _, r := range reqs {
		requested[r.Name] = true
		out.Tensors.Requested++
		t, ok := present[r.Name]
		if !ok {
			if !r.Optional {
				out.Tensors.Missing++
				out.refuse(newmodel.RefusalTensorMissing, r.Name, "the forward requests %s %v; the header has no such tensor", r.Name, r.Dims)
			}
			continue
		}
		if len(r.Dims) > 0 && !dimsEqual(t.Dims, r.Dims) {
			out.Tensors.Shape++
			out.refuse(newmodel.RefusalTensorShape, r.Name, "dims %v, want %v", t.Dims, r.Dims)
		}
	}
	for _, t := range f.Tensors {
		if requested[t.Name] {
			out.Tensors.Consumed++
			continue
		}
		out.Tensors.Orphan++
		out.refuse(newmodel.RefusalTensorOrphan, t.Name, "present but the %s forward never requests it", out.ModelType)
	}
}

func dimsEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type tensorRoute int

const (
	tensorUnmapped tensorRoute = iota
	tensorConsumed
	tensorDropped
)

// tensorDisposition replays the GGUF loaders' per-tensor routing (F32Tensors,
// QuantModelProfile) header-only: deliberate sidecar skips and retention drops
// are dropped, batched experts and split KV-b halves are consumed by their
// splitters, and everything else must have a canonical mapping.
func tensorDisposition(cfg model.Config, name string) tensorRoute {
	canon, handled := qwen35MTPMaterializationName(name, cfg)
	if handled {
		if canon == "" {
			return tensorDropped
		}
		return tensorConsumed
	}
	if archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(name) {
		return tensorDropped
	}
	if archUsesMLAMoELayout(cfg.ModelType) {
		if _, _, ok := glmMoeDsaSplitKVB(name); ok {
			return tensorConsumed
		}
	}
	if archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
		if _, _, ok := glmMoeDsaBatchedExpert(name); ok {
			return tensorConsumed
		}
	}
	canon, ok := CanonicalTensorNameArch(name, cfg.ModelType)
	if !ok {
		return tensorUnmapped
	}
	if _, keep := model.QuantSourceTensorName(cfg, canon); !keep {
		return tensorDropped
	}
	return tensorConsumed
}

// HostDequantSupported reports whether a host kernel decodes t to f32. It must
// agree with dequantF32IntoLimited's switch (TestHostDequantSupportedMatchesDequant).
func HostDequantSupported(t TensorType) bool {
	if _, ok := tensorTypeTraits(t); !ok {
		return false
	}
	switch t {
	case TensorF32, TensorF16, TensorBF16, TensorQ4_0, TensorQ4_1, TensorQ5_0, TensorQ5_1, TensorQ8_0,
		TensorQ2_K, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K,
		TensorIQ2_XXS, TensorIQ2_XS, TensorIQ2_S, TensorIQ3_XXS, TensorIQ3_S, TensorIQ1_S, TensorIQ1_M,
		TensorIQ4_NL, TensorIQ4_XS, TensorMXFP4, TensorQ2_0, TensorPQ2_0, TensorQ1_0, TensorHQQ4:
		return true
	default:
		return false
	}
}

// expertResidentSupported reports whether a routed-expert tensor of type t runs
// on the raw-resident host GEMV, the only host path that does not inflate every
// expert to Q8.
func expertResidentSupported(t TensorType) bool {
	if t == TensorF32 {
		return true
	}
	_, _, ok := residentExpertBlockGeometry(t)
	return ok
}

func checkDTypes(out *ManifestPreflight, f *File, cfg model.Config) {
	type gap struct{ op, dtype string }
	gaps := map[gap][]string{}
	tiers := map[string]bool{}
	for _, t := range f.Tensors {
		out.DTypes[t.Type.String()]++
		if !HostDequantSupported(t.Type) {
			g := gap{OpHostDequant, t.Type.String()}
			gaps[g] = append(gaps[g], t.Name)
			tiers[TierHost] = true
			continue
		}
		if _, _, ok := glmMoeDsaBatchedExpert(t.Name); ok && len(t.Dims) == 3 && !expertResidentSupported(t.Type) {
			g := gap{OpExpertResidentGEMV, t.Type.String()}
			gaps[g] = append(gaps[g], t.Name)
			tiers[TierHostResidentExperts] = true
		}
	}
	keys := make([]gap, 0, len(gaps))
	for g := range gaps {
		keys = append(keys, g)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].op != keys[j].op {
			return keys[i].op < keys[j].op
		}
		return keys[i].dtype < keys[j].dtype
	})
	for _, g := range keys {
		names := gaps[g]
		sort.Strings(names)
		shown := names
		if len(shown) > manifestRefusalDetail {
			shown = shown[:manifestRefusalDetail]
		}
		out.refuse(newmodel.RefusalUnsupportedDtype, g.op+"("+g.dtype+")", "%d tensors, e.g. %s", len(names), strings.Join(shown, ", "))
	}
	for t := range tiers {
		out.RefusedTiers = append(out.RefusedTiers, t)
	}
}

// Render writes a short human-readable summary.
func (p ManifestPreflight) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak model preflight: %s arch=%s keys=%d/%d read hf=%d/%d matched tensors=%s consumed=%d orphan=%d missing=%d shape=%d\n",
		p.Verdict, p.Arch, p.Keys.Read, p.Keys.ArchScoped, p.HF.Matched, p.HF.Bindings,
		p.Tensors.Mode, p.Tensors.Consumed, p.Tensors.Orphan, p.Tensors.Missing, p.Tensors.Shape)
	for _, r := range p.Refusals {
		fmt.Fprintf(&b, "  %s %s: %s\n", r.Reason, r.Axis, r.Detail)
	}
	return b.String()
}

// OpenHeaders parses only the GGUF header of path and, for a split checkpoint,
// of every sibling shard, returning shard 1's metadata with all shards' tensor
// directories merged. No tensor byte is read and no file stays open.
func OpenHeaders(path string) (*File, error) {
	first, err := Open(path)
	if err != nil {
		return nil, err
	}
	count, split := first.Uint64("split.count")
	if !split || count <= 1 {
		return first, nil
	}
	shard1 := path
	if _, hasArch := first.String("general.architecture"); !hasArch {
		if shard1, err = firstShardPath(path); err != nil {
			return nil, fmt.Errorf("gguf: %s is a split shard but its name is not a shard path: %w", path, err)
		}
	}
	paths, err := shardPaths(shard1, int(count))
	if err != nil {
		return nil, err
	}
	shards := make([]*File, len(paths))
	for i, p := range paths {
		if shards[i], err = Open(p); err != nil {
			return nil, fmt.Errorf("gguf: shard %d/%d: %w", i+1, len(paths), err)
		}
	}
	return MergeShardHeaders(shards[0], shards[1:]...)
}

// MergeShardHeaders merges split-shard headers: shard 1 keeps its metadata and
// every shard contributes its tensor directory, in shard order.
func MergeShardHeaders(first *File, rest ...*File) (*File, error) {
	if first == nil {
		return nil, fmt.Errorf("gguf: no first shard header")
	}
	if _, ok := first.String("general.architecture"); !ok {
		return nil, fmt.Errorf("gguf: first shard carries no general.architecture")
	}
	merged := *first
	merged.Tensors = append([]TensorInfo(nil), first.Tensors...)
	for _, s := range rest {
		if s == nil {
			return nil, fmt.Errorf("gguf: nil shard header")
		}
		merged.Tensors = append(merged.Tensors, s.Tensors...)
	}
	if want, ok := first.Uint64("split.tensors.count"); ok && want != uint64(len(merged.Tensors)) {
		return nil, fmt.Errorf("gguf: merged %d tensors, split.tensors.count says %d", len(merged.Tensors), want)
	}
	return &merged, nil
}
