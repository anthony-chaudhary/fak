package ggufload

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ArtifactQuant summarizes the tensor encodings actually present in a parsed GGUF.
// Inventory names every encoding in the header. Name identifies the quantized weight
// family independently of unquantized F32/F16/BF16 tensors. Recipe names the authoritative
// quantization mixture recipe (e.g. "UD-Q2_K_XL", "Q4_K_M") if matched, or empty if pure/unmatched.
type ArtifactQuant struct {
	Name        string
	Inventory   string
	Q4KResident bool
	Recipe      string
}

// ClassifyArtifactQuant opens a GGUF checkpoint and classifies its actual tensor inventory.
// Split checkpoints are assembled through the same OpenWeights path as the production loader,
// so encodings carried only by later shards cannot disappear from admission decisions.
func ClassifyArtifactQuant(path string) (ArtifactQuant, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return ArtifactQuant{}, err
	}
	defer ws.Close()
	return ClassifyTensorQuant(ws.File.Tensors), nil
}

// ClassifyTensorQuant classifies a parsed GGUF tensor inventory. Resident Q4_K is
// admitted only when the header actually contains Q4_K weights; ordinary F32/F16/BF16
// tensors and the other encodings supported by the Q4_K loader may coexist with them.
func ClassifyTensorQuant(tensors []TensorInfo) ArtifactQuant {
	if len(tensors) == 0 {
		return ArtifactQuant{Name: "unknown", Inventory: "unknown"}
	}
	types := make(map[TensorType]struct{})
	quantTypes := make(map[TensorType]struct{})
	for _, tensor := range tensors {
		types[tensor.Type] = struct{}{}
		switch tensor.Type {
		case TensorF32, TensorF16, TensorBF16:
		default:
			quantTypes[tensor.Type] = struct{}{}
		}
	}
	inventory := formatTensorTypes(types)
	name := formatTensorTypes(quantTypes)
	if len(quantTypes) == 0 {
		name = "unquantized"
	}
	_, hasQ4K := quantTypes[TensorQ4_K]
	recipe := ""
	if IsUDQ2KXL(tensors) {
		recipe = "UD-Q2_K_XL"
	} else if isQ4KM(quantTypes) {
		recipe = "Q4_K_M"
	}
	return ArtifactQuant{Name: name, Inventory: inventory, Q4KResident: hasQ4K, Recipe: recipe}
}

// ClassifyTargetTensorQuant classifies only the causal target model described by
// cfg. Parsed Qwen3.5-family GGUF configs already subtract nextn_predict_layers
// from NumLayers, so every tensor in the trailing decoder range belongs to the
// MTP sidecar even when its source name is an ordinary blk.N attention/MLP name.
// Known nextn and vision namespaces are also excluded unconditionally: this is a
// target-only view and therefore does not depend on the process retention flags.
//
// The input slice is never mutated. This makes the helper safe for a merged
// multi-shard inventory as well as for one config-carrying shard or a target-only
// shard subset.
func ClassifyTargetTensorQuant(cfg model.Config, tensors []TensorInfo) ArtifactQuant {
	target := make([]TensorInfo, 0, len(tensors))
	for _, tensor := range tensors {
		if targetQuantExcludesTensor(cfg, tensor.Name) {
			continue
		}
		target = append(target, tensor)
	}
	quant := ClassifyTensorQuant(target)
	if quant.Recipe == "" && isQwen35Q4KMTarget(cfg, target) {
		quant.Recipe = "Q4_K_M"
	}
	return quant
}

// isQwen35Q4KMTarget recognizes the parsed Qwen3.8 Q4_K_M target recipe's
// architecture-specific Q5_K linear-attention output band. The generic recipe
// intentionally remains unchanged: a full target+sidecar inventory or another
// architecture carrying Q5_K must not acquire this label by coincidence.
func isQwen35Q4KMTarget(cfg model.Config, tensors []TensorInfo) bool {
	if cfg.ModelType != "qwen35" && cfg.ModelType != "qwen35moe" {
		return false
	}
	quantTypes := make(map[TensorType]struct{})
	for _, tensor := range tensors {
		switch tensor.Type {
		case TensorF32, TensorF16, TensorBF16:
			continue
		case TensorQ4_K, TensorQ6_K, TensorQ8_0:
			quantTypes[tensor.Type] = struct{}{}
		case TensorQ5_K:
			if !strings.HasSuffix(tensor.Name, ".ssm_out.weight") {
				return false
			}
			quantTypes[tensor.Type] = struct{}{}
		default:
			return false
		}
	}
	_, hasQ4K := quantTypes[TensorQ4_K]
	_, hasQ5K := quantTypes[TensorQ5_K]
	_, hasQ6K := quantTypes[TensorQ6_K]
	return hasQ4K && hasQ5K && hasQ6K
}

func targetQuantExcludesTensor(cfg model.Config, name string) bool {
	if !archShipsMTPOrVisionSidecar(cfg.ModelType) {
		return false
	}
	if glmMoeDsaMTPOrVisionTensor(name) {
		return true
	}
	if (cfg.ModelType != "qwen35" && cfg.ModelType != "qwen35moe") || cfg.NumNextNPredictLayers <= 0 {
		return false
	}
	layer, _, ok := parseGLMBlkLayerSuffix(name)
	return ok && layer >= cfg.NumLayers && layer-cfg.NumLayers < cfg.NumNextNPredictLayers
}

// ErrRoutedExpertEncodingUnqualified names the refusal for a routed-expert blob whose GGUF
// encoding the host-offload loader cannot hold raw-resident. It is the typed key the admission
// planner and the serve arm selector share, so a genuinely unqualified MoE artifact refuses by a
// named reason instead of silently charging a transcoded (f32-inflated) weight or mis-selecting
// the device-resident arm.
var ErrRoutedExpertEncodingUnqualified = errors.New("gguf: routed-expert encoding not admitted for host offload")

// AdmittedRoutedExpertEncoding reports whether a GGUF encoding is one the routed-expert loader can
// hold raw-resident (residentExpertBlockGeometry returns ok) AND for which raw blobs are admitted
// to the host expert pool. It is an ENUMERATED admit-set — the loader's residentable rows
// restricted to the encodings a routed expert genuinely supports — never a wildcard, so an
// unrecognized future tag is refused rather than silently treated as device-scoped.
func AdmittedRoutedExpertEncoding(t TensorType) bool {
	switch t {
	case TensorQ2_K, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K, TensorQ8_0,
		TensorIQ3_XXS, TensorIQ2_XXS, TensorIQ2_XS, TensorIQ1_S, TensorIQ2_S,
		TensorIQ1_M, TensorIQ4_XS:
		return true
	}
	return false
}

// routedExpertResidencyEncoding is the single admission predicate the plan and the serve arm
// selector share: an encoding is host-chargeable at RAW payload bytes when the loader holds it
// raw-resident (AdmittedRoutedExpertEncoding) OR it is F32, whose dequant is the identity and so
// has no f32 inflation. Every other encoding (including F16/BF16, whose 2 B/weight would expand to
// 4 B/weight) is refused by name rather than charged at a footprint the loader would not honour.
func routedExpertResidencyEncoding(t TensorType) bool {
	return AdmittedRoutedExpertEncoding(t) || t == TensorF32
}

// routedExpertCanonicalName builds the per-expert canonical name the raw routed-expert loader
// emits (splitGLMMoeDsaExpertsRawQuant): model.layers.<layer>.mlp.experts.<e>.<proj>.weight for
// every non-deepseek41 arch, and the native-forward model.layers.<layer>.ffn.experts.<e>.w{1,3,2}.weight
// for deepseek41 (via batchedExpertCanonicalName). The classifier and the loader therefore ask
// model.ResidentKQuantEligible about the SAME name, so a packed artifact is admitted exactly when
// the loader would hold it raw — including the V4.1 Q2_K routed-expert bulk, whose residency would
// otherwise be checked against a name the native V4.1 forward never reads.
func routedExpertCanonicalName(arch string, layer int, proj string, expert int) string {
	return batchedExpertCanonicalName(arch, layer, expert, proj)
}

// RoutedExpertResidencyQualified reports whether every routed-expert tensor in a parsed GGUF is
// backed by an admitted encoding AND a canonical name the loader holds raw-resident
// (model.ResidentKQuantEligible). It classifies routed experts with the loader's own
// glmMoeDsaBatchedExpert predicate — never a substring heuristic — and returns the offending
// tensor name/type on the first unqualified entry. An artifact with NO routed expert is not
// qualified: ok=false with name=="" (the caller cannot confirm raw expert residency).
func RoutedExpertResidencyQualified(cfg model.Config, tensors []TensorInfo) (ok bool, offending TensorType, name string) {
	if !archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
		return false, 0, ""
	}
	found := false
	for _, info := range tensors {
		layer, proj, isExpert := glmMoeDsaBatchedExpert(info.Name)
		if !isExpert {
			continue
		}
		found = true
		if !routedExpertResidencyEncoding(info.Type) {
			return false, info.Type, info.Name
		}
		if info.Type != TensorF32 {
			canon := routedExpertCanonicalName(cfg.ModelType, layer, proj, 0)
			if !model.ResidentKQuantEligible(cfg, canon) {
				return false, info.Type, info.Name
			}
		}
	}
	if !found {
		return false, 0, ""
	}
	return true, 0, ""
}

// RoutedExpertEncodingRefusal returns nil when the artifact's routed experts are all admitted for
// host offload, and otherwise wraps ErrRoutedExpertEncodingUnqualified with the offending tensor
// and encoding. It is the single named-refusal seam the serve planner and arm selector share.
func RoutedExpertEncodingRefusal(cfg model.Config, tensors []TensorInfo) error {
	ok, typ, name := RoutedExpertResidencyQualified(cfg, tensors)
	if ok {
		return nil
	}
	if name == "" {
		return fmt.Errorf("%w: no routed-expert tensor in %s artifact", ErrRoutedExpertEncodingUnqualified, cfg.ModelType)
	}
	return fmt.Errorf("%w: tensor %s type %s", ErrRoutedExpertEncodingUnqualified, name, typ)
}

// AdmittedUDQ2KXLConstituents defines supported encodings for model-dependent
// Unsloth Dynamic Q2_K Extra Large (UD-Q2_K_XL) mixtures. Artifact-family
// admission does not establish resident or device-backend support.
var AdmittedUDQ2KXLConstituents = map[TensorType]bool{
	TensorQ2_K:    true,
	TensorQ3_K:    true,
	TensorIQ2_XXS: true,
	TensorIQ2_XS:  true,
	TensorIQ2_S:   true,
	TensorIQ1_S:   true,
	TensorIQ1_M:   true,
	TensorIQ3_XXS: true,
	TensorIQ3_S:   true,
	TensorIQ4_XS:  true,
	TensorQ4_K:    true,
	TensorQ5_K:    true,
	TensorQ6_K:    true,
	TensorQ8_0:    true,
	TensorQ4_0:    true,
	TensorF32:     true,
	TensorF16:     true,
	TensorBF16:    true,
}

// IsUDQ2KXL recognizes the supported UD-Q2_K_XL mixture shape; exact constituent
// choices vary by model:
//  1. Must contain TensorQ2_K.
//  2. Must contain standard attention/norm/boundary high-precision tensors (Q4_K, Q6_K, Q8_0, or F32/F16/BF16).
//  3. Must contain low-bit IQ tensors (IQ2_XXS, IQ2_XS, IQ1_S, etc.) or have predominantly Q2_K in MLP layers
//     with higher-precision attention layers.
//  4. Must be a mixture (pure Q2_K is not a dynamic mixture).
func IsUDQ2KXL(tensors []TensorInfo) bool {
	if len(tensors) == 0 {
		return false
	}
	var (
		hasQ2K      bool
		hasHighPrec bool
		hasLowBitIQ bool
		mlpTotal    int
		mlpQ2K      int
		attnNonQ2K  bool
		quantTypes  = make(map[TensorType]struct{})
	)
	for _, t := range tensors {
		switch t.Type {
		case TensorF32, TensorF16, TensorBF16:
			hasHighPrec = true
		default:
			quantTypes[t.Type] = struct{}{}
		}
		if t.Type == TensorQ2_K {
			hasQ2K = true
		}
		if isHighPrecBoundary(t.Type) {
			hasHighPrec = true
		}
		if isLowBitIQ(t.Type) {
			hasLowBitIQ = true
		}
		if isMLPTensor(t.Name) {
			mlpTotal++
			if t.Type == TensorQ2_K {
				mlpQ2K++
			}
		}
		if isAttnTensor(t.Name) && t.Type != TensorQ2_K {
			attnNonQ2K = true
		}
	}
	if !hasQ2K || !hasHighPrec {
		return false
	}
	if len(quantTypes) <= 1 {
		return false
	}
	if hasLowBitIQ {
		return true
	}
	if mlpTotal > 0 && mlpQ2K*2 >= mlpTotal && attnNonQ2K {
		return true
	}
	return false
}

// IsQ2KXL is an alias of IsUDQ2KXL for the Q2_K_XL quantization recipe.
func IsQ2KXL(tensors []TensorInfo) bool {
	return IsUDQ2KXL(tensors)
}

// ValidateUDQ2KXL verifies that all constituent tensors in a claimed UD-Q2_K_XL
// GGUF artifact are valid, non-empty, fall within the admitted constituent
// quantization types, and satisfy the supported model-dependent mixture shape.
func ValidateUDQ2KXL(tensors []TensorInfo) error {
	if len(tensors) == 0 {
		return fmt.Errorf("empty tensor inventory")
	}
	for i, t := range tensors {
		if strings.TrimSpace(t.Name) == "" {
			return fmt.Errorf("constituent tensor at index %d has empty name", i)
		}
		if len(t.Dims) == 0 {
			return fmt.Errorf("constituent tensor %q has empty dims", t.Name)
		}
		for dimIdx, d := range t.Dims {
			if d == 0 {
				return fmt.Errorf("constituent tensor %q has zero dim at index %d", t.Name, dimIdx)
			}
		}
		if !AdmittedUDQ2KXLConstituents[t.Type] {
			return fmt.Errorf("constituent tensor %q has unadmitted type %v (%d)", t.Name, t.Type, t.Type)
		}
	}
	var (
		hasQ2K      bool
		hasHighPrec bool
		quantTypes  = make(map[TensorType]struct{})
	)
	for _, t := range tensors {
		switch t.Type {
		case TensorF32, TensorF16, TensorBF16:
			hasHighPrec = true
		default:
			quantTypes[t.Type] = struct{}{}
		}
		if t.Type == TensorQ2_K {
			hasQ2K = true
		}
		if isHighPrecBoundary(t.Type) {
			hasHighPrec = true
		}
	}
	if !hasQ2K {
		return fmt.Errorf("claimed UD-Q2_K_XL missing required TensorQ2_K")
	}
	if !hasHighPrec {
		return fmt.Errorf("claimed UD-Q2_K_XL missing standard high-precision tensors (Q4_K, Q6_K, Q8_0, or F32/F16)")
	}
	if len(quantTypes) <= 1 {
		return fmt.Errorf("claimed UD-Q2_K_XL is pure Q2_K, not a mixed dynamic recipe")
	}
	if !IsUDQ2KXL(tensors) {
		return fmt.Errorf("tensor inventory does not satisfy UD-Q2_K_XL recipe specification (missing low-bit IQ or Q2_K MLP)")
	}
	return nil
}

func isHighPrecBoundary(t TensorType) bool {
	switch t {
	case TensorQ4_K, TensorQ5_K, TensorQ6_K, TensorQ8_0, TensorF32, TensorF16, TensorBF16:
		return true
	default:
		return false
	}
}

func isLowBitIQ(t TensorType) bool {
	switch t {
	case TensorIQ1_S, TensorIQ1_M, TensorIQ2_XXS, TensorIQ2_XS, TensorIQ2_S, TensorIQ3_XXS, TensorIQ4_XS, TensorIQ3_S, TensorIQ4_NL:
		return true
	default:
		return false
	}
}

func isMLPTensor(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "ffn") ||
		strings.Contains(lower, "mlp") ||
		strings.Contains(lower, "feed_forward") ||
		strings.Contains(lower, "gate_proj") ||
		strings.Contains(lower, "up_proj") ||
		strings.Contains(lower, "down_proj")
}

func isAttnTensor(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "attn") ||
		strings.Contains(lower, "attention") ||
		strings.Contains(lower, "q_proj") ||
		strings.Contains(lower, "k_proj") ||
		strings.Contains(lower, "v_proj") ||
		strings.Contains(lower, "o_proj") ||
		strings.Contains(lower, "qkv")
}

func isQ4KM(quantTypes map[TensorType]struct{}) bool {
	if len(quantTypes) == 0 {
		return false
	}
	if _, hasQ4K := quantTypes[TensorQ4_K]; !hasQ4K {
		return false
	}
	if _, hasQ6K := quantTypes[TensorQ6_K]; !hasQ6K {
		return false
	}
	for typ := range quantTypes {
		if typ != TensorQ4_K && typ != TensorQ6_K && typ != TensorQ8_0 {
			return false
		}
	}
	return true
}

func formatTensorTypes(types map[TensorType]struct{}) string {
	if len(types) == 0 {
		return "unknown"
	}
	names := make([]string, 0, len(types))
	for typ := range types {
		names = append(names, typ.String())
	}
	sort.Strings(names)
	if len(names) == 1 {
		return names[0]
	}
	return fmt.Sprintf("mixed(%s)", strings.Join(names, "+"))
}

// Output-head quantization classification, module sensitivity breakdown, weight-byte
// floor sweep, and kernel-path confirmation below are a clean-room adaptation of the
// weight-byte floor methodology from wandb/senpai (Apache-2.0, pinned at

// OutputHeadQuant summarizes the active LM output-head tensor encoding and footprint.
type OutputHeadQuant struct {
	TensorName    string
	Tied          bool
	Type          TensorType
	TypeName      string
	IsQuantized   bool
	PayloadBytes  uint64
	Elements      uint64
	BitsPerWeight float64
}

// AnalyzeOutputHeadQuant resolves the active LM head tensor ("output.weight" /
// "lm_head.weight", or "token_embd.weight" when tied per cfg.TieWordEmbeddings)
// and reports its encoding, quantized status, payload bytes, element count, and
// effective bits per weight. Fails closed on empty/malformed inventories or
// missing/conflicting head tensors.
func AnalyzeOutputHeadQuant(cfg model.Config, tensors []TensorInfo) (OutputHeadQuant, error) {
	headInfo, tied, err := resolveOutputHeadTensor(cfg, tensors)
	if err != nil {
		return OutputHeadQuant{}, err
	}
	elems, err := tensorElems(headInfo)
	if err != nil {
		return OutputHeadQuant{}, err
	}
	payload, err := tensorPayloadBytes(headInfo)
	if err != nil {
		return OutputHeadQuant{}, err
	}
	traits, ok := tensorTypeTraits(headInfo.Type)
	if !ok {
		return OutputHeadQuant{}, fmt.Errorf("gguf: tensor %s type %s: %w", headInfo.Name, headInfo.Type, ErrNoSimplePayload)
	}
	return OutputHeadQuant{
		TensorName:    headInfo.Name,
		Tied:          tied,
		Type:          headInfo.Type,
		TypeName:      headInfo.Type.String(),
		IsQuantized:   traits.IsQuantized,
		PayloadBytes:  payload,
		Elements:      elems,
		BitsPerWeight: float64(payload) * 8.0 / float64(elems),
	}, nil
}

func resolveOutputHeadTensor(cfg model.Config, tensors []TensorInfo) (TensorInfo, bool, error) {
	if len(tensors) == 0 {
		return TensorInfo{}, false, fmt.Errorf("gguf: empty tensor inventory")
	}
	var (
		explicitHead *TensorInfo
		embedTensor  *TensorInfo
	)
	for i := range tensors {
		t := &tensors[i]
		name := strings.TrimSpace(t.Name)
		if name == "" {
			return TensorInfo{}, false, fmt.Errorf("gguf: tensor at index %d has empty name", i)
		}
		switch name {
		case "output.weight", "lm_head.weight":
			if explicitHead != nil {
				return TensorInfo{}, false, fmt.Errorf("gguf: duplicate output head tensor (%s and %s)", explicitHead.Name, t.Name)
			}
			explicitHead = t
		case "token_embd.weight", "model.embed_tokens.weight":
			if embedTensor != nil {
				return TensorInfo{}, false, fmt.Errorf("gguf: duplicate embedding tensor (%s and %s)", embedTensor.Name, t.Name)
			}
			// Prefer GGUF canonical token_embd.weight if somehow ordered, though duplicates fail above.
			embedTensor = t
		}
	}
	if cfg.TieWordEmbeddings {
		if explicitHead != nil {
			return TensorInfo{}, false, fmt.Errorf("gguf: cfg.TieWordEmbeddings conflicts with explicit output head tensor %q", explicitHead.Name)
		}
		if embedTensor == nil {
			return TensorInfo{}, false, fmt.Errorf("gguf: tied output head requires token_embd.weight")
		}
		return *embedTensor, true, nil
	}
	if explicitHead == nil {
		return TensorInfo{}, false, fmt.Errorf("gguf: untied output head tensor (output.weight or lm_head.weight) not found")
	}
	return *explicitHead, false, nil
}

// ModuleSensitivityClass groups model tensors by architectural sensitivity to quantization
// so early output-head quantization leverage vs inner-layer leverage can be measured.
type ModuleSensitivityClass string

const (
	SensitivityOutputHead   ModuleSensitivityClass = "output_head"
	SensitivityEmbedding    ModuleSensitivityClass = "embedding"
	SensitivityAttention    ModuleSensitivityClass = "attention"
	SensitivityMLP          ModuleSensitivityClass = "mlp"
	SensitivityRoutedExpert ModuleSensitivityClass = "routed_expert"
	SensitivityNorm         ModuleSensitivityClass = "norm"
	SensitivityOther        ModuleSensitivityClass = "other"
)

var sensitivityClassOrder = []ModuleSensitivityClass{
	SensitivityOutputHead,
	SensitivityEmbedding,
	SensitivityAttention,
	SensitivityMLP,
	SensitivityRoutedExpert,
	SensitivityNorm,
	SensitivityOther,
}

// ModuleSensitivityStats aggregates element count, payload bytes, effective bits per weight,
// and unquantized share for one ModuleSensitivityClass.
type ModuleSensitivityStats struct {
	Class                       ModuleSensitivityClass
	TensorCount                 int
	Elements                    uint64
	PayloadBytes                uint64
	UnquantizedElements         uint64
	UnquantizedBytes            uint64
	EffectiveBPW                float64
	PayloadShare                float64
	UnquantizedShare            float64
	UnquantizedByteShareOfTotal float64
}

// ModuleSensitivityReport holds the per-class breakdown and model-wide totals across a tensor inventory.
type ModuleSensitivityReport struct {
	TotalTensors          int
	TotalElements         uint64
	TotalPayloadBytes     uint64
	TotalUnquantizedBytes uint64
	EffectiveBPW          float64
	UnquantizedShare      float64
	Classes               map[ModuleSensitivityClass]ModuleSensitivityStats
	Ordered               []ModuleSensitivityStats
}

// Class returns the stats for c, or a zero-value stats entry with Class set if c is absent.
func (r ModuleSensitivityReport) Class(c ModuleSensitivityClass) ModuleSensitivityStats {
	if s, ok := r.Classes[c]; ok {
		return s
	}
	return ModuleSensitivityStats{Class: c}
}

// ClassifyModuleSensitivity classifies tensors across module sensitivity classes
// (output_head, embedding, attention, mlp, routed_expert, norm, other) and aggregates
// element counts, payload bytes, effective bits per weight, and unquantized shares.
func ClassifyModuleSensitivity(cfg model.Config, tensors []TensorInfo) (ModuleSensitivityReport, error) {
	if len(tensors) == 0 {
		return ModuleSensitivityReport{}, fmt.Errorf("gguf: empty tensor inventory")
	}
	byClass := make(map[ModuleSensitivityClass]*ModuleSensitivityStats, len(sensitivityClassOrder))
	var (
		totalElems            uint64
		totalPayload          uint64
		totalUnquantizedBytes uint64
	)
	for i, t := range tensors {
		name := strings.TrimSpace(t.Name)
		if name == "" {
			return ModuleSensitivityReport{}, fmt.Errorf("gguf: tensor at index %d has empty name", i)
		}
		if (name == "output.weight" || name == "lm_head.weight") && cfg.TieWordEmbeddings {
			return ModuleSensitivityReport{}, fmt.Errorf("gguf: cfg.TieWordEmbeddings conflicts with explicit output head tensor %q", name)
		}
		elems, err := tensorElems(t)
		if err != nil {
			return ModuleSensitivityReport{}, err
		}
		payload, err := tensorPayloadBytes(t)
		if err != nil {
			return ModuleSensitivityReport{}, err
		}
		traits, ok := tensorTypeTraits(t.Type)
		if !ok {
			return ModuleSensitivityReport{}, fmt.Errorf("gguf: tensor %s type %s: %w", t.Name, t.Type, ErrNoSimplePayload)
		}
		if totalElems > math.MaxUint64-elems || totalPayload > math.MaxUint64-payload {
			return ModuleSensitivityReport{}, fmt.Errorf("gguf: sensitivity totals overflow uint64 at tensor %s", t.Name)
		}
		totalElems += elems
		totalPayload += payload

		class := classifyModuleSensitivityTensor(cfg, name)
		st := byClass[class]
		if st == nil {
			st = &ModuleSensitivityStats{Class: class}
			byClass[class] = st
		}
		st.TensorCount++
		st.Elements += elems
		st.PayloadBytes += payload
		if !traits.IsQuantized {
			st.UnquantizedElements += elems
			st.UnquantizedBytes += payload
			totalUnquantizedBytes += payload
		}
	}

	classes := make(map[ModuleSensitivityClass]ModuleSensitivityStats, len(byClass))
	ordered := make([]ModuleSensitivityStats, 0, len(byClass))
	for _, c := range sensitivityClassOrder {
		st := byClass[c]
		if st == nil {
			continue
		}
		if st.Elements > 0 {
			st.EffectiveBPW = float64(st.PayloadBytes) * 8.0 / float64(st.Elements)
		}
		if totalPayload > 0 {
			st.PayloadShare = float64(st.PayloadBytes) / float64(totalPayload)
		}
		if st.PayloadBytes > 0 {
			st.UnquantizedShare = float64(st.UnquantizedBytes) / float64(st.PayloadBytes)
		}
		if totalUnquantizedBytes > 0 {
			st.UnquantizedByteShareOfTotal = float64(st.UnquantizedBytes) / float64(totalUnquantizedBytes)
		}
		classes[c] = *st
		ordered = append(ordered, *st)
	}

	report := ModuleSensitivityReport{
		TotalTensors:          len(tensors),
		TotalElements:         totalElems,
		TotalPayloadBytes:     totalPayload,
		TotalUnquantizedBytes: totalUnquantizedBytes,
		Classes:               classes,
		Ordered:               ordered,
	}
	if totalElems > 0 {
		report.EffectiveBPW = float64(totalPayload) * 8.0 / float64(totalElems)
	}
	if totalPayload > 0 {
		report.UnquantizedShare = float64(totalUnquantizedBytes) / float64(totalPayload)
	}
	return report, nil
}

func classifyModuleSensitivityTensor(cfg model.Config, name string) ModuleSensitivityClass {
	switch name {
	case "output.weight", "lm_head.weight":
		return SensitivityOutputHead
	case "token_embd.weight", "model.embed_tokens.weight":
		if cfg.TieWordEmbeddings {
			return SensitivityOutputHead
		}
		return SensitivityEmbedding
	}
	// Check norm before routed_expert/attention/mlp because names like
	// blk.0.attn_norm.weight and blk.0.ffn_norm.weight contain "attn"/"ffn".
	if isSensitivityNormTensor(name) {
		return SensitivityNorm
	}
	// Check routed experts before MLP because batched GGUF expert names like
	// blk.0.ffn_gate_exps.weight match isMLPTensor's "ffn" substring.
	if isSensitivityRoutedExpertTensor(name) {
		return SensitivityRoutedExpert
	}
	if isSensitivityAttentionTensor(name) {
		return SensitivityAttention
	}
	if isSensitivityMLPTensor(name) {
		return SensitivityMLP
	}
	if strings.Contains(strings.ToLower(name), "engram_embd") {
		return SensitivityEmbedding
	}
	return SensitivityOther
}

func isSensitivityNormTensor(name string) bool {
	lower := strings.ToLower(name)
	return lower == "output_norm.weight" ||
		lower == "model.norm.weight" ||
		strings.HasSuffix(lower, "_norm.weight") ||
		strings.HasSuffix(lower, "_norm.bias") ||
		strings.HasSuffix(lower, ".norm.weight") ||
		strings.HasSuffix(lower, ".norm.bias") ||
		strings.HasSuffix(lower, ".enorm.weight") ||
		strings.HasSuffix(lower, ".hnorm.weight") ||
		strings.Contains(lower, "layernorm") ||
		strings.Contains(lower, "rmsnorm") ||
		strings.Contains(lower, "rms_norm")
}

func isSensitivityRoutedExpertTensor(name string) bool {
	if _, _, ok := glmMoeDsaBatchedExpert(name); ok {
		return true
	}
	lower := strings.ToLower(name)
	return strings.Contains(lower, ".mlp.experts.") ||
		strings.Contains(lower, ".ffn.experts.") ||
		strings.Contains(lower, ".block_sparse_moe.experts.") ||
		strings.HasSuffix(lower, "_exps.weight") ||
		strings.Contains(lower, "_exps.")
}

func isSensitivityAttentionTensor(name string) bool {
	if isAttnTensor(name) {
		return true
	}
	lower := strings.ToLower(name)
	return strings.Contains(lower, "linear_attn") ||
		strings.Contains(lower, ".ssm_") ||
		strings.Contains(lower, ".indexer.")
}

func isSensitivityMLPTensor(name string) bool {
	if isMLPTensor(name) {
		return true
	}
	return strings.Contains(strings.ToLower(name), "block_sparse_moe")
}

// WeightByteFloorSpec configures a weight-byte floor and INT4/INT8 group-size sweep.
type WeightByteFloorSpec struct {
	// BitWidths selects QA weight bit widths to sweep (valid: 4 and 8).
	// Nil defaults to []int{4, 8}; an explicitly empty slice fails closed.
	BitWidths []int
	// GroupSizes selects per-group quantization sizes (powers of two >= 8, e.g. 32, 64, 128).
	// Nil defaults to []int{32, 64, 128}; an explicitly empty slice fails closed.
	GroupSizes []int
	// HeadBits sets the QA bit width for the standalone HeadQuantLever summary (4 or 8; 0 defaults to 4).
	HeadBits int
	// HeadGroupSize sets the group size for HeadQuantLever (0 defaults to the largest swept GroupSize).
	HeadGroupSize int
	// ScaleBytesPerGroup sets per-group scale/metadata bytes (0 defaults to 2 for an FP16 scale).
	ScaleBytesPerGroup int
}

// DefaultWeightByteFloorSpec returns the standard INT4/INT8 group-size sweep (32, 64, 128)
// with an INT4 g128 output-head lever and 2-byte FP16 per-group scales.
func DefaultWeightByteFloorSpec() WeightByteFloorSpec {
	return WeightByteFloorSpec{
		BitWidths:          []int{4, 8},
		GroupSizes:         []int{32, 64, 128},
		HeadBits:           4,
		HeadGroupSize:      128,
		ScaleBytesPerGroup: 2,
	}
}

// HeadQuantLever models the explicit early-lever bytes/token reduction from quantizing
// the active output head independently of inner-layer weights.
type HeadQuantLever struct {
	Head                       OutputHeadQuant
	TargetBits                 int
	TargetGroupSize            int
	ScaleBytesPerGroup         int
	BaselineHeadBytes          uint64
	QuantizedHeadBytes         uint64
	SavedBytes                 uint64
	BaselineBytesPerToken      uint64
	WithHeadQuantBytesPerToken uint64
	HeadShareOfBaseline        float64
	TokenByteReductionFrac     float64
}

// WeightByteFloorPoint captures one (Bits, GroupSize) point on the QA weight-byte floor sweep.
type WeightByteFloorPoint struct {
	Bits                       int
	GroupSize                  int
	ScaleBytesPerGroup         int
	DenseQuantizedBytes        uint64
	HeadBaselineBytes          uint64
	HeadQuantizedBytes         uint64
	HeadSavedBytes             uint64
	BytesPerTokenWithBaseHead  uint64
	BytesPerTokenWithQuantHead uint64
	DenseEffectiveBPW          float64
	HeadEffectiveBPW           float64
	TotalEffectiveBPW          float64
	HeadShareWithoutHeadQuant  float64
	SpeedupFloorUpperBound     float64
}

// WeightByteFloorReport holds header-grounded baseline bytes/token, module sensitivity breakdown,
// the explicit early output-head quantization lever, and the INT4/INT8 group-size sweep points.
type WeightByteFloorReport struct {
	Head                     OutputHeadQuant
	Sensitivity              ModuleSensitivityReport
	BaselineDenseBytes       uint64
	BaselineHeadBytes        uint64
	BaselineEmbedGatherBytes uint64
	BaselineBytesPerToken    uint64
	ActiveSweptElements      uint64
	HeadLever                HeadQuantLever
	Points                   []WeightByteFloorPoint
}

// PointsForBits returns all sweep points matching bits in ascending group-size order.
func (r WeightByteFloorReport) PointsForBits(bits int) []WeightByteFloorPoint {
	var out []WeightByteFloorPoint
	for _, p := range r.Points {
		if p.Bits == bits {
			out = append(out, p)
		}
	}
	return out
}

// SweepWeightByteFloor computes header-grounded bytes/token for swept dense weights plus the
// active output head, isolates output-head quantization as an explicit early lever (HeadQuantLever),
// and sweeps INT4/INT8 QA weight quantization across the configured group sizes.
// Fails closed on empty/malformed tensors, invalid bit widths or group sizes, or invalid MoE routing config.
func SweepWeightByteFloor(cfg model.Config, tensors []TensorInfo, spec WeightByteFloorSpec) (WeightByteFloorReport, error) {
	head, err := AnalyzeOutputHeadQuant(cfg, tensors)
	if err != nil {
		return WeightByteFloorReport{}, err
	}
	sens, err := ClassifyModuleSensitivity(cfg, tensors)
	if err != nil {
		return WeightByteFloorReport{}, err
	}
	bitsList, groupList, headBits, headGroup, scaleBytes, err := normalizeWeightByteFloorSpec(spec)
	if err != nil {
		return WeightByteFloorReport{}, err
	}

	routedStats := sens.Class(SensitivityRoutedExpert)
	if routedStats.Elements > 0 && cfg.NumExperts > 0 {
		if cfg.NumExpertsPerTok <= 0 || cfg.NumExpertsPerTok > cfg.NumExperts {
			return WeightByteFloorReport{}, fmt.Errorf("gguf: invalid expert_used_count %d for expert_count %d", cfg.NumExpertsPerTok, cfg.NumExperts)
		}
	}

	var (
		denseQuantElems  []uint64
		baseDenseBytes   uint64
		activeDenseElems uint64
		baseEmbedGather  uint64
		fixedDenseBytes  uint64
	)
	for _, t := range tensors {
		name := strings.TrimSpace(t.Name)
		elems, err := tensorElems(t)
		if err != nil {
			return WeightByteFloorReport{}, err
		}
		payload, err := tensorPayloadBytes(t)
		if err != nil {
			return WeightByteFloorReport{}, err
		}
		class := classifyModuleSensitivityTensor(cfg, name)
		if class == SensitivityOutputHead {
			continue
		}
		if class == SensitivityEmbedding {
			if baseEmbedGather > math.MaxUint64-payload {
				return WeightByteFloorReport{}, fmt.Errorf("gguf: embedding gather bytes overflow uint64")
			}
			baseEmbedGather += payload
			continue
		}

		activeElems := elems
		activePayload := payload
		if class == SensitivityRoutedExpert && cfg.NumExperts > 0 {
			activeElems = activeMoEValue(elems, cfg.NumExpertsPerTok, cfg.NumExperts)
			activePayload = activeMoEValue(payload, cfg.NumExpertsPerTok, cfg.NumExperts)
		}
		if baseDenseBytes > math.MaxUint64-activePayload || activeDenseElems > math.MaxUint64-activeElems {
			return WeightByteFloorReport{}, fmt.Errorf("gguf: dense active totals overflow uint64")
		}
		baseDenseBytes += activePayload
		activeDenseElems += activeElems

		quantizable := (class == SensitivityAttention || class == SensitivityMLP || class == SensitivityRoutedExpert) &&
			!strings.HasSuffix(strings.ToLower(name), ".bias")
		if !quantizable {
			if fixedDenseBytes > math.MaxUint64-activePayload {
				return WeightByteFloorReport{}, fmt.Errorf("gguf: fixed dense bytes overflow uint64")
			}
			fixedDenseBytes += activePayload
			continue
		}
		denseQuantElems = append(denseQuantElems, activeElems)
	}

	if baseDenseBytes > math.MaxUint64-head.PayloadBytes || activeDenseElems > math.MaxUint64-head.Elements {
		return WeightByteFloorReport{}, fmt.Errorf("gguf: baseline bytes/token overflow uint64")
	}
	baseBytesPerToken := baseDenseBytes + head.PayloadBytes
	totalActiveElems := activeDenseElems + head.Elements

	leverHeadQuantBytes, err := quantizedWeightBytes(head.Elements, headBits, headGroup, scaleBytes)
	if err != nil {
		return WeightByteFloorReport{}, err
	}
	var leverSaved uint64
	if head.PayloadBytes > leverHeadQuantBytes {
		leverSaved = head.PayloadBytes - leverHeadQuantBytes
	}
	lever := HeadQuantLever{
		Head:                       head,
		TargetBits:                 headBits,
		TargetGroupSize:            headGroup,
		ScaleBytesPerGroup:         scaleBytes,
		BaselineHeadBytes:          head.PayloadBytes,
		QuantizedHeadBytes:         leverHeadQuantBytes,
		SavedBytes:                 leverSaved,
		BaselineBytesPerToken:      baseBytesPerToken,
		WithHeadQuantBytesPerToken: baseBytesPerToken - leverSaved,
	}
	if baseBytesPerToken > 0 {
		lever.HeadShareOfBaseline = float64(head.PayloadBytes) / float64(baseBytesPerToken)
		lever.TokenByteReductionFrac = float64(leverSaved) / float64(baseBytesPerToken)
	}

	points := make([]WeightByteFloorPoint, 0, len(bitsList)*len(groupList))
	for _, bits := range bitsList {
		for _, groupSize := range groupList {
			denseQuantBytes := fixedDenseBytes
			for _, elems := range denseQuantElems {
				qb, err := quantizedWeightBytes(elems, bits, groupSize, scaleBytes)
				if err != nil {
					return WeightByteFloorReport{}, err
				}
				if denseQuantBytes > math.MaxUint64-qb {
					return WeightByteFloorReport{}, fmt.Errorf("gguf: quantized dense bytes overflow uint64")
				}
				denseQuantBytes += qb
			}
			headQuantBytes, err := quantizedWeightBytes(head.Elements, bits, groupSize, scaleBytes)
			if err != nil {
				return WeightByteFloorReport{}, err
			}
			if denseQuantBytes > math.MaxUint64-head.PayloadBytes || denseQuantBytes > math.MaxUint64-headQuantBytes {
				return WeightByteFloorReport{}, fmt.Errorf("gguf: sweep bytes/token overflow uint64")
			}
			withBaseHead := denseQuantBytes + head.PayloadBytes
			withQuantHead := denseQuantBytes + headQuantBytes
			var headSaved uint64
			if head.PayloadBytes > headQuantBytes {
				headSaved = head.PayloadBytes - headQuantBytes
			}
			pt := WeightByteFloorPoint{
				Bits:                       bits,
				GroupSize:                  groupSize,
				ScaleBytesPerGroup:         scaleBytes,
				DenseQuantizedBytes:        denseQuantBytes,
				HeadBaselineBytes:          head.PayloadBytes,
				HeadQuantizedBytes:         headQuantBytes,
				HeadSavedBytes:             headSaved,
				BytesPerTokenWithBaseHead:  withBaseHead,
				BytesPerTokenWithQuantHead: withQuantHead,
			}
			if activeDenseElems > 0 {
				pt.DenseEffectiveBPW = float64(denseQuantBytes) * 8.0 / float64(activeDenseElems)
			}
			if head.Elements > 0 {
				pt.HeadEffectiveBPW = float64(headQuantBytes) * 8.0 / float64(head.Elements)
			}
			if totalActiveElems > 0 {
				pt.TotalEffectiveBPW = float64(withQuantHead) * 8.0 / float64(totalActiveElems)
			}
			if withBaseHead > 0 {
				pt.HeadShareWithoutHeadQuant = float64(head.PayloadBytes) / float64(withBaseHead)
			}
			if withQuantHead > 0 {
				pt.SpeedupFloorUpperBound = float64(baseBytesPerToken) / float64(withQuantHead)
			}
			points = append(points, pt)
		}
	}

	return WeightByteFloorReport{
		Head:                     head,
		Sensitivity:              sens,
		BaselineDenseBytes:       baseDenseBytes,
		BaselineHeadBytes:        head.PayloadBytes,
		BaselineEmbedGatherBytes: baseEmbedGather,
		BaselineBytesPerToken:    baseBytesPerToken,
		ActiveSweptElements:      totalActiveElems,
		HeadLever:                lever,
		Points:                   points,
	}, nil
}

func normalizeWeightByteFloorSpec(spec WeightByteFloorSpec) (bitsList, groupList []int, headBits, headGroup, scaleBytes int, err error) {
	if spec.BitWidths != nil && len(spec.BitWidths) == 0 {
		return nil, nil, 0, 0, 0, fmt.Errorf("gguf: bit-width sweep list is empty")
	}
	if spec.BitWidths == nil {
		bitsList = []int{4, 8}
	} else {
		seen := make(map[int]bool, len(spec.BitWidths))
		for _, b := range spec.BitWidths {
			if b != 4 && b != 8 {
				return nil, nil, 0, 0, 0, fmt.Errorf("gguf: invalid QA bit width %d (want 4 or 8)", b)
			}
			if seen[b] {
				return nil, nil, 0, 0, 0, fmt.Errorf("gguf: duplicate QA bit width %d", b)
			}
			seen[b] = true
			bitsList = append(bitsList, b)
		}
		sort.Ints(bitsList)
	}

	if spec.GroupSizes != nil && len(spec.GroupSizes) == 0 {
		return nil, nil, 0, 0, 0, fmt.Errorf("gguf: group-size sweep list is empty")
	}
	if spec.GroupSizes == nil {
		groupList = []int{32, 64, 128}
	} else {
		seen := make(map[int]bool, len(spec.GroupSizes))
		for _, g := range spec.GroupSizes {
			if !isValidQuantGroupSize(g) {
				return nil, nil, 0, 0, 0, fmt.Errorf("gguf: invalid quantization group size %d (want power of two >= 8)", g)
			}
			if seen[g] {
				return nil, nil, 0, 0, 0, fmt.Errorf("gguf: duplicate quantization group size %d", g)
			}
			seen[g] = true
			groupList = append(groupList, g)
		}
		sort.Ints(groupList)
	}

	scaleBytes = spec.ScaleBytesPerGroup
	if scaleBytes < 0 {
		return nil, nil, 0, 0, 0, fmt.Errorf("gguf: negative scale bytes per group %d", scaleBytes)
	}
	if scaleBytes == 0 {
		scaleBytes = 2
	}

	headBits = spec.HeadBits
	if headBits == 0 {
		headBits = 4
	} else if headBits != 4 && headBits != 8 {
		return nil, nil, 0, 0, 0, fmt.Errorf("gguf: invalid head QA bit width %d (want 4 or 8)", headBits)
	}

	headGroup = spec.HeadGroupSize
	if headGroup == 0 {
		headGroup = groupList[len(groupList)-1]
	} else if !isValidQuantGroupSize(headGroup) {
		return nil, nil, 0, 0, 0, fmt.Errorf("gguf: invalid head group size %d (want power of two >= 8)", headGroup)
	}

	return bitsList, groupList, headBits, headGroup, scaleBytes, nil
}

func isValidQuantGroupSize(g int) bool {
	return g >= 8 && (g&(g-1)) == 0
}

func activeMoEValue(total uint64, used, count int) uint64 {
	if count <= 0 || used <= 0 {
		return 0
	}
	q, r := total/uint64(count), total%uint64(count)
	return q*uint64(used) + (r*uint64(used))/uint64(count)
}

func quantizedWeightBytes(elems uint64, bits, groupSize, scaleBytesPerGroup int) (uint64, error) {
	if elems == 0 {
		return 0, nil
	}
	if bits != 4 && bits != 8 {
		return 0, fmt.Errorf("gguf: invalid QA bit width %d", bits)
	}
	if !isValidQuantGroupSize(groupSize) {
		return 0, fmt.Errorf("gguf: invalid group size %d", groupSize)
	}
	if scaleBytesPerGroup < 0 {
		return 0, fmt.Errorf("gguf: negative scale bytes per group %d", scaleBytesPerGroup)
	}
	b := uint64(bits)
	if elems > (math.MaxUint64-7)/b {
		return 0, fmt.Errorf("gguf: quantized weight code bytes overflow uint64")
	}
	codeBytes := (elems*b + 7) / 8
	g := uint64(groupSize)
	groups := (elems + g - 1) / g
	sb := uint64(scaleBytesPerGroup)
	if sb > 0 && groups > math.MaxUint64/sb {
		return 0, fmt.Errorf("gguf: quantized weight scale bytes overflow uint64")
	}
	scaleBytes := groups * sb
	if codeBytes > math.MaxUint64-scaleBytes {
		return 0, fmt.Errorf("gguf: quantized weight bytes overflow uint64")
	}
	return codeBytes + scaleBytes, nil
}

var (
	// ErrHeadKernelPathMissing reports an empty or missing kernel execution path string.
	ErrHeadKernelPathMissing = errors.New("gguf: output-head kernel path is empty")
	// ErrHeadKernelPathFloatFallback reports a silent fallback to an unquantized/float
	// head path when a quantized output-head execution path was requested.
	ErrHeadKernelPathFloatFallback = errors.New("gguf: output-head kernel fell back to unquantized float path")
	// ErrHeadKernelPathMismatch reports a quantized kernel path that disagrees with the
	// expected quantized head encoding, target bit width, or explicit path.
	ErrHeadKernelPathMismatch = errors.New("gguf: output-head kernel path does not match intended quantized path")
)

// HeadKernelExpectation specifies the intended output-head execution contract that a
// reported runtime kernel path must satisfy.
type HeadKernelExpectation struct {
	// RequireQuantized mandates a quantized output-head kernel path and refuses any
	// unquantized/float fallback (e.g. "lm_head_f32", "f16", "bf16").
	RequireQuantized bool
	// Head optionally supplies the GGUF-header OutputHeadQuant classification. When
	// Head.IsQuantized is true and TargetBits is zero, the reported kernel path must
	// match Head.Type's quantized execution family.
	Head OutputHeadQuant
	// TargetBits optionally specifies the target QA bit width (4 or 8) for the head.
	TargetBits int
	// ExpectedPath optionally pins a specific kernel path token (e.g. "lm_head_q4k").
	ExpectedPath string
}

type headKernelPathKind uint8

const (
	headPathUnknown headKernelPathKind = iota
	headPathFloat
	headPathQuant4
	headPathQuant8
	headPathQuantKQuant
	headPathQuant2
)

// ConfirmQuantizedHeadPath verifies fail-closed that a reported output-head kernel execution
// path matches the intended quantized path and did not silently fall back to an unquantized
// or mismatched path.
func ConfirmQuantizedHeadPath(expect HeadKernelExpectation, reportedPath string) error {
	trimmed := strings.TrimSpace(reportedPath)
	if trimmed == "" {
		return ErrHeadKernelPathMissing
	}
	if expect.TargetBits != 0 && expect.TargetBits != 4 && expect.TargetBits != 8 {
		return fmt.Errorf("%w: invalid target bits %d (want 4 or 8)", ErrHeadKernelPathMismatch, expect.TargetBits)
	}
	quantRequested := expect.RequireQuantized || expect.Head.IsQuantized || expect.TargetBits != 0
	expectedTrimmed := strings.TrimSpace(expect.ExpectedPath)
	if !quantRequested && expectedTrimmed == "" && strings.TrimSpace(expect.Head.TensorName) == "" {
		return fmt.Errorf("%w: empty head kernel expectation", ErrHeadKernelPathMismatch)
	}
	if expectedTrimmed != "" {
		expKind, _ := classifyHeadKernelPath(expectedTrimmed)
		if quantRequested && expKind == headPathFloat {
			return fmt.Errorf("%w: expected path %q is unquantized while quantized execution is required", ErrHeadKernelPathFloatFallback, expectedTrimmed)
		}
		if expKind == headPathUnknown {
			return fmt.Errorf("%w: unrecognized expected path %q", ErrHeadKernelPathMismatch, expectedTrimmed)
		}
		quantRequested = quantRequested || expKind != headPathFloat
	}

	kind, tokens := classifyHeadKernelPath(trimmed)
	if kind == headPathFloat {
		if quantRequested {
			return fmt.Errorf("%w: reported %q", ErrHeadKernelPathFloatFallback, trimmed)
		}
		if expectedTrimmed != "" && !strings.EqualFold(trimmed, expectedTrimmed) {
			return fmt.Errorf("%w: reported %q, want %q", ErrHeadKernelPathMismatch, trimmed, expectedTrimmed)
		}
		return nil
	}
	if kind == headPathUnknown {
		return fmt.Errorf("%w: unrecognized kernel path %q", ErrHeadKernelPathMismatch, trimmed)
	}

	if !quantRequested && expect.Head.TensorName != "" && !expect.Head.IsQuantized {
		return fmt.Errorf("%w: reported quantized path %q for unquantized %s head without quantized request", ErrHeadKernelPathMismatch, trimmed, expect.Head.Type)
	}

	if expectedTrimmed != "" && !strings.EqualFold(trimmed, expectedTrimmed) {
		return fmt.Errorf("%w: reported %q, want %q", ErrHeadKernelPathMismatch, trimmed, expectedTrimmed)
	}

	if expect.TargetBits == 4 {
		if kind != headPathQuant4 {
			return fmt.Errorf("%w: reported %q, want 4-bit quantized head path", ErrHeadKernelPathMismatch, trimmed)
		}
		return nil
	}
	if expect.TargetBits == 8 {
		if kind != headPathQuant8 {
			return fmt.Errorf("%w: reported %q, want 8-bit quantized head path", ErrHeadKernelPathMismatch, trimmed)
		}
		return nil
	}

	if expect.Head.IsQuantized {
		if !headPathMatchesTensorType(expect.Head.Type, kind, tokens) {
			return fmt.Errorf("%w: reported %q is incompatible with %s output head", ErrHeadKernelPathMismatch, trimmed, expect.Head.Type)
		}
	}
	return nil
}

func classifyHeadKernelPath(path string) (headKernelPathKind, map[string]bool) {
	lower := strings.ToLower(strings.TrimSpace(path))
	if lower == "" {
		return headPathUnknown, nil
	}
	parts := strings.FieldsFunc(lower, func(r rune) bool {
		return r == '_' || r == '-' || r == '/' || r == ':' || r == '.' || r == ' '
	})
	tokens := make(map[string]bool, len(parts)+1)
	for _, p := range parts {
		if p != "" {
			tokens[p] = true
		}
	}
	tokens[lower] = true

	// Explicit float or fallback-to-float tokens always classify as float so paths
	// such as "q4k_dequant_f32" or "lm_head_f32" are never misread as quantized.
	if tokens["f32"] || tokens["fp32"] || tokens["float32"] ||
		tokens["f16"] || tokens["fp16"] || tokens["float16"] ||
		tokens["bf16"] || tokens["bfloat16"] || tokens["float"] ||
		tokens["unquantized"] || tokens["fallback"] {
		return headPathFloat, tokens
	}

	switch {
	case tokens["q4k"] || tokens["q4_k"] || (tokens["q4"] && tokens["k"]) ||
		tokens["q4"] || tokens["q4_0"] || tokens["q4_1"] ||
		tokens["int4"] || tokens["gptq"] || tokens["awq"] ||
		tokens["mxfp4"] || tokens["hqq4"] || tokens["iq4"] ||
		(tokens["iq4"] && (tokens["nl"] || tokens["xs"])):
		return headPathQuant4, tokens
	case tokens["q8"] || tokens["q8_0"] || tokens["int8"]:
		return headPathQuant8, tokens
	case tokens["kquant"] ||
		tokens["q6k"] || tokens["q6_k"] || (tokens["q6"] && tokens["k"]) ||
		tokens["q5k"] || tokens["q5_k"] || (tokens["q5"] && tokens["k"]) ||
		tokens["q3k"] || tokens["q3_k"] || (tokens["q3"] && tokens["k"]) ||
		tokens["q2k"] || tokens["q2_k"] || (tokens["q2"] && tokens["k"]) ||
		tokens["q5_0"] || tokens["q5_1"] ||
		tokens["iq3"] || tokens["iq2"] || tokens["iq1"]:
		return headPathQuantKQuant, tokens
	case tokens["q2"] || tokens["q2_0"] || tokens["pq2_0"] || tokens["pq2"] || tokens["q1_0"] || tokens["q1"]:
		return headPathQuant2, tokens
	default:
		return headPathUnknown, tokens
	}
}

func headPathMatchesTensorType(t TensorType, kind headKernelPathKind, tokens map[string]bool) bool {
	switch t {
	case TensorQ4_K:
		return kind == headPathQuant4 && (tokens["q4k"] || tokens["q4_k"] || (tokens["q4"] && tokens["k"]) || tokens["int4"])
	case TensorQ4_0, TensorQ4_1:
		return kind == headPathQuant4 || kind == headPathQuantKQuant
	case TensorMXFP4, TensorIQ4_NL, TensorIQ4_XS, TensorHQQ4:
		return kind == headPathQuant4 || kind == headPathQuantKQuant
	case TensorQ8_0:
		return kind == headPathQuant8 || tokens["kquant"]
	case TensorQ6_K:
		return kind == headPathQuantKQuant && (tokens["kquant"] || tokens["q6k"] || tokens["q6_k"] || (tokens["q6"] && tokens["k"]))
	case TensorQ5_K:
		return kind == headPathQuantKQuant && (tokens["kquant"] || tokens["q5k"] || tokens["q5_k"] || (tokens["q5"] && tokens["k"]))
	case TensorQ3_K:
		return kind == headPathQuantKQuant && (tokens["kquant"] || tokens["q3k"] || tokens["q3_k"] || (tokens["q3"] && tokens["k"]))
	case TensorQ2_K:
		return kind == headPathQuantKQuant && (tokens["kquant"] || tokens["q2k"] || tokens["q2_k"] || (tokens["q2"] && tokens["k"]))
	case TensorQ5_0, TensorQ5_1,
		TensorIQ2_XXS, TensorIQ2_XS, TensorIQ1_S, TensorIQ2_S, TensorIQ1_M, TensorIQ3_S, TensorIQ3_XXS:
		return kind == headPathQuantKQuant
	case TensorQ2_0, TensorPQ2_0, TensorQ1_0:
		return kind == headPathQuant2
	default:
		return false
	}
}
