package ggufload

import (
	"fmt"
	"sort"
	"strings"
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

// ClassifyArtifactQuant opens a GGUF header and classifies its actual tensor inventory.
func ClassifyArtifactQuant(path string) (ArtifactQuant, error) {
	gg, err := Open(path)
	if err != nil {
		return ArtifactQuant{}, err
	}
	return ClassifyTensorQuant(gg.Tensors), nil
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

// AdmittedUDQ2KXLConstituents defines the 14 admitted constituent tensor types
// comprising the canonical Unsloth Dynamic Q2_K Extra Large (UD-Q2_K_XL) mixture.
var AdmittedUDQ2KXLConstituents = map[TensorType]bool{
	TensorQ2_K:    true,
	TensorIQ2_XXS: true,
	TensorIQ2_XS:  true,
	TensorIQ2_S:   true,
	TensorIQ1_S:   true,
	TensorIQ1_M:   true,
	TensorIQ3_XXS: true,
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

// IsUDQ2KXL reports whether tensors conform to the UD-Q2_K_XL recipe specification:
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
// quantization types, and satisfy the authoritative recipe specification.
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
