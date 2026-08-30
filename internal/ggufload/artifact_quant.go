package ggufload

import (
	"fmt"
	"sort"
	"strings"
)

// ArtifactQuant summarizes the tensor encodings actually present in a parsed GGUF.
// Inventory names every encoding in the header. Name identifies the quantized weight
// family independently of unquantized F32/F16/BF16 tensors.
type ArtifactQuant struct {
	Name        string
	Inventory   string
	Q4KResident bool
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
	return ArtifactQuant{Name: name, Inventory: inventory, Q4KResident: hasQ4K}
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
