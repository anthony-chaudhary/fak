// Package ggufinterop maps parsed GGUF artifacts into fak's neutral quantization contract.
package ggufinterop

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

// Outcome describes the result of mapping a GGUF artifact into the neutral quantization contract.
type Outcome string

const (
	// OutcomeSupported indicates the GGUF artifact was successfully mapped into a quant descriptor.
	OutcomeSupported Outcome = "supported"
	// OutcomeDelegate indicates multi-shard GGUF artifacts requiring the caller runtime to coordinate all shards.
	OutcomeDelegate Outcome = "delegate"
	// OutcomeAbstain indicates unsupported, missing, or unrecognized GGUF structures.
	OutcomeAbstain Outcome = "abstain"
	// OutcomeRefuse indicates invalid input such as a nil artifact.
	OutcomeRefuse Outcome = "refuse"
)

// Result represents the outcome and translated quantization descriptor from GGUF mapping.
type Result struct {
	Outcome    Outcome              `json:"outcome"`
	Reason     string               `json:"reason"`
	Descriptor quantmeta.Descriptor `json:"descriptor"`
	SplitCount int                  `json:"split_count,omitempty"`
}

// Map translates a parsed GGUF file into fak's neutral quantization descriptor.
func Map(f *ggufload.File) Result {
	if f == nil {
		return Result{Outcome: OutcomeRefuse, Reason: "nil GGUF artifact"}
	}
	arch := metaString(f, "general.architecture")
	if strings.TrimSpace(arch) == "" {
		return Result{Outcome: OutcomeAbstain, Reason: "general.architecture metadata is absent"}
	}
	families := map[string]bool{}
	for _, t := range f.Tensors {
		family, ok := familyFor(t.Type)
		if !ok {
			return Result{Outcome: OutcomeAbstain, Reason: fmt.Sprintf("unknown GGML tensor type %d", t.Type)}
		}
		families[family] = true
	}
	if len(families) == 0 {
		return Result{Outcome: OutcomeAbstain, Reason: "GGUF has no tensor quantization types"}
	}
	family := "mixed"
	if len(families) == 1 {
		for k := range families {
			family = k
		}
	}
	split := metaInt(f, "split.count")
	if split < 1 {
		split = 1
	}
	extra := map[string]json.RawMessage{"gguf_architecture": json.RawMessage(fmt.Sprintf("%q", arch)), "gguf_quant_family": json.RawMessage(fmt.Sprintf("%q", family))}
	d := quantmeta.Descriptor{Schema: "fak.quantmeta/1", Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"}, Provenance: quantmeta.ProvenanceSpec{MethodID: "gguf-header"}, Extra: extra}
	if split > 1 {
		return Result{Outcome: OutcomeDelegate, Reason: "split GGUF validated; runtime must open all shards", Descriptor: d, SplitCount: split}
	}
	return Result{Outcome: OutcomeSupported, Reason: "GGUF metadata mapped", Descriptor: d, SplitCount: split}
}
func metaString(f *ggufload.File, k string) string {
	if v, ok := f.Metadata[k]; ok {
		if s, ok := v.Value.(string); ok {
			return s
		}
	}
	return ""
}
func metaInt(f *ggufload.File, k string) int {
	if v, ok := f.Metadata[k]; ok {
		switch n := v.Value.(type) {
		case uint32:
			return int(n)
		case uint64:
			return int(n)
		case int:
			return n
		}
	}
	return 0
}
func familyFor(t ggufload.TensorType) (string, bool) {
	n := uint32(t)
	switch {
	case n <= 1:
		return "float", true
	case n == 28:
		return "ternary", true
	case n >= 10 && n <= 15:
		return "k-quant", true
	case n >= 16 && n <= 27 || n == 29:
		return "iq", true
	case n == 2 || n == 3 || n >= 6 && n <= 8:
		return "legacy", true
	default:
		return "", false
	}
}
