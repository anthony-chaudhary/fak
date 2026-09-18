package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rotationmeta"
)

// prism-ml/Ternary-Bonsai-2-27B-gguf rotated-basis contract (sha 6ed5e12b).
//
// The checkpoint folds a blockwise (1024) normalized Sylvester Walsh-Hadamard rotation
// into the stored ternary weights and declares the matching ACTIVATION transform as
// GGUF metadata. The loader must surface that declaration, and must refuse rather than
// ignore a malformed one: the model card states a runtime without the matching
// activation transform produces garbage, not an error.

// sumInts is the relation the sign payload must satisfy: one +/-1 sign for every
// weight in the declared row-width classes.
func sumInts(xs []int) int {
	n := 0
	for _, x := range xs {
		n += x
	}
	return n
}

// synthPrismHadamardMeta builds the real header's prism.hadamard.* keys, scaled down:
// sign_widths sum to the (short) sign_values length, matching the real file where the
// widths [5120,6144,17408] sum to the 28672-element sign vector.
func synthPrismHadamardMeta() map[string]Value {
	return map[string]Value{
		"general.architecture":                {Type: TypeString, Value: "qwen35"},
		"prism.hadamard.version":              {Type: TypeUint32, Value: uint32(1)},
		"prism.hadamard.block_size":           {Type: TypeUint32, Value: uint32(1024)},
		"prism.hadamard.transform":            {Type: TypeString, Value: "normalized-sylvester-walsh-hadamard"},
		"prism.hadamard.axis":                 {Type: TypeString, Value: "input-last-dimension"},
		"prism.hadamard.sign_mode":            {Type: TypeString, Value: "explicit"},
		"prism.hadamard.sign_widths":          {Type: TypeArray, Value: []Value{{Type: TypeInt32, Value: int32(4)}, {Type: TypeInt32, Value: int32(2)}}},
		"prism.hadamard.sign_values":          {Type: TypeArray, Value: []Value{{Type: TypeInt32, Value: int32(-1)}, {Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(-1)}, {Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(1)}}},
		"prism.hadamard.weight_names":         {Type: TypeArray, Value: []Value{{Type: TypeString, Value: "output.weight"}, {Type: TypeString, Value: "blk.0.ffn_down.weight"}}},
		"prism.hadamard.inverse_weight_names": {Type: TypeArray, Value: []Value{{Type: TypeString, Value: "token_embd.weight"}}},
		"prism.hadamard.gdn_v_grouped":        {Type: TypeBool, Value: true},
	}
}

func TestPrismHadamardMetaReadsDeclaredContract(t *testing.T) {
	t.Parallel()
	f := &File{Metadata: synthPrismHadamardMeta()}
	h, err := f.PrismHadamardMeta()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if h == nil {
		t.Fatal("declared contract must not read as absent")
	}
	if h.BlockSize != 1024 || h.Transform != "normalized-sylvester-walsh-hadamard" {
		t.Fatalf("bad header: %+v", h)
	}
	if len(h.SignWidths) != 2 || len(h.SignValues) != sumInts(h.SignWidths) {
		t.Fatalf("sign payload must be one sign per width-class weight: %+v", h)
	}
	if !h.GDNVGrouped {
		t.Fatal("gdn_v_grouped not read")
	}
	if len(h.WeightNames) != 2 || h.WeightNames[1] != "blk.0.ffn_down.weight" {
		t.Fatalf("bad weight names: %v", h.WeightNames)
	}
	if len(h.InverseNames) != 1 || h.InverseNames[0] != "token_embd.weight" {
		t.Fatalf("bad inverse names: %v", h.InverseNames)
	}
}

// A file with no prism.hadamard.* keys (every non-Bonsai-2 GGUF) must read as
// absent, not malformed: this is the overwhelmingly common case.
func TestPrismHadamardMetaAbsentOnPlainFile(t *testing.T) {
	t.Parallel()
	f := &File{Metadata: map[string]Value{"general.architecture": {Type: TypeString, Value: "qwen35"}}}
	h, err := f.PrismHadamardMeta()
	if err != nil || h != nil {
		t.Fatalf("want (nil,nil), got (%+v,%v)", h, err)
	}
}

// Once ANY prism.hadamard.* key is present the contract is required: a truncated
// declaration must refuse, never degrade to "no rotation declared".
func TestPrismHadamardMetaMalformedIsRefused(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		mutate func(map[string]Value)
	}{
		{"missing transform", func(m map[string]Value) { delete(m, "prism.hadamard.transform") }},
		{"missing signs", func(m map[string]Value) { delete(m, "prism.hadamard.sign_values") }},
		{"unknown transform", func(m map[string]Value) { m["prism.hadamard.transform"] = Value{Type: TypeString, Value: "rot13"} }},
		{"unknown axis", func(m map[string]Value) {
			m["prism.hadamard.axis"] = Value{Type: TypeString, Value: "output-last-dimension"}
		}},
		{"unversioned", func(m map[string]Value) { m["prism.hadamard.version"] = Value{Type: TypeUint32, Value: uint32(2)} }},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			m := synthPrismHadamardMeta()
			tc.mutate(m)
			f := &File{Metadata: m}
			h, err := f.PrismHadamardMeta()
			if !errors.Is(err, ErrPrismHadamardMalformed) {
				t.Fatalf("want ErrPrismHadamardMalformed, got (%+v,%v)", h, err)
			}
		})
	}
}

// The descriptor must be adjudication-compatible: a session without the declared
// activation fusion cannot claim support, and one with it can.
func TestPrismHadamardDescriptorAdjudicatesAgainstFusion(t *testing.T) {
	t.Parallel()
	f := &File{Metadata: synthPrismHadamardMeta()}
	h, err := f.PrismHadamardMeta()
	if err != nil {
		t.Fatal(err)
	}
	d := h.Descriptor()

	unsupported := rotationmeta.Adjudicate(d, rotationmeta.Capabilities{
		Recipes: map[rotationmeta.Recipe][]string{rotationmeta.RecipePrismHadamard: {"prism-bonsai2-gguf-v1"}},
	})
	if unsupported.Outcome != rotationmeta.OutcomeUnsupported || unsupported.Reason != rotationmeta.ReasonRuntimeTransformUnavailable {
		t.Fatalf("a runtime lacking the activation fusion must refuse, got %+v", unsupported)
	}

	supported := rotationmeta.Adjudicate(d, rotationmeta.Capabilities{
		Recipes: map[rotationmeta.Recipe][]string{rotationmeta.RecipePrismHadamard: {"prism-bonsai2-gguf-v1"}},
		Fusions: map[string]bool{PrismHadamardFusion: true},
	})
	if supported.Outcome != rotationmeta.OutcomeSupported {
		t.Fatalf("a runtime with the fusion must support, got %+v", supported)
	}
}
