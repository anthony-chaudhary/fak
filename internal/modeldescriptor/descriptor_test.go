package modeldescriptor

import (
	"errors"
	"reflect"
	"testing"
)

func base(id string) Descriptor {
	return Descriptor{Schema: Schema, ID: id, Revision: "1", Provenance: "fixture", Trust: "witnessed", Aliases: []string{id}, Topology: []string{"dense", "hybrid_attention"}, State: []Geometry{{Kind: "kv", Shape: []int{32, 128}, BytesPerElement: 2}, {Kind: "gdn", Shape: []int{64}, BytesPerElement: 4}}, Quantization: []string{"q4_k"}, Storage: []string{"gguf"}, Tokenizer: []string{"bpe"}, Tools: []string{"json_tools"}, Backends: []string{"metal"}, Kernels: []string{"gemm"}, Envelopes: []string{"m3"}, Oracles: []string{"quality_fixture"}, Readiness: []string{"ready"}, Migration: []string{"v1"}, NativeEngine: "fak-native"}
}
func TestExistingAndSyntheticDescriptorsStaySwitchFree(t *testing.T) {
	budget := Budget{OutsideLeafFiles: 1}
	for _, id := range []string{"qwen-dense", "qwen-hybrid", "synthetic-new"} {
		r := Check(Candidate{Descriptor: base(id)}, budget)
		if !r.WithinBudget || r.DescriptorDigest == "" {
			t.Fatalf("%s %+v", id, r)
		}
		again := Check(Candidate{Descriptor: base(id)}, budget)
		if !reflect.DeepEqual(r, again) {
			t.Fatal("nondeterministic report")
		}
	}
}
func TestMismatchAndForbiddenCombinationFailBeforeAllocation(t *testing.T) {
	d := base("bad")
	d.Forbidden = [][]string{{"hybrid_attention", "q4_k", "json_tools"}}
	if !errors.Is(Validate(d), ErrMismatch) {
		t.Fatal("forbidden combination admitted")
	}
	d = base("bad")
	d.State[0].Shape = nil
	d.Trust = "external"
	if !errors.Is(Validate(d), ErrMismatch) {
		t.Fatal("unwitnessed descriptor admitted")
	}
	r := Check(Candidate{Descriptor: base("coupled"), CoreSwitches: 1}, Budget{})
	if r.WithinBudget {
		t.Fatal("coupling budget not enforced")
	}
}
