package compute

import (
	"errors"
	"testing"
)

type independentDecodeKV struct{ KVStore }

func independentDecodeLane() Qwen35SequencePrefillRequest {
	return Qwen35SequencePrefillRequest{
		Path: Qwen35SequencePrefillPath, TokenIDs: []int{1}, NeedLogits: true,
		Hidden: 8, Intermediate: 16, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, RotaryDim: 2,
		NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 3,
		RMSNormEpsilon: 1e-5, RoPEThetaForLayer: []float64{10000},
		Layers: []Qwen35SequenceLayer{{}}, States: []Qwen35SequenceState{{}}, KV: &independentDecodeKV{},
	}
}

func TestQwen35IndependentDecodeBatchContract(t *testing.T) {
	a, b := independentDecodeLane(), independentDecodeLane()
	valid := Qwen35IndependentDecodeBatchRequest{Path: Qwen35IndependentDecodeBatchPath, Lanes: []Qwen35SequencePrefillRequest{a, b}}
	if err := validateQwen35IndependentDecodeBatchContract(valid); err != nil {
		t.Fatalf("valid contract: %v", err)
	}
	for name, mutate := range map[string]func(*Qwen35IndependentDecodeBatchRequest){
		"singleton":           func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes = r.Lanes[:1] },
		"wrong path":          func(r *Qwen35IndependentDecodeBatchRequest) { r.Path = "other" },
		"multi-token lane":    func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes[1].TokenIDs = []int{1, 2} },
		"missing logits":      func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes[1].NeedLogits = false },
		"mixed geometry":      func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes[1].Hidden++ },
		"mixed GDN geometry":  func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes[1].ValueHeadDim++ },
		"mixed RoPE geometry": func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes[1].RoPEThetaForLayer = []float64{500000} },
		"missing KV":          func(r *Qwen35IndependentDecodeBatchRequest) { r.Lanes[1].KV = nil },
	} {
		t.Run(name, func(t *testing.T) {
			r := valid
			r.Lanes = append([]Qwen35SequencePrefillRequest(nil), valid.Lanes...)
			mutate(&r)
			if err := validateQwen35IndependentDecodeBatchContract(r); err == nil {
				t.Fatal("expected refusal")
			}
		})
	}
	mutated := &Qwen35IndependentDecodeBatchError{Stage: "layer", Mutated: true, Cause: errors.New("injected")}
	if !Qwen35IndependentDecodeBatchMutated(mutated) || Qwen35IndependentDecodeBatchMutated(errors.New("plain")) {
		t.Fatal("mutation classification mismatch")
	}
}

func TestQwen35IndependentDecodeBatchContractRejectsAliasedKVOwner(t *testing.T) {
	a, b := independentDecodeLane(), independentDecodeLane()
	b.KV = a.KV
	err := validateQwen35IndependentDecodeBatchContract(Qwen35IndependentDecodeBatchRequest{Path: Qwen35IndependentDecodeBatchPath, Lanes: []Qwen35SequencePrefillRequest{a, b}})
	var batchErr *Qwen35IndependentDecodeBatchError
	if !errors.As(err, &batchErr) || batchErr.Stage != "lane-ownership" {
		t.Fatalf("aliased KV owner error=%v", err)
	}
}

func TestQwen35IndependentDecodeBatchContractRejectsMixedGDNGeometry(t *testing.T) {
	a, b := independentDecodeLane(), independentDecodeLane()
	b.NumKeyHeads++
	err := validateQwen35IndependentDecodeBatchContract(Qwen35IndependentDecodeBatchRequest{Path: Qwen35IndependentDecodeBatchPath, Lanes: []Qwen35SequencePrefillRequest{a, b}})
	var batchErr *Qwen35IndependentDecodeBatchError
	if !errors.As(err, &batchErr) || batchErr.Stage != "lane-geometry" {
		t.Fatalf("mixed GDN geometry error=%v", err)
	}
}
