package model

import (
	"context"
	"reflect"
	"testing"
)

func TestKVTreeCompactionVerifierRetainsComputedPathWithoutReplay(t *testing.T) {
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	prompt := []int{1, 2, 3}
	ref := m.NewSession()
	last := ref.Prefill(prompt)
	root := argmaxF32(last)
	rootLogits := ref.Step(root)
	child := argmaxF32(rootLogits)
	ref.Step(child)

	target := m.NewSession()
	last = target.Prefill(prompt)
	hookCalls := 0
	m.Cfg.EnableResidualHook = true
	m.SetResidualHook(func(int, []float32) { hookCalls++ })
	probe := m.NewSession()
	probe.Cache = target.Cache.Clone()
	probe.Step(root)
	if hookCalls == 0 {
		t.Fatal("test hook did not observe direct Session.Step replay")
	}
	hookCalls = 0
	tree := &CandidateTree{Nodes: []CandidateNode{
		{Token: root, Parent: -1, Depth: 0, Children: []int{2}},
		{Token: (root + 1) % m.Cfg.VocabSize, Parent: -1, Depth: 0},
		{Token: child, Parent: 0, Depth: 1},
	}}
	proposal := DraftProposal{Tree: tree, Tokens: tree.Tokens()}
	result, err := ParallelVerifyKernel(context.Background(), target, prompt, proposal, last, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.AcceptedTokens, []int{root, child}) {
		t.Fatalf("accepted=%v", result.AcceptedTokens)
	}
	if hookCalls != 0 {
		t.Fatalf("tree verification replayed accepted tokens through Session.Step: residual hook calls=%d", hookCalls)
	}
	if !reflect.DeepEqual(result.LastAcceptedLogits, result.TargetLogits[2]) {
		t.Fatal("terminal logits did not follow non-contiguous accepted panel row")
	}
	if reflect.DeepEqual(result.LastAcceptedLogits, result.TargetLogits[1]) {
		t.Fatal("terminal logits incorrectly used accepted-count row rather than terminal node row")
	}
	if target.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len=%d want %d", target.Cache.Len(), ref.Cache.Len())
	}
	for l := range ref.Cache.K {
		if !reflect.DeepEqual(target.Cache.K[l], ref.Cache.K[l]) || !reflect.DeepEqual(target.Cache.Kraw[l], ref.Cache.Kraw[l]) || !reflect.DeepEqual(target.Cache.V[l], ref.Cache.V[l]) {
			t.Fatalf("layer %d retained tree path differs from sequential reference", l)
		}
	}
	if !reflect.DeepEqual(target.Cache.pos, ref.Cache.pos) || !reflect.DeepEqual(target.Cache.lineage.ids, ref.Cache.lineage.ids) {
		t.Fatal("verifier compaction did not preserve linear position/token lineage")
	}
}
