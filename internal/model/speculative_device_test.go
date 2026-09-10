package model

import (
	"context"
	"reflect"
	"testing"
)

func TestSpecDecodeGreedyDeviceExecutesNGramProposalsWithSerialParity(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	target, backend := newDevicePanelTestSession(t, m)
	prompt := []int{1, 2, 3, 4, 1, 2, 3}
	const maxNew, depth = 11, 4

	serial := m.NewSession()
	logits := serial.Prefill(prompt)
	want := make([]int, 0, maxNew)
	for len(want) < maxNew {
		next := argmaxF32(logits)
		want = append(want, next)
		logits = serial.Step(next)
	}

	generator := NewNGramProposalGenerator(NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: depth})
	run, err := SpecDecodeGreedyDevice(context.Background(), target, prompt, maxNew, depth, generator)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(run.Output, want) {
		t.Fatalf("device speculative output=%v want serial=%v", run.Output, want)
	}
	if run.DraftedTokens == 0 || backend.calls < 2 {
		t.Fatalf("ngram route drafted=%d sequence_calls=%d", run.DraftedTokens, backend.calls)
	}
	for i, req := range backend.requests[1:] {
		if !req.NeedAllLogits || len(req.TokenIDs) < 1 || len(req.TokenIDs) > depth {
			t.Fatalf("verification request %d tokens=%d all_logits=%t", i, len(req.TokenIDs), req.NeedAllLogits)
		}
	}
	committed := append(append([]int(nil), prompt...), run.Output...)
	if _, err := target.VerifyTokenLineage(committed); err != nil {
		t.Fatalf("committed device lineage: %v", err)
	}
	assertFloat32BitsEqual(t, "post-run continuation", serial.Step(argmaxF32(logits)), target.Step(argmaxF32(logits)))
}

func TestVerifyGreedyDeviceDraftRejectsWithoutRetainingSuffix(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	target, _ := newDevicePanelTestSession(t, m)
	serial := m.NewSession()
	defer serial.Close()
	prefix := []int{3, 7, 11, 5, 17, 19}
	boundary := target.Prefill(prefix)
	serial.Prefill(prefix)
	correct := argmaxF32(boundary)
	wrong := (correct + 1) % m.Cfg.VocabSize

	result, err := target.VerifyGreedyDeviceDraft(context.Background(), []int{correct, wrong, wrong}, boundary)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Accepted, []int{correct}) {
		t.Fatalf("accepted=%v want [%d]", result.Accepted, correct)
	}
	wantCorrection := argmaxF32(serial.Step(correct))
	if result.Correction != wantCorrection {
		t.Fatalf("correction=%d want=%d", result.Correction, wantCorrection)
	}
	if result.Receipt.Path != targetVerificationQwen38DevicePanelPath || result.Receipt.TargetVerificationOperations != 1 {
		t.Fatalf("verification receipt=%+v", result.Receipt)
	}
	assertDeviceStateMatchesSerial(t, target, serial, append(append([]int(nil), prefix...), correct))
}
