//go:build darwin && arm64 && cgo

package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestQwen38MTPMixedQ4KMForwardExecutesResidentMetalWeights(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := qwen38MTPQ4KTestModels(t)
	forward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(forward.Close)
	if !forward.draft.MetalQ4K || forward.tensorFormat != Qwen38MTPFormatQ4K {
		t.Fatalf("draft mechanism format=%q metal=%v", forward.tensorFormat, forward.draft.MetalQ4K)
	}

	prior, embedding := qwen38MTPInputs(m.Cfg.HiddenSize, 0)
	if _, err := forward.Forward(0, prior, embedding); err != nil {
		t.Fatalf("execute Metal Q4_K MTP forward: %v", err)
	}

	metalQ4KMu.Lock()
	q4Resident := metalQ4KW[forward.draft.M]
	q8Resident := metalQ8KW[forward.draft.M]
	q6Resident := metalQ6KW[forward.draft.M]
	_, fc := q8Resident["mtp.fc.weight"]
	_, q := q4Resident["model.layers.0.self_attn.q_proj.weight"]
	_, v := q6Resident["model.layers.0.self_attn.v_proj.weight"]
	_, down := q6Resident["model.layers.0.mlp.down_proj.weight"]
	_, head := q6Resident["lm_head.weight"]
	metalQ4KMu.Unlock()
	if !fc || !q || !v || !down || !head {
		t.Fatalf("mixed Metal MTP residency fc_q8=%v q_q4=%v v_q6=%v down_q6=%v head_q6=%v", fc, q, v, down, head)
	}

	receipt := EvaluateQwen38MTPEligibility(Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true,
		MTPBackendReady:   true,
		Backend:           Qwen38MTPBackendMetal,
		Model:             m,
		Greedy:            true,
		Depth:             3,
		FreshSession:      true,
		MemoryHeadroomOK:  true,
		OperatorEnabled:   true,
	})
	if receipt.Engine != Qwen38EngineMTP ||
		receipt.Backend != Qwen38MTPBackendMetal ||
		receipt.MTPTensorFormat != Qwen38MTPFormatQ4K ||
		receipt.RequestedDepth != 3 ||
		!receipt.TargetEquivalent {
		t.Fatalf("Metal mechanism receipt=%+v", receipt)
	}
}

func TestQwen38MTPMixedQ4KMFeedbackExecutesExactResidentMetalOperations(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	q4, ref := qwen38MTPQ4KTestModels(t)
	forward, err := q4.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(forward.Close)
	oracle, err := ref.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(oracle.Close)

	retainedHead := q4.kqw["lm_head.weight"]
	if retainedHead == nil || forward.draft.M.kqw["lm_head.weight"] != retainedHead {
		t.Fatal("draft did not alias the target's retained Q6_K output.weight")
	}
	if _, hostHead := forward.draft.M.manifest["lm_head.weight"]; hostHead {
		t.Fatal("draft retained an F32 LM head beside the resident Q6_K head")
	}
	if forward.draft.M.q8w["mtp.fc.weight"] != q4.q8w["mtp.fc.weight"] ||
		forward.draft.M.q4kw["model.layers.0.self_attn.q_proj.weight"] != q4.q4kw["mtp.layers.0.self_attn.q_proj.weight"] ||
		forward.draft.M.kqw["model.layers.0.self_attn.v_proj.weight"] != q4.kqw["mtp.layers.0.self_attn.v_proj.weight"] ||
		forward.draft.M.kqw["model.layers.0.mlp.down_proj.weight"] != q4.kqw["mtp.layers.0.mlp.down_proj.weight"] {
		t.Fatal("draft mixed stores do not alias the target's exact retained projections")
	}
	forceQ8UploadBudget(t, forward.draft.M, true)
	forward.draft.PhaseProfiler = NewPhaseProfiler()

	prior, embedding := qwen38MTPInputs(q4.Cfg.HiddenSize, 0)
	gotFeedback, gotLogits, err := qwen35MTPForwardFeedback(forward, 0, prior, embedding)
	if err != nil {
		t.Fatalf("execute Metal Q4_K MTP feedback: %v", err)
	}
	wantFeedback, wantLogits, err := qwen35MTPForwardFeedback(oracle, 0, prior, embedding)
	if err != nil {
		t.Fatalf("execute F32 MTP feedback oracle: %v", err)
	}
	if cos := cosine(gotFeedback, wantFeedback); cos < 0.999 {
		t.Fatalf("Metal mixed-Q4_K_M feedback cosine=%.8f, want >= 0.999", cos)
	}
	if cos := cosine(gotLogits, wantLogits); cos < 0.999 {
		t.Fatalf("resident Q6_K head logits cosine=%.8f, want >= 0.999", cos)
	}
	if got, want := forward.Argmax(gotLogits), oracle.Argmax(wantLogits); got != want {
		t.Fatalf("resident Q4_K feedback argmax=%d, oracle=%d", got, want)
	}
	execution, err := forward.draft.PhaseProfiler.MetalExecutionReceipt()
	if err != nil {
		t.Fatalf("resident feedback Metal execution receipt: %v", err)
	}
	if err := metalgemm.ValidateExecutionReceipt(execution); err != nil {
		t.Fatalf("validate resident feedback Metal execution receipt: %v", err)
	}
	operations := make(map[metalgemm.ExecutionOperation]int)
	for _, event := range execution.Events {
		operations[event.Operation]++
	}
	wantOperations := map[metalgemm.ExecutionOperation]int{
		metalgemm.ExecutionQ8GEMV:            1, // mtp.fc
		metalgemm.ExecutionQ4KGEMV:           3, // q/k/o projections
		metalgemm.ExecutionQ6KGEMV:           2, // v projection and LM head
		metalgemm.ExecutionQ4KFusedMLPQ6Down: 1, // gate/up/down MLP
	}
	if !reflect.DeepEqual(operations, wantOperations) {
		t.Fatalf("Metal operations=%v, want exact mixed-layout operations=%v", operations, wantOperations)
	}
	if got := forward.draft.PhaseProfiler.MetalFallbackCount(); got != 0 {
		t.Fatalf("resident feedback recorded %d CPU fallback(s), want zero", got)
	}

	metalQ4KMu.Lock()
	q4Resident := metalQ4KW[forward.draft.M]
	q8Resident := metalQ8KW[forward.draft.M]
	q6Resident := metalQ6KW[forward.draft.M]
	_, fc := q8Resident["mtp.fc.weight"]
	_, q := q4Resident["model.layers.0.self_attn.q_proj.weight"]
	_, k := q4Resident["model.layers.0.self_attn.k_proj.weight"]
	_, o := q4Resident["model.layers.0.self_attn.o_proj.weight"]
	_, gate := q4Resident["model.layers.0.mlp.gate_proj.weight"]
	_, up := q4Resident["model.layers.0.mlp.up_proj.weight"]
	_, v := q6Resident["model.layers.0.self_attn.v_proj.weight"]
	_, down := q6Resident["model.layers.0.mlp.down_proj.weight"]
	_, head := q6Resident["lm_head.weight"]
	metalQ4KMu.Unlock()
	if !fc || !q || !k || !o || !gate || !up || !v || !down || !head {
		t.Fatalf("mixed Metal residency fc_q8=%v q4[q=%v k=%v o=%v gate=%v up=%v] q6[v=%v down=%v head=%v]", fc, q, k, o, gate, up, v, down, head)
	}
}

func TestQwen38MTPMixedQ4KMDraftVocabFilterProjectsResidentQ6KSubset(t *testing.T) {
	m, _ := qwen38MTPQ4KTestModels(t)
	forward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(forward.Close)

	// Metal has no row-subset GEMV contract. Keep the opt-in filter on the
	// deterministic resident CPU path and compare it with that same full-head
	// projection, rather than presenting it as a zero-fallback Metal witness.
	forward.draft.MetalQ4K = false
	if forward.draft.M.kqw["lm_head.weight"] == nil {
		t.Fatal("mixed-layout draft is missing its resident Q6_K LM head")
	}
	if _, ok := forward.draft.M.manifest["lm_head.weight"]; ok {
		t.Fatal("mixed-layout draft unexpectedly retained an F32 LM head")
	}

	x, _ := qwen38MTPInputs(m.Cfg.HiddenSize, 2)
	full := forward.ProjectHead(x)
	subset := []int{6, 1, 4, 1}
	forward.SetDraftVocabFilter(NewDraftVocabFilter(subset))
	got := forward.ProjectHead(x)
	if len(got) != len(subset) {
		t.Fatalf("filtered logits=%d, want %d", len(got), len(subset))
	}
	for i, token := range subset {
		if math.Float32bits(got[i]) != math.Float32bits(full[token]) {
			t.Fatalf("filtered logit[%d] token=%d bits=%08x, full bits=%08x", i, token, math.Float32bits(got[i]), math.Float32bits(full[token]))
		}
	}
	if got[1] != got[3] {
		t.Fatalf("duplicate token projection drifted: got[%d]=%v got[%d]=%v", 1, got[1], 3, got[3])
	}
	if token := forward.Argmax(got); token != subset[argmaxF32(got)] {
		t.Fatalf("filtered argmax token=%d, want remapped token=%d", token, subset[argmaxF32(got)])
	}
}
