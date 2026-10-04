package model

import (
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// qwen35_small_prompt_prefill_test.go pins the #13694 short-prompt floor levers that
// live in internal/model: the fresh-prompt batched-prefill threshold on the Metal
// resident session and the per-prefill Q4_K GEMM route observation the inkernel_chat
// line reports.
// [SW-VERIFIED] — Go-only, no GPU.

// TestQwen35MetalResidentPrefillGateAdmitsSmallPrompt pins the gate predicate: a fresh
// 2..15-token prompt on the Metal resident session takes the batched hybrid prefill,
// while every other session keeps the historical 16-token amortization threshold.
// fak-test:runtime fast est=10ms lane=default
func TestQwen35MetalResidentPrefillGateAdmitsSmallPrompt(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	for P := 2; P < qwen35HybridQBatchMinPrompt; P++ {
		if !q4kQwen35HybridPrefillAtPositionOKFor(cfg, P, 0, true) {
			t.Errorf("Metal resident fresh P=%d declined, want batched hybrid prefill", P)
		}
		if q4kQwen35HybridPrefillAtPositionOKFor(cfg, P, 0, false) {
			t.Errorf("CPU fresh P=%d admitted, want the historical %d-token threshold", P, qwen35HybridQBatchMinPrompt)
		}
		if q4kQwen35HybridPrefillAtPositionOK(cfg, P, 0) {
			t.Errorf("legacy gate fresh P=%d admitted, want the historical threshold", P)
		}
	}
	for _, metal := range []bool{true, false} {
		for _, P := range []int{0, 1} {
			if q4kQwen35HybridPrefillAtPositionOKFor(cfg, P, 0, metal) {
				t.Errorf("fresh P=%d metal=%v admitted, want the token step", P, metal)
			}
		}
		for _, P := range []int{qwen35HybridQBatchMinPrompt, 64} {
			if !q4kQwen35HybridPrefillAtPositionOKFor(cfg, P, 0, metal) {
				t.Errorf("fresh P=%d metal=%v declined, want batched", P, metal)
			}
		}
		if !q4kQwen35HybridPrefillAtPositionOKFor(cfg, 1, 7, metal) {
			t.Errorf("continuation P=1 metal=%v declined, want the resident append", metal)
		}
		if q4kQwen35HybridPrefillAtPositionOKFor(cfg, 4, -1, metal) {
			t.Errorf("negative base metal=%v admitted", metal)
		}
	}
	unsupported := cfg
	unsupported.AttnOutputGate = false
	if q4kQwen35HybridPrefillAtPositionOKFor(unsupported, 4, 0, true) {
		t.Error("unsupported geometry admitted at small P on the Metal resident session")
	}
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "1")
	if q4kQwen35HybridPrefillAtPositionOKFor(cfg, 4, 0, true) {
		t.Error("token-loop diagnostic escape hatch ignored at small P on the Metal resident session")
	}
}

// TestQwen35MetalResidentPrefillPredicateRequiresResidentMetal pins which sessions get
// the lowered threshold: only the backend-nil resident-Q4_K session with MetalQ4K on a
// live Metal device. A device-HAL session or a CPU Q4_K session keeps 16.
// fak-test:runtime fast est=2s lane=default
func TestQwen35MetalResidentPrefillPredicateRequiresResidentMetal(t *testing.T) {
	m := NewSynthetic(qwen35HybridQ4KTestCfg())
	s := m.NewSession()
	t.Cleanup(s.Close)
	if s.qwen35HybridMetalResidentPrefill() {
		t.Fatal("plain session reported Metal resident prefill")
	}
	s.Q4K = true
	if s.qwen35HybridMetalResidentPrefill() {
		t.Fatal("CPU resident-Q4_K session reported Metal resident prefill")
	}
	s.MetalQ4K = true
	if got, want := s.qwen35HybridMetalResidentPrefill(), metalgemm.Available(); got != want {
		t.Fatalf("MetalQ4K session predicate = %v, want metalgemm.Available() = %v", got, want)
	}
	var nilSession *Session
	if nilSession.qwen35HybridMetalResidentPrefill() {
		t.Fatal("nil session reported Metal resident prefill")
	}
}

// TestQwen35HybridSmallPromptPrefillMatchesTokenLoop is the small-P numerical parity
// gate for the batched resident-Q4_K hybrid prefill the lowered threshold now selects:
// at P in {2,3,9,15} (P=2 sits exactly at the conv history depth K-1) the batched body
// must build the same logits and KV/linear state as the per-token decode loop, with the
// same tolerances TestPrefillQwen35HybridQ4KMatchesTokenLoop documents for P=16. The CPU
// session's public Prefill must still take the token loop below 16 (threshold unchanged).
// fak-test:runtime medium est=30s lane=default
func TestQwen35HybridSmallPromptPrefillMatchesTokenLoop(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	base := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}
	for _, P := range []int{2, 3, 9, 15} {
		prompt := base[:P]

		ref := m.NewSession()
		ref.Q4K = true
		var refHidden []float32
		for _, id := range prompt {
			refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
		}
		want := ref.headResident(refHidden)

		got := m.NewSession()
		got.Q4K = true
		gotLogits := got.headResident(got.prefillQwen35HybridQ4KHidden(prompt))
		assertQ4KHybridPrefillMarker(t, got, 1, 0)

		label := "hybrid q4k small-P batched prefill P=" + strconv.Itoa(P)
		assertQuantLogitsClose(t, label+" logits", want, gotLogits)
		if argmax(want) != argmax(gotLogits) {
			t.Fatalf("%s greedy token = %d, want %d", label, argmax(gotLogits), argmax(want))
		}
		assertKVCacheQuantCloseTol(t, label, ref.Cache, got.Cache, prefillQ4KKTol(), prefillQ4KVTol())
		assertLinearAttnCacheQuantClose(t, label, ref.Cache.linear, got.Cache.linear)
		if got.Cache.Len() != P {
			t.Fatalf("%s cache len = %d, want %d", label, got.Cache.Len(), P)
		}

		cpu := m.NewSession()
		cpu.Q4K = true
		cpu.Prefill(prompt)
		assertQ4KHybridPrefillMarker(t, cpu, 0, 0)
		if obs := cpu.Q4KPrefillGEMMObservation(); obs != Q4KPrefillGEMMTokenLoop {
			t.Fatalf("CPU fresh P=%d prefill observation = %q, want %q", P, obs, Q4KPrefillGEMMTokenLoop)
		}
		ref.Close()
		got.Close()
		cpu.Close()
	}
}

// TestQwen35PrefillGEMMObservationRecordsRoute pins the per-prefill Q4_K GEMM route
// observation: distinct routes join in first-seen order, a reset clears it, and the CPU
// batched hybrid prefill reports "cpu".
// fak-test:runtime medium est=10s lane=default
func TestQwen35PrefillGEMMObservationRecordsRoute(t *testing.T) {
	var nilSession *Session
	if got := nilSession.Q4KPrefillGEMMObservation(); got != Q4KPrefillGEMMNone {
		t.Fatalf("nil session observation = %q, want %q", got, Q4KPrefillGEMMNone)
	}
	nilSession.ResetQ4KPrefillGEMMObservation()

	s := &Session{}
	if got := s.Q4KPrefillGEMMObservation(); got != Q4KPrefillGEMMNone {
		t.Fatalf("fresh observation = %q, want %q", got, Q4KPrefillGEMMNone)
	}
	s.observeQ4KPrefillGEMM("scalar")
	s.observeQ4KPrefillGEMM("scalar")
	s.observeQ4KPrefillGEMM("cpu")
	s.observeQ4KPrefillGEMM("")
	if got := s.Q4KPrefillGEMMObservation(); got != "scalar+cpu" {
		t.Fatalf("observation = %q, want scalar+cpu", got)
	}
	for _, label := range []string{"a", "b", "c", "d", "e"} {
		s.observeQ4KPrefillGEMM(label)
	}
	if got := s.Q4KPrefillGEMMObservation(); got != "scalar+cpu+a+b" {
		t.Fatalf("bounded observation = %q, want scalar+cpu+a+b", got)
	}
	s.ResetQ4KPrefillGEMMObservation()
	if got := s.Q4KPrefillGEMMObservation(); got != Q4KPrefillGEMMNone {
		t.Fatalf("reset observation = %q, want %q", got, Q4KPrefillGEMMNone)
	}

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	cpu := m.NewSession()
	t.Cleanup(cpu.Close)
	cpu.Q4K = true
	cpu.Prefill([]int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61})
	if got := cpu.Q4KPrefillGEMMObservation(); got != Q4KPrefillGEMMCPU {
		t.Fatalf("CPU batched hybrid prefill observation = %q, want %q", got, Q4KPrefillGEMMCPU)
	}
}
