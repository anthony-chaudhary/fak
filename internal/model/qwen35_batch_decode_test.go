package model

import "testing"

func TestStepBatchExecutionReceiptRequiresSharedPanel(t *testing.T) {
	llama := NewSynthetic(llamaArchConfig())
	shared := llama.NewBatchSession(2)
	_ = shared.StepBatch([]int{1, 2})
	if got := shared.LastStepSharedPanels(); got == 0 {
		t.Fatal("eligible batch did not receipt shared work")
	}

	qwen := NewSynthetic(qwen35HybridTestCfg())
	fallback := qwen.NewBatchSession(4)
	_ = fallback.StepBatch([]int{1, 2, 3, 4})
	if got := fallback.LastStepSharedPanels(); got != 0 {
		t.Fatalf("unsupported Qwen fallback claimed %d shared panels", got)
	}
}

func TestQwen35HybridQ4KBatchGateAndRaggedReceipt(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	old := qwen35HybridQ4KBatchStep
	defer func() { qwen35HybridQ4KBatchStep = old }()
	calls := 0
	qwen35HybridQ4KBatchStep = func(bs *BatchSession, ids []int) ([][]float32, bool) {
		calls++
		bs.lastStepSharedPanels = 7
		bs.recordStepMACs(len(ids))
		out := make([][]float32, len(ids))
		for i := range out {
			out[i] = []float32{float32(ids[i])}
		}
		return out, true
	}
	bs := m.NewBatchSession(4)
	got := bs.StepBatchActive([]int{4, 5, 6, 7}, []bool{true, false, true, true})
	if calls != 0 || bs.LastStepSharedPanels() != 0 {
		t.Fatalf("B=3 must stay serial: calls=%d receipt=%d", calls, bs.LastStepSharedPanels())
	}
	if got[1] != nil {
		t.Fatal("inactive lane produced logits")
	}

	got = bs.StepBatch([]int{8, 9, 10, 11})
	if calls != 1 || bs.LastStepSharedPanels() != 7 || bs.LastStepMACs() == 0 {
		t.Fatalf("calls=%d panels=%d macs=%d", calls, bs.LastStepSharedPanels(), bs.LastStepMACs())
	}
	for i := range got {
		if got[i][0] != float32(8+i) {
			t.Fatalf("row %d=%v", i, got[i])
		}
	}
}
