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
	var batches []int
	qwen35HybridQ4KBatchStep = func(bs *BatchSession, ids []int) ([][]float32, bool) {
		batches = append(batches, len(ids))
		bs.lastStepSharedPanels = 7
		bs.recordStepMACs(len(ids))
		out := make([][]float32, len(ids))
		for i := range out {
			out[i] = []float32{float32(ids[i])}
		}
		return out, true
	}
	for _, batch := range []int{2, 3, 8} {
		ids := make([]int, batch)
		for i := range ids {
			ids[i] = 8 + i
		}
		bs := m.NewBatchSession(batch)
		got := bs.StepBatch(ids)
		if batches[len(batches)-1] != batch || bs.LastStepSharedPanels() != 7 || bs.LastStepMACs() == 0 {
			t.Fatalf("B=%d batches=%v panels=%d macs=%d", batch, batches, bs.LastStepSharedPanels(), bs.LastStepMACs())
		}
		for i := range got {
			if got[i][0] != float32(ids[i]) {
				t.Fatalf("B=%d row %d=%v", batch, i, got[i])
			}
		}
	}

	bs := m.NewBatchSession(4)
	got := bs.StepBatchActive([]int{4, 5, 6, 7}, []bool{true, false, true, true})
	if batches[len(batches)-1] != 3 || bs.LastStepSharedPanels() != 7 || bs.LastStepMACs() == 0 {
		t.Fatalf("ragged batches=%v panels=%d macs=%d", batches, bs.LastStepSharedPanels(), bs.LastStepMACs())
	}
	if got[1] != nil {
		t.Fatal("inactive lane produced logits")
	}
}
