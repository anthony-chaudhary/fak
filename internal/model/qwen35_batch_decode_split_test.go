package model

import "testing"

// The batched Qwen request spans every layer on the device, so a lane with a dense layer split
// must fall back to its own split-aware Step rather than be decoded all-device.
//
// fak-test:runtime fast est=50ms
func TestQwen35HybridQ4KBatchDeclinesDenseLayerSplit(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prev := qwen35HybridQ4KBatchStep
	qwen35HybridQ4KBatchStep = func(*BatchSession, []int) ([][]float32, bool) { return nil, false }
	t.Cleanup(func() { qwen35HybridQ4KBatchStep = prev })

	bs := m.NewBatchSession(2)
	if !bs.qwen35HybridQ4KBatchOK(2) {
		t.Fatal("precondition: an unsplit hybrid batch must be eligible")
	}
	bs.Seqs[1].DenseGPULayers = 2
	if bs.qwen35HybridQ4KBatchOK(2) {
		t.Fatal("a lane with a dense layer split must decline the all-layer batch")
	}
	bs.Seqs[1].DenseGPULayers = m.Cfg.NumLayers
	if !bs.qwen35HybridQ4KBatchOK(2) {
		t.Fatal("--gpu-layers == NumLayers is full placement and must stay eligible")
	}
}
