package model

// qwen35HybridQ4KBatchStep is installed by the Apple Metal implementation. It
// stays nil elsewhere, preserving the established serial fallback.
var qwen35HybridQ4KBatchStep func(*BatchSession, []int) ([][]float32, bool)

func (bs *BatchSession) qwen35HybridQ4KBatchOK(batch int) bool {
	return bs != nil && bs.M != nil && batch >= 4 && batch <= 8 &&
		bs.M.Cfg.IsQwen35Hybrid() && bs.M.Cfg.BlockTopology == PreNorm && !bs.M.Cfg.IsMoE() && qwen35HybridQ4KBatchStep != nil
}

func (bs *BatchSession) stepBatchQwen35HybridQ4K(ids []int) [][]float32 {
	out, accepted := qwen35HybridQ4KBatchStep(bs, ids)
	if !accepted {
		// A decline occurs before any model state mutation, so scalar replay is safe.
		fallback := make([][]float32, len(ids))
		for i, id := range ids {
			fallback[i] = bs.Seqs[i].Step(id)
		}
		return fallback
	}
	return out
}
