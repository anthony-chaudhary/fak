package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Host-fallback stages for decode work a device backend ran on the host.
const (
	hostFallbackGreedyArgmax   = "greedy_argmax"
	hostFallbackEmbedRowUpload = "embedding_row_upload"
)

func init() {
	compute.DeclareHostFallbackStage(hostFallbackGreedyArgmax)
	compute.DeclareHostFallbackStage(hostFallbackEmbedRowUpload)
}

// q2kEmbeddingRowGatherHAL stages the raw Q2_K embedding table once as an immutable
// device weight and returns a gather that dequantizes one token row on the device, bit
// for bit as Q2KEmbedding.GatherRow at unit scale, as the resident [hidden] token input.
// It returns nil when the table encoding or the backend gather is unavailable; the
// caller keeps the host gather and upload then.
func (s *Session) q2kEmbeddingRowGatherHAL() func(id int) compute.Tensor {
	cfg := s.M.Cfg
	hidden, vocab := cfg.HiddenSize, cfg.VocabSize
	packed := s.M.Q2KEmbedding
	if packed == nil || packed.format != packedEmbeddingQ2K || packed.prism != nil ||
		packed.vocab != vocab || packed.hidden != hidden || hidden%qkK != 0 || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	gather, ok := s.Backend.(v41Q2KEmbeddingGatherBackend)
	if !ok || !gather.V41Q2KEmbeddingGatherAvailable() {
		return nil
	}
	key := fmt.Sprintf("v41-embedding:%s:Q2_K:%d:%d", v41EmbeddingTableName, vocab, hidden)
	table := s.cachedImmutableWeight(key, key, func() compute.Tensor {
		return s.Backend.Upload(compute.NewQ2K(compute.Default(), []int{vocab, hidden}, packed.raw), compute.Q2_K)
	})
	return func(id int) compute.Tensor {
		row, err := gather.V41Q2KEmbeddingGather(table, []int{id}, hidden)
		if err != nil {
			panic(err)
		}
		row.Shape = []int{hidden}
		return row
	}
}
