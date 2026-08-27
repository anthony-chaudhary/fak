//go:build darwin && arm64 && cgo

package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// Qwen35MetalDecodeMixer lowers one exact Qwen3.8 linear-attention layer to the
// caller-owned P=1 graph. Capability and all five weight handles are resolved
// before graph construction; missing support therefore remains a clean decline.
func (b *metalQwen35GDNSequenceBackend) Qwen35MetalDecodeMixer(s *Session, layer int, xn []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error) {
	if s == nil || s.M == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K || s.qwen35HAL == nil ||
		!s.qwen35HAL.decodeAccepted || layer < 0 || layer >= len(s.qwen35HAL.sequenceLayers) || !s.M.Cfg.isLinearAttnLayer(layer) {
		return nil, qwen35DecodeMixerReceipt{}, false, nil
	}
	if _, err := qwen38MetalQ8RuntimeNames(s.M.Cfg); err != nil {
		return nil, qwen35DecodeMixerReceipt{}, false, nil
	}
	state := b.state(s.qwen35HAL.sequenceLayers[layer])
	if state == nil {
		return nil, qwen35DecodeMixerReceipt{}, false, nil
	}
	prefix := func(suffix string) string { return layerName(layer, suffix) }
	names := []string{
		prefix("linear_attn.in_proj_qkv.weight"), prefix("linear_attn.in_proj_z.weight"),
		prefix("linear_attn.in_proj_b.weight"), prefix("linear_attn.in_proj_a.weight"),
		prefix("linear_attn.out_proj.weight"),
	}
	metalQ4KMu.Lock()
	table := metalQ8KW[s.M]
	handles := make([]*metalgemm.Q8Weight, len(names))
	for i, name := range names {
		handles[i] = table[name]
	}
	metalQ4KMu.Unlock()
	for _, handle := range handles {
		if handle == nil {
			return nil, qwen35DecodeMixerReceipt{}, false, nil
		}
	}
	cfg := s.M.Cfg
	if len(xn) != cfg.HiddenSize {
		return nil, qwen35DecodeMixerReceipt{}, false, nil
	}
	p := func(suffix string) string { return layerName(layer, suffix) }
	result, nativeReceipt, accepted, err := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
		Input:   xn,
		Weights: metalgemm.Qwen35DecodeWeights{InQKV: handles[0], InZ: handles[1], InB: handles[2], InA: handles[3], Out: handles[4]},
		State:   state,
		Panel: metalgemm.GDNPanel{
			Tokens: 1, Conv1D: s.M.tensor(p("linear_attn.conv1d.weight")), ALog: s.M.tensor(p("linear_attn.A_log")),
			DTBias: s.M.tensor(p("linear_attn.dt_bias")), Norm: s.M.tensor(p("linear_attn.norm.weight")),
			RMSNormEpsilon: float32(cfg.RMSNormEps),
		},
	})
	receipt := qwen35DecodeMixerReceipt{
		CommandBuffers: nativeReceipt.CommandBuffers, Commits: nativeReceipt.Commits, CompletionWaits: nativeReceipt.CompletionWaits,
		ProjectionDispatches: nativeReceipt.ProjectionDispatches, Quantizers: nativeReceipt.Quantizers, GDNEncoders: nativeReceipt.GDNEncoders,
		InputUploads: nativeReceipt.InputUploads, FinalReadbacks: nativeReceipt.FinalReadbacks,
		IntermediateReadbacks: nativeReceipt.IntermediateReadbacks, StateH2DTransfers: nativeReceipt.StateH2DTransfers, StateD2HTransfers: nativeReceipt.StateD2HTransfers,
		Encoders: nativeReceipt.Encoders, Committed: nativeReceipt.Committed, CompletedWait: nativeReceipt.CompletedWait,
	}
	if err == nil && accepted && len(result) != cfg.HiddenSize {
		return nil, receipt, true, fmt.Errorf("metalgemm: Qwen decode mixer output elements=%d, want %d", len(result), cfg.HiddenSize)
	}
	return result, receipt, accepted, err
}
