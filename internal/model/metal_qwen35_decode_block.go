//go:build darwin && arm64 && cgo

package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// Qwen35MetalDecodeBlock lowers one exact Qwen3.8 P=1 PreNorm
// linear-attention block to the caller-owned graph. Every model and resident
// weight condition is resolved before RunQwen35Decode constructs the graph.
func (b *metalQwen35GDNSequenceBackend) Qwen35MetalDecodeBlock(s *Session, layer int, x []float32) ([]float32, qwen35DecodeBlockReceipt, bool, error) {
	return b.qwen35MetalDecodeBlock(s, layer, x, false)
}

func (b *metalQwen35GDNSequenceBackend) qwen35MetalDecodeBlock(s *Session, layer int, x []float32, injectPostSubmitFailure bool) ([]float32, qwen35DecodeBlockReceipt, bool, error) {
	if s == nil || s.M == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K || s.qwen35HAL == nil ||
		!s.qwen35HAL.decodeAccepted || layer < 0 || layer >= len(s.qwen35HAL.sequenceLayers) || !s.M.Cfg.isLinearAttnLayer(layer) {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	cfg := s.M.Cfg
	if _, err := qwen38MetalQ8RuntimeNames(cfg); err != nil || cfg.BlockTopology != PreNorm || cfg.LayerNorm || cfg.IsMoE() || cfg.DenseMLP ||
		cfg.ActGeluTanh || cfg.ActGeluErf || len(x) != cfg.HiddenSize {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	state := b.state(s.qwen35HAL.sequenceLayers[layer])
	if state == nil {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	p := func(suffix string) string { return layerName(layer, suffix) }
	for _, suffix := range []string{"mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias"} {
		if s.M.has(p(suffix)) {
			return nil, qwen35DecodeBlockReceipt{}, false, nil
		}
	}
	mixerNames := []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
		p("linear_attn.out_proj.weight"),
	}
	metalQ4KMu.Lock()
	q8Table := metalQ8KW[s.M]
	q8 := make([]*metalgemm.Q8Weight, len(mixerNames))
	for i, name := range mixerNames {
		q8[i] = q8Table[name]
	}
	metalQ4KMu.Unlock()
	for _, weight := range q8 {
		if weight == nil {
			return nil, qwen35DecodeBlockReceipt{}, false, nil
		}
	}
	gateName, upName, downName := p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight")
	gateTensor, upTensor := s.M.q4kw[gateName], s.M.q4kw[upName]
	if gateTensor == nil || upTensor == nil {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	gate, up := s.M.metalQ4KWeight(gateName, gateTensor), s.M.metalQ4KWeight(upName, upTensor)
	if gate == nil || up == nil {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	weights := metalgemm.Qwen35DecodeWeights{
		InQKV: q8[0], InZ: q8[1], InB: q8[2], InA: q8[3], Out: q8[4], MLPActivation: gate, MLPUp: up,
	}
	if downTensor := s.M.q4kw[downName]; downTensor != nil {
		weights.MLPDownQ4 = s.M.metalQ4KWeight(downName, downTensor)
	} else if downTensor := s.M.kqw[downName]; downTensor != nil && downTensor.kind == kindQ6K {
		weights.MLPDownQ6 = s.M.metalQ6KWeight(downName, downTensor)
	}
	if weights.MLPDownQ4 == nil && weights.MLPDownQ6 == nil {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	attnNorm, mlpNorm := s.M.attentionNorms(layer), s.M.mlpNorms(layer)
	if len(attnNorm.pre) != cfg.HiddenSize || len(mlpNorm.pre) != cfg.HiddenSize || len(attnNorm.preBias) != 0 || len(mlpNorm.preBias) != 0 {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	result, nativeReceipt, accepted, err := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
		Input: x, Weights: weights, State: state,
		Panel: metalgemm.GDNPanel{
			Tokens: 1, Conv1D: s.M.tensor(p("linear_attn.conv1d.weight")), ALog: s.M.tensor(p("linear_attn.A_log")),
			DTBias: s.M.tensor(p("linear_attn.dt_bias")), Norm: s.M.tensor(p("linear_attn.norm.weight")),
			RMSNormEpsilon: float32(cfg.RMSNormEps),
		},
		Block: &metalgemm.Qwen35DecodeBlock{
			InputNorm: attnNorm.pre, MLPNorm: mlpNorm.pre,
			RMSNormEpsilon: float32(cfg.RMSNormEps), NormGain1p: cfg.NormGain1p,
		},
		InjectPostSubmitFailureForTest: injectPostSubmitFailure,
	})
	receipt := qwen35DecodeBlockReceipt{
		CommandBuffers: nativeReceipt.CommandBuffers, Commits: nativeReceipt.Commits, CompletionWaits: nativeReceipt.CompletionWaits,
		ProjectionDispatches: nativeReceipt.ProjectionDispatches, MixerProjectionDispatches: nativeReceipt.MixerProjectionDispatches,
		MLPProjectionDispatches: nativeReceipt.MLPProjectionDispatches, Quantizers: nativeReceipt.Quantizers, GDNEncoders: nativeReceipt.GDNEncoders,
		RMSNormEncoders: nativeReceipt.RMSNormEncoders, ResidualAddEncoders: nativeReceipt.ResidualAddEncoders, SwiGLUEncoders: nativeReceipt.SwiGLUEncoders,
		InputUploads: nativeReceipt.InputUploads, ConstantUploads: nativeReceipt.ConstantUploads, FinalReadbacks: nativeReceipt.FinalReadbacks,
		IntermediateReadbacks: nativeReceipt.IntermediateReadbacks, StateH2DTransfers: nativeReceipt.StateH2DTransfers, StateD2HTransfers: nativeReceipt.StateD2HTransfers,
		Encoders: nativeReceipt.Encoders, Committed: nativeReceipt.Committed, CompletedWait: nativeReceipt.CompletedWait,
	}
	if err == nil && accepted && len(result) != cfg.HiddenSize {
		return nil, receipt, true, fmt.Errorf("metalgemm: Qwen decode block output elements=%d, want %d", len(result), cfg.HiddenSize)
	}
	return result, receipt, accepted, err
}
