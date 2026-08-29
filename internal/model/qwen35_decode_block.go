package model

// qwen35DecodeBlockReceipt is the portable model view of one accepted P=1
// Metal linear-attention block. It distinguishes activation and immutable
// constant uploads so a zero-intermediate-transfer claim cannot hide H2D work.
type qwen35DecodeBlockReceipt struct {
	CommandBuffers, Commits, CompletionWaits                    int
	ProjectionDispatches, MixerProjectionDispatches             int
	MLPProjectionDispatches, Quantizers, GDNEncoders            int
	RMSNormEncoders, ResidualAddEncoders, SwiGLUEncoders        int
	InputUploads, ConstantUploads, FinalReadbacks               int
	IntermediateReadbacks, StateH2DTransfers, StateD2HTransfers int
	Encoders                                                    int
	Committed, CompletedWait                                    bool
}

type qwen35MetalDecodeBlock interface {
	Qwen35MetalDecodeBlock(*Session, int, []float32) ([]float32, qwen35DecodeBlockReceipt, bool, error)
}

// tryQwen35MetalDecodeBlock selects the complete block only after the recurrent
// owner has been promoted. Per-operation taps require the historical stage
// boundaries and therefore decline before submission. Layer taps remain valid:
// blockStep applies their steer/dump once to the returned complete residual.
func (s *Session) tryQwen35MetalDecodeBlock(layer int, x []float32) ([]float32, qwen35DecodeBlockReceipt, bool, error) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted || s.tapActive != nil && s.tapActive.ops || s.qwen35DecodeHandoffMode() != Qwen35DecodeHandoffAuto {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	block, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalDecodeBlock)
	if !ok {
		return nil, qwen35DecodeBlockReceipt{}, false, nil
	}
	out, receipt, accepted, err := block.Qwen35MetalDecodeBlock(s, layer, x)
	if !accepted {
		return nil, receipt, false, err
	}
	s.recordQwen35DecodeBlockAccepted()
	if err != nil {
		return nil, receipt, true, s.failQwen35GDNSequence(layer, "resident decode block", err)
	}
	return out, receipt, true, nil
}
