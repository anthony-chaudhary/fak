package model

// qwen35DecodeMixerReceipt is the model-owned view of one accepted P=1 Metal
// mixer. It remains portable so CGO-off builds preserve the selection contract.
type qwen35DecodeMixerReceipt struct {
	CommandBuffers, Commits, CompletionWaits                    int
	ProjectionDispatches, Quantizers, GDNEncoders               int
	InputUploads, FinalReadbacks                                int
	IntermediateReadbacks, StateH2DTransfers, StateD2HTransfers int
	Encoders                                                    int
	Committed, CompletedWait                                    bool
}

type qwen35MetalDecodeMixer interface {
	Qwen35MetalDecodeMixer(*Session, int, []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error)
}

// tryQwen35MetalDecodeMixer selects the whole P=1 operation only for an already
// promoted resident decode session. A pre-submit decline leaves the historical
// path untouched. An accepted error releases every resident owner and is returned
// to the caller, which must fail closed instead of replaying host projections.
func (s *Session) tryQwen35MetalDecodeMixer(layer int, xn []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted || s.tapActive != nil && s.tapActive.ops || s.qwen35DecodeHandoffMode() == Qwen35DecodeHandoffControl {
		return nil, qwen35DecodeMixerReceipt{}, false, nil
	}
	mixer, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalDecodeMixer)
	if !ok {
		return nil, qwen35DecodeMixerReceipt{}, false, nil
	}
	out, receipt, accepted, err := mixer.Qwen35MetalDecodeMixer(s, layer, xn)
	if !accepted {
		return nil, receipt, false, err
	}
	s.recordQwen35DecodeMixerAccepted()
	if err != nil {
		return nil, receipt, true, s.failQwen35GDNSequence(layer, "resident decode mixer", err)
	}
	return out, receipt, true, nil
}
