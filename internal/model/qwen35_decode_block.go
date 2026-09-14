package model

import "os"

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

// qwen35MetalDecodeTokenizer is implemented by the darwin sequence backend. It
// lowers a WHOLE decode token (all 64 Qwen3.8 layers) into one command buffer.
type qwen35MetalDecodeTokenizer interface {
	Qwen35MetalDecodeToken(*Session, int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error)
}

// tryQwen35MetalDecodeWholeToken selects the one-command-buffer whole-token decode
// graph after the resident GDN owners have been promoted. It declines before any
// submission when the owner, backend, or tap conditions are wrong; on acceptance
// the graph has advanced resident state and the caller must not replay. The route
// is ON by default: with the P=1 GEMV projection kernels and the per-shape buffer
// pool it measured decode 0.4-0.9 -> 4.2-4.6 tok/s on the M3 Pro (Qwen3.8-27B
// Q4_K_M) with bit-identical pooled/unpooled graph parity, versus the historical
// per-op path. Everything not exactly on this lane still declines to the
// historical decode. Set FAK_QWEN35_WHOLE_TOKEN_DECODE=0 to force the old path.
func (s *Session) tryQwen35MetalDecodeWholeToken(id int) ([]float32, bool, error) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted || s.tapActive != nil ||
		s.qwen35DecodeHandoffMode() == Qwen35DecodeHandoffControl {
		return nil, false, nil
	}
	if os.Getenv("FAK_QWEN35_WHOLE_TOKEN_DECODE") == "0" {
		return nil, false, nil
	}
	tok, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalDecodeTokenizer)
	if !ok {
		return nil, false, nil
	}
	hidden, receipt, accepted, err := tok.Qwen35MetalDecodeToken(s, id)
	if !accepted {
		return nil, false, err
	}
	s.recordQwen35DecodeBlockAccepted()
	if err != nil {
		return nil, true, s.failQwen35GDNSequence(-1, "whole-token decode graph", err)
	}
	// The one-command-buffer whole-token graph is the resident GDN sequence owner's
	// terminal evidence: record its executed receipt so Qwen35MetalForwardSequenceStatus
	// (and therefore the native-inference receipt + forward_path label) reports the
	// native route that actually ran, instead of silently falling back to the
	// host-recurrence label. Previously this receipt was discarded, so an
	// auto-admitted whole-token decode reported selector=on/evidence=unavailable and
	// the operator could not distinguish an active route from a declined one.
	if s.qwen35HAL != nil {
		s.qwen35HAL.setMetalForwardReceipt(receipt)
	}
	return hidden, true, nil
}
