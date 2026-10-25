package model

import (
	"fmt"
)

// PrefillMultimodal governs and ingests an expanded text+image prompt into a fresh
// ordinary f32 CPU session. It returns last-row logits, leaving the session ready for
// normal Step decode. Prefix reuse is intentionally a caller concern: image rows must
// never be restored from a token-only prefix snapshot.
func (s *Session) PrefillMultimodal(req MultimodalRequest, imageTokenID int) ([]float32, MultimodalVerdict, error) {
	if s == nil || s.M == nil {
		return nil, MultimodalVerdict{}, fmt.Errorf("%w: nil multimodal session", ErrMultimodalDenied)
	}
	if err := s.multimodalSessionSupportError(); err != nil {
		return nil, MultimodalVerdict{Decision: MultimodalDeny, Mode: req.Policy.withDefaults().Mode, Reason: err.Error()}, err
	}
	if imageTokenID < 0 || imageTokenID >= s.M.Cfg.VocabSize {
		err := fmt.Errorf("%w: image token id %d outside vocab size %d", ErrMultimodalDenied, imageTokenID, s.M.Cfg.VocabSize)
		return nil, MultimodalVerdict{Decision: MultimodalDeny, Mode: req.Policy.withDefaults().Mode, Reason: err.Error()}, err
	}
	if s.Cache == nil || s.Cache.Len() != 0 {
		err := fmt.Errorf("%w: multimodal prefill requires a fresh session", ErrMultimodalDenied)
		return nil, MultimodalVerdict{Decision: MultimodalDeny, Mode: req.Policy.withDefaults().Mode, Reason: err.Error()}, err
	}

	rows, verdict, err := s.M.prepareMultimodalRows(req)
	if err != nil {
		return nil, verdict, err
	}
	lineage := make([]int, 0, len(rows))
	for _, part := range req.Parts {
		if part.Image != nil {
			for range part.Image.Vectors {
				lineage = append(lineage, imageTokenID)
			}
			continue
		}
		lineage = append(lineage, part.TokenIDs...)
	}
	if len(lineage) != len(rows) {
		err := fmt.Errorf("%w: expanded lineage has %d tokens for %d rows", ErrMultimodalDenied, len(lineage), len(rows))
		verdict.Decision, verdict.Reason = MultimodalDeny, err.Error()
		return nil, verdict, err
	}

	var hidden []float32
	for i, row := range rows {
		hidden = s.tokenHiddenRow(lineage[i], i, append([]float32(nil), row...))
	}
	return s.head(hidden), verdict, nil
}

func (s *Session) multimodalSessionSupportError() error {
	switch {
	case s.M.Cfg.IsDeepSeekV41():
		return fmt.Errorf("%w: DeepSeek V4.1 session forward is not enabled", ErrV41VisionUnsupported)
	case s.Backend != nil:
		return fmt.Errorf("%w: device backend sessions are not enabled", ErrMultimodalDenied)
	case s.Quant || s.Q4 || s.Q4K || s.GPTQ || s.Metal || s.PrecisionPolicy != nil || s.M.Q2KEmbedding != nil:
		return fmt.Errorf("%w: resident quantized or accelerated sessions are not enabled", ErrMultimodalDenied)
	case s.M.Cfg.usesMLAMoELayout() || s.M.Cfg.isMiniMaxSparseAttn() || s.M.Cfg.isGemma4():
		return fmt.Errorf("%w: specialized architecture session is not enabled", ErrMultimodalDenied)
	default:
		return nil
	}
}
