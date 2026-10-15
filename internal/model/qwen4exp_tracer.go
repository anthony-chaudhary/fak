package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/qwen4exp"
)

func RunQwen4ExpDiagnosticStub(cfg Config, tokenIDs []int, seams qwen4exp.StubSeams) (qwen4exp.StubReceipt, error) {
	if !cfg.IsQwen4Exp() || cfg.Qwen4Exp == nil {
		return qwen4exp.StubReceipt{}, fmt.Errorf("%w: require qwen4exp config", qwen4exp.ErrDiagnosticStub)
	}
	if cfg.NumLayers != qwen4exp.StubLayers || cfg.HiddenSize != qwen4exp.StubHiddenSize || cfg.VocabSize != qwen4exp.StubVocabSize || cfg.Qwen4Exp.HCCount != qwen4exp.StubHCCount {
		return qwen4exp.StubReceipt{}, fmt.Errorf("%w: geometry layers=%d hidden=%d vocab=%d hc=%d, want %d/%d/%d/%d", qwen4exp.ErrDiagnosticStub, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize, cfg.Qwen4Exp.HCCount, qwen4exp.StubLayers, qwen4exp.StubHiddenSize, qwen4exp.StubVocabSize, qwen4exp.StubHCCount)
	}
	return qwen4exp.RunDiagnosticStub(tokenIDs, seams)
}
