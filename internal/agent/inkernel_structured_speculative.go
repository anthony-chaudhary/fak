package agent

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// structuredSpeculationSupported bounds rollback to ordinary host KV state.
// Recurrent, compressed, device and unknown architectures retain serial decode.
func (p *InKernelPlanner) structuredSpeculationSupported() bool {
	if p == nil || p.m == nil || p.backend != nil || p.metal || p.q4k || p.requireDeviceExecution || p.MetalMTPCoordinator() != nil || p.VulkanMTPEnabled() {
		return false
	}
	cfg := p.m.Cfg
	if cfg.IsQwen35Hybrid() || cfg.IsDeepSeekV41() || cfg.Qwen4Exp != nil || cfg.GLM5Next ||
		cfg.KVLoraRank != 0 || cfg.QLoraRank != 0 || cfg.LinearConvKernelDim != 0 ||
		len(cfg.CompressRatios) != 0 || len(cfg.LayerTypes) != 0 {
		return false
	}
	switch cfg.ModelType {
	case "", "llama", "qwen2", "qwen3":
		return true
	default:
		return false
	}
}

// acceptEmitted validates a target-selected token at the parent's current
// position. Scores are irrelevant here: selection already used masked logits.
func (c *nativeStructuredConstraint) acceptEmitted(token, vocab int, stops map[int]bool) error {
	if cap(c.maskBuffer) < vocab {
		c.maskBuffer = make([]float32, vocab)
	} else {
		c.maskBuffer = c.maskBuffer[:vocab]
		clear(c.maskBuffer)
	}
	effective, err := c.maskLogits(c.maskBuffer, stops)
	if err != nil {
		return err
	}
	return c.accept(token, effective)
}

// verifyStructuredSpeculativeRound adapts DeepInfer's per-position mask-before-
// verification and transactional grammar progress to the native linear verifier.
// Source: deepinfer/structure_output/matcher.py and model_executor/layers/sampler.py
// at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01 (Apache-2.0; vLLM lineage).
// Modified for host KV rollback and emission-owned state; no XGrammar code used.
func verifyStructuredSpeculativeRound(ctx context.Context, target *model.Session, proposal model.DraftProposal, last []float32, parent *nativeStructuredConstraint, stops map[int]bool) (result model.VerificationResult, err error) {
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if target == nil || target.M == nil || target.Cache == nil || proposal.Tree != nil {
		return result, ErrNativeStructuredUnsupported
	}
	if err = target.Cache.CanEvict(); err != nil {
		return result, fmt.Errorf("%w: %w", ErrNativeStructuredUnsupported, err)
	}
	fork := parent.fork()
	choice, err := fork.maskLogits(last, stops)
	if err != nil {
		return result, err
	}
	result.CorrectionToken = model.ArgmaxF32(choice)
	draft := proposal.Tokens
	for _, token := range draft {
		if token < 0 || token >= target.M.Cfg.VocabSize {
			return result, nil // malformed drafts cannot enter the target forward
		}
	}
	if len(draft) == 0 || draft[0] != result.CorrectionToken || stops[result.CorrectionToken] {
		return result, nil
	}
	base := target.Cache.Len()
	complete := false
	defer func() {
		if !complete {
			primary := recover()
			if primary != nil {
				defer func() { _ = recover(); panic(primary) }()
			}
			target.RollbackResidentSuffix(target.Cache.Len() - base)
		}
	}()
	rows := target.VerifyForward(draft, nil, nil)
	if len(rows) != len(draft) {
		return result, fmt.Errorf("%w: incomplete target verification rows", ErrNativeStructuredNoToken)
	}
	for i, token := range draft {
		if err = ctx.Err(); err != nil {
			return result, err
		}
		if token != result.CorrectionToken || stops[token] {
			break
		}
		if err = fork.accept(token, choice); err != nil {
			return result, err
		}
		result.AcceptedTokens = append(result.AcceptedTokens, token)
		choice, err = fork.maskLogits(rows[i], stops)
		if err != nil {
			return result, err
		}
		result.CorrectionToken = model.ArgmaxF32(choice)
	}
	result.NumAccepted = len(result.AcceptedTokens)
	result.RollbackKVCount = len(draft) - result.NumAccepted
	result.TargetLogits = rows
	target.RollbackResidentSuffix(result.RollbackKVCount)
	complete = true
	return result, nil
}
