// Conceptual INSPIRE-ONLY seam derived from NVIDIA/RULER
// Synthetic task generators only; no model training or benchmark score claim.
package ctxmmu

import (
	"errors"
	"fmt"
	"strings"
)

var ErrInvalidRulerTaskSpec = errors.New("ctxmmu: invalid RULER task spec")

// RulerTaskFamily names the synthetic long-context evaluation task families.
type RulerTaskFamily string

const (
	RulerNIAH        RulerTaskFamily = "niah"
	RulerVarTracking RulerTaskFamily = "variable_tracking"
	RulerAggregation RulerTaskFamily = "aggregation"
)

// RulerTaskSpec configures a deterministic synthetic long-context evaluation case.
type RulerTaskSpec struct {
	Family         RulerTaskFamily `json:"family"`
	ContextTokens  int             `json:"context_tokens"`
	NeedleDepthPct int             `json:"needle_depth_pct"`
	NumNeedles     int             `json:"num_needles"`
	Seed           int             `json:"seed"`
}

// Validate checks that family, context size, depth, and needle count are valid.
func (s RulerTaskSpec) Validate() error {
	switch s.Family {
	case RulerNIAH, RulerVarTracking, RulerAggregation:
	default:
		return fmt.Errorf("%w: unknown family %q", ErrInvalidRulerTaskSpec, s.Family)
	}
	if s.ContextTokens < 64 {
		return fmt.Errorf("%w: context_tokens=%d must be >= 64", ErrInvalidRulerTaskSpec, s.ContextTokens)
	}
	if s.NeedleDepthPct < 0 || s.NeedleDepthPct > 100 {
		return fmt.Errorf("%w: needle_depth_pct=%d must be in [0, 100]", ErrInvalidRulerTaskSpec, s.NeedleDepthPct)
	}
	if s.NumNeedles < 1 {
		return fmt.Errorf("%w: num_needles=%d must be >= 1", ErrInvalidRulerTaskSpec, s.NumNeedles)
	}
	return nil
}

// RulerTaskCase is one generated synthetic evaluation prompt and its ground-truth answer.
type RulerTaskCase struct {
	Family         RulerTaskFamily   `json:"family"`
	Prompt         string            `json:"prompt"`
	ExpectedAnswer string            `json:"expected_answer"`
	TargetTokens   int               `json:"target_tokens"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// GenerateRulerTask deterministically builds a synthetic long-context evaluation case.
func GenerateRulerTask(spec RulerTaskSpec) (RulerTaskCase, error) {
	if err := spec.Validate(); err != nil {
		return RulerTaskCase{}, err
	}
	const fillerUnit = "The quick brown fox jumps over the lazy dog. " // ~10 words
	fillerCount := max(4, spec.ContextTokens/10)
	insertIdx := (fillerCount * spec.NeedleDepthPct) / 100

	var needleText, expected string
	switch spec.Family {
	case RulerNIAH:
		key := fmt.Sprintf("KEY-%d", spec.Seed)
		val := fmt.Sprintf("VAL-%d", (spec.Seed*31+7)%10000)
		needleText = fmt.Sprintf("The secret passkey for %s is %s. ", key, val)
		expected = val
	case RulerVarTracking:
		hops := max(2, spec.NumNeedles)
		var b strings.Builder
		val := fmt.Sprintf("%d", (spec.Seed*17+3)%997)
		fmt.Fprintf(&b, "VAR_0 = %s. ", val)
		for i := 1; i <= hops; i++ {
			fmt.Fprintf(&b, "VAR_%d = VAR_%d. ", i, i-1)
		}
		needleText = b.String()
		expected = val
	case RulerAggregation:
		tag := fmt.Sprintf("ITEM_%d", spec.Seed%100)
		needleText = strings.Repeat(fmt.Sprintf("%s ", tag), spec.NumNeedles)
		expected = fmt.Sprintf("%s:%d", tag, spec.NumNeedles)
	}

	var prompt strings.Builder
	for i := 0; i < fillerCount; i++ {
		if i == insertIdx {
			prompt.WriteString(needleText)
		}
		prompt.WriteString(fillerUnit)
	}
	if insertIdx >= fillerCount {
		prompt.WriteString(needleText)
	}
	fmt.Fprintf(&prompt, "\nQuestion: report the target for %s.", spec.Family)

	return RulerTaskCase{
		Family:         spec.Family,
		Prompt:         prompt.String(),
		ExpectedAnswer: expected,
		TargetTokens:   spec.ContextTokens,
		Metadata: map[string]string{
			"depth_pct": fmt.Sprintf("%d", spec.NeedleDepthPct),
			"seed":      fmt.Sprintf("%d", spec.Seed),
		},
	}, nil
}

// RulerResidencyScore summarizes answer accuracy and effective MMU context residency.
type RulerResidencyScore struct {
	Matched        bool    `json:"matched"`
	ResidencyRatio float64 `json:"residency_ratio"`
}

// ScoreRulerResidency checks whether expected appears in actual and computes
// effective/target context residency ratio.
func ScoreRulerResidency(expected, actual string, effectiveCtxTokens, targetCtxTokens int) RulerResidencyScore {
	matched := expected != "" && strings.Contains(actual, expected)
	ratio := 0.0
	if targetCtxTokens > 0 && effectiveCtxTokens > 0 {
		ratio = float64(effectiveCtxTokens) / float64(targetCtxTokens)
		if ratio > 1.0 {
			ratio = 1.0
		}
	}
	return RulerResidencyScore{Matched: matched, ResidencyRatio: ratio}
}
