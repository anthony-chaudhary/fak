package model

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// Qwen38MTPShadowReceiptSchema identifies the schema version for MTP shadow receipts.
const Qwen38MTPShadowReceiptSchema = "fak/qwen38-mtp-shadow-receipt/v1"

// Qwen38MTPShadowReceipt witnesses the comparative shadow execution between native MTP
// and target-only decode.
type Qwen38MTPShadowReceipt struct {
	SchemaVersion    string          `json:"schema_version"`
	Engine           Qwen38MTPEngine `json:"engine"`
	PrimaryEngine    Qwen38MTPEngine `json:"primary_engine"`
	ShadowEngine     Qwen38MTPEngine `json:"shadow_engine"`
	TokensProduced   int             `json:"tokens_produced"`
	TokenMatchRate   float64         `json:"token_match_rate"`
	DivergenceCount  int             `json:"divergence_count"`
	DivergenceStep   int             `json:"divergence_step"` // -1 if no divergence
	MaxLogitDiff     float32         `json:"max_logit_diff"`
	CosineSimilarity float64         `json:"cosine_similarity"`
	PrimaryTokens    []int           `json:"primary_tokens"`
	ShadowTokens     []int           `json:"shadow_tokens"`
	ShadowDisrupted  bool            `json:"shadow_disrupted"`
	ShadowError      string          `json:"shadow_error,omitempty"`
}

// Validate verifies receipt schema invariants and internal consistency.
func (r Qwen38MTPShadowReceipt) Validate() error {
	if r.SchemaVersion != Qwen38MTPShadowReceiptSchema {
		return fmt.Errorf("model: invalid shadow receipt schema %q, want %q", r.SchemaVersion, Qwen38MTPShadowReceiptSchema)
	}
	if r.Engine != Qwen38EngineMTP {
		return fmt.Errorf("model: invalid shadow receipt engine %q, want %q", r.Engine, Qwen38EngineMTP)
	}
	if r.PrimaryEngine != Qwen38EngineMTP && r.PrimaryEngine != Qwen38EngineTargetDecode {
		return fmt.Errorf("model: invalid primary engine %q", r.PrimaryEngine)
	}
	if r.ShadowEngine != Qwen38EngineMTP && r.ShadowEngine != Qwen38EngineTargetDecode {
		return fmt.Errorf("model: invalid shadow engine %q", r.ShadowEngine)
	}
	if r.TokenMatchRate < 0.0 || r.TokenMatchRate > 1.00001 {
		return fmt.Errorf("model: invalid token match rate %g", r.TokenMatchRate)
	}
	if r.MaxLogitDiff < 0 {
		return fmt.Errorf("model: negative max logit diff %g", r.MaxLogitDiff)
	}
	if r.CosineSimilarity < -1.00001 || r.CosineSimilarity > 1.00001 {
		return fmt.Errorf("model: invalid cosine similarity %g", r.CosineSimilarity)
	}
	if r.DivergenceCount < 0 {
		return fmt.Errorf("model: negative divergence count %d", r.DivergenceCount)
	}
	if r.DivergenceCount == 0 && r.DivergenceStep != -1 {
		return fmt.Errorf("model: divergence count is 0 but divergence step is %d", r.DivergenceStep)
	}
	if r.DivergenceCount > 0 && r.DivergenceStep < 0 {
		return fmt.Errorf("model: divergence count is %d but divergence step is %d", r.DivergenceCount, r.DivergenceStep)
	}
	return nil
}

// Qwen38MTPShadowConfig configures execution and comparison parameters for shadow runner.
type Qwen38MTPShadowConfig struct {
	Concurrent      bool    `json:"concurrent"`
	ServeTargetOnly bool    `json:"serve_target_only"`
	LogitTolerance  float32 `json:"logit_tolerance"`
}

// DefaultQwen38MTPShadowConfig returns standard configuration for shadow comparison.
func DefaultQwen38MTPShadowConfig() Qwen38MTPShadowConfig {
	return Qwen38MTPShadowConfig{
		Concurrent:      false,
		ServeTargetOnly: true,
		LogitTolerance:  1e-4,
	}
}

// Qwen38MTPShadowRunner coordinates shadow evaluation between native MTP and target-only decode.
type Qwen38MTPShadowRunner struct {
	model  *Model
	config Qwen38MTPShadowConfig
}

// NewQwen38MTPShadowRunner constructs a shadow runner bound to the supplied Model.
func NewQwen38MTPShadowRunner(model *Model, cfg ...Qwen38MTPShadowConfig) *Qwen38MTPShadowRunner {
	c := DefaultQwen38MTPShadowConfig()
	if len(cfg) > 0 {
		c = cfg[0]
	}
	return &Qwen38MTPShadowRunner{
		model:  model,
		config: c,
	}
}

// Run executes native MTP and target-only decode concurrently or back-to-back from identical initial prefix.
// It serves the target-only (or configured primary) result while recording shadow telemetry;
// shadow failures never disrupt generation.
func (r *Qwen38MTPShadowRunner) Run(prompt []int, maxTokens int) ([]int, *Qwen38MTPShadowReceipt, error) {
	if r.model == nil {
		return nil, nil, errors.New("model: shadow runner requires non-nil model")
	}
	if len(prompt) == 0 {
		return nil, nil, errors.New("model: shadow runner requires non-empty prompt")
	}
	if maxTokens <= 0 {
		return nil, nil, errors.New("model: maxTokens must be positive")
	}

	var (
		primaryTokens, shadowTokens []int
		primaryLogits, shadowLogits [][]float32
		primaryErr, shadowErr       error
		shadowDisrupted             bool
		shadowErrMsg                string
	)

	primaryEngine := Qwen38EngineTargetDecode
	shadowEngine := Qwen38EngineMTP
	if !r.config.ServeTargetOnly {
		primaryEngine = Qwen38EngineMTP
		shadowEngine = Qwen38EngineTargetDecode
	}

	executePrimary := func() {
		if r.config.ServeTargetOnly {
			primaryTokens, primaryLogits, primaryErr = r.runTargetOnly(prompt, maxTokens)
		} else {
			primaryTokens, primaryLogits, primaryErr = r.runMTP(prompt, maxTokens)
		}
	}

	executeShadow := func() {
		defer func() {
			if rec := recover(); rec != nil {
				shadowDisrupted = true
				shadowErrMsg = fmt.Sprintf("panic in shadow execution: %v", rec)
			}
		}()

		if r.config.ServeTargetOnly {
			shadowTokens, shadowLogits, shadowErr = r.runMTP(prompt, maxTokens)
		} else {
			shadowTokens, shadowLogits, shadowErr = r.runTargetOnly(prompt, maxTokens)
		}
		if shadowErr != nil {
			shadowDisrupted = true
			shadowErrMsg = shadowErr.Error()
		}
	}

	if r.config.Concurrent {
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			executePrimary()
		}()
		go func() {
			defer wg.Done()
			executeShadow()
		}()
		wg.Wait()
	} else {
		executePrimary()
		executeShadow()
	}

	if primaryErr != nil {
		return nil, nil, fmt.Errorf("model: primary generation failed: %w", primaryErr)
	}

	// Compute comparison metrics
	receipt := CompareShadowOutputs(primaryTokens, shadowTokens, primaryLogits, shadowLogits)
	receipt.PrimaryEngine = primaryEngine
	receipt.ShadowEngine = shadowEngine
	receipt.ShadowDisrupted = shadowDisrupted
	receipt.ShadowError = shadowErrMsg

	return primaryTokens, receipt, nil
}

func (r *Qwen38MTPShadowRunner) runTargetOnly(prompt []int, maxTokens int) ([]int, [][]float32, error) {
	s := r.model.NewSession()
	s.captureTargetHidden = true
	defer s.Close()

	currLogits := s.Prefill(prompt)
	var tokens []int
	var logits [][]float32

	for i := 0; i < maxTokens; i++ {
		next := argmaxF32(currLogits)
		tokens = append(tokens, next)
		logits = append(logits, append([]float32(nil), currLogits...))
		currLogits = s.Step(next)
	}
	return tokens, logits, nil
}

func (r *Qwen38MTPShadowRunner) runMTP(prompt []int, maxTokens int) ([]int, [][]float32, error) {
	targetSess := r.model.NewSession()
	targetSess.captureTargetHidden = true
	defer targetSess.Close()

	mtpSess, err := NewQwen38MTPPersistentSession(targetSess)
	if err != nil {
		return nil, nil, err
	}
	defer mtpSess.Close()

	currLogits, err := mtpSess.Prefill(prompt)
	if err != nil {
		return nil, nil, err
	}

	var tokens []int
	var logits [][]float32

	for len(tokens) < maxTokens {
		snap, _ := targetSess.PrefixSnapshot()

		roundTokens, _, rErr := mtpSess.StepRound()
		if rErr != nil {
			if snap != nil {
				snap.Close()
			}
			return tokens, logits, rErr
		}

		if len(roundTokens) == 0 {
			if snap != nil {
				snap.Close()
			}
			break
		}

		// Token 0 of this round was evaluated with currLogits
		logits = append(logits, append([]float32(nil), currLogits...))
		tokens = append(tokens, roundTokens[0])

		// Step snap sequentially to get the exact per-token logits
		if snap != nil {
			tempSess := r.model.NewSession()
			clone, cErr := snap.Clone()
			if cErr == nil && clone != nil {
				_ = clone.Restore(tempSess)
				clone.Close()

				for i, tok := range roundTokens {
					stepLogits := tempSess.Step(tok)
					if i < len(roundTokens)-1 {
						if len(tokens) < maxTokens {
							logits = append(logits, append([]float32(nil), stepLogits...))
							tokens = append(tokens, roundTokens[i+1])
						}
					} else {
						// Last token's stepLogits is the boundary logits for the next round
						currLogits = stepLogits
					}
				}
			}
			tempSess.Close()
			snap.Close()
		}
	}

	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}
	if len(logits) > maxTokens {
		logits = logits[:maxTokens]
	}

	return tokens, logits, nil
}

// CompareShadowOutputs evaluates token sequences and logit distributions between two generation paths.
func CompareShadowOutputs(primaryTokens, shadowTokens []int, primaryLogits, shadowLogits [][]float32) *Qwen38MTPShadowReceipt {
	total := len(primaryTokens)
	if len(shadowTokens) > total {
		total = len(shadowTokens)
	}

	divergenceStep := -1
	divergenceCount := 0
	matched := 0

	limit := len(primaryTokens)
	if len(shadowTokens) < limit {
		limit = len(shadowTokens)
	}

	for i := 0; i < limit; i++ {
		if primaryTokens[i] == shadowTokens[i] {
			matched++
		} else {
			divergenceCount++
			if divergenceStep == -1 {
				divergenceStep = i
			}
		}
	}

	diffLen := total - limit
	if diffLen > 0 {
		divergenceCount += diffLen
		if divergenceStep == -1 {
			divergenceStep = limit
		}
	}

	var matchRate float64
	if total > 0 {
		matchRate = float64(matched) / float64(total)
	} else {
		matchRate = 1.0
	}

	maxDiff, cosineSim := ComputeShadowLogitMetrics(primaryLogits, shadowLogits)

	return &Qwen38MTPShadowReceipt{
		SchemaVersion:    Qwen38MTPShadowReceiptSchema,
		Engine:           Qwen38EngineMTP,
		PrimaryEngine:    Qwen38EngineTargetDecode,
		ShadowEngine:     Qwen38EngineMTP,
		TokensProduced:   len(primaryTokens),
		TokenMatchRate:   matchRate,
		DivergenceCount:  divergenceCount,
		DivergenceStep:   divergenceStep,
		MaxLogitDiff:     maxDiff,
		CosineSimilarity: cosineSim,
		PrimaryTokens:    append([]int(nil), primaryTokens...),
		ShadowTokens:     append([]int(nil), shadowTokens...),
	}
}

// ComputeShadowLogitMetrics calculates max absolute logit difference and average cosine similarity.
func ComputeShadowLogitMetrics(primaryLogits, shadowLogits [][]float32) (maxDiff float32, cosineSim float64) {
	if len(primaryLogits) == 0 || len(shadowLogits) == 0 {
		return 0, 1.0
	}

	steps := len(primaryLogits)
	if len(shadowLogits) < steps {
		steps = len(shadowLogits)
	}

	var maxD float32
	var totalCos float64
	validSteps := 0

	for s := 0; s < steps; s++ {
		pRow := primaryLogits[s]
		sRow := shadowLogits[s]
		n := len(pRow)
		if len(sRow) < n {
			n = len(sRow)
		}

		var dot, normP, normS float64
		for i := 0; i < n; i++ {
			pVal := pRow[i]
			sVal := sRow[i]
			d := float32(math.Abs(float64(pVal - sVal)))
			if d > maxD {
				maxD = d
			}
			p64 := float64(pVal)
			s64 := float64(sVal)
			dot += p64 * s64
			normP += p64 * p64
			normS += s64 * s64
		}

		if normP > 0 && normS > 0 {
			sim := dot / (math.Sqrt(normP) * math.Sqrt(normS))
			if sim > 1.0 {
				sim = 1.0
			}
			if sim < -1.0 {
				sim = -1.0
			}
			totalCos += sim
			validSteps++
		}
	}

	if validSteps > 0 {
		cosineSim = totalCos / float64(validSteps)
	} else {
		cosineSim = 1.0
	}
	return maxD, cosineSim
}
