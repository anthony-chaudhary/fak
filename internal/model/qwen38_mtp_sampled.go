package model

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

const (
	// Qwen38SampledVerificationReceiptSchema identifies the schema version for sampled MTP receipts.
	Qwen38SampledVerificationReceiptSchema = "fak/qwen38-mtp-sampled-receipt/v1"
)

// ErrSamplingModeUnsupported is returned when unsupported sampler options are provided.
var ErrSamplingModeUnsupported = errors.New("model: sampling mode unsupported")

// Qwen38SamplingDowngradeError reports a typed downgrade to ordinary fak-native target decode
// when an unsupported sampler option is specified.
type Qwen38SamplingDowngradeError struct {
	Reason Qwen38MTPDowngradeReason `json:"reason"`
	Detail string                   `json:"detail,omitempty"`
}

func (e *Qwen38SamplingDowngradeError) Error() string {
	if e == nil {
		return "model: sampling downgrade to target decode"
	}
	if e.Detail != "" {
		return fmt.Sprintf("model: sampling downgrade to target decode: %s (%s)", e.Reason, e.Detail)
	}
	return fmt.Sprintf("model: sampling downgrade to target decode: %s", e.Reason)
}

func (e *Qwen38SamplingDowngradeError) Unwrap() error {
	return ErrSamplingModeUnsupported
}

// Qwen38RNGState captures the complete state of the PRNG for bit-exact replay and rollback.
type Qwen38RNGState struct {
	State uint64 `json:"state"`
	Inc   uint64 `json:"inc"`
	Steps uint64 `json:"steps"`
}

// Qwen38PRNG provides a deterministic, thread-safe pseudo-random number generator
// based on PCG32 with explicit checkpointing and rollback capabilities.
type Qwen38PRNG struct {
	mu    sync.Mutex
	state uint64
	inc   uint64
	steps uint64
}

// NewQwen38PRNG constructs and seeds a new Qwen38PRNG.
func NewQwen38PRNG(seed, seq uint64) *Qwen38PRNG {
	rng := &Qwen38PRNG{}
	rng.Seed(seed, seq)
	return rng
}

// Seed reinitializes the PRNG with the specified seed and sequence selector.
func (r *Qwen38PRNG) Seed(seed, seq uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.state = 0
	r.inc = (seq << 1) | 1
	r.steps = 0
	r.stepLocked()
	r.state += seed
	r.stepLocked()
}

func (r *Qwen38PRNG) stepLocked() uint32 {
	oldState := r.state
	r.state = oldState*6364136223846793005 + r.inc
	xorshifted := uint32(((oldState >> 18) ^ oldState) >> 27)
	rot := uint32(oldState >> 59)
	r.steps++
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
}

// NextUint32 returns the next pseudo-random 32-bit unsigned integer.
func (r *Qwen38PRNG) NextUint32() uint32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stepLocked()
}

// NextFloat32 returns a uniform float32 in [0, 1).
func (r *Qwen38PRNG) NextFloat32() float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	val := r.stepLocked()
	// Keep 24 bits so conversion is exact and cannot round the upper endpoint to 1.
	return float32(val>>8) / 16777216.0
}

// NextFloat64 returns a uniform float64 in [0, 1).
func (r *Qwen38PRNG) NextFloat64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	v1 := uint64(r.stepLocked())
	v2 := uint64(r.stepLocked())
	val := (v1 << 32) | v2
	return float64(val) / 18446744073709551616.0
}

// Checkpoint returns an immutable snapshot of current PRNG state.
func (r *Qwen38PRNG) Checkpoint() Qwen38RNGState {
	r.mu.Lock()
	defer r.mu.Unlock()
	return Qwen38RNGState{
		State: r.state,
		Inc:   r.inc,
		Steps: r.steps,
	}
}

// Rollback restores the PRNG to a previously saved checkpoint.
func (r *Qwen38PRNG) Rollback(s Qwen38RNGState) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.state = s.State
	r.inc = s.Inc
	r.steps = s.Steps
}

// StepsConsumed returns the cumulative number of random values generated.
func (r *Qwen38PRNG) StepsConsumed() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.steps
}

// Qwen38RNGTx represents an active RNG transaction that can be committed or rolled back.
type Qwen38RNGTx struct {
	rng        *Qwen38PRNG
	checkpoint Qwen38RNGState
	committed  bool
	aborted    bool
}

// Commit finalizes the transaction, keeping advances to the PRNG.
func (tx *Qwen38RNGTx) Commit() {
	tx.committed = true
}

// Rollback restores the PRNG state to when the transaction was opened.
func (tx *Qwen38RNGTx) Rollback() {
	if !tx.committed && !tx.aborted {
		tx.rng.Rollback(tx.checkpoint)
		tx.aborted = true
	}
}

// Checkpoint returns the snapshot taken at transaction start.
func (tx *Qwen38RNGTx) Checkpoint() Qwen38RNGState {
	return tx.checkpoint
}

// Qwen38SamplerConfig defines the sampling options for speculative verification.
// Bounded supported sampler set:
// - Greedy: Temperature == 0 or Greedy == true
// - Sampled: Temperature > 0 and TopP in (0, 1]
// Any option outside this bounded set triggers typed downgrade `sampling_mode_unsupported`.
type Qwen38SamplerConfig struct {
	Temperature       float32 `json:"temperature"`
	TopP              float32 `json:"top_p"`
	Greedy            bool    `json:"greedy"`
	Seed              uint64  `json:"seed"`
	Seq               uint64  `json:"seq"`
	TopK              int     `json:"top_k,omitempty"`
	MinP              float32 `json:"min_p,omitempty"`
	RepetitionPenalty float32 `json:"repetition_penalty,omitempty"`
	PresencePenalty   float32 `json:"presence_penalty,omitempty"`
	FrequencyPenalty  float32 `json:"frequency_penalty,omitempty"`
	RollbackOnAbort   bool    `json:"rollback_on_abort"`
}

// Validate checks whether the sampler config falls within the supported bounded sampler set.
func (cfg Qwen38SamplerConfig) Validate() (Qwen38MTPDowngradeReason, error) {
	// Guard against unsupported foreign/extended sampling options
	if cfg.TopK > 0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("top_k=%d is unsupported; bounded set is {temperature, top_p, greedy}", cfg.TopK),
		}
	}
	if cfg.MinP > 0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("min_p=%g is unsupported", cfg.MinP),
		}
	}
	if cfg.RepetitionPenalty != 0 && cfg.RepetitionPenalty != 1.0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("repetition_penalty=%g is unsupported", cfg.RepetitionPenalty),
		}
	}
	if cfg.PresencePenalty != 0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("presence_penalty=%g is unsupported", cfg.PresencePenalty),
		}
	}
	if cfg.FrequencyPenalty != 0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("frequency_penalty=%g is unsupported", cfg.FrequencyPenalty),
		}
	}

	// Greedy mode: temperature == 0 or explicit greedy flag
	if cfg.Greedy || cfg.Temperature == 0 {
		if cfg.Temperature < 0 {
			return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
				Reason: Qwen38MTPSamplingUnsupported,
				Detail: fmt.Sprintf("negative temperature=%g", cfg.Temperature),
			}
		}
		if cfg.TopP != 0 && (cfg.TopP <= 0 || cfg.TopP > 1.0) {
			return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
				Reason: Qwen38MTPSamplingUnsupported,
				Detail: fmt.Sprintf("top_p=%g outside (0, 1]", cfg.TopP),
			}
		}
		return Qwen38MTPEligible, nil
	}

	// Sampled mode: temperature > 0 and top_p in (0, 1]
	if cfg.Temperature < 0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("negative temperature=%g", cfg.Temperature),
		}
	}
	if cfg.TopP <= 0 || cfg.TopP > 1.0 {
		return Qwen38MTPSamplingUnsupported, &Qwen38SamplingDowngradeError{
			Reason: Qwen38MTPSamplingUnsupported,
			Detail: fmt.Sprintf("top_p=%g outside (0, 1]", cfg.TopP),
		}
	}
	return Qwen38MTPEligible, nil
}

// Qwen38SampledDistributionMetrics records statistical and information-theoretic divergence metrics.
type Qwen38SampledDistributionMetrics struct {
	TotalVariationDistance float64   `json:"total_variation_distance"`
	ExpectedAcceptanceRate float64   `json:"expected_acceptance_rate"`
	ResidualMass           float64   `json:"residual_mass"`
	TargetEntropy          float64   `json:"target_entropy"`
	DraftEntropy           float64   `json:"draft_entropy"`
	KLDivergence           float64   `json:"kl_divergence"`
	PerStepAlpha           []float32 `json:"per_step_alpha"`
	PerStepAcceptProb      []float32 `json:"per_step_accept_prob"`
}

// Qwen38SampledTokenResult holds the verification verdict for an individual token position.
type Qwen38SampledTokenResult struct {
	Accepted         bool      `json:"accepted"`
	DraftToken       int       `json:"draft_token"`
	ReplacementToken int       `json:"replacement_token"`
	Alpha            float32   `json:"alpha"`
	UniformSample    float32   `json:"uniform_sample"`
	ResidualDist     []float32 `json:"residual_dist,omitempty"`
}

// Qwen38SampledVerificationReceipt witnesses the full speculative verification step.
type Qwen38SampledVerificationReceipt struct {
	SchemaVersion    string                           `json:"schema_version"`
	Engine           Qwen38MTPEngine                  `json:"engine"`
	Outcome          Qwen38MTPReceiptOutcome          `json:"outcome"`
	SamplingMode     string                           `json:"sampling_mode"`
	Temperature      float32                          `json:"temperature"`
	TopP             float32                          `json:"top_p"`
	RequestedDepth   int                              `json:"requested_depth"`
	EffectiveDepth   int                              `json:"effective_depth"`
	AcceptedTokens   []int                            `json:"accepted_tokens"`
	ReplacementToken int                              `json:"replacement_token"`
	RejectedAt       int                              `json:"rejected_at"`
	BonusToken       int                              `json:"bonus_token"`
	Tokens           Qwen38MTPTokenAccounting         `json:"tokens"`
	Metrics          Qwen38SampledDistributionMetrics `json:"metrics"`
	DowngradeReason  Qwen38MTPDowngradeReason         `json:"downgrade_reason,omitempty"`
	FailureReason    Qwen38MTPFailureReason           `json:"failure_reason,omitempty"`
	RNGStepsConsumed uint64                           `json:"rng_steps_consumed"`
}

// Validate verifies receipt schema invariants and internal consistency.
func (r Qwen38SampledVerificationReceipt) Validate() error {
	if r.SchemaVersion != Qwen38SampledVerificationReceiptSchema {
		return fmt.Errorf("model: qwen3.8 sampled receipt schema %q, want %q", r.SchemaVersion, Qwen38SampledVerificationReceiptSchema)
	}
	if r.Engine != Qwen38EngineMTP && r.Engine != Qwen38EngineTargetDecode {
		return fmt.Errorf("model: qwen3.8 sampled receipt engine %q is not fak-native", r.Engine)
	}
	if r.RequestedDepth < 0 || r.EffectiveDepth < 0 || r.EffectiveDepth > r.RequestedDepth {
		return fmt.Errorf("model: impossible depths requested=%d effective=%d", r.RequestedDepth, r.EffectiveDepth)
	}
	if err := r.Tokens.validate(r.EffectiveDepth); err != nil {
		return err
	}
	if r.Metrics.TotalVariationDistance < 0 || r.Metrics.TotalVariationDistance > 1.00001 {
		return fmt.Errorf("model: invalid TV distance %g", r.Metrics.TotalVariationDistance)
	}
	if r.Metrics.ExpectedAcceptanceRate < 0 || r.Metrics.ExpectedAcceptanceRate > 1.00001 {
		return fmt.Errorf("model: invalid expected acceptance rate %g", r.Metrics.ExpectedAcceptanceRate)
	}
	return nil
}

// Qwen38SampledSpeculativeVerifier executes mathematically rigorous speculative sampling
// under the Leviathan-Chen / Sun et al. formulation with deterministic PRNG transaction rollback.
type Qwen38SampledSpeculativeVerifier struct {
	mu     sync.Mutex
	config Qwen38SamplerConfig
	prng   *Qwen38PRNG
}

// NewQwen38SampledSpeculativeVerifier creates a verifier configured with the bounded sampler set.
// Returns a typed downgrade error if unsupported sampler options are provided.
func NewQwen38SampledSpeculativeVerifier(cfg Qwen38SamplerConfig) (*Qwen38SampledSpeculativeVerifier, error) {
	if reason, err := cfg.Validate(); err != nil {
		return nil, err
	} else if reason != Qwen38MTPEligible {
		return nil, &Qwen38SamplingDowngradeError{Reason: reason}
	}
	if cfg.TopP == 0 {
		cfg.TopP = 1.0
	}
	prng := NewQwen38PRNG(cfg.Seed, cfg.Seq)
	return &Qwen38SampledSpeculativeVerifier{
		config: cfg,
		prng:   prng,
	}, nil
}

// Config returns the immutable sampler configuration.
func (v *Qwen38SampledSpeculativeVerifier) Config() Qwen38SamplerConfig {
	return v.config
}

// PRNG returns the underlying transaction-aware PRNG.
func (v *Qwen38SampledSpeculativeVerifier) PRNG() *Qwen38PRNG {
	return v.prng
}

// CheckpointRNG captures the PRNG state.
func (v *Qwen38SampledSpeculativeVerifier) CheckpointRNG() Qwen38RNGState {
	return v.prng.Checkpoint()
}

// RollbackRNG restores the PRNG state to a checkpoint.
func (v *Qwen38SampledSpeculativeVerifier) RollbackRNG(s Qwen38RNGState) {
	v.prng.Rollback(s)
}

// BeginTx starts an explicit RNG transaction.
func (v *Qwen38SampledSpeculativeVerifier) BeginTx() *Qwen38RNGTx {
	return &Qwen38RNGTx{
		rng:        v.prng,
		checkpoint: v.prng.Checkpoint(),
	}
}

// AcceptanceProbability computes Leviathan-Chen acceptance probability:
//
//	alpha(x) = min(1, P_target(x) / P_draft(x))
func (v *Qwen38SampledSpeculativeVerifier) AcceptanceProbability(pTarget, pDraft float32) float32 {
	if pDraft <= 0 {
		if pTarget > 0 {
			return 1.0
		}
		return 0.0
	}
	ratio := pTarget / pDraft
	if ratio > 1.0 {
		return 1.0
	}
	if ratio < 0.0 {
		return 0.0
	}
	return ratio
}

// ResidualDistribution computes the normalized residual distribution upon rejection:
//
//	p_res(x) = max(0, p(x) - q(x)) / sum_{x'} max(0, p(x') - q(x'))
//
// If distributions are identical (denominator <= 0), it falls back to pTarget.
// Strictly normalizes the distribution so that sum_x p_res(x) == 1.0 within numerical precision.
func (v *Qwen38SampledSpeculativeVerifier) ResidualDistribution(pTarget, pDraft []float32) []float32 {
	n := len(pTarget)
	if len(pDraft) < n {
		n = len(pDraft)
	}
	residual := make([]float32, n)
	var sum float64
	maxIdx := 0
	maxVal := float32(-1.0)

	for i := 0; i < n; i++ {
		diff := float64(pTarget[i]) - float64(pDraft[i])
		if diff > 0 {
			residual[i] = float32(diff)
			sum += diff
			if residual[i] > maxVal {
				maxVal = residual[i]
				maxIdx = i
			}
		} else {
			residual[i] = 0
		}
	}

	if sum <= 1e-12 {
		copyDist := make([]float32, n)
		copy(copyDist, pTarget[:n])
		return copyDist
	}

	invSum := 1.0 / sum
	var checkSum float64
	for i := range residual {
		normalized := float64(residual[i]) * invSum
		residual[i] = float32(normalized)
		checkSum += normalized
	}

	// Correct any float rounding discrepancy to ensure exact sum-to-one invariant
	discrepancy := 1.0 - checkSum
	if math.Abs(discrepancy) > 0 {
		residual[maxIdx] += float32(discrepancy)
	}

	return residual
}

// SampleFromDistribution draws a token index from dist using uniform sample u in [0, 1).
func (v *Qwen38SampledSpeculativeVerifier) SampleFromDistribution(dist []float32, u float32) int {
	if len(dist) == 0 {
		return 0
	}
	if u < 0 {
		u = 0
	}
	if u >= 1.0 {
		u = 0.9999999
	}
	u64 := float64(u)
	var cum float64
	for i, p := range dist {
		cum += float64(p)
		if u64 < cum {
			return i
		}
	}
	return len(dist) - 1
}

// EffectiveDistribution calculates the exact theoretical output distribution:
//
//	P_effective(x) = q(x) * min(1, p(x)/q(x)) + (1 - alpha_total) * p_res(x)
//
// Leviathan & Chen (2023) Theorem 1 proves that P_effective(x) == P_target(x) for all x.
func (v *Qwen38SampledSpeculativeVerifier) EffectiveDistribution(pTarget, pDraft []float32) []float32 {
	n := len(pTarget)
	if len(pDraft) < n {
		n = len(pDraft)
	}
	effective := make([]float32, n)
	var alphaTotal float64
	for i := 0; i < n; i++ {
		pT := float64(pTarget[i])
		pD := float64(pDraft[i])
		alphaTotal += math.Min(pT, pD)
	}
	residual := v.ResidualDistribution(pTarget, pDraft)
	probReject := 1.0 - alphaTotal
	if probReject < 0 {
		probReject = 0
	}
	for i := 0; i < n; i++ {
		pT := float64(pTarget[i])
		pD := float64(pDraft[i])
		var acceptMass float64
		if pD > 0 {
			acceptProb := math.Min(1.0, pT/pD)
			acceptMass = pD * acceptProb
		}
		residualMass := probReject * float64(residual[i])
		effective[i] = float32(acceptMass + residualMass)
	}
	return effective
}

// ApplySamplerToLogits converts raw logits into a normalized probability distribution
// adhering to the configured temperature and Top-P filtering.
func (v *Qwen38SampledSpeculativeVerifier) ApplySamplerToLogits(logits []float32) ([]float32, error) {
	if len(logits) == 0 {
		return nil, errors.New("model: empty logits vector")
	}

	// Greedy mode: argmax gets probability 1.0, rest 0.0
	if v.config.Greedy || v.config.Temperature == 0 {
		out := make([]float32, len(logits))
		bestIdx := 0
		bestVal := logits[0]
		for i := 1; i < len(logits); i++ {
			if logits[i] > bestVal {
				bestVal = logits[i]
				bestIdx = i
			}
		}
		out[bestIdx] = 1.0
		return out, nil
	}

	// Temperature scaling
	temp := float64(v.config.Temperature)
	scaled := make([]float64, len(logits))
	maxVal := float64(logits[0]) / temp
	for i, l := range logits {
		val := float64(l) / temp
		scaled[i] = val
		if val > maxVal {
			maxVal = val
		}
	}

	// Numerically stable softmax
	var sum float64
	probs := make([]float32, len(logits))
	for i, s := range scaled {
		e := math.Exp(s - maxVal)
		probs[i] = float32(e)
		sum += e
	}
	invSum := 1.0 / sum
	for i := range probs {
		probs[i] = float32(float64(probs[i]) * invSum)
	}

	// Top-P (nucleus) filtering if TopP < 1.0
	if v.config.TopP > 0 && v.config.TopP < 1.0 {
		type pair struct {
			idx  int
			prob float32
		}
		pairs := make([]pair, len(probs))
		for i, p := range probs {
			pairs[i] = pair{idx: i, prob: p}
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].prob > pairs[j].prob
		})

		cutoff := float64(v.config.TopP)
		var cum float64
		keep := make(map[int]bool)
		var keptSum float64
		for _, p := range pairs {
			keep[p.idx] = true
			cum += float64(p.prob)
			keptSum += float64(p.prob)
			if cum >= cutoff {
				break
			}
		}

		filtered := make([]float32, len(probs))
		invKept := 1.0 / keptSum
		for i := range filtered {
			if keep[i] {
				filtered[i] = float32(float64(probs[i]) * invKept)
			}
		}
		return filtered, nil
	}

	return probs, nil
}

// VerifyToken verifies a single draft token position against target and draft distributions.
// If randVal >= 0, it is used for the uniform acceptance test; otherwise the PRNG is invoked.
func (v *Qwen38SampledSpeculativeVerifier) VerifyToken(
	token int,
	pTarget, pDraft []float32,
	randVal float32,
) Qwen38SampledTokenResult {
	// Greedy recovery mode: exact argmax verification
	if v.config.Greedy || v.config.Temperature == 0 {
		targetArgmax := 0
		maxTarget := float32(-1.0)
		for i, p := range pTarget {
			if p > maxTarget {
				maxTarget = p
				targetArgmax = i
			}
		}
		if token == targetArgmax {
			return Qwen38SampledTokenResult{
				Accepted:         true,
				DraftToken:       token,
				ReplacementToken: -1,
				Alpha:            1.0,
				UniformSample:    0.0,
			}
		}
		residual := make([]float32, len(pTarget))
		if targetArgmax < len(residual) {
			residual[targetArgmax] = 1.0
		}
		return Qwen38SampledTokenResult{
			Accepted:         false,
			DraftToken:       token,
			ReplacementToken: targetArgmax,
			Alpha:            0.0,
			UniformSample:    1.0,
			ResidualDist:     residual,
		}
	}

	if randVal < 0 {
		randVal = v.prng.NextFloat32()
	}

	pT := float32(0)
	if token >= 0 && token < len(pTarget) {
		pT = pTarget[token]
	}
	pD := float32(0)
	if token >= 0 && token < len(pDraft) {
		pD = pDraft[token]
	}

	alpha := v.AcceptanceProbability(pT, pD)
	if randVal < alpha {
		return Qwen38SampledTokenResult{
			Accepted:         true,
			DraftToken:       token,
			ReplacementToken: -1,
			Alpha:            alpha,
			UniformSample:    randVal,
		}
	}

	// Rejection: sample replacement token from residual distribution
	residual := v.ResidualDistribution(pTarget, pDraft)
	uReplace := v.prng.NextFloat32()
	replacement := v.SampleFromDistribution(residual, uReplace)

	return Qwen38SampledTokenResult{
		Accepted:         false,
		DraftToken:       token,
		ReplacementToken: replacement,
		Alpha:            alpha,
		UniformSample:    randVal,
		ResidualDist:     residual,
	}
}

// CalculateMetrics computes information-theoretic and acceptance metrics between target and draft.
func (v *Qwen38SampledSpeculativeVerifier) CalculateMetrics(
	pTarget, pDraft []float32,
	alphas []float32,
) Qwen38SampledDistributionMetrics {
	n := len(pTarget)
	if len(pDraft) < n {
		n = len(pDraft)
	}

	var tvSum, minSum, resSum, hTarget, hDraft, klSum float64
	for i := 0; i < n; i++ {
		pT := float64(pTarget[i])
		pD := float64(pDraft[i])
		diff := math.Abs(pT - pD)
		tvSum += diff
		minSum += math.Min(pT, pD)
		if pT > pD {
			resSum += (pT - pD)
		}
		if pT > 1e-12 {
			hTarget -= pT * math.Log2(pT)
		}
		if pD > 1e-12 {
			hDraft -= pD * math.Log2(pD)
		}
		if pT > 1e-12 && pD > 1e-12 {
			klSum += pT * math.Log2(pT/pD)
		}
	}

	perStepAlpha := make([]float32, len(alphas))
	copy(perStepAlpha, alphas)

	return Qwen38SampledDistributionMetrics{
		TotalVariationDistance: 0.5 * tvSum,
		ExpectedAcceptanceRate: minSum,
		ResidualMass:           resSum,
		TargetEntropy:          hTarget,
		DraftEntropy:           hDraft,
		KLDivergence:           klSum,
		PerStepAlpha:           perStepAlpha,
		PerStepAcceptProb:      perStepAlpha,
	}
}

// VerifyDraftSequence verifies a sequence of draft tokens against per-position target
// and draft distributions. Rejection halts speculative drafting and triggers residual correction.
// Explicit PRNG transaction rollback is executed on transaction abort.
func (v *Qwen38SampledSpeculativeVerifier) VerifyDraftSequence(
	draftTokens []int,
	pTargets [][]float32,
	pDrafts [][]float32,
) Qwen38SampledVerificationReceipt {
	k := len(draftTokens)
	initialSteps := v.prng.StepsConsumed()
	tx := v.BeginTx()

	mode := "sampled"
	if v.config.Greedy || v.config.Temperature == 0 {
		mode = "greedy"
	}

	receipt := Qwen38SampledVerificationReceipt{
		SchemaVersion:    Qwen38SampledVerificationReceiptSchema,
		Engine:           Qwen38EngineMTP,
		SamplingMode:     mode,
		Temperature:      v.config.Temperature,
		TopP:             v.config.TopP,
		RequestedDepth:   k,
		EffectiveDepth:   0,
		AcceptedTokens:   make([]int, 0, k),
		ReplacementToken: -1,
		RejectedAt:       -1,
		BonusToken:       -1,
	}

	alphas := make([]float32, 0, k)
	distribution := make([]Qwen38MTPAcceptanceBucket, 0, k)

	for i := 0; i < k; i++ {
		var targetDist, draftDist []float32
		if i < len(pTargets) {
			targetDist = pTargets[i]
		}
		if i < len(pDrafts) {
			draftDist = pDrafts[i]
		}

		ver := v.VerifyToken(draftTokens[i], targetDist, draftDist, -1)
		alphas = append(alphas, ver.Alpha)

		bucket := Qwen38MTPAcceptanceBucket{
			Depth:    i + 1,
			Proposed: 1,
		}

		if ver.Accepted {
			bucket.Accepted = 1
			receipt.AcceptedTokens = append(receipt.AcceptedTokens, draftTokens[i])
			distribution = append(distribution, bucket)
		} else {
			bucket.Rejected = 1
			receipt.RejectedAt = i
			receipt.ReplacementToken = ver.ReplacementToken
			distribution = append(distribution, bucket)
			break
		}
	}

	receipt.EffectiveDepth = len(distribution)

	// If all accepted and bonus distribution is supplied, sample bonus token
	if receipt.RejectedAt == -1 {
		if len(pTargets) > k {
			uBonus := v.prng.NextFloat32()
			receipt.BonusToken = v.SampleFromDistribution(pTargets[k], uBonus)
		} else if len(pTargets) > 0 {
			uBonus := v.prng.NextFloat32()
			receipt.BonusToken = v.SampleFromDistribution(pTargets[len(pTargets)-1], uBonus)
		}
	}

	proposed := len(distribution)
	accepted := len(receipt.AcceptedTokens)
	rejected := proposed - accepted

	receipt.Tokens = Qwen38MTPTokenAccounting{
		Proposed:     proposed,
		Accepted:     accepted,
		Rejected:     rejected,
		Distribution: distribution,
	}

	// Compute metrics across evaluated distributions
	var pT0, pD0 []float32
	if len(pTargets) > 0 {
		pT0 = pTargets[0]
	}
	if len(pDrafts) > 0 {
		pD0 = pDrafts[0]
	}
	receipt.Metrics = v.CalculateMetrics(pT0, pD0, alphas)

	if accepted > 0 {
		receipt.Outcome = Qwen38MTPOutcomeSucceeded
		receipt.DowngradeReason = Qwen38MTPEligible
		receipt.FailureReason = Qwen38MTPFailureNone
	} else {
		receipt.Outcome = Qwen38MTPOutcomeFailed
		receipt.DowngradeReason = Qwen38MTPAttemptFailed
		receipt.FailureReason = Qwen38MTPVerificationFailed
	}

	// If rollback on abort is requested and transaction failed completely, roll back PRNG
	if v.config.RollbackOnAbort && accepted == 0 {
		tx.Rollback()
	} else {
		tx.Commit()
	}

	receipt.RNGStepsConsumed = v.prng.StepsConsumed() - initialSteps
	return receipt
}

// VerifyDraftSequenceLogits applies the configured sampler to raw target and draft logits
// before executing speculative verification.
func (v *Qwen38SampledSpeculativeVerifier) VerifyDraftSequenceLogits(
	draftTokens []int,
	targetLogits, draftLogits [][]float32,
) (Qwen38SampledVerificationReceipt, error) {
	pTargets := make([][]float32, len(targetLogits))
	for i, logits := range targetLogits {
		dist, err := v.ApplySamplerToLogits(logits)
		if err != nil {
			return Qwen38SampledVerificationReceipt{}, fmt.Errorf("model: target logits position %d: %w", i, err)
		}
		pTargets[i] = dist
	}

	pDrafts := make([][]float32, len(draftLogits))
	for i, logits := range draftLogits {
		dist, err := v.ApplySamplerToLogits(logits)
		if err != nil {
			return Qwen38SampledVerificationReceipt{}, fmt.Errorf("model: draft logits position %d: %w", i, err)
		}
		pDrafts[i] = dist
	}

	return v.VerifyDraftSequence(draftTokens, pTargets, pDrafts), nil
}
