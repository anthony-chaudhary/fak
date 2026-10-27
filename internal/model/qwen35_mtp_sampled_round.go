package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
)

// ErrQwen35MTPSampledRoundActive requires the current round to finish first.
var ErrQwen35MTPSampledRoundActive = errors.New("model: Qwen3.8 MTP sampled round is already active")

// ErrQwen35MTPSampledNoProposal permits ordinary target-only sampling at this
// boundary: the first draft row was refused before any proposal RNG draw.
// Other round errors do NOT certify safe target-only fallback.
var ErrQwen35MTPSampledNoProposal = errors.New("model: Qwen3.8 MTP sampled first row is unsupported")

// Qwen35MTPSampledRound owns a sampled proposal and the corresponding live
// target verification. The caller must Commit or Abort before using either
// session again. Like Session, it requires exclusive, nonconcurrent ownership.
// It does not admit sampled requests into the serving planner.
type Qwen35MTPSampledRound struct {
	draft        *Qwen35MTPDraftSession
	target       *qwen35MTPTargetTransaction
	rng          Qwen38RNGState
	plan         SpecRejectionPlan
	closed       bool
	committed    bool
	targetLogits [][]float32
	draftElapsed time.Duration
}

// BeginSampledRound joins the native sampled MTP producer to exact K+1 target
// evidence and the keyed lossless rejection planner. Target state is provisional
// until Commit. A clean cancellation or Abort restores the proposal RNG and
// target state, so a retry with the same key and offset makes the same decision.
// Target or rollback failures latch an error and forbid session reuse.
// beforeLogits must be the unmodified target logits for committed. No grammar,
// logit bias or additional target score transform is supported by this seam.
// Target p uses the configured Qwen38 model sampler, not the ordinary agent
// sampler's float64/stable-tie policy; this API does not establish serving parity.
func (d *Qwen35MTPDraftSession) BeginSampledRound(ctx context.Context, committed []int, beforeLogits []float32, requestKey uint64, stepOffset int) (round *Qwen35MTPSampledRound, err error) {
	return d.BeginSampledRoundWithTargetProbabilities(ctx, committed, beforeLogits, requestKey, stepOffset, func(logits []float32) (SpecProbabilityRow, error) {
		return sampledMTPProbabilityRow(d.sampler, logits)
	})
}

// BeginSampledRoundWithTargetProbabilities uses the caller's canonical row-only
// target sampling policy instead of the Qwen38 model sampler. The callback must
// return full-vocabulary float64 weights and the actual inverse-CDF mass without
// a log/exp conversion, renormalization, drawing randomness,
// publishing output or changing request state. Stateful grammar/constraint
// transactions require a separate caller contract and are not admitted here.
// Both callback input and output are copied, so a reusable sampler workspace
// cannot alter target logits or alias a prior position's evidence.
// Draft row admission precedes every draw. An unsupported future row truncates
// the proposal while preserving its earlier valid prefix. Only the typed
// ErrQwen35MTPSampledNoProposal permits target-only fallback; other errors must
// not discard a sampled prefix and silently restart from the original boundary.
func (d *Qwen35MTPDraftSession) BeginSampledRoundWithTargetProbabilities(ctx context.Context, committed []int, beforeLogits []float32, requestKey uint64, stepOffset int, targetProbabilities func([]float32) (SpecProbabilityRow, error)) (round *Qwen35MTPSampledRound, err error) {
	return d.beginSampledRoundWithTargetProbabilities(ctx, committed, beforeLogits, requestKey, stepOffset, targetProbabilities, 0)
}

// BeginSampledRoundWithTargetProbabilitiesLimit preserves the canonical-policy
// transaction while bounding proposal growth by a positive per-call maxDraft.
// It never changes the configured depth. Metal retains its existing depth: its
// exact P=4 panel must not be redirected to another verifier by this ceiling.
func (d *Qwen35MTPDraftSession) BeginSampledRoundWithTargetProbabilitiesLimit(ctx context.Context, committed []int, beforeLogits []float32, requestKey uint64, stepOffset, maxDraft int, targetProbabilities func([]float32) (SpecProbabilityRow, error)) (round *Qwen35MTPSampledRound, err error) {
	if maxDraft <= 0 {
		return nil, ErrQwen35MTPInvalidDraftLength
	}
	return d.beginSampledRoundWithTargetProbabilities(ctx, committed, beforeLogits, requestKey, stepOffset, targetProbabilities, maxDraft)
}

func (d *Qwen35MTPDraftSession) beginSampledRoundWithTargetProbabilities(ctx context.Context, committed []int, beforeLogits []float32, requestKey uint64, stepOffset int, targetProbabilities func([]float32) (SpecProbabilityRow, error), maxDraft int) (round *Qwen35MTPSampledRound, err error) {
	if d == nil || d.closed {
		return nil, ErrQwen35MTPDrafterClosed
	}
	if d.runtimeErr != nil {
		return nil, d.runtimeErr
	}
	if d.sampledRound != nil {
		return nil, ErrQwen35MTPSampledRoundActive
	}
	if d.sampler == nil {
		return nil, ErrDraftEvidenceNotProcessedSamplingQ
	}
	if targetProbabilities == nil {
		return nil, fmt.Errorf("%w: target probability processor is required", ErrSpecRejectionProbability)
	}
	cfg := d.sampler.Config()
	if !(cfg.Temperature > 0) || math.IsInf(float64(cfg.Temperature), 0) || !(cfg.TopP > 0 && cfg.TopP <= 1) || d.effectiveThreshold() > 0 {
		return nil, &Qwen35MTPSpecDecodeUnsupportedError{Reason: "sampled rounds require finite positive temperature, top-p in (0,1], and no proposal coverage gate"}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if requestKey == 0 {
		return nil, ErrSpecRejectionUnkeyed
	}
	depth := d.depth
	if maxDraft > 0 && d.target != nil && !d.target.Metal && !d.target.MetalQ4K {
		depth = min(depth, maxDraft)
	}
	maxInt := int(^uint(0) >> 1)
	if stepOffset < 0 || stepOffset >= maxInt || depth > maxInt-stepOffset-1 {
		return nil, fmt.Errorf("%w: sampled round step range", ErrSpecRejectionShape)
	}
	if len(committed) != qwen35MTPResidentPrefixLen(d.target) || !qwen35MTPDepthNTargetHasEvaluatedPrefix(d.target, committed) {
		return nil, fmt.Errorf("%w: target does not match the committed prefix", ErrSpecRejectionShape)
	}
	vocab := d.target.M.Cfg.VocabSize
	processTarget := func(logits []float32) (SpecProbabilityRow, error) {
		if len(logits) != vocab || vocab == 0 {
			return SpecProbabilityRow{}, fmt.Errorf("%w: target vocabulary width", ErrSpecRejectionProbability)
		}
		for _, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 1) {
				return SpecProbabilityRow{}, fmt.Errorf("%w: non-finite target logit", ErrSpecRejectionProbability)
			}
		}
		row, err := targetProbabilities(append([]float32(nil), logits...))
		if err != nil {
			return SpecProbabilityRow{}, err
		}
		if err := validateSpecProbabilityRow(row, vocab); err != nil {
			return SpecProbabilityRow{}, fmt.Errorf("%w: %v", ErrSpecRejectionProbability, err)
		}
		return SpecProbabilityRow{Weights: append([]float64(nil), row.Weights...), Mass: row.Mass}, nil
	}
	boundary, err := processTarget(beforeLogits)
	if err != nil {
		return nil, err
	}

	r := &Qwen35MTPSampledRound{draft: d, rng: d.sampler.CheckpointRNG()}
	published := false
	defer func() {
		if !published {
			abortErr := r.Abort()
			if abortErr != nil && err == ErrQwen35MTPSampledNoProposal {
				// Failed cleanup must not retain the safe-fallback marker.
				err = abortErr
			} else {
				err = errors.Join(err, abortErr)
			}
		}
	}()
	draftStarted := time.Now()
	tokens, evidence := d.proposeWithLimit(committed, true, true, depth)
	r.draftElapsed = time.Since(draftStarted)
	if err := d.Err(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, ErrQwen35MTPSampledNoProposal
	}
	proposal := DraftProposal{Tokens: tokens, DraftEvidence: evidence}
	evidence, err = proposal.StochasticDraftEvidence()
	if err != nil {
		return nil, err
	}
	for _, row := range evidence {
		// Refusing a sampled token outside a full-logit subset changes the
		// proposal law. Only a genuinely compact, normalized q is admissible.
		if d.vocabFilter != nil && len(d.vocabFilter.Subset) > 0 && len(row.TokenIDs) == 0 {
			return nil, &Qwen35MTPSpecDecodeUnsupportedError{Reason: "sampled vocabulary filtering requires compact processed-q coordinates"}
		}
		if err := validateDraftProbabilityCoordinates(row, vocab); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrSpecRejectionProbability, err)
		}
	}
	d.sampledRound = r
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.target, err = beginQwen35MTPTargetTransaction(d.target, beforeLogits)
	if err != nil {
		return nil, err
	}
	rows, err := r.target.Verify(tokens)
	if err != nil {
		d.latch("sampled target verification", -1, err)
		return nil, err
	}
	if len(rows) != len(tokens) {
		return nil, ErrSpecRejectionTargetRows
	}
	r.targetLogits = make([][]float32, len(rows))
	for i, row := range rows {
		r.targetLogits[i] = append([]float32(nil), row...)
	}
	processed := make([]SpecProbabilityRow, len(tokens)+1)
	processed[0] = boundary
	for i, row := range rows {
		processed[i+1], err = processTarget(row)
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.plan, err = BuildSpecRejectionPlan(SpecRejectionPlanInput{
		DraftTokens: tokens, DraftEvidence: evidence, TargetProbabilityRows: processed,
		RequestKey: requestKey, StepOffset: stepOffset, ResidualMode: SpecResidualLossless,
	})
	if err != nil {
		return nil, err
	}
	published = true
	return r, nil
}

// Plan returns an independent decision. It does not publish tokens or advance
// the caller's keyed step offset; advance to NextStep only after committing.
func (r *Qwen35MTPSampledRound) Plan() SpecRejectionPlan {
	if r == nil {
		return SpecRejectionPlan{}
	}
	plan := r.plan
	plan.AcceptedTokens = append([]int(nil), plan.AcceptedTokens...)
	plan.AcceptProb = append([]float32(nil), plan.AcceptProb...)
	return plan
}

// TargetLogits returns independent copies of the K verification rows, where row
// i is the target boundary after draft token i. The pre-round boundary is not
// included. Rows remain available after termination and do not certify commit.
func (r *Qwen35MTPSampledRound) TargetLogits() [][]float32 {
	if r == nil {
		return nil
	}
	rows := make([][]float32, len(r.targetLogits))
	for i, row := range r.targetLogits {
		rows[i] = append([]float32(nil), row...)
	}
	return rows
}

// VerificationReceipt exposes the actual target transaction receipt only after
// successful Commit, including its actual retained-prefix repair/rollback costs.
// Before commit or after abort/failure it returns the zero receipt, not success.
func (r *Qwen35MTPSampledRound) VerificationReceipt() TargetVerificationReceipt {
	if r == nil || !r.committed {
		return TargetVerificationReceipt{}
	}
	receipt := r.target.VerificationReceipt()
	if receipt.Shape != nil {
		shape := *receipt.Shape
		shape.TokensPerSeq = append([]int(nil), shape.TokensPerSeq...)
		receipt.Shape = &shape
	}
	return receipt
}

// DraftElapsed measures only the actual proposal call (including draft catch-up
// and its local checkpoint work), not target processing, verification or commit.
func (r *Qwen35MTPSampledRound) DraftElapsed() time.Duration {
	if r == nil {
		return 0
	}
	return r.draftElapsed
}

// Commit adopts only an emitted prefix of Plan().AcceptedTokens, discarding the
// un-emitted accepted suffix as well as rejected draft state. The returned logits
// are the target boundary after that prefix. The correction/bonus token is NOT
// evaluated here: the caller may emit it only after the whole accepted prefix,
// then must Step it before another round. Commit consumes the proposal RNG;
// callers that publish no decision should Abort instead. Commit is deliberately
// not cancelable: after emission, cancellation cannot undo the external tokens.
func (r *Qwen35MTPSampledRound) Commit(accepted int) (logits []float32, err error) {
	if r == nil || r.closed {
		return nil, errors.New("model: Qwen3.8 MTP sampled round is closed")
	}
	committed := false
	defer func() {
		if !committed {
			err = errors.Join(err, r.Abort())
		}
	}()
	if accepted < 0 || accepted > len(r.plan.AcceptedTokens) {
		return nil, fmt.Errorf("%w: unverified sampled prefix %d", ErrSpecRejectionShape, accepted)
	}
	// Restore the draft while the target transaction can still abort on error.
	if err := r.draft.restoreProposal(); err != nil {
		r.draft.latch("sampled draft rollback", -1, err)
		return nil, err
	}
	logits, err = r.target.Commit(accepted)
	if err != nil {
		// The target error does not certify whether recovery succeeded. Refuse
		// further use rather than retry against possibly divergent state.
		r.draft.latch("sampled target commit", -1, err)
		return nil, err
	}
	r.finish(true)
	committed = true
	return logits, nil
}

// Abort discards the provisional target/draft state and restores the proposal
// RNG. It is idempotent, including after a successful Commit.
func (r *Qwen35MTPSampledRound) Abort() (err error) {
	if r == nil || r.closed {
		return nil
	}
	defer r.finish(false)
	// Both resources require cleanup even when either rollback fails.
	defer func() {
		err = errors.Join(err, r.draft.restoreProposal())
		if err != nil {
			r.draft.latch("sampled round rollback", -1, err)
		}
	}()
	if r.target != nil {
		err = r.target.Abort()
	}
	return err
}

func (r *Qwen35MTPSampledRound) finish(committed bool) {
	r.closed = true
	r.committed = committed
	if !committed {
		r.draft.sampler.RollbackRNG(r.rng)
	}
	if r.draft.sampledRound == r {
		r.draft.sampledRound = nil
	}
}

func sampledMTPProbabilityRow(sampler *Qwen38SampledSpeculativeVerifier, logits []float32) (SpecProbabilityRow, error) {
	probabilities, err := sampler.ApplySamplerToLogits(logits)
	if err != nil {
		return SpecProbabilityRow{}, err
	}
	row := SpecProbabilityRow{Weights: make([]float64, len(probabilities))}
	for i, p := range probabilities {
		if p < 0 || math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
			return SpecProbabilityRow{}, ErrSpecRejectionProbability
		}
		row.Weights[i] = float64(p)
		row.Mass += float64(p)
	}
	return row, nil
}
