package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type attemptCostAdvanceBackend struct {
	*mtpDiagnosticBackend
	advanceArmed bool
	delays       int
}

func (b *attemptCostAdvanceBackend) Read(t compute.Tensor) []float32 {
	if b.advanceArmed && len(t.Shape) == 1 && t.Shape[0] == b.vocab {
		b.advanceArmed = false
		b.delays++
		*b.now = b.now.Add(7 * time.Millisecond)
	}
	return b.mtpDiagnosticBackend.Read(t)
}

type attemptCostView struct {
	Attempts           int           `json:"attempts"`
	EmittedTokens      int           `json:"emitted_tokens"`
	MeasuredAttempts   int           `json:"measured_attempts"`
	UnmeasuredAttempts int           `json:"unmeasured_attempts"`
	DraftElapsed       time.Duration `json:"draft_elapsed_ns"`
	InclusiveElapsed   time.Duration `json:"inclusive_elapsed_ns"`
}

type speculativeCostView struct {
	Scope             string          `json:"scope"`
	Attempts          int             `json:"attempts"`
	Completed         attemptCostView `json:"completed"`
	ProposalError     attemptCostView `json:"proposal_error"`
	EmptyProposal     attemptCostView `json:"empty_proposal"`
	VerificationError attemptCostView `json:"verification_error"`
	Cancelled         attemptCostView `json:"cancelled"`
	Incomplete        attemptCostView `json:"incomplete"`
}

func readSpeculativeCost(t *testing.T, res inKernelGenerateResult) speculativeCostView {
	t.Helper()
	return readNativeAttemptCost(t, newNativeDecodeSummary(res))
}

func readNativeAttemptCost(t *testing.T, sum *NativeDecodeSummary) speculativeCostView {
	t.Helper()
	raw, err := json.Marshal(sum)
	if err != nil {
		t.Fatal(err)
	}
	var summary struct {
		Cost *speculativeCostView `json:"speculative_cost"`
	}
	if err := json.Unmarshal(raw, &summary); err != nil {
		t.Fatal(err)
	}
	if summary.Cost == nil {
		t.Fatal("normal decode summary omitted actual speculative attempt costs")
	}
	if summary.Cost.Scope != "proposal-through-emission-and-advance" {
		t.Fatalf("cost scope = %q", summary.Cost.Scope)
	}
	c := *summary.Cost
	if c.Attempts != c.Completed.Attempts+c.ProposalError.Attempts+c.EmptyProposal.Attempts+c.VerificationError.Attempts+c.Cancelled.Attempts+c.Incomplete.Attempts {
		t.Fatal("attempts do not belong to exactly one terminal outcome")
	}
	return c
}

// fak-test:runtime fast est=1s lane=default
func TestSpeculativeAttemptCostCompletePreservesCancelledMTP(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	tok := atomicThinkingTokenizer(t)
	cfg := speculativeHybridConfig()
	cfg.VocabSize = tok.Vocab()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	now := time.Unix(100, 0)
	base := newVulkanMTPPlannerBackend(t)
	base.vocab = cfg.VocabSize
	backend := &mtpDiagnosticBackend{vulkanMTPPlannerBackend: base, now: &now}
	p := NewInKernelPlanner(m, tok, "attempt-cost-completion", false, backend, false)
	if err := p.EnableVulkanMTP(1); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.WithValue(context.Background(), vulkanMTPDiagnosticClockKey{}, func() time.Time { return now }))
	defer cancel()
	calls := 0
	p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			calls++
			now = now.Add(3 * time.Millisecond)
			cancel()
			return nil, context.Canceled
		}), func() {}, nil
	}
	zero := 0.0
	comp, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "x"}}, nil, WithMaxTokens(3), WithTemperature(&zero))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Complete error = %v, expected cancellation sentinel", err)
	}
	if comp == nil || comp.NativeDecode == nil {
		t.Fatal("Complete discarded request-local accounting on its error path")
	}
	if comp.Model != "attempt-cost-completion" {
		t.Fatalf("error Completion model = %q", comp.Model)
	}
	c := readNativeAttemptCost(t, comp.NativeDecode)
	if calls != 1 || c.Attempts != 1 || c.Cancelled.Attempts != 1 || c.Cancelled.MeasuredAttempts != 1 || c.Cancelled.DraftElapsed != 3*time.Millisecond || c.Cancelled.InclusiveElapsed != 3*time.Millisecond || c.Completed.Attempts != 0 {
		t.Fatal("Complete cancellation lost the paid proposal or mislabeled it successful")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestSpeculativeAttemptCostPostVerificationFailureIsIncomplete(t *testing.T) {
	p, prompt, ctx, _ := mtpDiagnosticPlanner(t, 4, []int{7, 7, 7, 8})
	measurement := &nativeInferenceMeasurement{
		inferenceDisabled: true,
		traceNow:          decodeTraceClock(0, -time.Nanosecond),
	}
	res, err := p.generateReusedRecovering(ctx, prompt, 4, 0, 0, 0, nil, 0, 0, nil, nil, measurement)
	if err == nil || errors.Is(err, context.Canceled) || p.backend.(*mtpDiagnosticBackend).verificationCalls != 1 {
		t.Fatal("test did not reach a non-cancellation failure after actual target verification")
	}
	c := readSpeculativeCost(t, res)
	if c.Attempts != 1 || c.Completed.Attempts != 0 || c.Incomplete.Attempts != 1 || c.Incomplete.EmittedTokens != 1 || c.Incomplete.MeasuredAttempts != 1 || c.Incomplete.InclusiveElapsed != 13*time.Millisecond {
		t.Fatal("post-verification measurement failure was credited as completed work")
	}
}

// fak-test:runtime medium est=2s lane=default
func TestSpeculativeAttemptCostRecoveredInterruptionIsIncomplete(t *testing.T) {
	for _, stage := range []string{"after-verification", "draft-interrupted", "later-draft-interrupted"} {
		t.Run(stage, func(t *testing.T) {
			p, prompt, ctx, _ := mtpDiagnosticPlanner(t, 4, []int{7, 7, 7, 8})
			b := p.backend.(*mtpDiagnosticBackend)
			duringDraft, priorRound := stage != "after-verification", stage == "later-draft-interrupted"
			calls := 0
			if duringDraft {
				p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
					return model.NewMTPProposalGeneratorWithFn(func(_ context.Context, _ []int, maxDraft int) ([]int, error) {
						calls++
						*b.now = b.now.Add(3 * time.Millisecond)
						if !priorRound || calls == 2 {
							panic(model.ErrV4FlashAttentionUnimplemented)
						}
						draft := []int{7, 7, 7, 8}
						return draft[:min(maxDraft, len(draft))], nil
					}), func() {}, nil
				}
			}
			maxNew := 4
			if priorRound {
				maxNew = 8
			}
			res, err := p.generateReusedRecovering(ctx, prompt, maxNew, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
				if !duringDraft {
					panic(model.ErrV4FlashAttentionUnimplemented)
				}
				return false
			})
			if !errors.Is(err, model.ErrV4FlashAttentionUnimplemented) {
				t.Fatal("recoverable failure lost its sentinel")
			}
			c := readSpeculativeCost(t, res)
			wantAttempts, wantCompleted := 1, 0
			if priorRound {
				wantAttempts, wantCompleted = 2, 1
			}
			if c.Attempts != wantAttempts || c.Completed.Attempts != wantCompleted || c.Incomplete.Attempts != 1 || c.Incomplete.EmittedTokens != 0 {
				t.Fatal("recovered interruption lost paid work or credited successful work")
			}
			summary := newNativeDecodeSummary(res)
			if stage == "draft-interrupted" {
				if summary.Path != NativeDecodePathSerial || summary.Speculative != nil {
					t.Fatal("interrupted proposal invented a verified round")
				}
			} else if summary.Path != NativeDecodePathSpeculative || summary.Speculative == nil || summary.Speculative.Rounds != 1 || summary.Speculative.DraftTokens != 4 || summary.Speculative.AcceptedTokens != 3 {
				t.Fatal("recovery discarded known verification provenance or invented another round")
			}
			if priorRound && (res.gen != 4 || c.Completed.EmittedTokens != 4 || c.Completed.InclusiveElapsed != 13*time.Millisecond) {
				t.Fatal("later interrupted proposal changed the preceding closed round")
			}
			if duringDraft {
				if b.verificationCalls != wantCompleted || c.Incomplete.MeasuredAttempts != 0 || c.Incomplete.UnmeasuredAttempts != 1 || c.Incomplete.DraftElapsed != 0 || c.Incomplete.InclusiveElapsed != 0 {
					t.Fatal("interrupted proposal with unknown draft cost became a cheap measured sample")
				}
			} else if b.verificationCalls != 1 || c.Incomplete.MeasuredAttempts != 1 || c.Incomplete.DraftElapsed != 3*time.Millisecond || c.Incomplete.InclusiveElapsed != 13*time.Millisecond {
				t.Fatal("post-verification interruption lost its known proposal-inclusive cost")
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestSpeculativeAttemptCostIncludesDraftAndEmission(t *testing.T) {
	for _, stopEarly := range []bool{false, true} {
		t.Run(map[bool]string{false: "final-round", true: "partial-stop"}[stopEarly], func(t *testing.T) {
			p, prompt, ctx, calls := mtpDiagnosticPlanner(t, 4, []int{7, 7, 7, 8})
			res, err := p.generateReusedRecovering(ctx, prompt, 4, 0, 0, 0, nil, 0, 0, nil, func(int) bool { return stopEarly })
			if err != nil {
				t.Fatal(err)
			}
			c := readSpeculativeCost(t, res)
			wantTokens := 4
			if stopEarly {
				wantTokens = 1
			}
			if *calls != 1 || c.Attempts != 1 || c.Completed.Attempts != 1 || c.Completed.EmittedTokens != wantTokens || c.Completed.MeasuredAttempts != 1 || c.Completed.UnmeasuredAttempts != 0 {
				t.Fatalf("calls/attempts/completed/emitted/measured/unmeasured = %d/%d/%d/%d/%d/%d", *calls, c.Attempts, c.Completed.Attempts, c.Completed.EmittedTokens, c.Completed.MeasuredAttempts, c.Completed.UnmeasuredAttempts)
			}
			if c.Completed.DraftElapsed != 3*time.Millisecond || c.Completed.InclusiveElapsed != 13*time.Millisecond {
				t.Fatalf("draft/inclusive cost = %s/%s", c.Completed.DraftElapsed, c.Completed.InclusiveElapsed)
			}
		})
	}
	// Observe the same target output with and without drafting.
	p, prompt, ctx, _ := mtpDiagnosticPlanner(t, 4, []int{7, 7, 7, 8})
	var speculative, serial []int
	if _, err := p.generateReusedRecovering(ctx, prompt, 8, 0, 0, 0, nil, 0, 0, nil, func(tok int) bool { speculative = append(speculative, tok); return false }); err != nil {
		t.Fatal(err)
	}
	p.DisableSpeculativeDecoding()
	if _, err := p.generateReusedRecovering(ctx, prompt, 8, 0, 0, 0, nil, 0, 0, nil, func(tok int) bool { serial = append(serial, tok); return false }); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(speculative, serial) {
		t.Fatalf("drafting changed target tokens: %v versus %v", speculative, serial)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestSpeculativeAttemptCostIncludesBonusAdvanceAndEmission(t *testing.T) {
	now := time.Unix(100, 0)
	b := &attemptCostAdvanceBackend{mtpDiagnosticBackend: &mtpDiagnosticBackend{vulkanMTPPlannerBackend: newVulkanMTPPlannerBackend(t), now: &now}}
	p, prompt := vulkanMTPPlanner(t, b)
	if err := p.EnableVulkanMTP(4); err != nil {
		t.Fatal(err)
	}
	calls := 0
	p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
		return model.NewMTPProposalGeneratorWithFn(func(_ context.Context, _ []int, maxDraft int) ([]int, error) {
			calls++
			now = now.Add(3 * time.Millisecond)
			draft := []int{7, 7, 7, 8}
			return draft[:min(maxDraft, len(draft))], nil
		}), func() {}, nil
	}
	ctx := context.WithValue(context.Background(), vulkanMTPDiagnosticClockKey{}, func() time.Time { return now })
	emitted := 0
	res, err := p.generateReusedRecovering(ctx, prompt, 5, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
		emitted++
		now = now.Add(2 * time.Millisecond)
		if emitted == 4 {
			b.advanceArmed = true
		}
		return false
	})
	if err != nil {
		t.Fatal(err)
	}
	c := readSpeculativeCost(t, res)
	if calls != 2 || emitted != 5 || b.delays != 1 || b.advanceArmed || c.Attempts != 2 || c.Completed.EmittedTokens != 5 || c.Completed.MeasuredAttempts != 2 {
		t.Fatalf("calls/emitted/advance-delays/armed/attempts/counted-emits/measured = %d/%d/%d/%t/%d/%d/%d", calls, emitted, b.delays, b.advanceArmed, c.Attempts, c.Completed.EmittedTokens, c.Completed.MeasuredAttempts)
	}
	if c.Completed.DraftElapsed != 6*time.Millisecond || c.Completed.InclusiveElapsed != 43*time.Millisecond {
		t.Fatalf("draft/inclusive = %s/%s, expected 6ms/43ms including emission and actual bonus advance", c.Completed.DraftElapsed, c.Completed.InclusiveElapsed)
	}
}

// fak-test:runtime medium est=3s lane=default
func TestSpeculativeAttemptCostSeparatesFailuresAndFallback(t *testing.T) {
	draftFailure := errors.New("injected proposal failure")
	verifyFailure := errors.New("injected verification failure")
	for _, outcome := range []string{"proposal-error", "empty-proposal", "cancelled", "verification-error"} {
		t.Run(outcome, func(t *testing.T) {
			p, prompt, originalCtx, _ := mtpDiagnosticPlanner(t, 1, []int{7})
			ctx, cancel := context.WithCancel(originalCtx)
			defer cancel()
			b := p.backend.(*mtpDiagnosticBackend)
			calls := 0
			p.vulkanMTPDraftFactory = func(*model.Session, int) (model.ProposalGenerator, func(), error) {
				return model.NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
					calls++
					*b.now = b.now.Add(3 * time.Millisecond)
					switch outcome {
					case "proposal-error":
						return nil, draftFailure
					case "empty-proposal":
						return nil, nil
					case "cancelled":
						cancel()
						return nil, context.Canceled
					default:
						return []int{7}, nil
					}
				}), func() {}, nil
			}
			if outcome == "verification-error" {
				b.verificationErr = verifyFailure
			}
			res, err := p.generateReusedRecovering(ctx, prompt, 3, 0, 0, 0, nil, 0, 0, nil, func(int) bool { *b.now = b.now.Add(7 * time.Millisecond); return false })
			var wantErr error
			if outcome == "cancelled" {
				wantErr = context.Canceled
			} else if outcome == "verification-error" {
				wantErr = verifyFailure
			}
			if !errors.Is(err, wantErr) {
				t.Fatalf("returned error = %v; expected sentinel match %v", err, wantErr)
			}
			c := readSpeculativeCost(t, res)
			var bucket attemptCostView
			wantEmitted, wantElapsed := 1, 10*time.Millisecond
			switch outcome {
			case "proposal-error":
				bucket = c.ProposalError
			case "empty-proposal":
				bucket = c.EmptyProposal
			case "cancelled":
				bucket, wantEmitted, wantElapsed = c.Cancelled, 0, 3*time.Millisecond
			case "verification-error":
				bucket, wantEmitted, wantElapsed = c.VerificationError, 0, 13*time.Millisecond
			}
			if calls != 1 || c.Attempts != 1 || c.Completed.Attempts != 0 || bucket.Attempts != 1 || bucket.EmittedTokens != wantEmitted || bucket.MeasuredAttempts != 1 || bucket.UnmeasuredAttempts != 0 {
				t.Fatalf("calls/attempts/completed/outcome/emitted/measured/unmeasured = %d/%d/%d/%d/%d/%d/%d", calls, c.Attempts, c.Completed.Attempts, bucket.Attempts, bucket.EmittedTokens, bucket.MeasuredAttempts, bucket.UnmeasuredAttempts)
			}
			if bucket.DraftElapsed != 3*time.Millisecond || bucket.InclusiveElapsed != wantElapsed {
				t.Fatalf("draft/inclusive cost = %s/%s; expected 3ms/%s", bucket.DraftElapsed, bucket.InclusiveElapsed, wantElapsed)
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestSpeculativeAttemptCostGenericFailureAndUnmeasuredClock(t *testing.T) {
	for _, tc := range []struct {
		name        string
		draft, emit time.Duration
		measured    bool
	}{
		{"paid-fallback", 3 * time.Millisecond, 7 * time.Millisecond, true},
		{"nonpositive-clock", 0, 0, false},
		{"backwards-clock", -3 * time.Millisecond, 0, false},
		{"inconsistent-span", 3 * time.Millisecond, -2 * time.Millisecond, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Unix(100, 0)
			probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
			probe.after = func() { now = now.Add(tc.draft) }
			p, prompt := speculativeRequestPlanner(t, probe)
			ctx := context.WithValue(context.Background(), vulkanMTPDiagnosticClockKey{}, func() time.Time { return now })
			res, err := p.generateReusedRecovering(ctx, prompt, 3, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
				now = now.Add(tc.emit)
				return false
			})
			if err != nil {
				t.Fatal(err)
			}
			c := readSpeculativeCost(t, res)
			if probe.calls != 3 || c.Attempts != 3 || c.ProposalError.Attempts != 3 || c.ProposalError.EmittedTokens != 3 || c.Completed.Attempts != 0 {
				t.Fatalf("calls/attempts/proposal-errors/emitted/completed = %d/%d/%d/%d/%d", probe.calls, c.Attempts, c.ProposalError.Attempts, c.ProposalError.EmittedTokens, c.Completed.Attempts)
			}
			if tc.measured {
				if c.ProposalError.MeasuredAttempts != 3 || c.ProposalError.UnmeasuredAttempts != 0 || c.ProposalError.DraftElapsed != 9*time.Millisecond || c.ProposalError.InclusiveElapsed != 30*time.Millisecond {
					t.Fatal("generic fallback lost proposal/emission cost")
				}
			} else if c.ProposalError.MeasuredAttempts != 0 || c.ProposalError.UnmeasuredAttempts != 3 || c.ProposalError.DraftElapsed != 0 || c.ProposalError.InclusiveElapsed != 0 {
				t.Fatal("nonpositive clock observations became cheap measured attempts")
			}
		})
	}
}

// fak-test:runtime medium est=3s lane=default
func TestSpeculativeAttemptCostRetentionIsolationAndSerialRemainder(t *testing.T) {
	p, prompt, ctx, _ := mtpDiagnosticPlanner(t, 1, []int{7})
	first, err := p.generateReusedRecovering(ctx, prompt, 514, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := p.generateReusedRecovering(ctx, prompt, 2, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	a, b := readSpeculativeCost(t, first), readSpeculativeCost(t, second)
	if a.Attempts != 257 || a.Completed.EmittedTokens != 514 || a.Completed.DraftElapsed != 257*3*time.Millisecond || a.Completed.InclusiveElapsed != 257*13*time.Millisecond || b.Attempts != 1 || b.Completed.EmittedTokens != 2 {
		t.Fatal("cost retention was capped or a later request changed earlier costs")
	}
	p, prompt, ctx, _ = mtpDiagnosticPlanner(t, 4, []int{7, 8, 8, 8})
	fallback, err := p.generateReusedRecovering(ctx, prompt, 40, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	c := readSpeculativeCost(t, fallback)
	if fallback.gen != 40 || c.Attempts != 8 || c.Completed.Attempts != 8 || c.Completed.EmittedTokens != 16 || c.Completed.InclusiveElapsed != 104*time.Millisecond {
		t.Fatal("acceptance-floor serial remainder was charged as speculative work")
	}
}
