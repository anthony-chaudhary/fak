package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPSampledRound(t *testing.T) {
	ctx := context.Background()
	prompt := []int{0, 1}
	cfg := Qwen38SamplerConfig{Temperature: 0.8, TopP: 1, Seed: 17, Seq: 11}
	samePlan := func(got, want SpecRejectionPlan) bool {
		// Plan's defensive copy may represent an empty token sequence as nil;
		// the pure planner reserves a nonnil empty slice. Compare their values.
		got.AcceptedTokens = append([]int(nil), got.AcceptedTokens...)
		want.AcceptedTokens = append([]int(nil), want.AcceptedTokens...)
		return reflect.DeepEqual(got, want)
	}
	newDraft := func(t *testing.T, compact bool, samplerCfg ...Qwen38SamplerConfig) (*Qwen35MTPDraftSession, []float32) {
		t.Helper()
		selected := cfg
		if len(samplerCfg) > 0 {
			selected = samplerCfg[0]
		}
		m := qwen35MTPEnabledSyntheticModel(t)
		target := m.NewSession()
		t.Cleanup(target.Close)
		d, err := NewQwen35MTPDraftSessionWithConfig(target, Qwen35MTPDraftConfig{Depth: 2, SamplerConfig: &selected})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(d.Close)
		if compact {
			d.WithDraftVocabFilter([]int{2, 0})
		}
		return d, append([]float32(nil), target.Prefill(prompt)...)
	}
	serialTarget := func(t *testing.T, d *Qwen35MTPDraftSession) *Session {
		t.Helper()
		s := d.target.M.NewSession()
		s.captureTargetHidden = true
		t.Cleanup(s.Close)
		s.Prefill(prompt)
		normalizeSnapshotForTest(t, s)
		return s
	}

	t.Run("row refusal precedes draw and retains prefix", func(t *testing.T) {
		d, boundary := newDraft(t, false)
		if err := d.syncCommitted(prompt); err != nil {
			t.Fatal(err)
		}
		before := d.sampler.CheckpointRNG()
		// The raw-zero last coordinate has one reachable fallback grid point.
		// Refuse the WHOLE row before any draw, not just that unlucky outcome.
		if d.admitSampledProposalRow([]float32{0, 0, 0}, []float32{0.1, 0.8999999, 0}) {
			t.Fatal("conditioned zero-q fallback row admitted")
		}
		if !d.admitSampledProposalRow([]float32{0, 0, 0}, []float32{0.2, 0.8, 0}) {
			t.Fatal("zero-q tail with no reachable fallback was refused")
		}
		if d.sampler.CheckpointRNG() != before {
			t.Fatal("row admission consumed randomness")
		}
		// Fault-inject an unsupported FIRST row: no proposal draw or target
		// transaction is allowed, and the safe target-only marker is explicit.
		d.lastLogits = []float32{float32(math.NaN()), 0, 0}
		if _, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0); !errors.Is(err, ErrQwen35MTPSampledNoProposal) {
			t.Fatalf("first-row refusal = %v", err)
		}
		if d.sampler.CheckpointRNG() != before || d.Err() != nil {
			t.Fatal("clean first-row refusal changed RNG or latched an error")
		}
		// A later unsupported row must retain the valid earlier sample. It
		// must not abort the whole round and restart from the original boundary.
		d.lastLogits = []float32{0, 0, 0}
		qwen35MTPSetFixedDraftLogits(d, []float32{float32(math.NaN()), 0, 0})
		d.sampler.PRNG().NextFloat32()
		wantRNG := d.sampler.CheckpointRNG()
		d.sampler.RollbackRNG(before)
		round, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0)
		if err != nil {
			t.Fatal(err)
		}
		if round.Plan().NextStep != 2 || len(round.target.draft) != 1 || d.sampler.CheckpointRNG() != wantRNG {
			t.Fatalf("prefix discarded or extra draw consumed: plan=%+v draft=%v", round.Plan(), round.target.draft)
		}
		if err := round.Abort(); err != nil || d.sampler.CheckpointRNG() != before {
			t.Fatalf("prefix abort did not restore RNG: %v", err)
		}
		assertQwen35MTPTargetStateEqual(t, d.target, serialTarget(t, d))
	})

	for _, tc := range []struct {
		name    string
		compact bool
		topP    float32
	}{
		{"full vocabulary", false, 1},
		{"compact coordinates", true, 1},
		{"nucleus target and draft", false, 0.6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			samplerCfg := cfg
			samplerCfg.TopP = tc.topP
			d, boundary := newDraft(t, tc.compact, samplerCfg)
			beforeRNG := d.sampler.CheckpointRNG()
			round, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 5)
			if err != nil {
				t.Fatal(err)
			}
			plan := round.Plan()
			if !reflect.DeepEqual(round.VerificationReceipt(), TargetVerificationReceipt{}) {
				t.Fatal("provisional round published a success receipt")
			}
			choiceRows := round.TargetLogits()
			if len(choiceRows) != 2 {
				t.Fatalf("target row count = %d", len(choiceRows))
			}
			originalChoice := choiceRows[0][0]
			choiceRows[0][0] = 12345
			if round.TargetLogits()[0][0] != originalChoice {
				t.Fatal("target logit accessor aliases round evidence")
			}
			draftElapsed := round.DraftElapsed()
			if plan.NextStep != 8 || plan.CorrectionToken < 0 {
				t.Fatalf("incomplete real sampled round: %+v", plan)
			}
			oracle, oracleBoundary := newDraft(t, tc.compact, samplerCfg)
			tokens, evidence := oracle.ProposeWithEvidence(prompt)
			probabilities, err := sampledMTPProbabilityRow(oracle.sampler, oracleBoundary)
			if err != nil {
				t.Fatal(err)
			}
			rows := []SpecProbabilityRow{probabilities}
			for _, token := range tokens {
				probabilities, err = sampledMTPProbabilityRow(oracle.sampler, oracle.target.Step(token))
				if err != nil {
					t.Fatal(err)
				}
				rows = append(rows, probabilities)
			}
			wantPlan, err := BuildSpecRejectionPlan(SpecRejectionPlanInput{
				DraftTokens: tokens, DraftEvidence: evidence, TargetProbabilityRows: rows,
				RequestKey: 37, StepOffset: 5,
			})
			if err != nil || !samePlan(plan, wantPlan) {
				t.Fatalf("live target rows differ from serial K+1 oracle: got=%+v want=%+v err=%v", plan, wantPlan, err)
			}
			if _, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 5); !errors.Is(err, ErrQwen35MTPSampledRoundActive) {
				t.Fatalf("second round = %v", err)
			}
			if tokens := d.Propose(prompt); tokens != nil {
				t.Fatalf("proposal escaped active transaction: %v", tokens)
			}
			if err := round.Abort(); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(round.VerificationReceipt(), TargetVerificationReceipt{}) || round.DraftElapsed() != draftElapsed {
				t.Fatal("abort published success or rewrote proposal-only elapsed time")
			}
			if d.sampler.CheckpointRNG() != beforeRNG || d.sampledRound != nil || d.pending != nil {
				t.Fatal("abort retained RNG or provisional draft ownership")
			}
			serial := serialTarget(t, d)
			assertQwen35MTPTargetStateEqual(t, d.target, serial)

			// This replay uses the actual MTP forward and target transaction,
			// not fabricated q rows or a replacement verifier callback.
			replay, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 5)
			if err != nil {
				t.Fatal(err)
			}
			if !samePlan(plan, replay.Plan()) {
				t.Fatalf("aborted retry changed decision: first=%+v retry=%+v", plan, replay.Plan())
			}
			got, err := replay.Commit(len(plan.AcceptedTokens))
			if err != nil {
				t.Fatal(err)
			}
			receipt := replay.VerificationReceipt()
			if receipt.Schema == "" || receipt.AcceptedTokens != len(plan.AcceptedTokens) || !reflect.DeepEqual(receipt, replay.target.VerificationReceipt()) {
				t.Fatalf("missing actual committed-prefix receipt: %+v", receipt)
			}
			want := boundary
			for _, token := range plan.AcceptedTokens {
				want = serial.Step(token)
			}
			assertFloat32BitsEqual(t, "sampled commit boundary", got, want)
			assertQwen35MTPTargetStateEqual(t, d.target, serial)
			if d.sampler.CheckpointRNG() == beforeRNG || d.sampledRound != nil || d.pending != nil {
				t.Fatal("commit lost RNG advancement or retained provisional ownership")
			}
			if err := replay.Abort(); err != nil {
				t.Fatalf("terminal abort: %v", err)
			}
			// The correction is deliberately still unevaluated at Commit.
			assertFloat32BitsEqual(t, "sampled correction continuation", d.target.Step(plan.CorrectionToken), serial.Step(plan.CorrectionToken))
		})
	}

	t.Run("invalid prefix and pre-publication cancellation abort", func(t *testing.T) {
		for _, cancelled := range []bool{false, true} {
			d, boundary := newDraft(t, false)
			beforeRNG := d.sampler.CheckpointRNG()
			round, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0)
			if err != nil {
				t.Fatal(err)
			}
			if cancelled {
				cctx, cancel := context.WithCancel(ctx)
				cancel()
				if err := round.Abort(); err != nil {
					t.Fatal(err)
				}
				if _, err := d.BeginSampledRound(cctx, prompt, boundary, 37, 0); !errors.Is(err, context.Canceled) {
					t.Fatalf("cancelled begin = %v", err)
				}
			} else if _, err := round.Commit(len(round.Plan().AcceptedTokens) + 1); !errors.Is(err, ErrSpecRejectionShape) {
				t.Fatalf("invalid commit = %v", err)
			}
			if d.sampler.CheckpointRNG() != beforeRNG || d.sampledRound != nil {
				t.Fatal("failed commit consumed RNG or left a live round")
			}
			assertQwen35MTPTargetStateEqual(t, d.target, serialTarget(t, d))
		}
	})

	t.Run("emitted strict prefix survives cancellation", func(t *testing.T) {
		d, boundary := newDraft(t, false)
		cctx, cancel := context.WithCancel(ctx)
		defer cancel()
		var round *Qwen35MTPSampledRound
		for key := uint64(1); key <= 128; key++ {
			var err error
			round, err = d.BeginSampledRound(cctx, prompt, boundary, key, 0)
			if err != nil {
				t.Fatal(err)
			}
			if len(round.Plan().AcceptedTokens) == 2 {
				break
			}
			if err := round.Abort(); err != nil {
				t.Fatal(err)
			}
			round = nil
		}
		if round == nil {
			t.Fatal("fixture found no full-acceptance keyed decision")
		}
		first := round.Plan().AcceptedTokens[0]
		// Model one token already emitted, followed by cancellation/max-budget.
		// The terminal state must retain it and discard the second accepted token.
		cancel()
		got, err := round.Commit(1)
		if err != nil {
			t.Fatal(err)
		}
		serial := serialTarget(t, d)
		assertFloat32BitsEqual(t, "emitted-prefix boundary", got, serial.Step(first))
		assertQwen35MTPTargetStateEqual(t, d.target, serial)
	})

	t.Run("failed rollback refuses session reuse", func(t *testing.T) {
		d, boundary := newDraft(t, false)
		beforeRNG := d.sampler.CheckpointRNG()
		round, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0)
		if err != nil {
			t.Fatal(err)
		}
		if d.pending == nil {
			t.Fatal("depth-two fixture has no proposal checkpoint")
		}
		d.pending.snapshot.Close() // deterministic restoration failure
		if err := round.Abort(); err == nil || d.Err() == nil {
			t.Fatalf("failed rollback was not terminal: abort=%v session=%v", err, d.Err())
		}
		if d.sampler.CheckpointRNG() != beforeRNG || d.sampledRound != nil || d.pending != nil {
			t.Fatal("failed rollback stranded RNG or provisional ownership")
		}
		if next, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0); next != nil || err == nil {
			t.Fatalf("round admitted after failed restore: %v %v", next, err)
		}
		if tokens := d.Propose(prompt); tokens != nil {
			t.Fatalf("proposal admitted after failed restore: %v", tokens)
		}
		assertQwen35MTPTargetStateEqual(t, d.target, serialTarget(t, d))
	})

	t.Run("admission remains fail closed", func(t *testing.T) {
		d, boundary := newDraft(t, false)
		beforeRNG := d.sampler.CheckpointRNG()
		badBoundary := append([]float32(nil), boundary...)
		badBoundary[0] = float32(math.NaN())
		if _, err := d.BeginSampledRound(ctx, prompt, badBoundary, 37, 0); !errors.Is(err, ErrSpecRejectionProbability) {
			t.Fatalf("non-finite target admission = %v", err)
		}
		if _, err := d.BeginSampledRound(ctx, prompt, boundary, 0, 0); !errors.Is(err, ErrSpecRejectionUnkeyed) {
			t.Fatalf("unkeyed admission = %v", err)
		}
		if _, err := d.BeginSampledRound(ctx, prompt[:1], boundary, 37, 0); !errors.Is(err, ErrSpecRejectionShape) {
			t.Fatalf("shorter-than-resident prefix admission = %v", err)
		}
		d.SetCoverageThreshold(0.1)
		if _, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0); !errors.Is(err, ErrQwen35MTPSpecDecodeUnsupported) {
			t.Fatalf("conditioned-q admission = %v", err)
		}
		if d.sampler.CheckpointRNG() != beforeRNG || d.target.Cache.Len() != len(prompt) {
			t.Fatal("refused admission mutated target or RNG")
		}
	})

	t.Run("target callback failure rolls back and permits clean retry", func(t *testing.T) {
		d, boundary := newDraft(t, false)
		beforeRNG := d.sampler.CheckpointRNG()
		injected := errors.New("target policy failed after verification")
		calls := 0
		processor := func(logits []float32) (SpecProbabilityRow, error) {
			calls++
			if calls == 2 {
				return SpecProbabilityRow{}, injected
			}
			return sampledMTPProbabilityRow(d.sampler, logits)
		}
		if round, err := d.BeginSampledRoundWithTargetProbabilities(ctx, prompt, boundary, 37, 0, processor); round != nil || !errors.Is(err, injected) {
			t.Fatalf("callback failure = %v, %v", round, err)
		}
		if calls != 2 || d.Err() != nil || d.sampler.CheckpointRNG() != beforeRNG || d.sampledRound != nil {
			t.Fatal("clean callback refusal changed RNG or poisoned the restored session")
		}
		assertQwen35MTPTargetStateEqual(t, d.target, serialTarget(t, d))
		round, err := d.BeginSampledRound(ctx, prompt, boundary, 37, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := round.Abort(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("canonical callback retains float64 rows across workspace reuse", func(t *testing.T) {
		d, boundary := newDraft(t, false)
		before := append([]float32(nil), boundary...)
		makeProcessor := func() func([]float32) (SpecProbabilityRow, error) {
			workspace := make([]float64, len(boundary))
			return func(logits []float32) (SpecProbabilityRow, error) {
				maximum := logits[argmaxF32(logits)]
				var mass float64
				for i, value := range logits {
					workspace[i] = math.Exp(float64(value-maximum) / 1.3)
					mass += workspace[i]
				}
				logits[0] = 12345 // callback input is not target-owned storage
				return SpecProbabilityRow{Weights: workspace, Mass: mass}, nil
			}
		}
		round, err := d.BeginSampledRoundWithTargetProbabilities(ctx, prompt, boundary, 37, 0, makeProcessor())
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(before, boundary) {
			t.Fatal("callback modified caller-owned boundary logits")
		}
		oracle, oracleBoundary := newDraft(t, false)
		tokens, evidence := oracle.ProposeWithEvidence(prompt)
		processor := makeProcessor()
		rows := make([]SpecProbabilityRow, 0, len(tokens)+1)
		for i := 0; i <= len(tokens); i++ {
			if i > 0 {
				oracleBoundary = oracle.target.Step(tokens[i-1])
			}
			row, err := processor(append([]float32(nil), oracleBoundary...))
			if err != nil {
				t.Fatal(err)
			}
			rows = append(rows, SpecProbabilityRow{Weights: append([]float64(nil), row.Weights...), Mass: row.Mass})
		}
		want, err := BuildSpecRejectionPlan(SpecRejectionPlanInput{DraftTokens: tokens, DraftEvidence: evidence, TargetProbabilityRows: rows, RequestKey: 37})
		if err != nil || !samePlan(round.Plan(), want) {
			t.Fatalf("canonical callback evidence = %+v, want %+v, error %v", round.Plan(), want, err)
		}
		if err := round.Abort(); err != nil {
			t.Fatal(err)
		}
	})
}
