package model

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestQwen35MTPDepthNDraftSequentialFeedbackAndCacheCoherence(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{2, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)

	type call struct {
		pos      int
		prior    []float32
		feedback []float32
	}
	var calls []call
	realStep := d.step
	d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
		feedback, logits, err := realStep(f, pos, prior, embedding)
		calls = append(calls, call{
			pos:      pos,
			prior:    append([]float32(nil), prior...),
			feedback: append([]float32(nil), feedback...),
		})
		return feedback, logits, err
	}

	first := d.Propose(prompt)
	if err := d.Err(); err != nil {
		t.Fatalf("first depth-N proposal: %v", err)
	}
	if len(first) != 4 {
		t.Fatalf("first draft = %v, want depth 4", first)
	}
	if got := d.forward.draft.Cache.Len(); got != len(prompt)+3 {
		t.Fatalf("speculative MTP cache length = %d, want committed %d + speculative 3", got, len(prompt))
	}
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("drafting mutated target cache length to %d, want %d", target.Cache.Len(), len(prompt))
	}
	if d.pending == nil {
		t.Fatal("successful multi-step proposal retained no rollback checkpoint")
	}
	if len(calls) != len(prompt)+3 {
		t.Fatalf("forward calls = %d, want %d committed catch-up + speculative steps", len(calls), len(prompt)+3)
	}
	for i, call := range calls {
		if call.pos != i {
			t.Fatalf("call %d position = %d, want %d", i, call.pos, i)
		}
	}
	if got := calls[0].prior; !reflect.DeepEqual(got, make([]float32, m.Cfg.HiddenSize)) {
		t.Fatalf("first committed prior hidden = %v, want zero boundary", got)
	}
	targetHidden0, err := target.TargetHiddenAt(0)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "shifted committed target hidden", targetHidden0, calls[1].prior)
	targetHidden1, err := target.TargetHiddenAt(1)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "first speculative target seed", targetHidden1, calls[2].prior)
	assertFloat32BitsEqual(t, "draft feedback step 2", calls[2].feedback, calls[3].prior)
	assertFloat32BitsEqual(t, "draft feedback step 3", calls[3].feedback, calls[4].prior)

	target.Step(first[0])
	extended := append(append([]int(nil), prompt...), first[0])
	beforeSecond := len(calls)
	second := d.Propose(extended)
	if err := d.Err(); err != nil {
		t.Fatalf("second depth-N proposal: %v", err)
	}
	if len(second) != 4 {
		t.Fatalf("second draft = %v, want depth 4", second)
	}
	secondCalls := calls[beforeSecond:]
	if len(secondCalls) != 4 {
		t.Fatalf("second proposal calls = %d, want one committed catch-up + three speculative", len(secondCalls))
	}
	if secondCalls[0].pos != len(prompt) {
		t.Fatalf("committed rebase position = %d, want %d", secondCalls[0].pos, len(prompt))
	}
	assertFloat32BitsEqual(t, "committed rebase uses target hidden", targetHidden1, secondCalls[0].prior)
	if got := d.forward.draft.Cache.Len(); got != len(extended)+3 {
		t.Fatalf("second speculative MTP cache length = %d, want %d", got, len(extended)+3)
	}
	if !reflect.DeepEqual(d.processed, extended) {
		t.Fatalf("processed committed tokens = %v, want %v", d.processed, extended)
	}
}

func TestQwen35MTPDepthNPartialFailureRollsBackDraftCache(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)

	injected := errors.New("injected depth-N draft failure")
	realStep := d.step
	d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
		feedback, logits, err := realStep(f, pos, prior, embedding)
		if err == nil && pos == len(prompt)+1 {
			return nil, nil, injected
		}
		return feedback, logits, err
	}

	if got := d.Propose(prompt); got != nil {
		t.Fatalf("proposal after partial failure = %v, want nil", got)
	}
	if !errors.Is(d.Err(), injected) {
		t.Fatalf("runtime error = %v, want injected failure", d.Err())
	}
	if d.pending != nil {
		t.Fatal("failed proposal retained an active rollback checkpoint")
	}
	if got := d.forward.draft.Cache.Len(); got != len(prompt) {
		t.Fatalf("draft cache after failure = %d, want committed boundary %d", got, len(prompt))
	}
	if got := d.forward.lastPos; got != len(prompt)-1 {
		t.Fatalf("draft last position after failure = %d, want %d", got, len(prompt)-1)
	}
	if !reflect.DeepEqual(d.processed, prompt) {
		t.Fatalf("processed tokens after failure = %v, want %v", d.processed, prompt)
	}
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("draft failure mutated target cache length to %d", target.Cache.Len())
	}
}

func TestSpecDecodeGreedyQwen35MTPDepthAdmission(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	n := 12

	for depth := 1; depth <= Qwen35MTPMaxDraftDepth; depth++ {
		t.Run(fmt.Sprintf("depth-%d", depth), func(t *testing.T) {
			want := m.NewSession()
			want.captureTargetHidden = true
			t.Cleanup(want.Close)
			wantOutput := want.Generate(prompt, n)

			target := m.NewSession()
			t.Cleanup(target.Close)
			run, err := SpecDecodeGreedyQwen35MTPDepthN(target, prompt, n, depth)
			if err != nil {
				t.Fatalf("native depth-%d MTP decode: %v", depth, err)
			}
			if !reflect.DeepEqual(run.Output, wantOutput) {
				t.Fatalf("depth-%d output = %v, want target greedy %v", depth, run.Output, wantOutput)
			}
			if run.DraftedTokens == 0 || run.Rounds == 0 {
				t.Fatalf("depth-%d run did no drafting: %+v", depth, run)
			}
			if run.AcceptedDrafts+run.EvictKV != run.DraftedTokens {
				t.Fatalf("depth-%d accounting accepted %d + rejected %d != drafted %d", depth, run.AcceptedDrafts, run.EvictKV, run.DraftedTokens)
			}
			assertQwen35MTPTargetStateEqual(t, target, want)
		})
	}

	for _, tc := range []struct {
		name   string
		depth  int
		mutate func(*Session)
		reason string
	}{
		{name: "zero", depth: 0},
		{name: "above bound", depth: Qwen35MTPMaxDraftDepth + 1, reason: "exceeds the witnessed native bound"},
		{name: "quant format", depth: 2, mutate: func(s *Session) { s.Quant = true }, reason: "#9985"},
		{name: "malformed shape", depth: 2, mutate: func(s *Session) {
			meta := s.M.manifest["mtp.fc.weight"]
			meta.Shape = []int{1, 1}
			s.M.manifest["mtp.fc.weight"] = meta
		}, reason: "weight shape"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := qwen35MTPEnabledSyntheticModel(t)
			target := model.NewSession()
			t.Cleanup(target.Close)
			if tc.mutate != nil {
				tc.mutate(target)
			}
			_, err := SpecDecodeGreedyQwen35MTPDepthN(target, prompt, 2, tc.depth)
			if tc.depth == 0 {
				if !errors.Is(err, ErrQwen35MTPInvalidDraftLength) {
					t.Fatalf("zero-depth error = %v, want ErrQwen35MTPInvalidDraftLength", err)
				}
			} else {
				unsupported := assertQwen35MTPUnsupported(t, err)
				if tc.reason != "" && !strings.Contains(unsupported.Reason, tc.reason) {
					t.Fatalf("unsupported reason = %q, want substring %q", unsupported.Reason, tc.reason)
				}
			}
			if target.Cache.Len() != 0 {
				t.Fatalf("typed refusal mutated target cache length to %d", target.Cache.Len())
			}
		})
	}
}

func TestSpecDecodeGreedyQwen35MTPDepthNDeterministicAcceptance(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3
	wantSession := m.NewSession()
	wantSession.captureTargetHidden = true
	t.Cleanup(wantSession.Close)
	wantOutput := wantSession.Generate(prompt, depth+1)
	history := append(append([]int(nil), prompt...), wantOutput...)

	builder := func(target *Session, depth int) (*Qwen35MTPDraftSession, error) {
		d, err := NewQwen35MTPDraftSession(target, depth)
		if err != nil {
			return nil, err
		}
		realStep := d.step
		d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
			feedback, logits, err := realStep(f, pos, prior, embedding)
			if err != nil || pos+1 >= len(history) {
				return feedback, logits, err
			}
			for i := range logits {
				logits[i] = 0
			}
			logits[history[pos+1]] = 1
			return feedback, logits, nil
		}
		return d, nil
	}

	target := m.NewSession()
	t.Cleanup(target.Close)
	run, err := specDecodeGreedyQwen35MTPDepthN(target, prompt, depth+1, depth, builder)
	if err != nil {
		t.Fatalf("full-acceptance native depth-N decode: %v", err)
	}
	if !reflect.DeepEqual(run.Output, wantOutput) {
		t.Fatalf("full-acceptance output = %v, want %v", run.Output, wantOutput)
	}
	if run.Rounds != 1 || run.DraftedTokens != depth || run.AcceptedDrafts != depth || run.EvictKV != 0 {
		t.Fatalf("full-acceptance accounting = %+v, want one round and %d/%d/0 drafted/accepted/rejected", run, depth, depth)
	}
	assertQwen35MTPTargetStateEqual(t, target, wantSession)

	partialBuilder := func(target *Session, depth int) (*Qwen35MTPDraftSession, error) {
		d, err := builder(target, depth)
		if err != nil {
			return nil, err
		}
		perfectStep := d.step
		d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
			feedback, logits, err := perfectStep(f, pos, prior, embedding)
			if err == nil && pos == len(prompt) {
				for i := range logits {
					logits[i] = 0
				}
				logits[(history[pos+1]+1)%len(logits)] = 1
			}
			return feedback, logits, err
		}
		return d, nil
	}

	partialWant := m.NewSession()
	partialWant.captureTargetHidden = true
	t.Cleanup(partialWant.Close)
	partialOutput := partialWant.Generate(prompt, 2)
	partialTarget := m.NewSession()
	t.Cleanup(partialTarget.Close)
	partial, err := specDecodeGreedyQwen35MTPDepthN(partialTarget, prompt, 2, depth, partialBuilder)
	if err != nil {
		t.Fatalf("partial-acceptance native depth-N decode: %v", err)
	}
	if !reflect.DeepEqual(partial.Output, partialOutput) {
		t.Fatalf("partial-acceptance output = %v, want %v", partial.Output, partialOutput)
	}
	if partial.Rounds != 1 || partial.DraftedTokens != depth || partial.AcceptedDrafts != 1 || partial.EvictKV != depth-1 {
		t.Fatalf("partial-acceptance accounting = %+v, want one accepted and %d rejected", partial, depth-1)
	}
	assertQwen35MTPTargetStateEqual(t, partialTarget, partialWant)
}
