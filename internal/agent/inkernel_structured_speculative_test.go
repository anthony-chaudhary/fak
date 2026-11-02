package agent

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type structuredSpeculativeProbe struct {
	calls   int
	propose func([]int, int) (model.DraftProposal, error)
}

type structuredFailAfterForwardMask struct{ calls int }

func (m *structuredFailAfterForwardMask) MaskLogits(_ []int, logits []float32) {
	m.calls++
	for i := range logits {
		logits[i] = float32(math.Inf(-1))
	}
	if m.calls == 1 {
		logits[7] = 1
	}
}

func (*structuredSpeculativeProbe) Name() string { return "structured-test" }
func (p *structuredSpeculativeProbe) Propose(_ context.Context, committed []int, limit int) (model.DraftProposal, error) {
	p.calls++
	return p.propose(committed, limit)
}

// fak-test:runtime fast est=300ms lane=default
func TestNativeStructuredSpeculativeCompleteAndStream(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	temperature := 0.0
	options := []SampleOpt{WithResponseFormat(nativeStructuredTestFormat), WithTemperature(&temperature), WithPerTokenStream(true)}
	messages := []Message{{Role: RoleUser, Content: "Return the requested object."}}
	serial, err := p.Complete(context.Background(), messages, nil, options...)
	if err != nil {
		t.Fatal(err)
	}
	want, err := p.tok.Encode(serial.Message.Content)
	if err != nil {
		t.Fatal(err)
	}
	proposalError := errors.New("draft unavailable")
	for _, mode := range []string{"invalid", "accepted", "empty", "error", "tree"} {
		t.Run(mode, func(t *testing.T) {
			base := -1
			probe := &structuredSpeculativeProbe{propose: func(committed []int, limit int) (model.DraftProposal, error) {
				if base < 0 {
					base = len(committed)
				}
				switch mode {
				case "tree":
					return model.DraftProposal{Tree: &model.CandidateTree{Nodes: []model.CandidateNode{{Token: want[0], Parent: -1}}}}, nil
				case "empty":
					return model.DraftProposal{}, nil
				case "error":
					return model.DraftProposal{}, proposalError
				case "invalid":
					return model.NewLinearProposal([]int{p.m.Cfg.VocabSize + 1}, nil), nil
				}
				at := len(committed) - base
				if at >= len(want) {
					return model.DraftProposal{}, nil
				}
				return model.NewLinearProposal(slices.Clone(want[at:min(at+limit, len(want))]), nil), nil
			}}
			p.EnableSpeculativeDecoding(probe, 4)
			var streamed strings.Builder
			got, err := p.CompleteStream(context.Background(), func(piece string) error { streamed.WriteString(piece); return nil }, messages, nil, options...)
			if err != nil {
				t.Fatal(err)
			}
			if probe.calls == 0 || got.Message.Content != serial.Message.Content || streamed.String() != serial.Message.Content {
				t.Fatalf("proposals=%d content=%q stream=%q want=%q", probe.calls, got.Message.Content, streamed.String(), serial.Message.Content)
			}
			// Reusing the planner must not reuse the previous request's matcher.
			base = -1
			again, err := p.Complete(context.Background(), messages, nil, options...)
			if err != nil || again.Message.Content != serial.Message.Content {
				t.Fatalf("fresh request=%v err=%v", again, err)
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeStructuredSpeculativeEmissionOwnsGrammarAndKV(t *testing.T) {
	for _, mode := range []string{"accepted-callback", "accepted-cancel", "bonus-budget", "fallback-budget", "eos"} {
		t.Run(mode, func(t *testing.T) {
			p := nativeStructuredTestPlanner(t)
			p.tree, p.scopedTree = nil, nil
			stops := StopIDs(p.tok, p.m.Cfg)
			constraint, err := p.newNativeRequestConstraint(SampleParams{GuidedDecode: map[string]json.RawMessage{"guided_regex": json.RawMessage(`"abcd"`)}}, stops)
			if err != nil {
				t.Fatal(err)
			}
			want, err := p.tok.Encode("abcd")
			if err != nil || len(want) != 4 {
				t.Fatalf("tokens=%v err=%v", want, err)
			}
			prompt := []int{1, 2, 3}
			oracle := p.m.NewSession()
			p.configureNativeSession(oracle)
			defer oracle.Close()
			oracle.Prefill(prompt)
			probe := &structuredSpeculativeProbe{propose: func(committed []int, limit int) (model.DraftProposal, error) {
				if mode == "fallback-budget" {
					return model.DraftProposal{}, errors.New("draft unavailable")
				}
				at := len(committed) - len(prompt)
				if at >= len(want) {
					return model.NewLinearProposal([]int{0}, nil), nil
				}
				return model.NewLinearProposal(slices.Clone(want[at:min(at+limit, len(want))]), nil), nil
			}}
			p.EnableSpeculativeDecoding(probe, 2)
			ctx, cancel := context.WithCancel(context.WithValue(context.Background(), nativeStructuredConstraintContextKey{}, constraint))
			defer cancel()
			var emitted []int
			observed := false
			p.specSessionCloseHook = func(target *model.Session) {
				observed = true
				if target.Cache.Len() != len(prompt)+len(emitted) || !slices.Equal(constraint.history, emitted) {
					t.Fatalf("KV=%d prompt=%d emitted=%v grammar=%v", target.Cache.Len(), len(prompt), emitted, constraint.history)
				}
				for _, token := range emitted {
					oracle.Step(token)
				}
				if got, expected := target.Step(7), oracle.Step(7); !reflect.DeepEqual(got, expected) {
					t.Fatal("retained target continuation differs from emitted-token oracle")
				}
			}
			budget := 8
			if mode == "bonus-budget" {
				budget = 3
			}
			if mode == "fallback-budget" {
				budget = 1
			}
			result, err := p.generateReusedRecovering(ctx, prompt, budget, 0, 0, 0, nil, 0, 0, stops, func(token int) bool {
				emitted = append(emitted, token)
				if mode == "accepted-cancel" {
					cancel()
				}
				return mode == "accepted-callback"
			})
			if mode == "accepted-cancel" {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancel error=%v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			count := 1
			if mode == "bonus-budget" {
				count = 3
			}
			if mode == "eos" {
				count = 4
			}
			if !observed || probe.calls == 0 || result.gen != count || !slices.Equal(emitted, want[:count]) {
				t.Fatalf("observed=%t calls=%d generated=%d emitted=%v", observed, probe.calls, result.gen, emitted)
			}
		})
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestNativeStructuredSpeculativeForkAndMaskedBoundary(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	stops := StopIDs(p.tok, p.m.Cfg)
	c, err := p.newNativeStructuredConstraint(nativeStructuredTestFormat, stops)
	if err != nil {
		t.Fatal(err)
	}
	fork := c.fork()
	fork.phase, fork.closed = nativeStructuredReasoning, true
	fork.history = append(fork.history, 17)
	fork.maskBuffer = []float32{123}
	if c.phase != nativeStructuredJSON || c.closed || len(c.history) != 0 || len(c.maskBuffer) != 0 {
		t.Fatal("fork mutated parent state")
	}
	s := p.m.NewSession()
	p.configureNativeSession(s)
	defer s.Close()
	s.Prefill([]int{1, 2, 3})
	open, _ := p.tok.Encode("{")
	invalid, _ := p.tok.Encode("x")
	last := make([]float32, p.m.Cfg.VocabSize)
	last[invalid[0]], last[open[0]] = 1000, 1
	before := s.Cache.Len()
	result, err := verifyStructuredSpeculativeRound(context.Background(), s, model.NewLinearProposal([]int{invalid[0]}, nil), last, c, stops)
	if err != nil || result.NumAccepted != 0 || result.CorrectionToken != open[0] || s.Cache.Len() != before || len(c.history) != 0 || last[invalid[0]] != 1000 {
		t.Fatalf("result=%v err=%v KV=%d parent=%v", result, err, s.Cache.Len(), c.history)
	}
	failing := &structuredFailAfterForwardMask{}
	c.mask = failing
	_, err = verifyStructuredSpeculativeRound(context.Background(), s, model.NewLinearProposal([]int{7}, nil), last, c, stops)
	if !errors.Is(err, ErrNativeStructuredNoToken) || failing.calls != 2 || s.Cache.Len() != before || len(c.history) != 0 {
		t.Fatalf("failed round: err=%v masks=%d KV=%d history=%v", err, failing.calls, s.Cache.Len(), c.history)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeStructuredSpeculativeExclusions(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	probe := &structuredSpeculativeProbe{propose: func([]int, int) (model.DraftProposal, error) { return model.DraftProposal{}, nil }}
	p.EnableSpeculativeDecoding(probe, 2)
	temperature, seed := 0.8, int64(7)
	for index, option := range []SampleOpt{WithTemperature(&temperature), WithSeed(&seed), WithNativeDecodeTopLogprobs(2)} {
		got, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Return the requested object."}}, nil, WithResponseFormat(nativeStructuredTestFormat), option)
		if index == 2 {
			engine := p.speculativeEngine
			p.speculativeEngine = nil
			serial, serialErr := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Return the requested object."}}, nil, WithResponseFormat(nativeStructuredTestFormat), option)
			p.speculativeEngine = engine
			// Preserve serial refusal as well as successful masked-score output.
			if (err == nil) != (serialErr == nil) {
				t.Fatalf("logprob error parity: speculative=%v serial=%v", err, serialErr)
			}
			if err == nil && !reflect.DeepEqual(got.NativeDecodeTopLogprobs, serial.NativeDecodeTopLogprobs) {
				t.Fatal("structured top-logprob receipt differs from serial")
			}
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if probe.calls != 0 {
		t.Fatal("target-only requests invoked proposals")
	}
	for _, kind := range []string{"qwen3_5", "deepseek_v4", "mamba", "unknown"} {
		p.m.Cfg.ModelType = kind
		if p.structuredSpeculationSupported() {
			t.Fatalf("admitted %s", kind)
		}
	}
	p.m.Cfg.ModelType = "llama"
	p.metal = true
	if p.structuredSpeculationSupported() {
		t.Fatal("admitted Metal")
	}
}
