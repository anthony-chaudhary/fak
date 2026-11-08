package agent

// Pinned behavior for the grammar/completion-token budget adapted from
// TensorFold's grammar_budget flag (Apache-2.0): arming, the per-step budget
// math, the closed refusal contracts, and the no-grammar no-op.

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1ms lane=default
func TestNativeGrammarBudgetKeyPreservesLimitWidth(t *testing.T) {
	t.Parallel()
	prefix := []int{1, 7, 9}
	key := nativeGrammarBudgetKey(prefix, 2047)
	if key != nativeGrammarBudgetKey(append([]int(nil), prefix...), 2047) {
		t.Fatal("equal memo coordinates produced different keys")
	}
	if key == nativeGrammarBudgetKey(prefix, 2048) || key == nativeGrammarBudgetKey([]int{1, 7, 8}, 2047) || key == nativeGrammarBudgetKey([]int{1, 7}, 2047) {
		t.Fatal("distinct memo coordinates collided")
	}
	large := int64(1<<32) + 2047
	if large > int64(int(^uint(0)>>1)) {
		t.Skip("large limit is not representable on this native-int width")
	}
	largeKey := nativeGrammarBudgetKey(prefix, int(large))
	if largeKey == key {
		t.Fatal("64-bit remaining budget reused a truncated 32-bit key")
	}
	memo := map[string]int{key: 2048, largeKey: 4096}
	if len(memo) != 2 || memo[key] != 2048 || memo[largeKey] != 4096 {
		t.Fatal("distinct capped completion results aliased")
	}
}

func nativeGrammarBudgetGBNF(t *testing.T, p *InKernelPlanner, source string) *nativeStructuredConstraint {
	t.Helper()
	stops := StopIDs(p.tok, p.m.Cfg)
	stopID, err := p.nativeStructuredStopID(stops)
	if err != nil {
		t.Fatal(err)
	}
	mask, err := grammar.CompileGBNF(source, nativeRegexTokenDecoder{tok: p.tok, special: p.tok.SpecialTokens()}, grammar.CompileOptions{EOS: stopID})
	if err != nil {
		t.Fatalf("CompileGBNF: %v", err)
	}
	return &nativeStructuredConstraint{mask: mask, stopID: stopID, openID: -1, closeID: -1}
}

func nativeGrammarBudgetEncode(t *testing.T, p *InKernelPlanner, text string) int {
	t.Helper()
	ids, err := p.tok.Encode(text)
	if err != nil || len(ids) != 1 {
		t.Fatalf("encode %q: ids=%v err=%v", text, ids, err)
	}
	return ids[0]
}

// nativeGrammarBudgetEndlessMask allows exactly one non-EOS token at every
// position: an infinite frontier with no completion within the caps.
type nativeGrammarBudgetEndlessMask struct{ eos int }

func (m nativeGrammarBudgetEndlessMask) MaskLogits(history []int, logits []float32) {
	for i := range logits {
		logits[i] = float32(math.Inf(-1))
	}
	survivor := 1
	if survivor == m.eos {
		survivor++
	}
	if survivor < len(logits) {
		logits[survivor] = 0
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestGrammarBudgetWithinMaxTokens(t *testing.T) {
	t.Run("fork_probe_and_emission_isolation", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		stops := StopIDs(p.tok, p.m.Cfg)
		c := nativeGrammarBudgetGBNF(t, p, `root ::= "ok"`)
		if err := c.armGrammarBudget(p, 3); err != nil {
			t.Fatal(err)
		}
		step := func(c *nativeStructuredConstraint, token string) {
			t.Helper()
			logits, err := c.maskLogits(make([]float32, p.m.Cfg.VocabSize), stops)
			if err != nil {
				t.Fatal(err)
			}
			id := nativeGrammarBudgetEncode(t, p, token)
			if !nativeStructuredLogitFinite(logits[id]) {
				t.Fatalf("token %q masked", token)
			}
			if err := c.accept(id, logits); err != nil {
				t.Fatal(err)
			}
		}
		step(c, "o")
		fork := c.fork()
		if fork.budget == c.budget || fork.budget.probe != c.budget.probe || fork.budget.emitted != 1 || fork.budget.maxTokens != 3 {
			t.Fatal("fork must share probe/memo but copy budget state")
		}
		step(fork, "k")
		if fork.budget.emitted != 2 || c.budget.emitted != 1 || len(c.history) != 1 {
			t.Fatal("fork acceptance changed parent emission/history")
		}
		fork.reset()
		if fork.budget.emitted != 0 || c.budget.emitted != 1 {
			t.Fatal("fork reset changed parent")
		}
		step(c, "k")
		if c.budget.emitted != 2 || fork.budget.emitted != 0 {
			t.Fatal("parent acceptance changed fork")
		}
	})
	t.Run("out_of_range_eos", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		c := nativeGrammarBudgetGBNF(t, p, `root ::= "a"`)
		c.stopID = p.m.Cfg.VocabSize
		if err := c.armGrammarBudget(p, 16); !errors.Is(err, ErrNativeGrammarBudgetABI) || !errors.Is(err, ErrNativeGrammarBudget) {
			t.Fatalf("err = %v, want ABI refusal", err)
		}
	})
	t.Run("refusals", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		raw := nativeGBNFRaw(t, `root ::= "a"`)
		sampled := 0
		p.cachePrimeSamplerHook = func() { sampled++ }
		for name, value := range map[string]json.RawMessage{
			"string": json.RawMessage(`"true"`),
			"number": json.RawMessage(`1`),
			"object": json.RawMessage(`{"on":true}`),
			"array":  json.RawMessage(`[true]`),
		} {
			t.Run(name, func(t *testing.T) {
				opts := []SampleOpt{
					WithGuidedDecode(map[string]json.RawMessage{"guided_grammar": raw, nativeGuidedBudgetKey: value}),
					WithMaxTokens(16),
				}
				comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil, opts...)
				if !errors.Is(err, ErrNativeGrammarBudgetNotBoolean) || !errors.Is(err, ErrNativeGrammarBudget) || !errors.Is(err, ErrNativeStructuredUnsupported) {
					t.Fatalf("err = %v, want grammar-budget not-boolean refusal", err)
				}
				if comp != nil || sampled != 0 {
					t.Fatalf("refusal ran decode: completion=%v sampled=%d", comp != nil, sampled)
				}
			})
		}

		t.Run("uncompiled", func(t *testing.T) {
			p := nativeStructuredTestPlanner(t)
			stops := StopIDs(p.tok, p.m.Cfg)
			stopID, err := p.nativeStructuredStopID(stops)
			if err != nil {
				t.Fatal(err)
			}
			c := &nativeStructuredConstraint{stopID: stopID, openID: -1, closeID: -1}
			err = c.armGrammarBudget(p, 16)
			if !errors.Is(err, ErrNativeGrammarBudgetUncompiled) || !errors.Is(err, ErrNativeGrammarBudget) {
				t.Fatalf("err = %v, want uncompiled-grammar refusal", err)
			}
		})

		t.Run("abi", func(t *testing.T) {
			p := nativeStructuredTestPlanner(t)
			c := nativeGrammarBudgetGBNF(t, p, `root ::= "a"`)
			vocab := p.m.Cfg.VocabSize
			p.m.Cfg.VocabSize = 0
			defer func() { p.m.Cfg.VocabSize = vocab }()
			err := c.armGrammarBudget(p, 16)
			if !errors.Is(err, ErrNativeGrammarBudgetABI) || !errors.Is(err, ErrNativeGrammarBudget) {
				t.Fatalf("err = %v, want missing-ABI refusal", err)
			}
		})

		t.Run("rules", func(t *testing.T) {
			p := nativeStructuredTestPlanner(t)
			stops := StopIDs(p.tok, p.m.Cfg)
			stopID, err := p.nativeStructuredStopID(stops)
			if err != nil {
				t.Fatal(err)
			}
			c := &nativeStructuredConstraint{mask: nativeGrammarBudgetEndlessMask{eos: stopID}, stopID: stopID, openID: -1, closeID: -1}
			err = c.armGrammarBudget(p, 16)
			if !errors.Is(err, ErrNativeGrammarBudgetRules) || !errors.Is(err, ErrNativeGrammarBudget) {
				t.Fatalf("err = %v, want unsupported-rules refusal", err)
			}
		})
	})

	t.Run("budget_math", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		stops := StopIDs(p.tok, p.m.Cfg)
		source := `root ::= short | long
short ::= "ok"
long ::= "verbose" "more"`
		c := nativeGrammarBudgetGBNF(t, p, source)
		if err := c.armGrammarBudget(p, 3); err != nil {
			t.Fatal(err)
		}
		if c.budget.minDraws != 3 {
			t.Fatalf("minDraws = %d, want 3 (o, k, EOS)", c.budget.minDraws)
		}
		o, v, k := nativeGrammarBudgetEncode(t, p, "o"), nativeGrammarBudgetEncode(t, p, "v"), nativeGrammarBudgetEncode(t, p, "k")

		logits := make([]float32, p.m.Cfg.VocabSize)
		effective, err := c.maskLogits(logits, stops)
		if err != nil {
			t.Fatal(err)
		}
		if !nativeStructuredLogitFinite(effective[o]) {
			t.Fatal("short alternative cleared although it completes within the budget")
		}
		if nativeStructuredLogitFinite(effective[v]) {
			t.Fatal("long alternative kept although it cannot complete within the budget")
		}
		if nativeStructuredLogitFinite(effective[c.stopID]) {
			t.Fatal("EOS admitted before the reply is complete")
		}

		if err := c.accept(o, effective); err != nil {
			t.Fatal(err)
		}
		effective, err = c.maskLogits(logits, stops)
		if err != nil {
			t.Fatal(err)
		}
		if !nativeStructuredLogitFinite(effective[k]) {
			t.Fatal("closing token of the short alternative cleared")
		}
		if nativeStructuredLogitFinite(effective[c.stopID]) {
			t.Fatal("EOS admitted before the reply is complete")
		}

		if err := c.accept(k, effective); err != nil {
			t.Fatal(err)
		}
		effective, err = c.maskLogits(logits, stops)
		if err != nil {
			t.Fatal(err)
		}
		if !nativeStructuredLogitFinite(effective[c.stopID]) {
			t.Fatal("EOS withheld after the reply completed")
		}
	})

	t.Run("fail_open_when_nothing_fits", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		stops := StopIDs(p.tok, p.m.Cfg)
		c := nativeGrammarBudgetGBNF(t, p, `root ::= "ok"`)
		if err := c.armGrammarBudget(p, 2); err != nil {
			t.Fatal(err)
		}
		o := nativeGrammarBudgetEncode(t, p, "o")
		logits := make([]float32, p.m.Cfg.VocabSize)
		effective, err := c.maskLogits(logits, stops)
		if err != nil {
			t.Fatal(err)
		}
		// Two draws cannot hold o, k, EOS: the budget must leave the grammar
		// mask untouched rather than dead-end the reply.
		if !nativeStructuredLogitFinite(effective[o]) {
			t.Fatal("budget dead-ended a reply with no fitting completion")
		}
		if nativeStructuredLogitFinite(effective[c.stopID]) {
			t.Fatal("EOS admitted before the reply is complete")
		}
	})

	t.Run("no_grammar_noop", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil,
			WithGuidedDecode(map[string]json.RawMessage{nativeGuidedBudgetKey: json.RawMessage(`true`)}),
			WithMaxTokens(8))
		if err != nil {
			t.Fatalf("grammar_budget without a grammar must be a no-op: %v", err)
		}
		if comp == nil {
			t.Fatal("no completion returned")
		}
	})
}

// fak-test:runtime fast est=400ms lane=default
func TestGrammarBudgetArmedCompletion(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	raw := nativeGBNFRaw(t, `root ::= "ok" | "verbose"`)
	zero := 0.0
	v := nativeGrammarBudgetEncode(t, p, "v")
	msgs := []Message{{Role: RoleUser, Content: "pick one"}}
	run := func(budget bool) (string, string, error) {
		guided := map[string]json.RawMessage{"guided_grammar": raw}
		if budget {
			guided[nativeGuidedBudgetKey] = json.RawMessage(`true`)
		}
		comp, err := p.Complete(context.Background(), msgs, nil,
			WithGuidedDecode(guided),
			WithTemperature(&zero),
			WithLogitBias(model.LogitBias{v: 100}),
			WithMaxTokens(3))
		if err != nil {
			return "", "", err
		}
		return comp.Message.Content, comp.FinishReason, nil
	}
	// Control: greedy toward "verbose" truncates at three tokens.
	content, reason, err := run(false)
	if err != nil {
		t.Fatal(err)
	}
	if content != "ver" || reason != "length" {
		t.Fatalf("control = (%q, %q), want truncated verbose prefix", content, reason)
	}
	// Armed: the budget clears "v" at the first step and the reply completes.
	content, reason, err = run(true)
	if err != nil {
		t.Fatal(err)
	}
	if content != "ok" || reason != "stop" {
		t.Fatalf("armed = (%q, %q), want (\"ok\", stop)", content, reason)
	}
}
