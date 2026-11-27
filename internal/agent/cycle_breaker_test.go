// fak-test:runtime fast est=1s lane=default
package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestTokenCycleBreaker(t *testing.T) {
	t.Run("breaks at exact period times min repeats and resets", func(t *testing.T) {
		var b tokenCycleBreaker
		const period = 10
		total := 0
		for rep := 0; rep < 10; rep++ {
			for i := 0; i < period; i++ {
				total++
				if b.push(100 + i) {
					if total != period*tokenCycleMinRepeats {
						t.Fatalf("broke at token %d, want %d", total, period*tokenCycleMinRepeats)
					}
					if !b.broken || b.period != period {
						t.Fatalf("broken=%v period=%d, want true/%d", b.broken, b.period, period)
					}
					b.reset()
					if b.broken || b.period != 0 || len(b.recent) != 0 {
						t.Fatalf("reset did not clear breaker state: %+v", b)
					}
					return
				}
			}
		}
		t.Fatal("expected cycle breaker to halt repeating token sequence")
	})

	t.Run("ignores non-repeating and short-period tokens", func(t *testing.T) {
		var b tokenCycleBreaker
		for i := 0; i < 200; i++ {
			if b.push(i) {
				t.Fatalf("unexpected break on monotonic sequence at %d", i)
			}
		}
		b.reset()
		// Period 4 is below tokenCycleMinPeriod (8), though 4*2=8 matches period 8
		// after 32 tokens; a 3-repeat period-10 sequence must not trigger.
		for rep := 0; rep < tokenCycleMinRepeats-1; rep++ {
			for i := 0; i < 10; i++ {
				if b.push(200 + i) {
					t.Fatalf("broke too early at rep=%d i=%d", rep, i)
				}
			}
		}
	})
}

func TestInKernelDecodeLaneBreaksRepeatingCycle(t *testing.T) {
	const (
		vocab  = 32
		period = 10
		maxNew = 200
	)
	var cycle tokenCycleBreaker
	ln := &decodeLane{
		logits: make([]float32, vocab),
		maxNew: maxNew,
		stops:  map[int]bool{},
		emit: func(next int) bool {
			return cycle.push(next)
		},
	}
	ctx := context.Background()
	for step := 0; step < maxNew; step++ {
		for j := range ln.logits {
			ln.logits[j] = -10
		}
		wantTok := 5 + (step % period)
		ln.logits[wantTok] = 10
		_, advance := ln.decodeOne(ctx)
		if !advance {
			break
		}
	}
	if !cycle.broken || cycle.period != period {
		t.Fatalf("cycle.broken=%v period=%d, want true/%d", cycle.broken, cycle.period, period)
	}
	if !ln.stopped || !ln.done {
		t.Fatalf("lane stopped=%v done=%v, want true/true", ln.stopped, ln.done)
	}
	if ln.gen != period*tokenCycleMinRepeats {
		t.Fatalf("lane generated %d tokens, want exact cycle break at %d", ln.gen, period*tokenCycleMinRepeats)
	}
}

func TestInKernelPlannerCompleteSmokeWithCycleBreaker(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-cycle-smoke", false, nil, false)
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, nil, WithMaxTokens(8))
	if err != nil {
		t.Fatal(err)
	}
	if comp == nil || comp.Usage.CompletionTokens == 0 {
		t.Fatalf("unexpected empty completion: %+v", comp)
	}
	if comp.Message.ReasoningContent == "" && comp.Usage.CompletionTokensDetails != nil {
		t.Fatalf("expected nil CompletionTokensDetails when ReasoningContent is empty, got %+v", comp.Usage.CompletionTokensDetails)
	}
}

func TestNativeReasoningTokenDetailsAndInkling2LoopUsage(t *testing.T) {
	tok := loadProbeTok(t)
	if got := nativeReasoningTokenDetails(tok, "", 10); got != nil {
		t.Fatalf("empty reasoning returned %+v, want nil", got)
	}
	if got := nativeReasoningTokenDetails(tok, "thought", 0); got != nil {
		t.Fatalf("zero completion tokens returned %+v, want nil", got)
	}
	if got := nativeReasoningTokenDetails(tok, "deep reasoning trace here", 2); got == nil || got.ReasoningTokens != 2 {
		t.Fatalf("expected clamp to completionTokens=2, got %+v", got)
	}

	raw := inkling2RoleModel + inkling2ContentThinking + "step one then step two" + inkling2EndMessage +
		inkling2RoleModel + inkling2ContentText + "final answer" + inkling2EndMessage
	gen := &inkling2StreamGen{}
	p := inkling2StreamPlanner(t, gen)
	ids, err := p.tok.Encode(raw)
	if err != nil {
		t.Fatal(err)
	}
	gen.ids = ids
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, WithMaxTokens(len(ids)+8))
	if err != nil {
		t.Fatal(err)
	}
	if comp.Message.ReasoningContent != "step one then step two" {
		t.Fatalf("ReasoningContent = %q, want %q", comp.Message.ReasoningContent, "step one then step two")
	}
	if comp.Usage.CompletionTokensDetails == nil || comp.Usage.CompletionTokensDetails.ReasoningTokens <= 0 {
		t.Fatalf("expected ReasoningTokens > 0, got %+v", comp.Usage.CompletionTokensDetails)
	}
	if comp.Usage.CompletionTokensDetails.ReasoningTokens > comp.Usage.CompletionTokens {
		t.Fatalf("ReasoningTokens (%d) > CompletionTokens (%d)", comp.Usage.CompletionTokensDetails.ReasoningTokens, comp.Usage.CompletionTokens)
	}
}

func TestReusedToolCallIDsAcrossTurns(t *testing.T) {
	t.Run("ElideStaleReadMessages does not elide earlier non-Read sharing reused ID", func(t *testing.T) {
		msgs := []Message{
			{Role: RoleUser, Content: "inspect"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_text_0", Function: Func{Name: "Bash", Arguments: `{"command":"ls"}`}}}},
			{Role: RoleTool, ToolCallID: "call_text_0", Content: "important bash output that must not be elided"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_text_0", Function: Func{Name: "Read", Arguments: `{"file_path":"foo.go"}`}}}},
			{Role: RoleTool, ToolCallID: "call_text_0", Content: "package foo // old content"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_text_1", Function: Func{Name: "Edit", Arguments: `{"file_path":"foo.go"}`}}}},
			{Role: RoleTool, ToolCallID: "call_text_1", Content: "edited"},
			// 6 trailing messages to sit inside defaultStaleProtectTail
			{Role: RoleUser, Content: "t1"},
			{Role: RoleAssistant, Content: "a1"},
			{Role: RoleUser, Content: "t2"},
			{Role: RoleAssistant, Content: "a2"},
			{Role: RoleUser, Content: "t3"},
			{Role: RoleAssistant, Content: "a3"},
		}
		got := ElideStaleReadMessages(msgs, nil)
		if got[2].Content != msgs[2].Content {
			t.Fatalf("earlier Bash result with reused call_text_0 was corrupted/elided: %q", got[2].Content)
		}
		if !strings.Contains(got[4].Content, "superseded by a later in-session edit") {
			t.Fatalf("superseded Read result was not elided: %q", got[4].Content)
		}
	})

	t.Run("quarantineOutboundMessages pairs reused ID with nearest preceding assistant call", func(t *testing.T) {
		// Verify progressive callNames pairing does not crash or mispair across reused IDs.
		msgs := []Message{
			{Role: RoleUser, Content: "run"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_text_0", Function: Func{Name: "Read", Arguments: `{"file_path":"a.go"}`}}}},
			{Role: RoleTool, ToolCallID: "call_text_0", Content: "ok"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_text_0", Function: Func{Name: "Bash", Arguments: `{"command":"pwd"}`}}}},
			{Role: RoleTool, ToolCallID: "call_text_0", Content: "/work"},
		}
		out, _ := QuarantineOutboundMessages(msgs)
		if len(out) != len(msgs) || out[2].Content != "ok" || out[4].Content != "/work" {
			t.Fatalf("unexpected quarantine output: %+v", out)
		}
	})
}
