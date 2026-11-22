package agent

import (
	"context"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestInKernelStreamProjectorFeedsThinkSpanToReasoningSinkLive(t *testing.T) {
	for _, tc := range []struct {
		name string
		// seeded opens the turn inside the prompt's <think>; budgeted means Complete
		// counts that seed when it splits (a thinking budget is armed).
		seeded, budgeted bool
		// proseFirst rows keep the existing prose-before-span content stream, which
		// the final split rewrites; only content parity with no sink is asserted.
		proseFirst bool
		pieces     []string
		liveAfter  int
		wantLive   string // reasoning sink after liveAfter pieces
		wantSink   string // reasoning sink at end of turn; the caller's remainder adds the rest
		wantFinal  string
	}{
		{
			name:      "open_and_close",
			pieces:    []string{"<th", "ink>", "\n", "two", " plus", " two", "\n</th", "ink>", "\n\n", "4"},
			liveAfter: 6, wantLive: "two plus two", wantSink: "two plus two", wantFinal: "two plus two",
		},
		{
			name:   "seeded_budget_close_only",
			seeded: true, budgeted: true,
			pieces:    []string{"Let me", " think", "\n</think>", "\n\nok"},
			liveAfter: 2, wantLive: "Let me think", wantSink: "Let me think", wantFinal: "Let me think",
		},
		{
			name:   "seeded_budget_unclosed",
			seeded: true, budgeted: true,
			pieces:    []string{"Let me", " think", " more"},
			liveAfter: 2, wantLive: "Let me think", wantSink: "Let me think more", wantFinal: "Let me think more",
		},
		{
			name:      "seeded_nobudget_close_only",
			seeded:    true,
			pieces:    []string{"Let me", " think", "\n</think>", "\n\nok"},
			liveAfter: 2, wantLive: "", wantSink: "Let me think", wantFinal: "Let me think",
		},
		{
			name:      "seeded_unclosed_nobudget",
			seeded:    true,
			pieces:    []string{"Let me", " think", " more"},
			liveAfter: 3, wantLive: "", wantSink: "", wantFinal: "",
		},
		{
			name:   "seeded_reopened_nobudget",
			seeded: true,
			pieces: []string{"<think>", "abc", "</think>", "ok"},
			// The split's reasoning starts at the inner open only without a budget,
			// so the sink leaves the whole span to the remainder.
			wantSink: "", wantFinal: "abc",
		},
		{
			name:   "seeded_reopened_budget",
			seeded: true, budgeted: true,
			pieces:   []string{"<think>", "abc", "</think>", "ok"},
			wantSink: "<think>abc", wantFinal: "<think>abc",
		},
		{
			name:      "later_span_stays_held",
			pieces:    []string{"<think>", "a", "</think>", "b", "<think>c</think>"},
			liveAfter: 2, wantLive: "a", wantSink: "a", wantFinal: "a",
		},
		{
			name:      "unclosed_at_cutoff",
			pieces:    []string{"<think>", "abc ", "def "},
			liveAfter: 3, wantLive: "abc def", wantSink: "abc def", wantFinal: "abc def",
		},
		{
			name:       "prose_close_before_open",
			proseFirst: true,
			pieces:     []string{"x</think>y", "<think>", "abc"},
			wantSink:   "", wantFinal: "x",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run := func(withSink bool) (content, reasoning, live string) {
				var c, r strings.Builder
				p := newInKernelStreamProjector(func(piece string) error {
					c.WriteString(piece)
					return nil
				}, nil, tc.seeded)
				p.spans.deferTools = true
				if withSink {
					p.spans.reasoning = func(piece string) error {
						r.WriteString(piece)
						return nil
					}
					p.spans.reasoningAtClose = tc.seeded && !tc.budgeted
				}
				for i, piece := range tc.pieces {
					if err := p.feed(piece); err != nil {
						t.Fatal(err)
					}
					if i+1 == tc.liveAfter {
						live = r.String()
					}
				}
				_, finalContent := splitNativeBudgetReasoning(strings.Join(tc.pieces, ""), tc.seeded && tc.budgeted)
				if err := p.finish(finalContent, false); err != nil {
					t.Fatal(err)
				}
				return c.String(), r.String(), live
			}
			final, finalContent := splitNativeBudgetReasoning(strings.Join(tc.pieces, ""), tc.seeded && tc.budgeted)
			if final != tc.wantFinal {
				t.Fatalf("final split reasoning = %q, want %q", final, tc.wantFinal)
			}
			content, reasoning, live := run(true)
			if tc.liveAfter > 0 && live != tc.wantLive {
				t.Fatalf("after %d pieces reasoning sink = %q, want %q", tc.liveAfter, live, tc.wantLive)
			}
			if reasoning != tc.wantSink {
				t.Fatalf("reasoning sink = %q, want %q", reasoning, tc.wantSink)
			}
			if !strings.HasPrefix(final, reasoning) || reasoning+StreamReasoningRemainder(reasoning, final) != final {
				t.Fatalf("reasoning sink %q diverged from final split %q", reasoning, final)
			}
			if plain, _, _ := run(false); content != plain {
				t.Fatalf("content sink = %q with a reasoning sink, %q without", content, plain)
			}
			if !tc.proseFirst && content != finalContent {
				t.Fatalf("content sink = %q, want split content %q with no reasoning bytes", content, finalContent)
			}
		})
	}
}

// TestInKernelCompleteStreamReasoningSinkNeverDivergesFromCompletion drives the
// production CompleteStream, so the final split is Complete's own, not a copy.
// fak-test:runtime fast est=2s lane=default
func TestInKernelCompleteStreamReasoningSinkNeverDivergesFromCompletion(t *testing.T) {
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	zero := 0.0
	for _, tc := range []struct {
		name        string
		seeded      bool
		biasTo      string // the token every greedy step picks
		budget      int
		maxTokens   int
		wantFinal   string
		wantContent string
		wantSink    string
	}{
		// No budget: Complete splits an unclosed seeded span as content.
		{name: "seeded_unclosed_nobudget", seeded: true, biasTo: "x", maxTokens: 4, wantFinal: "", wantContent: "xxxx", wantSink: ""},
		{name: "seeded_reopened_nobudget", seeded: true, biasTo: thinkOpen, maxTokens: 3, wantFinal: thinkOpen + thinkOpen, wantSink: ""},
		{name: "seeded_explicit_budget_live", seeded: true, biasTo: "x", budget: 2, maxTokens: 4, wantFinal: "xx", wantContent: "x", wantSink: "xx"},
		{name: "unseeded_open_live", biasTo: thinkOpen, maxTokens: 3, wantFinal: thinkOpen + thinkOpen, wantSink: thinkOpen + thinkOpen},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := nativeStructuredReasoningPlanner(t, tc.seeded)
			open, close, body, eos := nativeStructuredReasoningIDs(t, p)
			bias := map[int]float64{close: -100, eos: -100}
			if tc.biasTo == thinkOpen {
				bias[open] = 100
			} else {
				bias[body] = 100
			}
			opts := []SampleOpt{WithTemperature(&zero), WithLogitBias(bias), WithMaxTokens(tc.maxTokens), WithPerTokenStream(true)}
			if tc.budget > 0 {
				opts = append(opts, WithThinkingBudget(tc.budget))
			}
			var content, reasoning strings.Builder
			ctx := WithStreamReasoningSink(context.Background(), func(delta string) error {
				reasoning.WriteString(delta)
				return nil
			})
			comp, err := p.CompleteStream(ctx, func(delta string) error {
				content.WriteString(delta)
				return nil
			}, []Message{{Role: RoleUser, Content: "Answer."}}, nil, opts...)
			if err != nil {
				t.Fatal(err)
			}
			final := comp.Message.ReasoningContent
			if final != tc.wantFinal || comp.Message.Content != tc.wantContent {
				t.Fatalf("completion reasoning=%q content=%q, want %q and %q", final, comp.Message.Content, tc.wantFinal, tc.wantContent)
			}
			if got := reasoning.String(); got != tc.wantSink || !strings.HasPrefix(final, got) || got+StreamReasoningRemainder(got, final) != final {
				t.Fatalf("reasoning sink = %q, want %q and a prefix of the completion's %q", got, tc.wantSink, final)
			}
			if content.String() != comp.Message.Content {
				t.Fatalf("content sink = %q, completion content %q", content.String(), comp.Message.Content)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestHTTPPlannerCompleteStreamFeedsReasoningSinkLive(t *testing.T) {
	const body = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"two plus\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" two\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"4\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	srv, _ := sseServer(t, body)
	var events []string
	held := 0
	ctx := WithStreamReasoningSink(WithStreamHeldActivity(context.Background(), func() { held++ }), func(delta string) error {
		events = append(events, "reasoning:"+delta)
		return nil
	})
	comp, err := NewHTTPPlanner(srv.URL, "m", "").CompleteStream(ctx, func(delta string) error {
		events = append(events, "content:"+delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "two plus two"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(events, "|"), "reasoning:two plus|reasoning: two|content:4"; got != want {
		t.Fatalf("sink events = %q, want %q", got, want)
	}
	if comp.Message.ReasoningContent != "two plus two" || comp.Message.Content != "4" || held != 2 {
		t.Fatalf("completion reasoning=%q content=%q held=%d", comp.Message.ReasoningContent, comp.Message.Content, held)
	}
	if rest := StreamReasoningRemainder("two plus two", comp.Message.ReasoningContent); rest != "" {
		t.Fatalf("remainder after a complete live stream = %q, want empty", rest)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestStreamReasoningRemainderNeverRepeatsDivergedPrefix(t *testing.T) {
	for _, tc := range []struct{ streamed, final, want string }{
		{"", "two plus two", "two plus two"},
		{"two plus", "two plus two", " two"},
		{"two plus two", "two plus two", ""},
		{"three", "two plus two", ""},
	} {
		if got := StreamReasoningRemainder(tc.streamed, tc.final); got != tc.want {
			t.Fatalf("StreamReasoningRemainder(%q, %q) = %q, want %q", tc.streamed, tc.final, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSerialStreamBoundaryObservesReasoningSink(t *testing.T) {
	var emitted bool
	var got string
	ctx := WithStreamReasoningSink(context.Background(), func(delta string) error {
		got += delta
		return nil
	})
	ctx = WithSerialStreamBoundary(ctx, func() { emitted = true })
	if err := streamReasoningSink(ctx)(""); err != nil || emitted {
		t.Fatalf("empty reasoning delta crossed the replay boundary: emitted=%v err=%v", emitted, err)
	}
	if err := streamReasoningSink(ctx)("x"); err != nil || !emitted || got != "x" {
		t.Fatalf("reasoning delta emitted=%v got=%q err=%v, want boundary set and delta forwarded", emitted, got, err)
	}
}
