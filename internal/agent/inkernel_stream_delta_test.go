package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// streamProjectionSink is the post-decode projection runArmStream's loop wraps around
// the planner sink (loop_project.go): raw decode deltas are replayed as post-processed
// content deltas at completion. It is reproduced here as a test seam because it is the
// contract CompleteStream's raw sink feeding is defined against.
type streamProjectionSink struct {
	fragments []string
	err       error
}

func (s *streamProjectionSink) rawObserve(piece string) {
	if s.err == nil && piece != "" {
		s.fragments = append(s.fragments, piece)
	}
}

// completeFinalize is called by the test with the returned Completion; it drops the raw
// accumulation and projects the final Content as one delta, the way the loop does.
func (s *streamProjectionSink) completeFinalize(comp *Completion) {
	s.fragments = nil
	if comp != nil && comp.Message.Content != "" {
		s.fragments = append(s.fragments, comp.Message.Content)
	}
}

// TestInKernelPlannerStreamsPerTokenFragments is the fragment-count witness: a decoded
// N-token turn must reach the sink as N>=2 fragments, not one post-hoc blob. Fail if a
// regression collapses the per-token emit seam back into the single-blob projection.
func TestInKernelPlannerStreamsPerTokenFragments(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-stream-fragments", false, nil, false)

	var fragments []string
	comp, err := p.CompleteStream(context.Background(), func(delta string) error {
		fragments = append(fragments, delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) < 2 {
		t.Fatalf("a %d-token completion produced %d sink fragments; the per-token emit seam collapsed to one post-hoc emit (fragments=%q)", comp.Usage.CompletionTokens, len(fragments), fragments)
	}
}

// TestInKernelPlannerStreamParityConcatenatedDeltasEqualsBufferedText is the parity
// witness: same prompt + seed, the completion RETURNED by CompleteStream is byte-for-byte
// the buffered Complete result (same text, same reasoning content, same usage), and the
// RAW decode deltas fed to the sink must be exactly reproducible per token once they are
// re-decoded against the same tokenizer — the property the loop's projection needs to
// replay them as content deltas.
func TestInKernelPlannerStreamParityConcatenatedDeltasEqualsBufferedText(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	tok := loadProbeTok(t)

	messages := []Message{{Role: RoleUser, Content: "alpha beta gamma"}}

	buffered, err := NewInKernelPlanner(m, tok, "tiny-stream-parity-buffered", false, nil, false).Complete(
		context.Background(), messages, nil, WithMaxTokens(8))
	if err != nil {
		t.Fatal(err)
	}

	var raw strings.Builder
	fragmentCount := 0
	streamed, err := NewInKernelPlanner(m, tok, "tiny-stream-parity-streamed", false, nil, false).CompleteStream(
		context.Background(), func(delta string) error {
			raw.WriteString(delta)
			fragmentCount++
			return nil
		}, messages, nil, WithMaxTokens(8))
	if err != nil {
		t.Fatal(err)
	}

	// The planner-level completion parity: what the streaming CALLER of this framework
	// receives from CompleteStream must be exactly what Complete returned.
	if streamed.Message.Content != buffered.Message.Content {
		t.Fatalf("CompleteStream content %q != buffered content %q", streamed.Message.Content, buffered.Message.Content)
	}
	if streamed.Message.ReasoningContent != buffered.Message.ReasoningContent {
		t.Fatalf("CompleteStream reasoning %q != buffered reasoning %q", streamed.Message.ReasoningContent, buffered.Message.ReasoningContent)
	}
	if streamed.FinishReason != buffered.FinishReason {
		t.Fatalf("CompleteStream finish %q != buffered finish %q", streamed.FinishReason, buffered.FinishReason)
	}
	if !reflect.DeepEqual(streamed.Usage, buffered.Usage) {
		t.Fatalf("CompleteStream usage %+v != buffered usage %+v", streamed.Usage, buffered.Usage)
	}

	// The raw decode delta identity: if a non-empty completion was produced, the raw
	// stream must have carried at least one character's worth of token deltas, and every
	// streaming character must be one of the buffered tokens' characters (the decode
	// ran through the same emit seam the buffered Complete uses).
	streamedText := raw.String()
	if streamedText == "" && buffered.Message.Content != "" {
		t.Fatal("a non-empty buffered completion produced an empty raw token stream")
	}
	if streamedText != buffered.Message.Content {
		t.Fatalf("streamed bytes %q != buffered content %q", streamedText, buffered.Message.Content)
	}
	if buffered.Message.Content != "" && fragmentCount < 2 {
		t.Fatalf("the streamed raw text was a single %d-char blob, not per-token deltas", len(streamedText))
	}
}

// TestInKernelTokenStreamFlagGatesPerToken is the #920 witness for the opt-in gate:
// FAK_STREAM_INKERNEL_PER_TOKEN=1 forwards the live per-token decode seam (>=2
// fragments for a multi-token turn), while the default OFF projects the finished turn
// as AT MOST one post-hoc delta whose bytes equal the returned Completion's Content.
func TestInKernelTokenStreamFlagGatesPerToken(t *testing.T) {
	const prompt = "a b c d e"

	t.Run("on-per-token-forwarding", func(t *testing.T) {
		t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
		m := model.NewSynthetic(tinyConcurrencyConfig())
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-token-stream-on", false, nil, false)

		var fragments []string
		comp, err := p.CompleteStream(context.Background(), func(delta string) error {
			fragments = append(fragments, delta)
			return nil
		}, []Message{{Role: RoleUser, Content: prompt}}, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatal(err)
		}
		if len(fragments) < 2 {
			t.Fatalf("flag ON: %d-token completion produced %d sink fragments, want >=2 (fragments=%q)", comp.Usage.CompletionTokens, len(fragments), fragments)
		}
	})

	t.Run("off-one-posthoc-delta", func(t *testing.T) {
		t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "")
		m := model.NewSynthetic(tinyConcurrencyConfig())
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-token-stream-off", false, nil, false)

		var fragments []string
		comp, err := p.CompleteStream(context.Background(), func(delta string) error {
			fragments = append(fragments, delta)
			return nil
		}, []Message{{Role: RoleUser, Content: prompt}}, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatal(err)
		}
		if len(fragments) > 1 {
			t.Fatalf("flag OFF: sink received %d fragments, want at most 1 (%q)", len(fragments), fragments)
		}
		if comp.Message.Content != "" {
			if len(fragments) != 1 {
				t.Fatalf("flag OFF: non-empty completion %q produced %d fragments, want exactly 1", comp.Message.Content, len(fragments))
			}
			if fragments[0] != comp.Message.Content {
				t.Fatalf("flag OFF: post-hoc delta %q != Completion content %q", fragments[0], comp.Message.Content)
			}
		}
	})
}

// Sink failures must be returned unchanged and stop further deliveries in either mode.
func TestInKernelTokenStreamSinkError(t *testing.T) {
	for _, flag := range []string{"", "1"} {
		t.Run("flag="+flag, func(t *testing.T) {
			t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", flag)
			m := model.NewSynthetic(tinyConcurrencyConfig())
			m.Quantize()
			p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-stream-sink-error", false, nil, false)
			want := errors.New("sink disconnected")
			calls := 0
			_, err := p.CompleteStream(context.Background(), func(string) error {
				calls++
				return want
			}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6))
			if !errors.Is(err, want) || calls != 1 {
				t.Fatalf("sink error = %v, calls = %d; want %v and one call", err, calls, want)
			}
		})
	}
}

// TestInKernelPerTokenStreamOptInDefaultsOn is the turnkey-default witness: with the env
// gate OFF, a per-call WithPerTokenStream(true) (what cmd/fak up's streaming call site
// passes) must still forward the live per-token seam — >=2 fragments for a multi-token
// turn — so `fak up` streams true incremental deltas without an env var.
func TestInKernelPerTokenStreamOptInDefaultsOn(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-per-token-optin", false, nil, false)

	var fragments []string
	comp, err := p.CompleteStream(context.Background(), func(delta string) error {
		fragments = append(fragments, delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6), WithPerTokenStream(true))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) < 2 {
		t.Fatalf("WithPerTokenStream(true) with env off: %d-token completion produced %d fragments, want >=2", comp.Usage.CompletionTokens, len(fragments))
	}
}

// TestInKernelPerTokenStreamOptOutStaysBuffered is the byte-identical witness for the
// explicit off-switch: WithPerTokenStream(false) overrides an env ON gate and degrades to
// the single post-hoc projection, exactly the pre-seam behavior.
func TestInKernelPerTokenStreamOptOutStaysBuffered(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-per-token-optout", false, nil, false)

	var fragments []string
	comp, err := p.CompleteStream(context.Background(), func(delta string) error {
		fragments = append(fragments, delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6), WithPerTokenStream(false))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) > 1 {
		t.Fatalf("WithPerTokenStream(false) with env on: sink received %d fragments, want at most 1", len(fragments))
	}
	if comp.Message.Content != "" && (len(fragments) != 1 || fragments[0] != comp.Message.Content) {
		t.Fatalf("explicit off: post-hoc delta %q != Completion content %q", fragments, comp.Message.Content)
	}
}

// TestToolSpanGuardSuppressesToolCallMarkup is the tool-safety witness: a fake decode
// observer emits a Hermes <tool_call> JSON span token-piece by token-piece between prose.
// The guard must forward the leading and trailing PROSE and drop the ENTIRE span — the
// opener, the JSON body, and the closer must never reach the client as content.
func TestToolSpanGuardSuppressesToolCallMarkup(t *testing.T) {
	const (
		prefix = "I will read the file now. "
		tool   = `<tool_call>{"name": "Read", "arguments": {"filePath": "secret.go"}}</tool_call>`
		suffix = " Done."
	)
	// Split the tool span across pieces mid-token to prove partial-openers are held.
	pieces := []string{prefix, `<tool_`, `call>{"name": "Read", `, `"arguments": {"filePath": "secret.go"}}`, `</tool_call>`, suffix}

	var got strings.Builder
	g := newToolSpanGuard(func(delta string) error {
		got.WriteString(delta)
		return nil
	})
	for _, piece := range pieces {
		if err := g.feed(piece); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.flush(); err != nil {
		t.Fatal(err)
	}

	streamed := got.String()
	if strings.Contains(streamed, "<tool_call>") || strings.Contains(streamed, "secret.go") || strings.Contains(streamed, `"name"`) {
		t.Fatalf("tool-call markup leaked into streamed content: %q", streamed)
	}
	want := prefix + suffix
	if streamed != want {
		t.Fatalf("streamed prose = %q, want %q (only prose, no span)", streamed, want)
	}
}

// TestToolSpanGuardProseUnaffected proves the guard is inert on markup-free prose,
// including legitimate braces, brackets, and partial tag-like text that never becomes a
// tool call — so a normal turn streams byte-identically to the raw seam.
func TestToolSpanGuardProseUnaffected(t *testing.T) {
	const prose = "Here is JSON {\"a\": 1} and an array [1,2] and a fence ```go``` plus a <b>tag</b>."
	chunks := []string{"Here is JSON {", "\"a\": 1} and an ", "array [1,2] and a fence ```go``` plus a <b>tag</b>."}

	var got strings.Builder
	g := newToolSpanGuard(func(delta string) error { got.WriteString(delta); return nil })
	for _, c := range chunks {
		if err := g.feed(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.flush(); err != nil {
		t.Fatal(err)
	}
	if got.String() != prose {
		t.Fatalf("markup-free prose was altered: %q, want %q", got.String(), prose)
	}
}

// TestToolSpanGuardUnclosedSpanDropsToEnd proves an unclosed opener (a truncated or
// malformed tool call) is dropped to end of turn and the leading prose still flows, so a
// half-formed span can never leak its raw syntax.
func TestToolSpanGuardUnclosedSpanDropsToEnd(t *testing.T) {
	var got strings.Builder
	g := newToolSpanGuard(func(delta string) error { got.WriteString(delta); return nil })
	for _, c := range []string{"answer: ", "<tool_call>{\"name\": \"Bash\""} {
		if err := g.feed(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.flush(); err != nil {
		t.Fatal(err)
	}
	if got.String() != "answer: " {
		t.Fatalf("unclosed span: streamed = %q, want %q", got.String(), "answer: ")
	}
}
