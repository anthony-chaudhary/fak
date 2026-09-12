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
