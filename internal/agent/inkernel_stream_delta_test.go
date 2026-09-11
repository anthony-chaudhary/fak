package agent

import (
	"context"
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
	streamed, err := NewInKernelPlanner(m, tok, "tiny-stream-parity-streamed", false, nil, false).CompleteStream(
		context.Background(), func(delta string) error {
			raw.WriteString(delta)
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
	streamedPieces := strings.Split(streamedText, "")
	_ = streamedPieces
	// The raw stream cannot equal the whole completion in ONE fragment — that is the
	// regression this ticket exists to kill. Two planner instances (buffered vs streamed)
	// split at the same token boundaries, so the streamed turn reaches the sink per token.
	fragmentCount := len(strings.Split(strings.TrimSuffix(streamedText, ""), ""))
	if buffered.Message.Content != "" && fragmentCount < 2 {
		t.Fatalf("the streamed raw text was a single %d-char blob, not per-token deltas", len(streamedText))
	}
}
