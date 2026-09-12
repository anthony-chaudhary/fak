package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeCompactionObserverPublishesOnlySuccessfulComplete(t *testing.T) {
	newPlanner := func(contextTokens int) *InKernelPlanner {
		p := NewInKernelPlannerWithConfig(
			model.NewSynthetic(tinyConcurrencyConfig()), loadProbeTok(t), "native-compaction-observer",
			false, nil, false, InKernelPlannerConfig{ContextTokens: contextTokens},
		)
		p.quant = false
		p.SetPromptShrinkLevers(80, false, false)
		return p
	}
	activeMessages := nativeCompactionObserverMessages()

	t.Run("successful Complete publishes quantitative proof in registration order", func(t *testing.T) {
		p := newPlanner(0)
		var order []string
		var observations []NativeCompactionObservation
		ctx := WithNativeCompactionObserver(context.Background(), func(value NativeCompactionObservation) {
			order = append(order, "first")
			observations = append(observations, value)
		})
		ctx = WithNativeCompactionObserver(ctx, func(value NativeCompactionObservation) {
			order = append(order, "second")
			observations = append(observations, value)
		})

		if _, err := p.Complete(ctx, activeMessages, nil, WithMaxTokens(1)); err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(order, ","); got != "first,second" {
			t.Fatalf("observer order = %q, want first,second", got)
		}
		if len(observations) != 2 || observations[0] != observations[1] {
			t.Fatalf("composed observers = %#v, want the same single observation each", observations)
		}
		assertNativeCompactionObservation(t, observations[0])
		raw, err := json.Marshal(observations[0])
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(raw), "native-compaction-secret-marker") {
			t.Fatalf("observation leaked compacted message content: %s", raw)
		}
	})

	t.Run("EncodePrompt prepares compaction without publishing", func(t *testing.T) {
		p := newPlanner(0)
		calls := 0
		ctx := WithNativeCompactionObserver(context.Background(), func(NativeCompactionObservation) { calls++ })
		if _, err := p.EncodePrompt(ctx, activeMessages, nil, WithMaxTokens(1)); err != nil {
			t.Fatal(err)
		}
		if calls != 0 {
			t.Fatalf("EncodePrompt observer calls = %d, want 0", calls)
		}
	})

	t.Run("idle successful Complete does not publish", func(t *testing.T) {
		p := newPlanner(0)
		calls := 0
		ctx := WithNativeCompactionObserver(context.Background(), func(NativeCompactionObservation) { calls++ })
		if _, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "short"}}, nil, WithMaxTokens(1)); err != nil {
			t.Fatal(err)
		}
		if calls != 0 {
			t.Fatalf("idle Complete observer calls = %d, want 0", calls)
		}
	})

	t.Run("pre-cancelled Complete does not publish", func(t *testing.T) {
		p := newPlanner(0)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		calls := 0
		ctx = WithNativeCompactionObserver(ctx, func(NativeCompactionObservation) { calls++ })
		_, err := p.Complete(ctx, activeMessages, nil, WithMaxTokens(1))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Complete error = %v, want context.Canceled", err)
		}
		if calls != 0 {
			t.Fatalf("cancelled Complete observer calls = %d, want 0", calls)
		}
	})

	t.Run("generation failure does not publish", func(t *testing.T) {
		p := newPlanner(0)
		ctx, cancel := context.WithCancel(context.Background())
		calls := 0
		ctx = WithNativeCompactionObserver(ctx, func(NativeCompactionObservation) { calls++ })
		_, err := p.Complete(ctx, activeMessages, nil,
			WithMaxTokens(4),
			WithDecodeTokenObserver(func(string, string) { cancel() }),
		)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Complete error = %v, want generation-time context.Canceled", err)
		}
		if calls != 0 {
			t.Fatalf("failed generation observer calls = %d, want 0", calls)
		}
	})
}

func nativeCompactionObserverMessages() []Message {
	return []Message{
		{Role: RoleSystem, Content: "System prompt invariant instructions."},
		{Role: RoleUser, Content: "Read the file first."},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "r1", Type: "function", Function: Func{Name: "Read", Arguments: `{"path":"large.txt"}`}}}},
		{Role: RoleTool, ToolCallID: "r1", Content: strings.Repeat("native-compaction-secret-marker line\n", 30)},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "e1", Type: "function", Function: Func{Name: "Edit", Arguments: `{"path":"large.txt"}`}}}},
		{Role: RoleTool, ToolCallID: "e1", Content: "ok"},
		{Role: RoleAssistant, Content: "File edited."},
		{Role: RoleUser, Content: "Middle turn 1"},
		{Role: RoleAssistant, Content: "Middle reply 1"},
		{Role: RoleUser, Content: "Middle turn 2"},
		{Role: RoleAssistant, Content: "Middle reply 2"},
		{Role: RoleUser, Content: "Latest query: what is final state?"},
	}
}

func assertNativeCompactionObservation(t *testing.T, got NativeCompactionObservation) {
	t.Helper()
	if got.DroppedMessages <= 0 || got.PreBytes <= got.PostBytes || got.ShedBytes != got.PreBytes-got.PostBytes {
		t.Fatalf("invalid compaction quantities: %#v", got)
	}
	if got.PreEstimatedTokens <= got.PostEstimatedTokens || got.PostEstimatedTokens <= 0 {
		t.Fatalf("invalid token estimates: pre=%d post=%d", got.PreEstimatedTokens, got.PostEstimatedTokens)
	}
	if got.TokenMethod != "typed_message_estimate" || got.SerializationMethod != "typed_messages_json" || got.ProofFailure != "" {
		t.Fatalf("invalid proof vocabulary: %#v", got)
	}
	for name, hash := range map[string]string{
		"pre": got.PreSHA256, "post": got.PostSHA256,
		"pre-prefix": got.PrePrefixSHA256, "post-prefix": got.PostPrefixSHA256,
		"pre-suffix": got.PreSuffixSHA256, "post-suffix": got.PostSuffixSHA256,
	} {
		if len(hash) != 64 {
			t.Fatalf("%s hash length = %d, want 64", name, len(hash))
		}
	}
	if got.PreSHA256 == got.PostSHA256 {
		t.Fatal("pre/post hashes unexpectedly match")
	}
	if got.PrefixBytes <= 0 || got.SuffixBytes <= 0 {
		t.Fatalf("structural regions missing: prefix=%d suffix=%d", got.PrefixBytes, got.SuffixBytes)
	}
	if got.PrePrefixSHA256 != got.PostPrefixSHA256 || got.PreSuffixSHA256 != got.PostSuffixSHA256 {
		t.Fatalf("structural identity mismatch: %#v", got)
	}
}
