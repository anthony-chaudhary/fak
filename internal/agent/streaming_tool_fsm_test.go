package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// TestStreamingToolFSM_TokenFeedsAndIdentification tests incremental token feeds
// and confirms tool identification occurs within 15 tokens across formats.
func TestStreamingToolFSM_TokenFeedsAndIdentification(t *testing.T) {
	t.Run("json_subword_tokens", func(t *testing.T) {
		fsm := NewStreamingToolFSM()
		tokens := []string{
			"{\"",
			"name",
			"\": \"",
			"Read",
			"\", ",
			"\"arguments",
			"\": {",
			"\"filePath",
			"\": \"",
			"internal/agent/streaming_tool_fsm.go",
			"\"}}",
		}

		for _, tok := range tokens {
			if err := fsm.Feed(tok); err != nil {
				t.Fatalf("Feed(%q) error: %v", tok, err)
			}
		}

		if fsm.ToolName() != "Read" {
			t.Fatalf("ToolName = %q, want %q", fsm.ToolName(), "Read")
		}
		if fsm.ToolIdentifiedToken() > 15 {
			t.Fatalf("ToolIdentifiedToken = %d, want <= 15", fsm.ToolIdentifiedToken())
		}
		if !fsm.IsSpeculatable() {
			t.Fatalf("Read tool must be speculatable, got specReason: %v", fsm.SpecReason())
		}
		if fsm.State() != StateCompleted {
			t.Fatalf("State = %v, want StateCompleted", fsm.State())
		}
		if !strings.Contains(fsm.RawArgs(), "streaming_tool_fsm.go") {
			t.Fatalf("RawArgs = %q, expected path", fsm.RawArgs())
		}
	})

	t.Run("xml_hermes_tokens", func(t *testing.T) {
		fsm := NewStreamingToolFSM()
		tokens := []string{
			"<tool_call>\n",
			"{\"name\": ",
			"\"Read\", ",
			"\"arguments\": ",
			"{\"filePath\": \"internal/agent/main.go\"}",
			"}\n</tool_call>",
		}

		for _, tok := range tokens {
			if err := fsm.Feed(tok); err != nil {
				t.Fatalf("Feed error: %v", err)
			}
		}

		if fsm.ToolName() != "Read" {
			t.Fatalf("ToolName = %q, want Read", fsm.ToolName())
		}
		if fsm.ToolIdentifiedToken() > 15 {
			t.Fatalf("ToolIdentifiedToken = %d, want <= 15", fsm.ToolIdentifiedToken())
		}
		if fsm.State() != StateCompleted {
			t.Fatalf("State = %v, want StateCompleted", fsm.State())
		}
	})

	t.Run("qwen_parameter_tokens", func(t *testing.T) {
		fsm := NewStreamingToolFSM()
		tokens := []string{
			"<tool_call>\n",
			"<function=Read>\n",
			"<parameter=filePath>\n",
			"internal/agent/main.go\n",
			"</parameter>\n",
			"</function>\n",
			"</tool_call>",
		}

		for _, tok := range tokens {
			if err := fsm.Feed(tok); err != nil {
				t.Fatalf("Feed error: %v", err)
			}
		}

		if fsm.ToolName() != "Read" {
			t.Fatalf("ToolName = %q, want Read", fsm.ToolName())
		}
		if fsm.ToolIdentifiedToken() > 15 {
			t.Fatalf("ToolIdentifiedToken = %d, want <= 15", fsm.ToolIdentifiedToken())
		}
		if fsm.State() != StateCompleted {
			t.Fatalf("State = %v, want StateCompleted", fsm.State())
		}
		if !strings.Contains(fsm.RawArgs(), "filePath") {
			t.Fatalf("RawArgs = %q, expected filePath", fsm.RawArgs())
		}
	})

	t.Run("prose_preceding_tool_call", func(t *testing.T) {
		fsm := NewStreamingToolFSM()
		tokens := []string{
			"I will ",
			"now inspect ",
			"the requested file.\n",
			"<tool_call>\n",
			"{\"name\": \"Read\", \"arguments\": {\"filePath\": \"foo.go\"}}",
			"\n</tool_call>",
		}

		for _, tok := range tokens {
			_ = fsm.Feed(tok)
		}

		if fsm.ToolName() != "Read" {
			t.Fatalf("ToolName = %q, want Read", fsm.ToolName())
		}
		if fsm.State() != StateCompleted {
			t.Fatalf("State = %v, want StateCompleted", fsm.State())
		}
	})
}

// TestStreamingToolFSM_SpeculativeDispatchAndCommit tests read-only speculative
// dispatch and instantaneous commit on exact argument match.
func TestStreamingToolFSM_SpeculativeDispatchAndCommit(t *testing.T) {
	ctx := context.Background()
	fsm := NewStreamingToolFSM()

	tokens := []string{
		"{\"name\": \"Read\", ",
		"\"arguments\": {\"filePath\": \"internal/agent/turn.go\"}",
		"}",
	}

	for _, tok := range tokens {
		_ = fsm.Feed(tok)
	}

	if !fsm.IsSpeculatable() {
		t.Fatalf("Read tool should be speculatable")
	}

	executed := false
	runner := func(ctx context.Context, tool, args string) (string, error) {
		executed = true
		if tool != "Read" {
			return "", fmt.Errorf("unexpected tool: %s", tool)
		}
		if !strings.Contains(args, "turn.go") {
			return "", fmt.Errorf("unexpected args: %s", args)
		}
		time.Sleep(10 * time.Millisecond) // simulate file read latency
		return "file-content: package agent", nil
	}

	err := fsm.SpeculativeDispatch(ctx, runner)
	if err != nil {
		t.Fatalf("SpeculativeDispatch error: %v", err)
	}
	if !fsm.IsDispatched() {
		t.Fatalf("IsDispatched should be true")
	}

	// Allow speculative execution to complete in background while decode finishes
	time.Sleep(20 * time.Millisecond)

	// Authoritative commit
	t0 := time.Now()
	res, err := fsm.Commit("Read", "{\"filePath\": \"internal/agent/turn.go\"}")
	duration := time.Since(t0)

	if err != nil {
		t.Fatalf("Commit error: %v", err)
	}
	if res != "file-content: package agent" {
		t.Fatalf("Commit result = %q, want 'file-content: package agent'", res)
	}
	if !executed {
		t.Fatalf("runner was not executed")
	}
	// Verify fast path (0ms wait)
	if duration > 50*time.Millisecond {
		t.Fatalf("Commit took %v, expected fast path near 0ms", duration)
	}
}

// TestStreamingToolFSM_SpeculativeDispatchRunningWhenCommitted tests that Commit
// cleanly waits for an in-flight speculative execution when decode finishes early.
func TestStreamingToolFSM_SpeculativeDispatchRunningWhenCommitted(t *testing.T) {
	ctx := context.Background()
	fsm := NewStreamingToolFSM()

	_ = fsm.Feed("{\"name\": \"Read\", \"arguments\": {\"filePath\": \"foo.go\"}}")

	runner := func(ctx context.Context, tool, args string) (string, error) {
		time.Sleep(25 * time.Millisecond)
		return "delayed-content", nil
	}

	if err := fsm.SpeculativeDispatch(ctx, runner); err != nil {
		t.Fatalf("SpeculativeDispatch failed: %v", err)
	}

	// Commit immediately before runner finishes
	res, err := fsm.Commit("Read", "{\"filePath\": \"foo.go\"}")
	if err != nil {
		t.Fatalf("Commit failed: %v", err)
	}
	if res != "delayed-content" {
		t.Fatalf("res = %q, want 'delayed-content'", res)
	}
}

// TestStreamingToolFSM_Squash tests squash behavior on argument mismatch,
// non-speculatable tools, and explicit cancellation.
func TestStreamingToolFSM_Squash(t *testing.T) {
	t.Run("squash_on_argument_mismatch", func(t *testing.T) {
		ctx := context.Background()
		fsm := NewStreamingToolFSM()

		_ = fsm.Feed("{\"name\": \"Read\", \"arguments\": {\"filePath\": \"expected.go\"}}")

		runner := func(ctx context.Context, tool, args string) (string, error) {
			return "expected-content", nil
		}
		if err := fsm.SpeculativeDispatch(ctx, runner); err != nil {
			t.Fatalf("dispatch error: %v", err)
		}

		time.Sleep(10 * time.Millisecond)

		// Authoritative model output diverged to different file
		res, err := fsm.Commit("Read", "{\"filePath\": \"diverged.go\"}")
		if !errors.Is(err, ErrSpeculationMismatch) {
			t.Fatalf("Commit err = %v, want ErrSpeculationMismatch", err)
		}
		if res != "" {
			t.Fatalf("Commit res = %q, want empty", res)
		}
		if fsm.State() != StateSquashed {
			t.Fatalf("State = %v, want StateSquashed", fsm.State())
		}
		if !strings.Contains(fsm.SquashReason(), "argument mismatch") {
			t.Fatalf("SquashReason = %q, expected 'argument mismatch'", fsm.SquashReason())
		}
	})

	t.Run("squash_on_tool_name_mismatch", func(t *testing.T) {
		ctx := context.Background()
		fsm := NewStreamingToolFSM()

		_ = fsm.Feed("{\"name\": \"Read\", \"arguments\": {\"filePath\": \"a.go\"}}")

		runner := func(ctx context.Context, tool, args string) (string, error) {
			return "content", nil
		}
		_ = fsm.SpeculativeDispatch(ctx, runner)

		_, err := fsm.Commit("Grep", "{\"filePath\": \"a.go\"}")
		if !errors.Is(err, ErrSpeculationMismatch) {
			t.Fatalf("expected mismatch on tool name, got: %v", err)
		}
		if fsm.State() != StateSquashed {
			t.Fatalf("State = %v, want StateSquashed", fsm.State())
		}
	})

	t.Run("non_speculatable_mutating_tool_refused", func(t *testing.T) {
		ctx := context.Background()
		fsm := NewStreamingToolFSM()

		_ = fsm.Feed("{\"name\": \"Write\", \"arguments\": {\"filePath\": \"secret.txt\", \"content\": \"leak\"}}")

		if fsm.ToolName() != "Write" {
			t.Fatalf("ToolName = %q, want Write", fsm.ToolName())
		}
		if fsm.IsSpeculatable() {
			t.Fatalf("Write tool must NOT be speculatable")
		}

		executed := false
		runner := func(ctx context.Context, tool, args string) (string, error) {
			executed = true
			return "ok", nil
		}

		err := fsm.SpeculativeDispatch(ctx, runner)
		if !errors.Is(err, ErrNotSpeculatable) {
			t.Fatalf("SpeculativeDispatch error = %v, want ErrNotSpeculatable", err)
		}
		if executed {
			t.Fatalf("runner must not have been executed for mutating tool")
		}
		if fsm.IsDispatched() {
			t.Fatalf("IsDispatched should be false")
		}
	})

	t.Run("explicit_squash_cancels_background_context", func(t *testing.T) {
		ctx := context.Background()
		fsm := NewStreamingToolFSM()

		_ = fsm.Feed("{\"name\": \"Read\", \"arguments\": {\"filePath\": \"bigfile.txt\"}}")

		cancelledCh := make(chan struct{})
		runner := func(runCtx context.Context, tool, args string) (string, error) {
			select {
			case <-runCtx.Done():
				close(cancelledCh)
				return "", runCtx.Err()
			case <-time.After(1 * time.Second):
				return "should-not-reach", nil
			}
		}

		if err := fsm.SpeculativeDispatch(ctx, runner); err != nil {
			t.Fatalf("dispatch error: %v", err)
		}

		// Explicit squash
		fsm.Squash("turn cancelled by operator")
		if fsm.State() != StateSquashed {
			t.Fatalf("State = %v, want StateSquashed", fsm.State())
		}
		if fsm.SquashReason() != "turn cancelled by operator" {
			t.Fatalf("SquashReason = %q", fsm.SquashReason())
		}

		select {
		case <-cancelledCh:
			// Background runner context was cancelled promptly
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("background runner was not cancelled on Squash")
		}

		_, err := fsm.Commit("Read", "{\"filePath\": \"bigfile.txt\"}")
		if !errors.Is(err, ErrSpeculationSquashed) {
			t.Fatalf("Commit on squashed FSM must return ErrSpeculationSquashed, got %v", err)
		}
	})
}

// TestStreamingToolFSM_Bash_ReadOnlyVsMutating verifies Bash tool speculatability
// gating based on command safety (e.g. cat/ls vs rm/curl).
func TestStreamingToolFSM_Bash_ReadOnlyVsMutating(t *testing.T) {
	t.Run("bash_read_only_ls", func(t *testing.T) {
		fsm := NewStreamingToolFSM()
		_ = fsm.Feed("{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls -la\"}}")
		if !fsm.IsSpeculatable() {
			t.Fatalf("Bash with 'ls -la' should be speculatable")
		}
		if fsm.SpecReason() != vdso.SpecOK {
			t.Fatalf("SpecReason = %v, want SpecOK", fsm.SpecReason())
		}
	})

	t.Run("bash_mutating_rm", func(t *testing.T) {
		fsm := NewStreamingToolFSM()
		_ = fsm.Feed("{\"name\": \"Bash\", \"arguments\": {\"command\": \"rm -rf /tmp/test\"}}")
		if fsm.IsSpeculatable() {
			t.Fatalf("Bash with 'rm -rf' must NOT be speculatable")
		}
		if !fsm.SpecReason().Refused() {
			t.Fatalf("SpecReason must refuse mutating bash call")
		}
	})
}

// TestStreamingToolFSM_ConcurrentSafety exercises the state machine under heavy
// concurrent access to verify race freedom with go test -race.
func TestStreamingToolFSM_ConcurrentSafety(t *testing.T) {
	const goroutines = 25
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			fsm := NewStreamingToolFSM()

			feedDone := make(chan struct{})
			go func() {
				defer close(feedDone)
				tokens := []string{
					"{\"",
					"name",
					"\": \"",
					"Read",
					"\", \"arguments\": {\"filePath\": \"test.txt\"}}",
				}
				for _, tok := range tokens {
					_ = fsm.Feed(tok)
					_ = fsm.State()
					_ = fsm.ToolName()
					_ = fsm.IsSpeculatable()
				}
			}()

			go func() {
				_ = fsm.Buffer()
				_ = fsm.TokenCount()
				_ = fsm.EarlyArgs()
				_ = fsm.RawArgs()
			}()

			<-feedDone

			if idx%2 == 0 {
				_ = fsm.SpeculativeDispatch(context.Background(), func(ctx context.Context, tool, args string) (string, error) {
					return "ok", nil
				})
				_, _ = fsm.Commit("Read", "{\"filePath\": \"test.txt\"}")
			} else {
				fsm.Squash("concurrent test squash")
			}
		}(i)
	}

	wg.Wait()
}

// TestStreamingToolFSM_LoopTurnIntegration verifies integration with RunArmStream
// when speculative execution is enabled.
func TestStreamingToolFSM_LoopTurnIntegration(t *testing.T) {
	planner := &scriptedStreamingPlanner{
		turns: []*Completion{
			{
				Message: Message{
					Content: "I will read the file: <tool_call>{\"name\": \"fetch_doc\", \"arguments\": {\"id\": \"doc1\"}}</tool_call>",
					ToolCalls: []ToolCall{
						{
							ID: "call_fetch_1",
							Function: Func{
								Name:      "fetch_doc",
								Arguments: `{"id": "doc1"}`,
							},
						},
					},
				},
			},
			{
				Message: Message{
					Content: "Doc 1 content processed successfully.",
				},
			},
		},
	}

	var capturedFSMs []*StreamingToolFSM
	metrics, err := RunArmStream(
		context.Background(),
		planner,
		"Fetch documentation doc1",
		false,
		3,
		nil,
		nil,
		WithStreamingSpeculation(true),
		WithStreamingFSMHook(func(f *StreamingToolFSM) {
			capturedFSMs = append(capturedFSMs, f)
		}),
	)
	if err != nil {
		t.Fatalf("RunArmStream failed: %v", err)
	}
	if len(capturedFSMs) == 0 {
		t.Fatalf("StreamingToolFSM hook was not called")
	}
	// Turn 1 had the tool call
	if capturedFSMs[0].ToolName() != "fetch_doc" {
		t.Fatalf("Turn 1 FSM ToolName = %q, want fetch_doc", capturedFSMs[0].ToolName())
	}
	if metrics.Turns != 2 {
		t.Fatalf("metrics.Turns = %d, want 2", metrics.Turns)
	}
}
