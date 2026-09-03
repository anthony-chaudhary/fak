package agentopt

import (
	"fmt"
	"strings"
	"testing"
)

func TestReflexionTraceSummarizer(t *testing.T) {
	summarizer := NewReflexionSummarizer()

	t.Run("ToolErrorExtraction", func(t *testing.T) {
		failed := FailedAttemptInfo{
			AttemptIndex: 1,
			Goal:         "Read confidential configuration file",
			Action:       "read /etc/shadow",
			ToolName:     "bash",
			ToolArgs:     "cat /etc/shadow",
			ToolError:    "cat: /etc/shadow: Permission denied",
		}

		record := summarizer.SummarizeFailure(failed)

		if record.AttemptIndex != 1 {
			t.Fatalf("expected AttemptIndex 1, got %d", record.AttemptIndex)
		}
		if !strings.Contains(record.FailedAction, "read /etc/shadow") {
			t.Errorf("expected FailedAction to contain 'read /etc/shadow', got %q", record.FailedAction)
		}
		if !strings.Contains(strings.ToLower(record.RootCause), "permission denied") {
			t.Errorf("expected RootCause to mention permission denied, got %q", record.RootCause)
		}
		if len(record.ConcreteMitigation) == 0 {
			t.Errorf("expected non-empty ConcreteMitigation")
		}
		if !strings.Contains(record.ConstraintRule, "DO NOT") {
			t.Errorf("expected ConstraintRule to contain 'DO NOT', got %q", record.ConstraintRule)
		}

		// Ensure strictly < 200 tokens
		toks := record.Tokens()
		if toks >= MaxReflexionTokens {
			t.Errorf("expected record tokens < %d, got %d", MaxReflexionTokens, toks)
		}
	})

	t.Run("AssertionFailureExtraction", func(t *testing.T) {
		failed := FailedAttemptInfo{
			AttemptIndex:     2,
			Goal:             "Query payment gateway endpoint",
			Action:           "http_get /v1/payments/pay_123",
			AssertionFailure: "AssertionError: expected status code 200 but received 404 Not Found",
		}

		record := summarizer.SummarizeFailure(failed)

		if record.AttemptIndex != 2 {
			t.Fatalf("expected AttemptIndex 2, got %d", record.AttemptIndex)
		}
		if !strings.Contains(record.RootCause, "404 Not Found") {
			t.Errorf("expected RootCause to include 404 Not Found, got %q", record.RootCause)
		}
		if len(record.ConcreteMitigation) == 0 {
			t.Errorf("expected non-empty ConcreteMitigation")
		}
		if record.Tokens() >= MaxReflexionTokens {
			t.Errorf("expected tokens < %d, got %d", MaxReflexionTokens, record.Tokens())
		}
	})

	t.Run("ExceptionTraceExtraction", func(t *testing.T) {
		panicTrace := `panic: runtime error: index out of range [5] with length 2
goroutine 1 [running]:
main.processItems(0xc0000a4000, 0x2, 0x2)
	/workspace/processor.go:42 +0x3f
main.main()
	/workspace/main.go:12 +0x22`

		failed := FailedAttemptInfo{
			AttemptIndex:   3,
			Goal:           "Process items in batch queue",
			Action:         "processItems(queue)",
			ExceptionTrace: panicTrace,
		}

		record := summarizer.SummarizeFailure(failed)

		if record.AttemptIndex != 3 {
			t.Fatalf("expected AttemptIndex 3, got %d", record.AttemptIndex)
		}
		if !strings.Contains(record.RootCause, "index out of range") {
			t.Errorf("expected RootCause to capture panic reason, got %q", record.RootCause)
		}
		if !strings.Contains(strings.ToLower(record.ConstraintRule), "bounds") &&
			!strings.Contains(strings.ToLower(record.ConcreteMitigation), "bounds") {
			t.Errorf("expected mitigation or rule to address bounds, got mitigation=%q rule=%q",
				record.ConcreteMitigation, record.ConstraintRule)
		}
		if record.Tokens() >= MaxReflexionTokens {
			t.Errorf("expected tokens < %d, got %d", MaxReflexionTokens, record.Tokens())
		}
	})

	t.Run("MemoryConstraintInjection", func(t *testing.T) {
		wm := NewWorkingMemory(1000)

		failed := FailedAttemptInfo{
			AttemptIndex: 1,
			ToolName:     "database_query",
			ToolArgs:     "SELECT * FROM secure_vault",
			ToolError:    "Access denied for user 'guest'@'localhost'",
		}

		record := summarizer.SummarizeFailure(failed)
		constraintText := summarizer.FormatAsMemoryConstraint(record)

		if !strings.Contains(constraintText, "[REFLEXION_CONSTRAINT_ATTEMPT_1]") {
			t.Errorf("expected constraint tag in formatted string, got %q", constraintText)
		}

		// Inject into operational working memory
		err := summarizer.InjectWorkingMemory(wm, record)
		if err != nil {
			t.Fatalf("failed to inject into working memory: %v", err)
		}

		// Retrieve from working memory
		key := fmt.Sprintf("reflexion_constraint_attempt_%d", record.AttemptIndex)
		storedVal, ok := wm.Get(key)
		if !ok {
			t.Fatalf("expected working memory to contain key %q", key)
		}
		if storedVal != constraintText {
			t.Errorf("stored value does not match formatted constraint:\ngot:  %q\nwant: %q", storedVal, constraintText)
		}

		// Token ceiling check
		toks := EstimateTokens(constraintText)
		if toks >= MaxReflexionTokens {
			t.Errorf("expected constraint tokens < %d, got %d", MaxReflexionTokens, toks)
		}
	})

	t.Run("SubsequentAttemptConstraintAccumulation", func(t *testing.T) {
		wm := NewWorkingMemory(1000)

		// Attempt 1: fails on timeout
		failed1 := FailedAttemptInfo{
			AttemptIndex: 1,
			Action:       "batch_fetch_all_records",
			ToolError:    "context deadline exceeded: operation timed out after 30s",
		}
		rec1 := summarizer.SummarizeFailure(failed1)
		if err := summarizer.InjectMemoryConstraint(wm, rec1); err != nil {
			t.Fatalf("unexpected error injecting attempt 1: %v", err)
		}

		// Attempt 2: fails on invalid syntax
		failed2 := FailedAttemptInfo{
			AttemptIndex: 2,
			Action:       "batch_fetch_chunks",
			ToolError:    "invalid JSON payload: unexpected token '}' at line 10",
		}
		rec2 := summarizer.SummarizeFailure(failed2)
		if err := summarizer.InjectMemoryConstraint(wm, rec2); err != nil {
			t.Fatalf("unexpected error injecting attempt 2: %v", err)
		}

		// Check both constraints live in memory
		v1, ok1 := wm.Get("reflexion_constraint_attempt_1")
		v2, ok2 := wm.Get("reflexion_constraint_attempt_2")
		if !ok1 || !ok2 {
			t.Fatalf("expected both attempts in working memory: ok1=%v ok2=%v", ok1, ok2)
		}
		if !strings.Contains(v1, "Action: batch_fetch_all_records") {
			t.Errorf("unexpected content in v1: %s", v1)
		}
		if !strings.Contains(v2, "Action: batch_fetch_chunks") {
			t.Errorf("unexpected content in v2: %s", v2)
		}

		// Format operational constraint block for attempt 3
		block := summarizer.FormatOperationalMemoryBlock([]ReflexionRecord{rec1, rec2})
		if !strings.Contains(block, "Attempt 1:") || !strings.Contains(block, "Attempt 2:") {
			t.Errorf("expected operational constraint block to mention both attempts, got:\n%s", block)
		}
	})

	t.Run("LongTraceCompressionUnderTokenCap", func(t *testing.T) {
		// Generate an oversized, noisy failure trace (~2000 chars)
		var longBuilder strings.Builder
		longBuilder.WriteString("panic: runtime error: slice bounds out of range [:9999] with capacity 100\n")
		for i := 0; i < 40; i++ {
			longBuilder.WriteString(fmt.Sprintf("goroutine 42 [running]:\nruntime/debug.Stack()\n\t/usr/local/go/src/runtime/debug/stack.go:%d +0x80\n", i*10))
		}

		failed := FailedAttemptInfo{
			AttemptIndex:   4,
			Action:         "slice_reallocation_loop",
			ExceptionTrace: longBuilder.String(),
		}

		record := summarizer.SummarizeFailure(failed)
		constraintText := summarizer.FormatAsMemoryConstraint(record)
		tokens := EstimateTokens(constraintText)

		if tokens >= MaxReflexionTokens {
			t.Errorf("expected compressed constraint tokens < %d, got %d", MaxReflexionTokens, tokens)
		}
		if !strings.Contains(record.RootCause, "slice bounds out of range") {
			t.Errorf("expected extracted root cause to retain core panic message, got %q", record.RootCause)
		}
	})

	t.Run("ReflectHelper", func(t *testing.T) {
		failed := FailedAttemptInfo{
			AttemptIndex: 5,
			Action:       "api_call",
			ToolError:    "HTTP 429: Too Many Requests - rate limit exceeded",
		}

		ref := summarizer.Reflect(failed)
		if ref.Record.AttemptIndex != 5 {
			t.Errorf("expected AttemptIndex 5, got %d", ref.Record.AttemptIndex)
		}
		if ref.EstimatedTokens >= MaxReflexionTokens {
			t.Errorf("expected tokens < %d, got %d", MaxReflexionTokens, ref.EstimatedTokens)
		}
		if !strings.Contains(ref.Constraint, "[REFLEXION_CONSTRAINT_ATTEMPT_5]") {
			t.Errorf("expected constraint tag in reflection output, got %q", ref.Constraint)
		}

		wm := NewWorkingMemory(500)
		if err := summarizer.InjectReflection(wm, ref); err != nil {
			t.Fatalf("failed to inject reflection: %v", err)
		}
		val, ok := wm.Get("reflexion_constraint_attempt_5")
		if !ok || val != ref.Constraint {
			t.Fatalf("expected stored reflection constraint in memory, got ok=%v val=%q", ok, val)
		}
	})

	t.Run("FromExecutionSteps", func(t *testing.T) {
		steps := []ExecutionStep{
			{
				StepIndex: 1,
				Thought:   "Check file status",
				ToolName:  "stat_file",
				Receipt: &ToolReceipt{
					ToolName: "stat_file",
					Output:   "File exists",
				},
			},
			{
				StepIndex: 2,
				Thought:   "Open target file for writing",
				ToolName:  "open_file",
				Receipt: &ToolReceipt{
					ToolName: "open_file",
					Error:    "open /root/secret.key: permission denied",
				},
			},
		}

		failedInfo := FromExecutionSteps(1, "Access secret keys", steps, "Execution terminated with errors")
		if failedInfo.ToolName != "open_file" {
			t.Errorf("expected ToolName 'open_file', got %q", failedInfo.ToolName)
		}
		if failedInfo.ToolError != "open /root/secret.key: permission denied" {
			t.Errorf("expected ToolError 'open /root/secret.key: permission denied', got %q", failedInfo.ToolError)
		}

		rec := summarizer.SummarizeFailure(failedInfo)
		if !strings.Contains(strings.ToLower(rec.RootCause), "permission denied") {
			t.Errorf("expected RootCause to reflect permission denied, got %q", rec.RootCause)
		}
		if rec.Tokens() >= MaxReflexionTokens {
			t.Errorf("expected tokens < %d, got %d", MaxReflexionTokens, rec.Tokens())
		}
	})
}
