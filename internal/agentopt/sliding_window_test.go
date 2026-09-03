package agentopt

import (
	"fmt"
	"strings"
	"testing"
)

// Family 5: Context-window management & compression.
// Tests for pinned sliding-window context reduction with preserved head and tail.

func TestSlidingWindowContextCompaction(t *testing.T) {
	t.Run("ExtremeContextPressurePreservesHeadAndTail", func(t *testing.T) {
		// 1. Define immutable head context: system prompt, tool schemas, objective declaration.
		systemPrompt := "You are a senior compiler and systems engineer operating inside the kernel runtime. " +
			"Maintain absolute correctness, zero external dependencies, and strict token economy."
		toolSchemas := []string{
			"read_file(path string) string",
			"edit_file(path string, old string, new string) bool",
			"execute_cmd(command string) (string, int)",
			"grep_code(pattern string, path string) []string",
		}
		tools := []ToolSchema{
			{
				Name:        "read_file",
				Description: "Read file contents from workspace disk",
				Required:    []string{"path"},
			},
			{
				Name:        "edit_file",
				Description: "Perform exact string replacements in workspace files",
				Required:    []string{"path"},
			},
		}
		objective := "Implement Family 5 sliding-window context reduction with preserved head and tail."

		head := PinnedPreamble{
			SystemPrompt:         systemPrompt,
			ToolSchemas:          toolSchemas,
			Tools:                tools,
			ObjectiveDeclaration: objective,
		}

		headTokens := head.TotalTokens()
		if headTokens <= 0 {
			t.Fatalf("expected positive head tokens, got %d", headTokens)
		}

		// 2. Generate intermediate turns with heavy bodies (large file contents, diffs, tool traces).
		// We generate 20 intermediate turns + 4 tail turns = 24 turns total.
		var turns []ConversationTurn
		turnIdx := 0

		for i := 0; i < 10; i++ {
			// User query turn
			turns = append(turns, ConversationTurn{
				Index:   turnIdx,
				Role:    "user",
				Content: fmt.Sprintf("Step %d: Inspect intermediate AST outlines and summarize structural symbols in pkg %d.", i, i),
			})
			turnIdx++

			// Assistant response with tool invocation
			toolCmd := fmt.Sprintf("grep_code(pattern=\"FuncOutline\", path=\"pkg%d/ast.go\")", i)
			turns = append(turns, ConversationTurn{
				Index:     turnIdx,
				Role:      "assistant",
				Content:   fmt.Sprintf("I will run grep_code across pkg%d to locate symbol definitions and interface boundaries.", i),
				ToolCalls: []string{toolCmd},
				ToolResult: strings.Repeat(fmt.Sprintf(
					"pkg%d/ast.go:42: type FuncOutline struct { Name string; Signature string; Exported bool; Tokens int }\n"+
						"pkg%d/ast.go:88: func ParseOutline(src string) (*CodeOutline, error) { return nil, nil }\n", i, i), 4),
			})
			turnIdx++
		}

		// 3. Tail context: 4 recent turns that the agent needs to continue ongoing work.
		tailTurnA := ConversationTurn{
			Index:   turnIdx,
			Role:    "user",
			Content: "Now wire the reducer into the pipeline and verify token reduction metrics.",
		}
		turnIdx++
		tailTurnB := ConversationTurn{
			Index:      turnIdx,
			Role:       "assistant",
			Content:    "Wiring sliding-window reducer and running unit test verification.",
			ToolCalls:  []string{"execute_cmd(command=\"go test ./internal/agentopt\")"},
			ToolResult: "PASS: TestSlidingWindowContextCompaction (0.01s)",
		}
		turnIdx++
		tailTurnC := ConversationTurn{
			Index:   turnIdx,
			Role:    "user",
			Content: "Confirm that head context and tail turns remain completely unchanged.",
		}
		turnIdx++
		tailTurnD := ConversationTurn{
			Index:   turnIdx,
			Role:    "assistant",
			Content: "Verification complete: head context and 4 recent tail turns are pinned and intact.",
		}
		turnIdx++

		turns = append(turns, tailTurnA, tailTurnB, tailTurnC, tailTurnD)

		totalOriginalTokens := headTokens
		for _, tTurn := range turns {
			totalOriginalTokens += tTurn.TotalTokens()
		}

		if totalOriginalTokens < 2000 {
			t.Fatalf("expected heavy dialog (>2000 tokens), got %d", totalOriginalTokens)
		}

		// 4. Apply extreme context pressure: ceiling = 500 tokens (down from >2000 tokens).
		reducer := NewSlidingWindowReducer(4, WithMinTailSize(2), WithSummaryBudget(150))
		maxTokenCeiling := 500

		result := reducer.ReduceWindow(head, turns, maxTokenCeiling)

		// 5. Verify immutable head preservation.
		if result.Head.SystemPrompt != head.SystemPrompt {
			t.Fatalf("head system prompt corrupted: expected %q, got %q", head.SystemPrompt, result.Head.SystemPrompt)
		}
		if result.Head.ObjectiveDeclaration != head.ObjectiveDeclaration {
			t.Fatalf("head objective corrupted: expected %q, got %q", head.ObjectiveDeclaration, result.Head.ObjectiveDeclaration)
		}
		if result.Head.GetObjective() != head.GetObjective() {
			t.Fatalf("head GetObjective corrupted: expected %q, got %q", head.GetObjective(), result.Head.GetObjective())
		}
		if len(result.Head.ToolSchemas) != len(head.ToolSchemas) {
			t.Fatalf("head tool schemas altered: expected %d, got %d", len(head.ToolSchemas), len(result.Head.ToolSchemas))
		}
		for i, ts := range head.ToolSchemas {
			if result.Head.ToolSchemas[i] != ts {
				t.Fatalf("head tool schema %d altered: expected %q, got %q", i, ts, result.Head.ToolSchemas[i])
			}
		}
		if len(result.Head.Tools) != len(head.Tools) {
			t.Fatalf("head tools altered: expected %d, got %d", len(head.Tools), len(result.Head.Tools))
		}
		for i, tool := range head.Tools {
			if result.Head.Tools[i].Name != tool.Name || result.Head.Tools[i].Description != tool.Description {
				t.Fatalf("head tool %d altered: expected %+v, got %+v", i, tool, result.Head.Tools[i])
			}
		}
		if result.HeadTokens != headTokens {
			t.Fatalf("expected headTokens %d, got %d", headTokens, result.HeadTokens)
		}

		// 6. Verify pinned tail preservation.
		if len(result.PinnedTail) < 2 || len(result.PinnedTail) > 4 {
			t.Fatalf("expected pinned tail length between 2 and 4, got %d", len(result.PinnedTail))
		}
		tailLen := len(result.PinnedTail)
		for i := 0; i < tailLen; i++ {
			expected := turns[len(turns)-tailLen+i]
			actual := result.PinnedTail[i]
			if actual.Index != expected.Index || actual.Role != expected.Role || actual.Content != expected.Content {
				t.Fatalf("tail turn %d mismatch: expected %+v, got %+v", i, expected, actual)
			}
		}

		// 7. Verify intermediate turns were folded into structured summary receipt.
		if !result.ReductionApplied {
			t.Fatalf("expected reduction to be applied under context pressure")
		}
		expectedFoldedCount := len(turns) - tailLen
		if result.TurnsFolded != expectedFoldedCount {
			t.Fatalf("expected %d turns folded, got %d", expectedFoldedCount, result.TurnsFolded)
		}
		if result.TurnsPreserved != tailLen {
			t.Fatalf("expected %d turns preserved, got %d", tailLen, result.TurnsPreserved)
		}
		if len(result.FoldedReceipts) != 1 {
			t.Fatalf("expected 1 folded receipt, got %d", len(result.FoldedReceipts))
		}

		receipt := result.FoldedReceipts[0]
		if receipt.StartIndex != turns[0].Index {
			t.Fatalf("expected receipt StartIndex=%d, got %d", turns[0].Index, receipt.StartIndex)
		}
		if receipt.EndIndex != turns[expectedFoldedCount-1].Index {
			t.Fatalf("expected receipt EndIndex=%d, got %d", turns[expectedFoldedCount-1].Index, receipt.EndIndex)
		}
		if receipt.TurnCount != expectedFoldedCount {
			t.Fatalf("expected receipt TurnCount=%d, got %d", expectedFoldedCount, receipt.TurnCount)
		}
		if receipt.TokensSaved <= 0 {
			t.Fatalf("expected positive tokens saved in receipt, got %d", receipt.TokensSaved)
		}
		if receipt.OriginalTokens <= receipt.FoldedTokens {
			t.Fatalf("expected original tokens (%d) > folded tokens (%d)", receipt.OriginalTokens, receipt.FoldedTokens)
		}
		if receipt.CompressionRatio <= 0.5 {
			t.Fatalf("expected compression ratio > 0.5, got %f", receipt.CompressionRatio)
		}
		if len(receipt.ToolsInvoked) == 0 {
			t.Fatalf("expected extracted tools invoked in receipt, got none")
		}
		if !strings.Contains(receipt.FormatContent(), "[Folded Turn Receipt:") {
			t.Fatalf("receipt formatted content missing header: %s", receipt.FormatContent())
		}

		// 8. Verify active turns assemble folded receipt turn followed by pinned tail.
		if len(result.ActiveTurns) != 1+tailLen {
			t.Fatalf("expected %d active turns (1 receipt + %d tail), got %d", 1+tailLen, tailLen, len(result.ActiveTurns))
		}
		receiptTurn := result.ActiveTurns[0]
		if receiptTurn.Role != "system" || receiptTurn.Metadata["folded_receipt"] != "true" {
			t.Fatalf("expected active turn 0 to be folded receipt turn, got %+v", receiptTurn)
		}
		for i := 0; i < tailLen; i++ {
			if result.ActiveTurns[1+i].Index != result.PinnedTail[i].Index {
				t.Fatalf("active turn %d index mismatch: expected %d, got %d", 1+i, result.PinnedTail[i].Index, result.ActiveTurns[1+i].Index)
			}
		}

		// 9. Verify token budget compliance.
		if result.TotalTokens > maxTokenCeiling {
			t.Fatalf("result total tokens (%d) exceeds maxTokenCeiling (%d)", result.TotalTokens, maxTokenCeiling)
		}
		if result.TokensSaved <= 0 {
			t.Fatalf("expected positive tokens saved overall, got %d", result.TokensSaved)
		}
	})

	t.Run("NoReductionWhenUnderBudget", func(t *testing.T) {
		head := PinnedPreamble{
			SystemPrompt: "Concise assistant",
		}
		turns := []ConversationTurn{
			{Index: 0, Role: "user", Content: "Hello"},
			{Index: 1, Role: "assistant", Content: "Hi there"},
			{Index: 2, Role: "user", Content: "Short task"},
			{Index: 3, Role: "assistant", Content: "Done"},
		}

		reducer := NewSlidingWindowReducer(2)
		result := reducer.ReduceWindow(head, turns, 2000)

		if result.ReductionApplied {
			t.Fatalf("expected ReductionApplied=false when comfortably under budget")
		}
		if result.TurnsFolded != 0 {
			t.Fatalf("expected 0 turns folded, got %d", result.TurnsFolded)
		}
		if result.TurnsPreserved != len(turns) {
			t.Fatalf("expected all %d turns preserved, got %d", len(turns), result.TurnsPreserved)
		}
		if len(result.ActiveTurns) != len(turns) {
			t.Fatalf("expected ActiveTurns length %d, got %d", len(turns), len(result.ActiveTurns))
		}
		if len(result.FoldedReceipts) != 0 {
			t.Fatalf("expected 0 folded receipts, got %d", len(result.FoldedReceipts))
		}
		if result.TokensSaved != 0 {
			t.Fatalf("expected 0 tokens saved, got %d", result.TokensSaved)
		}
		if len(result.PinnedTail) != 2 {
			t.Fatalf("expected pinned tail length 2, got %d", len(result.PinnedTail))
		}
	})

	t.Run("SeverePressureShrinksTailToMinTail", func(t *testing.T) {
		head := PinnedPreamble{
			SystemPrompt: "System instruction for memory-constrained worker.",
		}
		// 8 heavy turns
		var turns []ConversationTurn
		for i := 0; i < 8; i++ {
			turns = append(turns, ConversationTurn{
				Index:   i,
				Role:    "user",
				Content: strings.Repeat(fmt.Sprintf("User prompt line %d with substantial payload details. ", i), 5),
			})
		}

		// tailSize=4, minTailSize=1, very tight ceiling: 120 tokens
		reducer := NewSlidingWindowReducer(4, WithMinTailSize(1), WithSummaryBudget(50))
		result := reducer.ReduceWindow(head, turns, 120)

		if !result.ReductionApplied {
			t.Fatalf("expected reduction applied")
		}
		if len(result.PinnedTail) < 1 {
			t.Fatalf("expected at least minTailSize (1) pinned turn, got %d", len(result.PinnedTail))
		}
		// The most recent turn must always be in the pinned tail
		lastOrigTurn := turns[len(turns)-1]
		lastPinnedTurn := result.PinnedTail[len(result.PinnedTail)-1]
		if lastPinnedTurn.Index != lastOrigTurn.Index {
			t.Fatalf("expected last turn index %d, got %d", lastOrigTurn.Index, lastPinnedTurn.Index)
		}
		if result.Head.SystemPrompt != head.SystemPrompt {
			t.Fatalf("head system prompt must remain intact")
		}
	})

	t.Run("ShrinkWindowParity", func(t *testing.T) {
		head := PinnedPreamble{
			SystemPrompt:         "Kernel agent prompt",
			ObjectiveDeclaration: "Verify ShrinkWindow parity",
		}
		turns := []ConversationTurn{
			{Index: 0, Role: "user", Content: "Turn 0"},
			{Index: 1, Role: "assistant", Content: "Turn 1"},
			{Index: 2, Role: "user", Content: "Turn 2"},
			{Index: 3, Role: "assistant", Content: "Turn 3"},
			{Index: 4, Role: "user", Content: "Turn 4"},
			{Index: 5, Role: "assistant", Content: "Turn 5"},
		}

		r := NewSlidingWindowReducer(2)
		resReduce := r.ReduceWindow(head, turns, 80)
		resShrink := r.ShrinkWindow(head, turns, 80)

		if resReduce.TotalTokens != resShrink.TotalTokens {
			t.Fatalf("ReduceWindow tokens (%d) != ShrinkWindow tokens (%d)", resReduce.TotalTokens, resShrink.TotalTokens)
		}
		if resReduce.TurnsFolded != resShrink.TurnsFolded {
			t.Fatalf("ReduceWindow folded (%d) != ShrinkWindow folded (%d)", resReduce.TurnsFolded, resShrink.TurnsFolded)
		}
		if resReduce.TurnsPreserved != resShrink.TurnsPreserved {
			t.Fatalf("ReduceWindow preserved (%d) != ShrinkWindow preserved (%d)", resReduce.TurnsPreserved, resShrink.TurnsPreserved)
		}

		// Package-level functions
		pkgResReduce := ReduceWindow(head, turns, 80)
		pkgResShrink := ShrinkWindow(head, turns, 80)
		if pkgResReduce.TotalTokens != pkgResShrink.TotalTokens {
			t.Fatalf("pkg ReduceWindow tokens (%d) != pkg ShrinkWindow tokens (%d)", pkgResReduce.TotalTokens, pkgResShrink.TotalTokens)
		}
	})

	t.Run("EmptyAndSingleTurnEdgeCases", func(t *testing.T) {
		head := PinnedPreamble{
			SystemPrompt: "Test empty edge case",
		}

		// 0 turns
		resEmpty := ReduceWindow(head, nil, 500)
		if resEmpty.ReductionApplied {
			t.Fatalf("expected ReductionApplied=false for nil turns")
		}
		if resEmpty.Head.SystemPrompt != head.SystemPrompt {
			t.Fatalf("expected head preserved on nil turns")
		}

		// 1 turn
		singleTurn := []ConversationTurn{
			{Index: 0, Role: "user", Content: "Single turn question"},
		}
		resSingle := ReduceWindow(head, singleTurn, 500)
		if resSingle.ReductionApplied {
			t.Fatalf("expected ReductionApplied=false for single turn under budget")
		}
		if len(resSingle.PinnedTail) != 1 {
			t.Fatalf("expected single turn in pinned tail, got %d", len(resSingle.PinnedTail))
		}
	})

	t.Run("CustomSummarizerOption", func(t *testing.T) {
		head := PinnedPreamble{
			SystemPrompt: "Agent prompt with custom summarizer",
		}
		turns := []ConversationTurn{
			{Index: 0, Role: "user", Content: "Detailed question 0 with extensive context requirements."},
			{Index: 1, Role: "assistant", Content: "Detailed answer 1 analyzing multi-stage pipeline performance."},
			{Index: 2, Role: "user", Content: "Detailed question 2 exploring memory-bound bottlenecks."},
			{Index: 3, Role: "assistant", Content: "Detailed answer 3 discussing memory bandwidth and latency."},
			{Index: 4, Role: "user", Content: "Recent question 4 for pinned tail."},
			{Index: 5, Role: "assistant", Content: "Recent answer 5 for pinned tail."},
		}

		customCalled := false
		customSummarizer := func(intTurns []ConversationTurn, targetBudget int) FoldedTurnReceipt {
			customCalled = true
			return FoldedTurnReceipt{
				StartIndex:     intTurns[0].Index,
				EndIndex:       intTurns[len(intTurns)-1].Index,
				TurnCount:      len(intTurns),
				Summary:        "Custom summarized receipt",
				FoldedTokens:   15,
				OriginalTokens: 100,
				TokensSaved:    85,
			}
		}

		reducer := NewSlidingWindowReducer(2, WithSummarizer(customSummarizer))
		result := reducer.ReduceWindow(head, turns, 30)

		if !customCalled {
			t.Fatalf("expected custom summarizer to be invoked")
		}
		if len(result.FoldedReceipts) != 1 || result.FoldedReceipts[0].Summary != "Custom summarized receipt" {
			t.Fatalf("expected custom receipt in result, got %+v", result.FoldedReceipts)
		}
	})
}
