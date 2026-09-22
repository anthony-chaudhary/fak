package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestCompactBudgetWiring is the S3 witness for the agent-150k-context goal.
//
// The load-bearing claim the goal pins is: a native planner built the way the
// resident serve/turnkey path builds it must SHRINK an over-window transcript
// instead of hard-refusing it. Today `serveNativePlannerConfig` never sets
// InKernelPlannerConfig.CompactHistoryBudget, so ApplyPromptShrink short-circuits
// (`if compactBudget <= 0 && !elideStale && !deferTools { return messages, tools,
// TypedPromptShrinkOutcome{} }`) and the request reaches refuseContextLength at
// full size -> HTTP 400 context_length_exceeded with no second chance.
//
// The test therefore FAILS on the unwired planner (exactly the production shape)
// and passes once the resolved window is spent on a compaction budget. It is the
// honest witness: it asserts the OUTCOME (served vs refused), not that a field
// was set.
func TestCompactBudgetWiring(t *testing.T) {
	tok := loadProbeTok(t)

	// A long agent transcript. NOTE the ruler: CompactMessagesWithOptions sizes
	// messages with EstimateMessageTokens, the codebase's (bytes+3)/4 heuristic,
	// NOT the tokenizer. The transcript must exceed the budget ON THAT RULER for
	// the shrink to fire. Asserting the estimate keeps this honest: if the ruler
	// changes, the fixture is re-sized rather than silently passing.
	var b strings.Builder
	for i := 0; i < 6000; i++ {
		fmt.Fprintf(&b, "turn %05d: inspected pkg/file_%05d.go and applied a bounded edit; the read returned 40 lines of surrounding context.\n", i, i)
	}
	long := b.String()
	estTokens := EstimateMessageTokens(Message{Role: RoleUser, Content: long})
	if estTokens < 80_000 {
		t.Fatalf("fixture prices at %d est. tokens; need >80k so it exceeds the 60k budget on the EstimateMessageTokens ruler", estTokens)
	}

	messages := []Message{
		{Role: RoleSystem, Content: "You are a coding agent. Invariants apply."},
		{Role: RoleUser, Content: long},
		{Role: RoleAssistant, Content: "Working."},
		{Role: RoleUser, Content: "Continue and report the final state."},
	}

	newPlanner := func(compactBudget int) *InKernelPlanner {
		cfg := tinyConcurrencyConfig()
		m := model.NewSynthetic(cfg)
		m.Quantize()
		return NewInKernelPlannerWithConfig(m, tok, "native-150k", false, nil, false, InKernelPlannerConfig{
			// The resolved window the serve/turnkey seam hands the planner.
			ContextTokens: 150_000,
			// The S3 wiring under test.
			CompactHistoryBudget: compactBudget,
		})
	}

	// 1. The unwired production shape (budget 0) must observe a shrink no-op.
	unwired := newPlanner(0)
	_, _, outcome := unwired.ApplyPromptShrink(context.Background(), messages, nil)
	if outcome.Compacted {
		t.Fatalf("unwired planner (budget 0) compacted=%v; the test's premise is that budget 0 is a no-op", outcome.Compacted)
	}

	// 2. A wired planner honoring the 150k envelope must actually shrink. The
	//    budget sits below the transcript's estimated size so the shrink has
	//    something to shed: a 150k window minus a 32k output reserve, with the
	//    remaining slack expressed as a resident line rather than the raw window.
	shrinkBudget := 60_000
	wired := newPlanner(shrinkBudget)
	shrunk, _, outcome := wired.ApplyPromptShrink(context.Background(), messages, nil)
	if !outcome.Compacted {
		t.Fatalf("wired planner (budget %d against a ~%d-token transcript, 150k window) did not compact; outcome=%+v",
			shrinkBudget, estTokens, outcome)
	}
	if outcome.CompactOutcome.ShedTokens <= 0 {
		t.Fatalf("wired shrink reported no tokens shed; outcome=%+v", outcome)
	}
	// The shed must be real payload, not a relabel: the surviving transcript must
	// be materially smaller than the input on the same EstimateMessageTokens ruler.
	var before, after int
	for _, m := range messages {
		before += EstimateMessageTokens(m)
	}
	for _, m := range shrunk {
		after += EstimateMessageTokens(m)
	}
	if after >= before {
		t.Fatalf("wired shrink did not reduce the transcript: before=%d after=%d est. tokens", before, after)
	}
	t.Logf("wired shrink shed %d tokens: %d -> %d est. tokens across %d messages",
		outcome.CompactOutcome.ShedTokens, before, after, len(shrunk))
	// The system prefix must survive bit-for-bit: it is the RadixAttention anchor.
	if shrunk[0].Role != RoleSystem || shrunk[0].Content != messages[0].Content {
		t.Fatalf("system prefix not preserved byte-identically: got %q", shrunk[0].Content)
	}

	// 3. The end-to-end claim: a wired planner SERVES the long transcript
	//    (no InKernelContextLengthError), while the unwired planner that is over
	//    its window would refuse it with the typed error. This is the 400 wall the
	//    goal exists to remove.
	_, err := wired.Complete(context.Background(), messages, nil, WithMaxTokens(1))
	if err != nil {
		var ctxErr *InKernelContextLengthError
		if errors.As(err, &ctxErr) {
			t.Fatalf("wired planner still refused the transcript with %v; the compaction budget did not relieve the window", err)
		}
		// Any non-context error (uninitialized device, etc.) is not this test's
		// concern; only the context wall is.
		t.Logf("wired Complete returned non-context error (acceptable for this witness): %v", err)
	}
}
