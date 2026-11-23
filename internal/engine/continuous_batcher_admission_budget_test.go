package engine

// Budgeted-admission witnesses: a prompt whose post-reuse remainder exceeds
// PrefillBudget is deferred into SlotStatePrefilling and consumed in budget-sized
// chunks, so no full-length forward runs inside the batcher lock. Short prompts and
// a zero budget keep the eager single-forward admission and its observables.

import (
	"context"
	"fmt"
	"testing"
)

// admissionBudgetPrompt builds n deterministic token ids from start.
func admissionBudgetPrompt(start, n int) []int {
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = start + i
	}
	return prompt
}

// admissionBudgetAnyPrefilling reports whether any slot is still consuming prompt
// chunks; the prefill arm is done once every slot has reached active decode.
func admissionBudgetAnyPrefilling(slots []*Slot) bool {
	for _, slot := range slots {
		if slot.State == SlotStatePrefilling {
			return true
		}
	}
	return false
}

// admissionBudgetAllReached reports whether every slot produced at least target tokens.
func admissionBudgetAllReached(slots []*Slot, target int) bool {
	for _, slot := range slots {
		if slot.TokensGenerated() < target {
			return false
		}
	}
	return true
}

// TestContinuousBatcher_AdmissionDefersOverBudgetUnflaggedPrompt pins the admission
// state of each budget/prompt-length combination: over-budget is deferred, a
// remainder equal to the budget is not, and budget 0 stays eager.
func TestContinuousBatcher_AdmissionDefersOverBudgetUnflaggedPrompt(t *testing.T) {
	const target = 2
	cases := []struct {
		name        string
		budget      int
		prompt      []int
		chunked     bool
		wantState   SlotState
		wantLastTok int
	}{
		{
			name:        "over-budget-unflagged",
			budget:      3,
			prompt:      admissionBudgetPrompt(10, 10),
			wantState:   SlotStatePrefilling,
			wantLastTok: 0,
		},
		{
			name:        "short-unflagged-stays-eager",
			budget:      3,
			prompt:      admissionBudgetPrompt(20, 2),
			wantState:   SlotStateActiveDecode,
			wantLastTok: 21,
		},
		{
			name:        "exactly-budget-unflagged-stays-eager",
			budget:      3,
			prompt:      admissionBudgetPrompt(30, 3),
			wantState:   SlotStateActiveDecode,
			wantLastTok: 32,
		},
		{
			name:        "zero-budget-long-unflagged-stays-eager",
			budget:      0,
			prompt:      admissionBudgetPrompt(40, 10),
			wantState:   SlotStateActiveDecode,
			wantLastTok: 49,
		},
		{
			name:        "over-budget-flagged-prefills",
			budget:      3,
			prompt:      admissionBudgetPrompt(50, 10),
			chunked:     true,
			wantState:   SlotStatePrefilling,
			wantLastTok: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cb := wireBatcher(t, 2, tc.budget)
			if _, err := cb.Submit(&SubagentRequest{
				SessionID:      "req",
				PromptTokens:   tc.prompt,
				TargetTokens:   target,
				ChunkedPrefill: tc.chunked,
			}); err != nil {
				t.Fatalf("Submit failed: %v", err)
			}

			slot, ok := cb.GetSlot("req")
			if !ok {
				t.Fatal("submitted slot not found")
			}
			if slot.State != tc.wantState {
				t.Fatalf("state after Submit = %s, want %s", slot.State, tc.wantState)
			}
			// Admission never forwards a token: decode starts only after the prompt
			// (or its final chunk) has been consumed.
			if got := slot.TokensGenerated(); got != 0 {
				t.Fatalf("TokensGenerated after Submit = %d, want 0", got)
			}
			if slot.LastToken != tc.wantLastTok {
				t.Fatalf("LastToken after Submit = %d, want %d", slot.LastToken, tc.wantLastTok)
			}
		})
	}
}

// TestContinuousBatcher_AdmissionBudgetedStepIsLinearInBudget pins that an unflagged
// over-budget prompt is consumed at no more than PrefillBudget tokens per step, that
// the steps sum to exactly the prompt tokens, and that ceil(total/budget) steps
// converge every slot to active decode and then to its target tokens.
func TestContinuousBatcher_AdmissionBudgetedStepIsLinearInBudget(t *testing.T) {
	const (
		budget    = 3
		promptLen = 10
		requests  = 3
		target    = 2
		total     = requests * promptLen
		wantSteps = (total + budget - 1) / budget
	)
	ctx := context.Background()
	cb := wireBatcher(t, requests+1, budget)

	slots := make([]*Slot, requests)
	for i := 0; i < requests; i++ {
		sessionID := fmt.Sprintf("req-%d", i)
		if _, err := cb.Submit(&SubagentRequest{
			SessionID:    sessionID,
			PromptTokens: admissionBudgetPrompt(100+20*i, promptLen),
			TargetTokens: target,
		}); err != nil {
			t.Fatalf("Submit(%s) failed: %v", sessionID, err)
		}
		slot, ok := cb.GetSlot(sessionID)
		if !ok {
			t.Fatalf("%s slot not found", sessionID)
		}
		if slot.State != SlotStatePrefilling {
			t.Fatalf("%s state after Submit = %s, want %s", sessionID, slot.State, SlotStatePrefilling)
		}
		slots[i] = slot
	}

	consumed, prefillSteps := 0, 0
	for admissionBudgetAnyPrefilling(slots) {
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("StepPhase %d failed: %v", prefillSteps, err)
		}
		if res.Phase != PhasePrefill {
			t.Fatalf("step %d phase = %q while prefilling, want %q", prefillSteps, res.Phase, PhasePrefill)
		}
		if res.PrefillTokens <= 0 || res.PrefillTokens > budget {
			t.Fatalf("step %d PrefillTokens = %d, want 0 < x <= %d", prefillSteps, res.PrefillTokens, budget)
		}
		consumed += res.PrefillTokens
		prefillSteps++
		if prefillSteps == 1 {
			// No slot may advance past the budget in a single step: the whole point
			// of deferral is that no full-length forward runs.
			advanced := 0
			for _, slot := range slots {
				if slot.PrefillPos > budget {
					t.Fatalf("slot %s PrefillPos after first step = %d, want <= %d", slot.SessionID, slot.PrefillPos, budget)
				}
				advanced += slot.PrefillPos
			}
			if advanced != budget {
				t.Fatalf("first step advanced %d prompt tokens across slots, want exactly the budget %d", advanced, budget)
			}
		}
		if prefillSteps > 10*requests {
			t.Fatal("prefill did not terminate")
		}
	}

	if consumed != total {
		t.Fatalf("prefill tokens summed over steps = %d, want %d", consumed, total)
	}
	if prefillSteps != wantSteps {
		t.Fatalf("prefill steps = %d, want ceil(%d/%d) = %d", prefillSteps, total, budget, wantSteps)
	}
	for _, slot := range slots {
		if slot.State != SlotStateActiveDecode {
			t.Fatalf("slot %s state after %d prefill steps = %s, want %s", slot.SessionID, prefillSteps, slot.State, SlotStateActiveDecode)
		}
		if slot.TokensGenerated() != 0 {
			t.Fatalf("slot %s generated %d tokens before any decode step, want 0", slot.SessionID, slot.TokensGenerated())
		}
		if slot.PrefillPos != len(slot.PendingPrompt) {
			t.Fatalf("slot %s PrefillPos = %d, want %d", slot.SessionID, slot.PrefillPos, len(slot.PendingPrompt))
		}
	}

	// Every prompt is fully consumed, so no step below may run the prefill arm:
	// the scheduler has no prompt work left and must decode one token per resident
	// slot per step until each reaches its target.
	decodeSteps := 0
	for i := 0; i < 50 && !admissionBudgetAllReached(slots, target); i++ {
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("decode StepPhase %d failed: %v", i, err)
		}
		if res.Phase != PhaseDecode {
			t.Fatalf("step %d phase = %q with no prompt work left, want %q", i, res.Phase, PhaseDecode)
		}
		if res.DecodeTokens != requests {
			t.Fatalf("decode step %d advanced %d slots, want %d", i, res.DecodeTokens, requests)
		}
		decodeSteps++
	}
	if decodeSteps != target {
		t.Fatalf("decode steps = %d, want %d", decodeSteps, target)
	}
	for _, slot := range slots {
		if got := slot.TokensGenerated(); got != target {
			t.Fatalf("slot %s generated %d tokens, want %d", slot.SessionID, got, target)
		}
	}
}

// TestContinuousBatcher_StepDrainsDeferredSlot pins that the decode-only driver
// (Step, which is stepWithBudget(ctx, 0)) can still advance a slot that admission
// deferred into SlotStatePrefilling: the prefill arm runs unbounded with no budget,
// so a deferred slot is never stranded with no arm able to move it. The first Step
// must report a PhasePrefill that consumes the whole remainder, and later Steps
// must decode to the target.
func TestContinuousBatcher_StepDrainsDeferredSlot(t *testing.T) {
	const (
		budget    = 4
		promptLen = 10
		target    = 3
	)
	ctx := context.Background()
	cb := wireBatcher(t, 1, budget)

	prompt := admissionBudgetPrompt(200, promptLen)
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:    "deferred",
		PromptTokens: prompt,
		TargetTokens: target,
	}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
	slot, ok := cb.GetSlot("deferred")
	if !ok {
		t.Fatal("deferred slot not found")
	}
	if slot.State != SlotStatePrefilling {
		t.Fatalf("state after Submit = %s, want %s", slot.State, SlotStatePrefilling)
	}

	first, err := cb.Step(ctx)
	if err != nil {
		t.Fatalf("first Step failed: %v", err)
	}
	// A decode-only step has no budget, so the arm drains the entire remainder in
	// one pass: the leftover work is the prompt fold admission used to do eagerly.
	if first.Phase != PhasePrefill {
		t.Fatalf("first Step phase = %q, want %q", first.Phase, PhasePrefill)
	}
	if first.PrefillTokens <= 0 || first.PrefillTokens > promptLen {
		t.Fatalf("first Step PrefillTokens = %d, want 0 < x <= %d", first.PrefillTokens, promptLen)
	}
	if slot.State != SlotStateActiveDecode {
		t.Fatalf("state after first Step = %s, want %s", slot.State, SlotStateActiveDecode)
	}
	if slot.PrefillPos != len(slot.PendingPrompt) {
		t.Fatalf("PrefillPos after first Step = %d, want %d", slot.PrefillPos, len(slot.PendingPrompt))
	}
	if got := slot.TokensGenerated(); got != 0 {
		t.Fatalf("TokensGenerated after the prefill Step = %d, want 0", got)
	}

	decodeSteps := 0
	for i := 0; i < 50 && slot.TokensGenerated() < target; i++ {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("decode Step %d failed: %v", i, err)
		}
		if res.Phase != PhaseDecode {
			t.Fatalf("Step %d phase = %q after the prompt was drained, want %q", i, res.Phase, PhaseDecode)
		}
		decodeSteps++
	}
	if got := slot.TokensGenerated(); got != target {
		t.Fatalf("generated %d tokens after %d decode steps, want %d", got, decodeSteps, target)
	}
	if decodeSteps != target {
		t.Fatalf("decode steps = %d, want %d (one token per step)", decodeSteps, target)
	}
}

// TestContinuousBatcher_AdmissionDefersOnReusedSuffix pins the remainder
// arithmetic the deferral decision rests on. Admission defers on the prompt
// suffix left after prefix reuse (len(prompt) - matched), not on the raw prompt
// length: a reused turn whose suffix exceeds the budget defers and resumes at
// PrefillPos == matched so the reused tokens are never re-folded, while a reused
// turn that is longer than the budget overall but whose suffix fits admits
// eagerly. Both must generate exactly the tokens of a cold from-zero fold.
func TestContinuousBatcher_AdmissionDefersOnReusedSuffix(t *testing.T) {
	const (
		budget = 2
		target = 3
	)
	prefix := []int{1, 2, 3}
	suffix := []int{4, 5, 6}
	full := append(append([]int(nil), prefix...), suffix...)

	newHybridBatcher := func(t *testing.T, disableReuse bool) *ContinuousBatcher {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cfg.PrefillBudget = budget
		cfg.Model = rpHybridModel()
		cfg.DisableRecurrentPrefixReuse = disableReuse
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })
		return cb
	}

	// The cold oracle is one from-zero fold of a turn-2 prompt on a fresh batcher.
	coldTokens := func(t *testing.T, prompt []int) []int {
		t.Helper()
		cold := newHybridBatcher(t, true)
		if _, err := cold.Submit(&SubagentRequest{SessionID: "cold", PromptTokens: prompt, TargetTokens: target}); err != nil {
			t.Fatalf("Submit(cold) failed: %v", err)
		}
		return rpDrive(t, cold, "cold", target)
	}

	cases := []struct {
		name        string
		prompt      []int
		wantDefer   bool
		wantMatched int
	}{
		{
			// Suffix 3 > budget 2: deferred, resuming after the reused prefix.
			name:        "suffix-over-budget-defers",
			prompt:      full,
			wantDefer:   true,
			wantMatched: len(prefix),
		},
		{
			// Raw length 5 > budget 2, but the reused suffix is one token: only the
			// remainder arithmetic keeps this on the eager single-forward admit.
			name:        "raw-over-budget-suffix-within-budget-stays-eager",
			prompt:      append(append([]int(nil), prefix...), 7),
			wantDefer:   false,
			wantMatched: len(prefix),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			remainder := len(tc.prompt) - tc.wantMatched
			if tc.wantDefer && remainder <= budget {
				t.Fatalf("suffix %d must exceed budget %d for the deferred case", remainder, budget)
			}
			if !tc.wantDefer && (len(tc.prompt) <= budget || remainder > budget) {
				t.Fatalf("eager case needs raw length %d > budget %d and suffix %d <= budget",
					len(tc.prompt), budget, remainder)
			}
			wantState := SlotStateActiveDecode
			if tc.wantDefer {
				wantState = SlotStatePrefilling
			}

			warm := newHybridBatcher(t, false)
			if _, enabled := warm.PrefixCacheStats(); !enabled {
				t.Fatal("hybrid batcher without reuse disabled has no prefix cache")
			}
			if _, err := warm.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: prefix, TargetTokens: 2}); err != nil {
				t.Fatalf("Submit(turn1) failed: %v", err)
			}
			rpDrive(t, warm, "conv", 2)
			if got := warm.PrefixCacheLen(); got != 1 {
				t.Fatalf("turn 1 retained %d boundaries, want 1; the reuse witness would be vacuous", got)
			}

			if _, err := warm.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: tc.prompt, TargetTokens: target}); err != nil {
				t.Fatalf("Submit(turn2) failed: %v", err)
			}
			slot, ok := warm.GetSlot("conv")
			if !ok {
				t.Fatal("turn-2 slot not found")
			}
			if slot.State != wantState {
				t.Fatalf("turn-2 state after Submit = %s, want %s (raw %d, reused suffix %d, budget %d)",
					slot.State, wantState, len(tc.prompt), remainder, budget)
			}
			if slot.PrefixMatchedTokens != tc.wantMatched || slot.PrefixHits != 1 {
				t.Fatalf("turn-2 reuse record = hits %d / matched %d, want 1 / %d",
					slot.PrefixHits, slot.PrefixMatchedTokens, tc.wantMatched)
			}
			if tc.wantDefer {
				// The deferred slot resumes after the reused prefix, so the reused
				// tokens are never part of the pending chunk sequence.
				if slot.PrefillPos != tc.wantMatched {
					t.Fatalf("turn-2 PrefillPos after Submit = %d, want the reused prefix length %d", slot.PrefillPos, tc.wantMatched)
				}
			}

			got := rpDrive(t, warm, "conv", target)
			if stats, _ := warm.PrefixCacheStats(); stats.Hits != 1 {
				t.Fatalf("turn 2 cache hits = %d, want exactly 1 (the witness would be vacuous)", stats.Hits)
			}
			if want := coldTokens(t, tc.prompt); !equalTokenTimeline(got, want) {
				t.Fatalf("reused-suffix timeline = %v, cold from-zero fold = %v; admission must not change output", got, want)
			}
		})
	}
}

// TestContinuousBatcher_ChunkedOutputEquivalenceUnflagged pins that an unflagged
// over-budget prompt and its ChunkedPrefill twin produce the same token timeline,
// matching an oracle replay of the same chunked prefill.
func TestContinuousBatcher_ChunkedOutputEquivalenceUnflagged(t *testing.T) {
	const (
		budget = 3
		target = 3
	)
	mdl := prefillOutputSyntheticModel()
	prompt := discriminatingModelPrompt(t, mdl, budget)
	if len(prompt) <= budget {
		t.Fatalf("prompt length %d must exceed budget %d for the over-budget path", len(prompt), budget)
	}
	expected := expectedModelTokenTimeline(t, mdl, prompt, target, budget)
	if len(expected) != target {
		t.Fatalf("oracle timeline length = %d, want %d", len(expected), target)
	}

	run := func(t *testing.T, chunked bool) []int {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 1
		cfg.PrefillBudget = budget
		cfg.Model = mdl
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })

		if _, err := cb.Submit(&SubagentRequest{
			SessionID:      "req",
			PromptTokens:   prompt,
			TargetTokens:   target,
			ChunkedPrefill: chunked,
		}); err != nil {
			t.Fatalf("Submit failed: %v", err)
		}
		slot, ok := cb.GetSlot("req")
		if !ok {
			t.Fatal("req slot not found")
		}
		if !chunked && slot.State != SlotStatePrefilling {
			t.Fatalf("unflagged over-budget state = %s, want %s", slot.State, SlotStatePrefilling)
		}
		for i := 0; i < 50 && slot.TokensGenerated() < target; i++ {
			if _, err := cb.StepPhase(context.Background()); err != nil {
				t.Fatalf("StepPhase %d failed: %v", i, err)
			}
		}
		if got := slot.TokensGenerated(); got != target {
			t.Fatalf("generated %d tokens, want %d", got, target)
		}
		streamed := drainSlotTokens(slot)
		if !equalTokenTimeline(streamed, slot.GeneratedTokens) {
			t.Fatalf("streamed timeline = %v, want %v", streamed, slot.GeneratedTokens)
		}
		return append([]int(nil), slot.GeneratedTokens...)
	}

	unflagged := run(t, false)
	flagged := run(t, true)
	if !equalTokenTimeline(unflagged, flagged) {
		t.Fatalf("unflagged timeline = %v, chunked twin = %v; the admission gate must not change output", unflagged, flagged)
	}
	if !equalTokenTimeline(unflagged, expected) {
		t.Fatalf("generated timeline = %v, want oracle %v", unflagged, expected)
	}
}
