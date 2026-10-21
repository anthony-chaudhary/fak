package engine

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestContinuousBatcher_ModelEmitsPrefillTokenFirst(t *testing.T) {
	for _, testCase := range []struct {
		name          string
		chunked       bool
		prefillBudget int
	}{
		{name: "eager"},
		{name: "chunked", chunked: true, prefillBudget: 2},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			mdl := prefillOutputSyntheticModel()
			prompt := discriminatingModelPrompt(t, mdl, testCase.prefillBudget)
			const targetTokens = 3
			expected := expectedModelTokenTimeline(t, mdl, prompt, targetTokens, testCase.prefillBudget)

			cfg := DefaultContinuousBatcherConfig()
			cfg.MaxSlots = 1
			cfg.PrefillBudget = testCase.prefillBudget
			cfg.Model = mdl
			cb, err := NewContinuousBatcher(cfg)
			if err != nil {
				t.Fatalf("NewContinuousBatcher failed: %v", err)
			}
			t.Cleanup(func() { _ = cb.Close() })

			if _, err := cb.Submit(&SubagentRequest{
				SessionID:      testCase.name,
				PromptTokens:   prompt,
				TargetTokens:   targetTokens,
				ChunkedPrefill: testCase.chunked,
			}); err != nil {
				t.Fatalf("Submit failed: %v", err)
			}

			slot, ok := cb.GetSlot(testCase.name)
			if !ok {
				t.Fatal("submitted slot not found")
			}
			for step := 0; step < 20 && slot.TokensGenerated() < targetTokens; step++ {
				if _, err := cb.StepPhase(context.Background()); err != nil {
					t.Fatalf("StepPhase failed: %v", err)
				}
			}
			if !equalTokenTimeline(slot.GeneratedTokens, expected) {
				t.Fatalf("generated token timeline = %v, want prefill-first timeline %v", slot.GeneratedTokens, expected)
			}
		})
	}

	t.Run("target-one", func(t *testing.T) {
		mdl := prefillOutputSyntheticModel()
		prompt := discriminatingModelPrompt(t, mdl, 0)
		expected := expectedModelTokenTimeline(t, mdl, prompt, 1, 0)

		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 1
		cfg.Model = mdl
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })

		if _, err := cb.Submit(&SubagentRequest{
			SessionID:    "target-one",
			PromptTokens: prompt,
			TargetTokens: 1,
		}); err != nil {
			t.Fatalf("Submit failed: %v", err)
		}
		slot, ok := cb.GetSlot("target-one")
		if !ok {
			t.Fatal("target-one slot not found")
		}
		result, err := cb.StepPhase(context.Background())
		if err != nil {
			t.Fatalf("StepPhase failed: %v", err)
		}
		if result.GeneratedTokens["target-one"] != expected[0] {
			t.Fatalf("step token = %d, want prefill prediction %d", result.GeneratedTokens["target-one"], expected[0])
		}
		if !equalTokenTimeline(slot.GeneratedTokens, expected) {
			t.Fatalf("generated token timeline = %v, want %v", slot.GeneratedTokens, expected)
		}
		if slot.State != SlotStateFinished {
			t.Fatalf("slot state = %s, want %s", slot.State, SlotStateFinished)
		}
		if streamed := drainSlotTokens(slot); !equalTokenTimeline(streamed, expected) {
			t.Fatalf("streamed token timeline = %v, want %v", streamed, expected)
		}
	})

	t.Run("mixed-fresh-ongoing", func(t *testing.T) {
		mdl := prefillOutputSyntheticModel()
		ongoingPrompt := discriminatingModelPrompt(t, mdl, 0)
		ongoingExpected := expectedModelTokenTimeline(t, mdl, ongoingPrompt, 2, 0)
		freshPrompt := discriminatingModelPromptExcludingFirst(t, mdl, 0, ongoingExpected[1])
		freshExpected := expectedModelTokenTimeline(t, mdl, freshPrompt, 1, 0)

		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cfg.Model = mdl
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })

		if _, err := cb.Submit(&SubagentRequest{
			SessionID:    "ongoing",
			PromptTokens: ongoingPrompt,
			TargetTokens: 2,
		}); err != nil {
			t.Fatalf("Submit(ongoing) failed: %v", err)
		}
		ongoing, ok := cb.GetSlot("ongoing")
		if !ok {
			t.Fatal("ongoing slot not found")
		}
		firstResult, err := cb.StepPhase(context.Background())
		if err != nil {
			t.Fatalf("first StepPhase failed: %v", err)
		}
		if got := firstResult.GeneratedTokens["ongoing"]; got != ongoingExpected[0] {
			t.Fatalf("ongoing first step token = %d, want %d", got, ongoingExpected[0])
		}

		if _, err := cb.Submit(&SubagentRequest{
			SessionID:    "fresh",
			PromptTokens: freshPrompt,
			TargetTokens: 1,
		}); err != nil {
			t.Fatalf("Submit(fresh) failed: %v", err)
		}
		fresh, ok := cb.GetSlot("fresh")
		if !ok {
			t.Fatal("fresh slot not found")
		}
		secondResult, err := cb.StepPhase(context.Background())
		if err != nil {
			t.Fatalf("second StepPhase failed: %v", err)
		}
		if got := secondResult.GeneratedTokens["ongoing"]; got != ongoingExpected[1] {
			t.Fatalf("ongoing second step token = %d, want %d", got, ongoingExpected[1])
		}
		if got := secondResult.GeneratedTokens["fresh"]; got != freshExpected[0] {
			t.Fatalf("fresh first step token = %d, want prefill prediction %d", got, freshExpected[0])
		}
		if !equalTokenTimeline(ongoing.GeneratedTokens, ongoingExpected) {
			t.Fatalf("ongoing generated timeline = %v, want %v", ongoing.GeneratedTokens, ongoingExpected)
		}
		if !equalTokenTimeline(fresh.GeneratedTokens, freshExpected) {
			t.Fatalf("fresh generated timeline = %v, want %v", fresh.GeneratedTokens, freshExpected)
		}
		if streamed := drainSlotTokens(ongoing); !equalTokenTimeline(streamed, ongoingExpected) {
			t.Fatalf("ongoing streamed timeline = %v, want %v", streamed, ongoingExpected)
		}
		if streamed := drainSlotTokens(fresh); !equalTokenTimeline(streamed, freshExpected) {
			t.Fatalf("fresh streamed timeline = %v, want %v", streamed, freshExpected)
		}
	})
}

func prefillOutputSyntheticModel() *model.Model {
	return model.NewSynthetic(model.Config{
		Name:             "prefill-output-synthetic",
		HiddenSize:       64,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          16,
		IntermediateSize: 128,
		VocabSize:        200,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	})
}

func expectedModelTokenTimeline(t *testing.T, mdl *model.Model, prompt []int, targetTokens, prefillBudget int) []int {
	t.Helper()
	oracle := mdl.NewSession()
	defer oracle.Close()

	var logits []float32
	if prefillBudget <= 0 || len(prompt) <= prefillBudget {
		logits = oracle.Prefill(prompt)
	} else {
		position := 0
		for len(prompt)-position > prefillBudget {
			oracle.PrefillNoLogits(prompt[position : position+prefillBudget])
			position += prefillBudget
		}
		logits = oracle.Prefill(prompt[position:])
	}

	timeline := make([]int, 0, targetTokens)
	nextToken := testArgmax(logits)
	timeline = append(timeline, nextToken)
	for len(timeline) < targetTokens {
		nextToken = testArgmax(oracle.Step(nextToken))
		timeline = append(timeline, nextToken)
	}
	return timeline
}

func discriminatingModelPrompt(t *testing.T, mdl *model.Model, prefillBudget int) []int {
	return discriminatingModelPromptExcludingFirst(t, mdl, prefillBudget, -1)
}

func discriminatingModelPromptExcludingFirst(t *testing.T, mdl *model.Model, prefillBudget, excludedFirst int) []int {
	t.Helper()
	for seed := 0; seed < mdl.Cfg.VocabSize; seed++ {
		vocabSize := mdl.Cfg.VocabSize
		prompt := []int{
			seed,
			(seed + 1) % vocabSize,
			(seed + 7) % vocabSize,
			(seed + 19) % vocabSize,
			(seed + 41) % vocabSize,
		}
		timeline := expectedModelTokenTimeline(t, mdl, prompt, 2, prefillBudget)
		if timeline[0] != timeline[1] && timeline[0] != excludedFirst {
			return prompt
		}
	}
	t.Fatal("synthetic model produced no prompt that distinguishes the prefill prediction from the next step prediction")
	return nil
}

func drainSlotTokens(slot *Slot) []int {
	var tokens []int
	for token := range slot.Tokens() {
		tokens = append(tokens, token)
	}
	return tokens
}

func testArgmax(values []float32) int {
	bestIndex := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[bestIndex] {
			bestIndex = index
		}
	}
	return bestIndex
}

func equalTokenTimeline(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if got[index] != want[index] {
			return false
		}
	}
	return true
}
