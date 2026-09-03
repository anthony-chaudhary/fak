package polymodel

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestDraftMechanismConstants(t *testing.T) {
	mechanisms := []DraftMechanism{
		DraftMechanismLinearMTP,
		DraftMechanismDraftModel,
		DraftMechanismTreeSpec,
		DraftMechanismPromptLookup,
	}

	expected := []string{
		"linear_mtp",
		"draft_model",
		"tree_spec",
		"prompt_lookup",
	}

	for i, m := range mechanisms {
		if string(m) != expected[i] {
			t.Errorf("DraftMechanism[%d] = %q, want %q", i, m, expected[i])
		}
	}
}

func TestDraftSpecJSON(t *testing.T) {
	spec := DraftSpec{
		Method:       DraftMechanismDraftModel,
		DraftModelID: "qwen-0.5b",
		DraftDepth:   4,
		TreeTopology: "linear",
		NGramWindow:  3,
	}

	data, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("Marshal DraftSpec error: %v", err)
	}

	var decoded DraftSpec
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal DraftSpec error: %v", err)
	}

	if !reflect.DeepEqual(spec, decoded) {
		t.Fatalf("DraftSpec round-trip mismatch: got %+v, want %+v", decoded, spec)
	}
}

func TestOfflineTraceSampleJSON(t *testing.T) {
	sample := OfflineTraceSample{
		SampleID:     "test-sample-1",
		Domain:       "code",
		PromptTokens: []int{1, 2, 3},
		TargetTokens: []int{10, 20, 30, 40},
		DraftTokens:  []int{10, 20, 30},
		DraftTree:    GenerateLinearTree(3),
	}

	data, err := json.Marshal(sample)
	if err != nil {
		t.Fatalf("Marshal OfflineTraceSample error: %v", err)
	}

	var decoded OfflineTraceSample
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal OfflineTraceSample error: %v", err)
	}

	if decoded.SampleID != sample.SampleID || decoded.Domain != sample.Domain {
		t.Fatalf("Sample metadata mismatch: %+v vs %+v", decoded, sample)
	}
	if !reflect.DeepEqual(decoded.DraftTokens, sample.DraftTokens) {
		t.Fatalf("Draft tokens mismatch: got %v, want %v", decoded.DraftTokens, sample.DraftTokens)
	}
}

func TestEvaluateLinearTraceSingleRound(t *testing.T) {
	tests := []struct {
		name          string
		draft         []int
		target        []int
		wantAccepted  int
		wantAdvance   int
		wantPositions []int // expected accepted counts per position
	}{
		{
			name:          "all accepted with bonus",
			draft:         []int{10, 20, 30},
			target:        []int{10, 20, 30, 40},
			wantAccepted:  3,
			wantAdvance:   4,
			wantPositions: []int{1, 1, 1},
		},
		{
			name:          "partial prefix accepted",
			draft:         []int{10, 20, 99},
			target:        []int{10, 20, 30, 40},
			wantAccepted:  2,
			wantAdvance:   3,
			wantPositions: []int{1, 1, 0},
		},
		{
			name:          "zero accepted",
			draft:         []int{99, 98, 97},
			target:        []int{10, 20, 30, 40},
			wantAccepted:  0,
			wantAdvance:   1,
			wantPositions: []int{0, 0, 0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sample := OfflineTraceSample{
				SampleID:     "single-round",
				Domain:       "code",
				DraftTokens:  tc.draft,
				TargetTokens: tc.target,
			}

			profile := EvaluateLinearTrace(sample, len(tc.draft))

			if profile.Schema != SpecAcceptanceProfileSchema {
				t.Errorf("schema = %q, want %q", profile.Schema, SpecAcceptanceProfileSchema)
			}
			if profile.TotalRounds != 1 {
				t.Errorf("rounds = %d, want 1", profile.TotalRounds)
			}
			if profile.MeanAcceptanceLength != float64(tc.wantAdvance) {
				t.Errorf("MeanAcceptanceLength = %f, want %f", profile.MeanAcceptanceLength, float64(tc.wantAdvance))
			}
			if len(profile.Positions) != len(tc.draft) {
				t.Fatalf("positions len = %d, want %d", len(profile.Positions), len(tc.draft))
			}

			for pos := range tc.wantPositions {
				if profile.Positions[pos].Accepted != tc.wantPositions[pos] {
					t.Errorf("pos %d accepted = %d, want %d", pos, profile.Positions[pos].Accepted, tc.wantPositions[pos])
				}
				expectedAlpha := float64(tc.wantPositions[pos])
				if profile.PositionalAlpha[pos] != expectedAlpha {
					t.Errorf("pos %d alpha = %f, want %f", pos, profile.PositionalAlpha[pos], expectedAlpha)
				}
			}
		})
	}
}

func TestEvaluateLinearTraceMultiRound(t *testing.T) {
	// A long target sequence simulated over multiple 4-token speculative rounds.
	target := make([]int, 40)
	draft := make([]int, 40)
	for i := 0; i < 40; i++ {
		target[i] = 100 + i
		if i%4 == 3 {
			draft[i] = 999 // Mismatch every 4th token to force correction step
		} else {
			draft[i] = 100 + i
		}
	}

	sample := OfflineTraceSample{
		SampleID:     "multi-round-seq",
		Domain:       "math",
		TargetTokens: target,
		DraftTokens:  draft,
	}

	profile := EvaluateLinearTrace(sample, 4)

	if profile.TotalRounds <= 1 {
		t.Fatalf("expected multiple rounds, got %d", profile.TotalRounds)
	}
	if profile.MeanAcceptanceLength < 1.0 {
		t.Fatalf("MeanAcceptanceLength = %f, must be >= 1.0", profile.MeanAcceptanceLength)
	}
	if len(profile.PositionalAlpha) != 4 {
		t.Fatalf("PositionalAlpha len = %d, want 4", len(profile.PositionalAlpha))
	}
	// Positions 0..2 should have higher acceptance than position 3 (which always mismatches)
	if profile.PositionalAlpha[0] <= profile.PositionalAlpha[3] {
		t.Errorf("pos 0 alpha (%f) should exceed pos 3 alpha (%f)", profile.PositionalAlpha[0], profile.PositionalAlpha[3])
	}
}

func TestEvaluateLinearTracePromptLookup(t *testing.T) {
	// Prompt contains a repeating phrase: [10, 20, 30, 40, 50]
	prompt := []int{1, 2, 3, 10, 20, 30, 40, 50, 99}
	// Target sequence continues with the same prefix [10, 20, 30], expecting prompt lookup to draft [40, 50]
	target := []int{10, 20, 30, 40, 50, 60}

	sample := OfflineTraceSample{
		SampleID:     "prompt-lookup",
		Domain:       "code",
		PromptTokens: prompt,
		TargetTokens: target,
	}

	spec := DraftSpec{
		Method:      DraftMechanismPromptLookup,
		DraftDepth:  2,
		NGramWindow: 2,
	}

	evaluator := NewOfflineProfileEvaluator("target-test", spec)
	profile := evaluator.Evaluate([]OfflineTraceSample{sample})

	if profile.TotalRounds == 0 {
		t.Fatal("expected positive rounds for prompt lookup evaluation")
	}
	if profile.MeanAcceptanceLength < 1.0 {
		t.Fatalf("MeanAcceptanceLength = %f, want >= 1.0", profile.MeanAcceptanceLength)
	}
}

func TestEvaluateTreeTraceSingleRound(t *testing.T) {
	// Create a tree with root 0 -> child 1 (token 10) -> child 2 (token 20)
	tree := SpecTree{
		Nodes: []TreeNode{
			{TargetArgmax: 10, Children: []int{1}},
			{Token: 10, TargetArgmax: 20, Children: []int{2}},
			{Token: 20, TargetArgmax: 99},
		},
	}

	sample := OfflineTraceSample{
		SampleID:  "tree-fixture",
		Domain:    "code",
		DraftTree: tree,
	}

	profile := EvaluateTreeTrace(sample)

	if profile.TotalRounds != 1 {
		t.Fatalf("TotalRounds = %d, want 1", profile.TotalRounds)
	}
	// Both speculative nodes (index 1 and 2) matched their parent's TargetArgmax.
	// Advance = 2 accepted + 1 bonus = 3.
	if profile.MeanAcceptanceLength != 3.0 {
		t.Errorf("MeanAcceptanceLength = %f, want 3.0", profile.MeanAcceptanceLength)
	}

	// TreeDepthYield should reflect depth 1 and depth 2 accepted
	if yield1, ok := profile.TreeDepthYield[1]; !ok || yield1 != 1.0 {
		t.Errorf("depth 1 yield = %v, want 1.0", yield1)
	}
	if yield2, ok := profile.TreeDepthYield[2]; !ok || yield2 != 1.0 {
		t.Errorf("depth 2 yield = %v, want 1.0", yield2)
	}
}

func TestEvaluateTreeTraceBranchingAdvantage(t *testing.T) {
	// Branching tree: root 0 has two children:
	// Child 1 (token 11 - wrong), Child 2 (token 10 - correct).
	// Child 2 has Child 3 (token 20 - correct).
	tree := SpecTree{
		Nodes: []TreeNode{
			{Children: []int{1, 2}},
			{Token: 11},
			{Token: 10, Children: []int{3}},
			{Token: 20},
		},
	}

	sample := OfflineTraceSample{
		SampleID:     "branch-advantage",
		Domain:       "code",
		DraftTree:    tree,
		TargetTokens: []int{10, 20, 30},
	}

	profile := EvaluateTreeTrace(sample)

	// Since Child 2 (token 10) matches target[0] and Child 3 (token 20) matches target[1],
	// the tree accepts 2 nodes and advances 3 tokens.
	if profile.MeanAcceptanceLength != 3.0 {
		t.Errorf("MeanAcceptanceLength = %f, want 3.0 (branching exploration)", profile.MeanAcceptanceLength)
	}
	if profile.TreeDepthYield[1] != 1.0 {
		t.Errorf("depth 1 yield = %f, want 1.0", profile.TreeDepthYield[1])
	}
	if profile.TreeDepthYield[2] != 1.0 {
		t.Errorf("depth 2 yield = %f, want 1.0", profile.TreeDepthYield[2])
	}
}

func TestOfflineProfileEvaluatorBatch(t *testing.T) {
	samples := []OfflineTraceSample{
		{
			SampleID:     "s1",
			Domain:       "code",
			DraftTokens:  []int{10, 20},
			TargetTokens: []int{10, 20, 30},
		},
		{
			SampleID:     "s2",
			Domain:       "code",
			DraftTokens:  []int{10, 99},
			TargetTokens: []int{10, 20, 30},
		},
	}

	spec := DraftSpec{
		Method:       DraftMechanismLinearMTP,
		DraftDepth:   2,
		DraftModelID: "mtp-qwen",
	}

	eval := NewOfflineProfileEvaluator("qwen-38b", spec)
	eval.Domain = "code"
	eval.Temperature = 0.0

	profile := eval.Evaluate(samples)

	if profile.Schema != SpecAcceptanceProfileSchema {
		t.Errorf("Schema = %q, want %q", profile.Schema, SpecAcceptanceProfileSchema)
	}
	if profile.TargetModel != "qwen-38b" {
		t.Errorf("TargetModel = %q, want 'qwen-38b'", profile.TargetModel)
	}
	if profile.SampleCount != 2 {
		t.Errorf("SampleCount = %d, want 2", profile.SampleCount)
	}
	if profile.TotalRounds != 2 {
		t.Errorf("TotalRounds = %d, want 2", profile.TotalRounds)
	}

	// s1 advances 3 (2 accepted + bonus); s2 advances 2 (1 accepted + correction)
	// Mean acceptance length = (3 + 2) / 2 = 2.5
	if profile.MeanAcceptanceLength != 2.5 {
		t.Errorf("MeanAcceptanceLength = %f, want 2.5", profile.MeanAcceptanceLength)
	}

	// Position 0: 2 proposed, 2 accepted -> alpha = 1.0
	// Position 1: 2 proposed, 1 accepted -> alpha = 0.5
	if len(profile.PositionalAlpha) != 2 {
		t.Fatalf("PositionalAlpha len = %d, want 2", len(profile.PositionalAlpha))
	}
	if profile.PositionalAlpha[0] != 1.0 {
		t.Errorf("PositionalAlpha[0] = %f, want 1.0", profile.PositionalAlpha[0])
	}
	if profile.PositionalAlpha[1] != 0.5 {
		t.Errorf("PositionalAlpha[1] = %f, want 0.5", profile.PositionalAlpha[1])
	}
}

func TestEvaluateTraceSamplesHelper(t *testing.T) {
	samples := []OfflineTraceSample{
		{
			SampleID:     "helper-1",
			Domain:       "dialog",
			DraftTokens:  []int{5, 6},
			TargetTokens: []int{5, 6, 7},
		},
	}

	spec := DraftSpec{
		Method:     DraftMechanismDraftModel,
		DraftDepth: 2,
	}

	profile := EvaluateTraceSamples(samples, spec)
	if profile.Domain != "dialog" {
		t.Errorf("inferred domain = %q, want 'dialog'", profile.Domain)
	}
	if profile.MeanAcceptanceLength != 3.0 {
		t.Errorf("MeanAcceptanceLength = %f, want 3.0", profile.MeanAcceptanceLength)
	}
}

func TestGenerateSyntheticEvalCorpus(t *testing.T) {
	if samples := GenerateSyntheticEvalCorpus("code", 0); samples != nil {
		t.Errorf("expected nil for 0 samples, got %v", samples)
	}

	nSamples := 15
	codeCorpus := GenerateSyntheticEvalCorpus("code", nSamples)
	mathCorpus := GenerateSyntheticEvalCorpus("math", nSamples)
	dialogCorpus := GenerateSyntheticEvalCorpus("dialog", nSamples)

	if len(codeCorpus) != nSamples || len(mathCorpus) != nSamples || len(dialogCorpus) != nSamples {
		t.Fatalf("corpus lengths mismatch: code=%d, math=%d, dialog=%d",
			len(codeCorpus), len(mathCorpus), len(dialogCorpus))
	}

	// Verify deterministic reproducibility
	codeCorpus2 := GenerateSyntheticEvalCorpus("code", nSamples)
	if !reflect.DeepEqual(codeCorpus[0].TargetTokens, codeCorpus2[0].TargetTokens) {
		t.Fatal("GenerateSyntheticEvalCorpus must be deterministic across repeated calls")
	}

	// Evaluate acceptance profiles across domains
	spec := DraftSpec{
		Method:     DraftMechanismLinearMTP,
		DraftDepth: 4,
	}

	codeProfile := EvaluateTraceSamples(codeCorpus, spec)
	mathProfile := EvaluateTraceSamples(mathCorpus, spec)
	dialogProfile := EvaluateTraceSamples(dialogCorpus, spec)

	if codeProfile.MeanAcceptanceLength <= dialogProfile.MeanAcceptanceLength {
		t.Errorf("code mean acceptance (%f) should exceed dialog mean acceptance (%f)",
			codeProfile.MeanAcceptanceLength, dialogProfile.MeanAcceptanceLength)
	}
	if mathProfile.MeanAcceptanceLength <= dialogProfile.MeanAcceptanceLength {
		t.Errorf("math mean acceptance (%f) should exceed dialog mean acceptance (%f)",
			mathProfile.MeanAcceptanceLength, dialogProfile.MeanAcceptanceLength)
	}
}

func TestSpecAcceptanceProfileJSONRoundTrip(t *testing.T) {
	rate1 := 0.8
	rate2 := 0.6
	profile := SpecAcceptanceProfile{
		Schema:      SpecAcceptanceProfileSchema,
		TargetModel: "fak-target",
		DraftSpec: DraftSpec{
			Method:     DraftMechanismLinearMTP,
			DraftDepth: 2,
		},
		Domain:               "code",
		Temperature:          0.2,
		SampleCount:          10,
		TotalRounds:          25,
		MeanAcceptanceLength: 2.75,
		PositionalAlpha:      []float64{0.8, 0.6},
		TreeDepthYield: map[int]float64{
			1: 0.8,
			2: 0.6,
		},
		Positions: []AcceptancePosition{
			{Position: 0, Proposed: 25, Accepted: 20, Rate: &rate1},
			{Position: 1, Proposed: 20, Accepted: 12, Rate: &rate2},
		},
	}

	jsonBytes, err := profile.JSON()
	if err != nil {
		t.Fatalf("profile.JSON() error: %v", err)
	}

	var parsed SpecAcceptanceProfile
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}

	if parsed.Schema != SpecAcceptanceProfileSchema {
		t.Errorf("parsed.Schema = %q, want %q", parsed.Schema, SpecAcceptanceProfileSchema)
	}
	if parsed.MeanAcceptanceLength != profile.MeanAcceptanceLength {
		t.Errorf("parsed.MeanAcceptanceLength = %f, want %f", parsed.MeanAcceptanceLength, profile.MeanAcceptanceLength)
	}
	if len(parsed.PositionalAlpha) != len(profile.PositionalAlpha) {
		t.Fatalf("parsed.PositionalAlpha len = %d, want %d", len(parsed.PositionalAlpha), len(profile.PositionalAlpha))
	}
	if parsed.TreeDepthYield[1] != 0.8 || parsed.TreeDepthYield[2] != 0.6 {
		t.Errorf("parsed.TreeDepthYield mismatch: %+v", parsed.TreeDepthYield)
	}
}

func TestEmptyAndEdgeCaseSamples(t *testing.T) {
	eval := NewOfflineProfileEvaluator("target", DraftSpec{Method: DraftMechanismLinearMTP})

	emptyProfile := eval.Evaluate(nil)
	if emptyProfile.SampleCount != 0 || emptyProfile.TotalRounds != 0 || emptyProfile.MeanAcceptanceLength != 0 {
		t.Errorf("empty profile = %+v, expected zero counts", emptyProfile)
	}

	sampleWithEmptyTarget := OfflineTraceSample{
		SampleID:     "empty-target",
		TargetTokens: []int{},
	}
	profile := eval.Evaluate([]OfflineTraceSample{sampleWithEmptyTarget})
	if profile.TotalRounds != 0 || profile.MeanAcceptanceLength != 0 {
		t.Errorf("empty target profile = %+v, expected zero rounds", profile)
	}
}

func TestOfflineProfileGreedySemantics(t *testing.T) {
	t.Run("single_round_match_accept_greedy", func(t *testing.T) {
		cases := []struct {
			name   string
			draft  []int
			target []int
		}{
			{
				name:   "full_match",
				draft:  []int{10, 20, 30, 40},
				target: []int{10, 20, 30, 40, 50},
			},
			{
				name:   "partial_match_early_diverge",
				draft:  []int{10, 99, 30, 40},
				target: []int{10, 20, 30, 40, 50},
			},
			{
				name:   "zero_match",
				draft:  []int{99, 88, 77, 66},
				target: []int{10, 20, 30, 40, 50},
			},
			{
				name:   "full_match_exact_target_length",
				draft:  []int{10, 20, 30},
				target: []int{10, 20, 30},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				expectedResult := AcceptGreedy(tc.draft, tc.target)

				sample := OfflineTraceSample{
					SampleID:     tc.name,
					Domain:       "code",
					DraftTokens:  tc.draft,
					TargetTokens: tc.target,
				}

				profile := EvaluateLinearTrace(sample, len(tc.draft))

				if profile.TotalRounds != 1 {
					t.Fatalf("TotalRounds = %d, want 1", profile.TotalRounds)
				}
				if profile.MeanAcceptanceLength != float64(expectedResult.Advance) {
					t.Errorf("MeanAcceptanceLength = %f, want %f (matching AcceptGreedy.Advance)",
						profile.MeanAcceptanceLength, float64(expectedResult.Advance))
				}

				if len(profile.PositionalAlpha) != len(tc.draft) {
					t.Fatalf("PositionalAlpha len = %d, want %d", len(profile.PositionalAlpha), len(tc.draft))
				}

				for pos := 0; pos < len(tc.draft); pos++ {
					var expectedAlpha float64
					if pos < expectedResult.Accepted {
						expectedAlpha = 1.0
					} else {
						expectedAlpha = 0.0
					}
					if profile.PositionalAlpha[pos] != expectedAlpha {
						t.Errorf("position %d alpha = %f, want %f", pos, profile.PositionalAlpha[pos], expectedAlpha)
					}
				}
			})
		}
	})

	t.Run("multi_step_simulation_matches_accept_greedy", func(t *testing.T) {
		depth := 4
		targetLen := 48
		target := make([]int, targetLen)
		draft := make([]int, targetLen)

		// Create target sequence and draft with known match/divergence pattern:
		// Every 6th token diverges.
		for i := 0; i < targetLen; i++ {
			target[i] = 1000 + i
			if i%6 == 3 {
				draft[i] = 9999
			} else {
				draft[i] = 1000 + i
			}
		}

		// Simulate step-by-step with AcceptGreedy manually
		var expectedRounds, expectedAdvance int
		expectedProposed := make([]int, depth)
		expectedAccepted := make([]int, depth)

		committed := 0
		for committed < len(target) {
			var d []int
			if committed < len(draft) {
				end := committed + depth
				if end > len(draft) {
					end = len(draft)
				}
				d = draft[committed:end]
			}
			if len(d) == 0 {
				committed++
				expectedRounds++
				expectedAdvance++
				continue
			}

			targetLimit := committed + len(d) + 1
			if targetLimit > len(target) {
				targetLimit = len(target)
			}
			targetArgmax := target[committed:targetLimit]

			res := AcceptGreedy(d, targetArgmax)
			expectedRounds++
			expectedAdvance += res.Advance

			for pos := 0; pos < len(d); pos++ {
				expectedProposed[pos]++
				if pos < res.Accepted {
					expectedAccepted[pos]++
				}
			}

			committed += res.Advance
			if res.Advance <= 0 {
				committed++
			}
		}

		sample := OfflineTraceSample{
			SampleID:     "multi-step-trace",
			Domain:       "math",
			TargetTokens: target,
			DraftTokens:  draft,
		}

		eval := NewOfflineProfileEvaluator("target-test", DraftSpec{
			Method:     DraftMechanismDraftModel,
			DraftDepth: depth,
		})
		profile := eval.Evaluate([]OfflineTraceSample{sample})

		if profile.TotalRounds != expectedRounds {
			t.Fatalf("TotalRounds = %d, want %d", profile.TotalRounds, expectedRounds)
		}

		expectedMeanLen := float64(expectedAdvance) / float64(expectedRounds)
		if math.Abs(profile.MeanAcceptanceLength-expectedMeanLen) > 1e-9 {
			t.Errorf("MeanAcceptanceLength = %f, want %f", profile.MeanAcceptanceLength, expectedMeanLen)
		}

		if len(profile.PositionalAlpha) != depth {
			t.Fatalf("PositionalAlpha len = %d, want %d", len(profile.PositionalAlpha), depth)
		}

		for pos := 0; pos < depth; pos++ {
			expectedAlpha := float64(expectedAccepted[pos]) / float64(expectedProposed[pos])
			if math.Abs(profile.PositionalAlpha[pos]-expectedAlpha) > 1e-9 {
				t.Errorf("pos %d alpha = %f, want %f", pos, profile.PositionalAlpha[pos], expectedAlpha)
			}
			if profile.Positions[pos].Proposed != expectedProposed[pos] {
				t.Errorf("pos %d proposed = %d, want %d", pos, profile.Positions[pos].Proposed, expectedProposed[pos])
			}
			if profile.Positions[pos].Accepted != expectedAccepted[pos] {
				t.Errorf("pos %d accepted = %d, want %d", pos, profile.Positions[pos].Accepted, expectedAccepted[pos])
			}
		}
	})
}

func TestOfflineProfileTreeSemantics(t *testing.T) {
	t.Run("direct_tree_fixture_matches_accept_tree", func(t *testing.T) {
		// Tree structure:
		// Root 0 (TargetArgmax 10)
		//   -> Child 1 (Token 10, TargetArgmax 20)
		//        -> Child 3 (Token 20, TargetArgmax 30)
		//             -> Child 5 (Token 99, TargetArgmax 40) [mismatch: token 99 != want 30]
		//   -> Child 2 (Token 11, TargetArgmax 99) [alternate branch, not taken]
		//        -> Child 4 (Token 22, TargetArgmax 99)
		tree := SpecTree{
			Nodes: []TreeNode{
				{TargetArgmax: 10, Children: []int{1, 2}},
				{Token: 10, TargetArgmax: 20, Children: []int{3}},
				{Token: 11, TargetArgmax: 99, Children: []int{4}},
				{Token: 20, TargetArgmax: 30, Children: []int{5}},
				{Token: 22, TargetArgmax: 99},
				{Token: 99, TargetArgmax: 40},
			},
		}

		expectedRes := AcceptTree(tree) // Path should be [1, 3] (2 nodes accepted), Advance = 3
		if len(expectedRes.Path) != 2 || expectedRes.Advance != 3 {
			t.Fatalf("unexpected AcceptTree result: path=%v, advance=%d", expectedRes.Path, expectedRes.Advance)
		}

		sample := OfflineTraceSample{
			SampleID:  "tree-direct-semantic",
			Domain:    "code",
			DraftTree: tree,
		}

		profile := EvaluateTreeTrace(sample)

		if profile.TotalRounds != 1 {
			t.Fatalf("TotalRounds = %d, want 1", profile.TotalRounds)
		}
		if profile.MeanAcceptanceLength != float64(expectedRes.Advance) {
			t.Errorf("MeanAcceptanceLength = %f, want %f", profile.MeanAcceptanceLength, float64(expectedRes.Advance))
		}

		if yield1, ok := profile.TreeDepthYield[1]; !ok || yield1 != 1.0 {
			t.Errorf("TreeDepthYield[1] = %v, want 1.0", yield1)
		}
		if yield2, ok := profile.TreeDepthYield[2]; !ok || yield2 != 1.0 {
			t.Errorf("TreeDepthYield[2] = %v, want 1.0", yield2)
		}
		if yield3, ok := profile.TreeDepthYield[3]; ok && yield3 != 0.0 {
			t.Errorf("TreeDepthYield[3] = %v, want 0.0", yield3)
		}
	})

	t.Run("multi_round_tree_matches_accept_tree", func(t *testing.T) {
		// Tree template (without TargetArgmax, will be assigned from TargetTokens)
		// Root 0:
		//   -> Child 1 (Token 100) -> Child 3 (Token 200)
		//   -> Child 2 (Token 101) -> Child 4 (Token 201)
		tree := SpecTree{
			Nodes: []TreeNode{
				{Children: []int{1, 2}},
				{Token: 100, Children: []int{3}},
				{Token: 101, Children: []int{4}},
				{Token: 200},
				{Token: 201},
			},
		}

		// Target sequence designed to take branch 1 then branch 2 alternately
		targetTokens := []int{
			100, 200, 300, // round 1: chooses child 1 (100) -> child 3 (200), advance=3
			101, 201, 301, // round 2: chooses child 2 (101) -> child 4 (201), advance=3
			999, 998, 997, // round 3: matches neither child (root token 999), advance=1
		}

		var expectedRounds, expectedAdvance int
		expectedDepthAcc := make(map[int]int)

		committed := 0
		for committed < len(targetTokens) {
			subTree := assignTreeTargetTokens(tree, targetTokens[committed:])
			res := AcceptTree(subTree)
			if res.Advance <= 0 {
				committed++
				expectedRounds++
				expectedAdvance++
				continue
			}
			expectedRounds++
			expectedAdvance += res.Advance
			for d := 1; d <= 2; d++ {
				if len(res.Path) >= d {
					expectedDepthAcc[d]++
				}
			}
			committed += res.Advance
		}

		sample := OfflineTraceSample{
			SampleID:     "multi-round-tree",
			Domain:       "code",
			DraftTree:    tree,
			TargetTokens: targetTokens,
		}

		eval := NewOfflineProfileEvaluator("target-test", DraftSpec{
			Method:     DraftMechanismTreeSpec,
			DraftDepth: 2,
		})
		profile := eval.Evaluate([]OfflineTraceSample{sample})

		if profile.TotalRounds != expectedRounds {
			t.Fatalf("TotalRounds = %d, want %d", profile.TotalRounds, expectedRounds)
		}

		expectedMeanLen := float64(expectedAdvance) / float64(expectedRounds)
		if math.Abs(profile.MeanAcceptanceLength-expectedMeanLen) > 1e-9 {
			t.Errorf("MeanAcceptanceLength = %f, want %f", profile.MeanAcceptanceLength, expectedMeanLen)
		}

		for d := 1; d <= 2; d++ {
			expectedYield := float64(expectedDepthAcc[d]) / float64(expectedRounds)
			actualYield := profile.TreeDepthYield[d]
			if math.Abs(actualYield-expectedYield) > 1e-9 {
				t.Errorf("TreeDepthYield[%d] = %f, want %f", d, actualYield, expectedYield)
			}
		}
	})
}

func TestThreeReferenceConfigurations(t *testing.T) {
	// Prepare reference corpus with non-empty target and draft sequences
	nSamples := 8
	corpus := make([]OfflineTraceSample, nSamples)
	for i := 0; i < nSamples; i++ {
		target := make([]int, 24)
		draft := make([]int, 24)
		for j := 0; j < 24; j++ {
			target[j] = 200 + j*3
			if (i+j)%3 != 0 {
				draft[j] = target[j]
			} else {
				draft[j] = 50000 + j
			}
		}
		corpus[i] = OfflineTraceSample{
			SampleID:     fmt.Sprintf("ref-corpus-%d", i+1),
			Domain:       "code",
			PromptTokens: []int{10, 20, 30},
			TargetTokens: target,
			DraftTokens:  draft,
		}
	}

	configs := []struct {
		name             string
		spec             DraftSpec
		expectedAlphaLen int
	}{
		{
			name: "Linear MTP (depth 1)",
			spec: DraftSpec{
				Method:     DraftMechanismLinearMTP,
				DraftDepth: 1,
			},
			expectedAlphaLen: 1,
		},
		{
			name: "Draft Model (depth 4)",
			spec: DraftSpec{
				Method:       DraftMechanismDraftModel,
				DraftDepth:   4,
				DraftModelID: "qwen-0.5b",
			},
			expectedAlphaLen: 4,
		},
		{
			name: "Eagle Tree",
			spec: DraftSpec{
				Method:       DraftMechanismTreeSpec,
				DraftDepth:   4,
				TreeTopology: "deep_narrow",
			},
			expectedAlphaLen: 0, // tree speculation computes alpha based on generated tree depth
		},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			eval := NewOfflineProfileEvaluator("qwen-38b", cfg.spec)
			eval.Domain = "code"
			profile := eval.Evaluate(corpus)

			// 1. Verify schema "fak-spec-acceptance/1"
			if profile.Schema != SpecAcceptanceProfileSchema {
				t.Errorf("Schema = %q, want %q", profile.Schema, SpecAcceptanceProfileSchema)
			}
			if profile.Schema != "fak-spec-acceptance/1" {
				t.Errorf("Schema = %q, want \"fak-spec-acceptance/1\"", profile.Schema)
			}

			// 2. Verify non-zero samples and valid rounds
			if profile.SampleCount != nSamples {
				t.Errorf("SampleCount = %d, want %d", profile.SampleCount, nSamples)
			}
			if profile.TotalRounds <= 0 {
				t.Errorf("TotalRounds = %d, want > 0", profile.TotalRounds)
			}
			if profile.MeanAcceptanceLength < 1.0 {
				t.Errorf("MeanAcceptanceLength = %f, want >= 1.0", profile.MeanAcceptanceLength)
			}

			// 3. Verify valid PositionalAlpha
			if len(profile.PositionalAlpha) == 0 {
				t.Fatal("PositionalAlpha is empty, want at least one position")
			}
			if cfg.expectedAlphaLen > 0 && len(profile.PositionalAlpha) != cfg.expectedAlphaLen {
				t.Errorf("PositionalAlpha len = %d, want %d", len(profile.PositionalAlpha), cfg.expectedAlphaLen)
			}

			for i, alpha := range profile.PositionalAlpha {
				if math.IsNaN(alpha) || math.IsInf(alpha, 0) {
					t.Errorf("PositionalAlpha[%d] is NaN or Inf: %f", i, alpha)
				}
				if alpha < 0.0 || alpha > 1.0 {
					t.Errorf("PositionalAlpha[%d] = %f, want in [0.0, 1.0]", i, alpha)
				}
			}

			// 4. Verify Tree specific fields when applicable
			if cfg.spec.Method == DraftMechanismTreeSpec {
				if len(profile.TreeDepthYield) == 0 {
					t.Error("TreeDepthYield is empty for Eagle Tree")
				}
				for d, yield := range profile.TreeDepthYield {
					if math.IsNaN(yield) || yield < 0.0 || yield > 1.0 {
						t.Errorf("TreeDepthYield[%d] = %f, want in [0.0, 1.0]", d, yield)
					}
				}
			}
		})
	}
}

func TestOfflineProfileJSONRoundTrip(t *testing.T) {
	rate1 := 0.85
	rate2 := 0.60
	rate3 := 0.40
	original := SpecAcceptanceProfile{
		Schema:      SpecAcceptanceProfileSchema,
		TargetModel: "fak-target-38b",
		DraftSpec: DraftSpec{
			Method:       DraftMechanismDraftModel,
			DraftModelID: "qwen-0.5b",
			DraftDepth:   3,
			TreeTopology: "linear",
			NGramWindow:  4,
		},
		Domain:               "code",
		Temperature:          0.2,
		SampleCount:          12,
		TotalRounds:          30,
		MeanAcceptanceLength: 2.85,
		PositionalAlpha:      []float64{0.85, 0.60, 0.40},
		TreeDepthYield: map[int]float64{
			1: 0.85,
			2: 0.60,
			3: 0.40,
		},
		Positions: []AcceptancePosition{
			{Position: 0, Proposed: 30, Accepted: 25, Rate: &rate1},
			{Position: 1, Proposed: 25, Accepted: 15, Rate: &rate2},
			{Position: 2, Proposed: 15, Accepted: 6, Rate: &rate3},
		},
	}

	// 1. Validate profile.JSON() method
	data, err := original.JSON()
	if err != nil {
		t.Fatalf("profile.JSON() failed: %v", err)
	}

	var decoded SpecAcceptanceProfile
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if decoded.Schema != SpecAcceptanceProfileSchema {
		t.Errorf("decoded.Schema = %q, want %q", decoded.Schema, SpecAcceptanceProfileSchema)
	}
	if decoded.TargetModel != original.TargetModel {
		t.Errorf("decoded.TargetModel = %q, want %q", decoded.TargetModel, original.TargetModel)
	}
	if !reflect.DeepEqual(decoded.DraftSpec, original.DraftSpec) {
		t.Errorf("decoded.DraftSpec = %+v, want %+v", decoded.DraftSpec, original.DraftSpec)
	}
	if decoded.Domain != original.Domain {
		t.Errorf("decoded.Domain = %q, want %q", decoded.Domain, original.Domain)
	}
	if decoded.Temperature != original.Temperature {
		t.Errorf("decoded.Temperature = %f, want %f", decoded.Temperature, original.Temperature)
	}
	if decoded.SampleCount != original.SampleCount {
		t.Errorf("decoded.SampleCount = %d, want %d", decoded.SampleCount, original.SampleCount)
	}
	if decoded.TotalRounds != original.TotalRounds {
		t.Errorf("decoded.TotalRounds = %d, want %d", decoded.TotalRounds, original.TotalRounds)
	}
	if decoded.MeanAcceptanceLength != original.MeanAcceptanceLength {
		t.Errorf("decoded.MeanAcceptanceLength = %f, want %f", decoded.MeanAcceptanceLength, original.MeanAcceptanceLength)
	}
	if !reflect.DeepEqual(decoded.PositionalAlpha, original.PositionalAlpha) {
		t.Errorf("decoded.PositionalAlpha = %v, want %v", decoded.PositionalAlpha, original.PositionalAlpha)
	}
	if !reflect.DeepEqual(decoded.TreeDepthYield, original.TreeDepthYield) {
		t.Errorf("decoded.TreeDepthYield = %v, want %v", decoded.TreeDepthYield, original.TreeDepthYield)
	}
	if len(decoded.Positions) != len(original.Positions) {
		t.Fatalf("decoded.Positions len = %d, want %d", len(decoded.Positions), len(original.Positions))
	}
	for i := range original.Positions {
		if decoded.Positions[i].Position != original.Positions[i].Position ||
			decoded.Positions[i].Proposed != original.Positions[i].Proposed ||
			decoded.Positions[i].Accepted != original.Positions[i].Accepted ||
			*decoded.Positions[i].Rate != *original.Positions[i].Rate {
			t.Errorf("decoded.Positions[%d] = %+v, want %+v", i, decoded.Positions[i], original.Positions[i])
		}
	}

	// 2. Validate standard json.Marshal matches profile.JSON() semantically
	stdData, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var stdDecoded SpecAcceptanceProfile
	if err := json.Unmarshal(stdData, &stdDecoded); err != nil {
		t.Fatalf("json.Unmarshal stdData failed: %v", err)
	}
	if !reflect.DeepEqual(decoded, stdDecoded) {
		t.Errorf("profile.JSON() and json.Marshal() unmarshaled representations differ: %+v vs %+v", decoded, stdDecoded)
	}

	// 3. Validate DraftSpec and OfflineTraceSample round trip
	specData, err := json.Marshal(original.DraftSpec)
	if err != nil {
		t.Fatalf("Marshal DraftSpec error: %v", err)
	}
	var roundTripSpec DraftSpec
	if err := json.Unmarshal(specData, &roundTripSpec); err != nil {
		t.Fatalf("Unmarshal DraftSpec error: %v", err)
	}
	if !reflect.DeepEqual(roundTripSpec, original.DraftSpec) {
		t.Errorf("DraftSpec round trip mismatch: %+v vs %+v", roundTripSpec, original.DraftSpec)
	}

	sample := OfflineTraceSample{
		SampleID:     "roundtrip-sample",
		Domain:       "code",
		PromptTokens: []int{1, 2, 3},
		TargetTokens: []int{10, 20, 30},
		DraftTokens:  []int{10, 20},
		DraftTree:    GenerateWideShallowTree(2, 2),
	}
	sampleData, err := json.Marshal(sample)
	if err != nil {
		t.Fatalf("Marshal OfflineTraceSample error: %v", err)
	}
	var roundTripSample OfflineTraceSample
	if err := json.Unmarshal(sampleData, &roundTripSample); err != nil {
		t.Fatalf("Unmarshal OfflineTraceSample error: %v", err)
	}
	if roundTripSample.SampleID != sample.SampleID ||
		!reflect.DeepEqual(roundTripSample.PromptTokens, sample.PromptTokens) ||
		!reflect.DeepEqual(roundTripSample.TargetTokens, sample.TargetTokens) ||
		!reflect.DeepEqual(roundTripSample.DraftTokens, sample.DraftTokens) ||
		len(roundTripSample.DraftTree.Nodes) != len(sample.DraftTree.Nodes) {
		t.Errorf("OfflineTraceSample round trip mismatch: %+v vs %+v", roundTripSample, sample)
	}
}
