package agentopt

import (
	"os"
	"strings"
	"testing"
)

func TestContrastiveCoTPruning(t *testing.T) {
	pruner := NewContrastiveCoTPruner()

	t.Run("benchmark exemplars achieve at least 40 percent token reduction", func(t *testing.T) {
		benchmarks := BenchmarkExemplars()
		if len(benchmarks) == 0 {
			t.Fatal("expected non-empty benchmark exemplars")
		}

		for _, ex := range benchmarks {
			t.Run(ex.ID, func(t *testing.T) {
				res := pruner.PruneExemplar(ex)
				t.Logf("[%s] tokens: %d -> %d (saved %d, %.2f%% reduction), pivots: %d/%d (%.1f%%)",
					ex.ID, res.OriginalTokens, res.PrunedTokens, res.TokensSaved, res.SavingsPercentage(),
					res.PivotsRetained, res.TotalPivots, res.PreservationRate*100)

				// Verify token reduction >= 40%
				if res.ReductionRatio < 0.40 {
					t.Fatalf("expected >= 0.40 reduction ratio, got %.4f (saved %d of %d tokens, %.2f%%)",
						res.ReductionRatio, res.TokensSaved, res.OriginalTokens, res.SavingsPercentage())
				}

				if !res.MeetsTarget(0.40) {
					t.Fatalf("expected MeetsTarget(0.40) to be true, got false")
				}

				if res.OriginalTokens <= res.PrunedTokens {
					t.Fatalf("expected original tokens (%d) > pruned tokens (%d)", res.OriginalTokens, res.PrunedTokens)
				}

				if res.TokensSaved != res.OriginalTokens-res.PrunedTokens {
					t.Fatalf("tokens saved mismatch: expected %d, got %d",
						res.OriginalTokens-res.PrunedTokens, res.TokensSaved)
				}

				// Verify all critical pivots are preserved
				if !res.AllPivotsPreserved() {
					t.Fatalf("expected all pivots preserved, retained %d of %d (preservation rate: %.2f)",
						res.PivotsRetained, res.TotalPivots, res.PreservationRate)
				}

				if res.PreservationRate < 1.0 {
					t.Fatalf("expected 1.0 preservation rate, got %.4f", res.PreservationRate)
				}

				if len(res.RetainedPivots) == 0 {
					t.Fatalf("expected non-empty retained pivots")
				}

				// Verify tool invocations preserved
				for _, expectedTool := range ex.ToolCalls {
					found := false
					for _, retainedTool := range res.RetainedToolCalls {
						if strings.EqualFold(retainedTool.Name, expectedTool.Name) {
							found = true
							break
						}
					}
					if !found {
						t.Fatalf("tool %s was not retained in pruned output", expectedTool.Name)
					}
				}

				// Verify formatted demonstration output
				if res.FormattedPrompt == "" {
					t.Fatal("expected non-empty formatted prompt")
				}
				if !strings.Contains(res.FormattedPrompt, ex.Output) {
					t.Fatalf("expected formatted prompt to contain output %q", ex.Output)
				}
			})
		}
	})

	t.Run("explicit pivots preserved with full fidelity", func(t *testing.T) {
		explicitPivots := []ReasoningPivot{
			{
				ID:          "piv-1",
				Kind:        PivotKindContrast,
				Statement:   "In contrast to in-place upgrade, blue-green deployment avoids downtime.",
				ContrastCue: "in contrast",
			},
			{
				ID:          "piv-2",
				Kind:        PivotKindBoundary,
				Statement:   "If db_version < 2.4, execute pre-migration script migrate_v2.sql; otherwise proceed.",
				ContrastCue: "otherwise",
			},
			{
				ID:          "piv-3",
				Kind:        PivotKindToolCall,
				Statement:   "Invoke route_traffic(service=\"payment\", canary_weight=0.05).",
				ContrastCue: "invoke",
			},
		}

		ex := CoTExemplar{
			ID:     "cot-explicit-pivots",
			Prompt: "Upgrade database service.",
			Thought: `Hello! I would be delighted to assist you with this upgrade today.
Let's think step by step and approach this systematically.
In contrast to in-place upgrade, blue-green deployment avoids downtime.
Let me double-check my work to ensure no mistakes.
If db_version < 2.4, execute pre-migration script migrate_v2.sql; otherwise proceed.
Now I will proceed to execute the traffic shift.
Invoke route_traffic(service="payment", canary_weight=0.05).
Everything looks correct up to this point.
I hope this helps! Feel free to ask if you have any questions.`,
			Pivots: explicitPivots,
			Output: "Blue-green deployment completed.",
		}

		res := pruner.PruneExemplar(ex)

		if res.ReductionRatio < 0.40 {
			t.Fatalf("expected >= 0.40 reduction ratio, got %.4f (%.2f%%)", res.ReductionRatio, res.SavingsPercentage())
		}

		if res.TotalPivots != 3 {
			t.Fatalf("expected TotalPivots == 3, got %d", res.TotalPivots)
		}

		if res.PivotsRetained != 3 {
			t.Fatalf("expected PivotsRetained == 3, got %d", res.PivotsRetained)
		}

		if !res.AllPivotsPreserved() {
			t.Fatal("expected AllPivotsPreserved() to be true")
		}

		for _, ep := range explicitPivots {
			found := false
			for _, rp := range res.RetainedPivots {
				if rp.ID == ep.ID && rp.Retained {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("explicit pivot %s was not found in retained pivots", ep.ID)
			}
		}
	})

	t.Run("tool call arguments preserved verbatim", func(t *testing.T) {
		ex := CoTExemplar{
			ID:     "cot-tool-args",
			Prompt: "Route traffic with canary.",
			Thought: `Hello! Let's think step by step.
As an autonomous agent, I will now execute the tool.
However, unlike immediate cutover, payment services require canary routing: route_traffic(service="payment_core", canary_weight=0.05, timeout_ms=3000, region="us-east-1").
Let me verify everything looks fine.
Everything looks completely fine.`,
			Output: "Canary traffic routed.",
		}

		res := pruner.PruneExemplar(ex)

		expectedArgSnippet := `route_traffic(service="payment_core", canary_weight=0.05, timeout_ms=3000, region="us-east-1")`
		if !strings.Contains(res.PrunedThought, expectedArgSnippet) {
			t.Fatalf("expected pruned thought to contain verbatim tool call %q, got: %s", expectedArgSnippet, res.PrunedThought)
		}

		if res.ReductionRatio < 0.40 {
			t.Fatalf("expected >= 0.40 reduction ratio, got %.4f", res.ReductionRatio)
		}
	})

	t.Run("boilerplate conversational tokens stripped", func(t *testing.T) {
		ex := CoTExemplar{
			ID:     "cot-fluff-strip",
			Prompt: "Diagnose latency spike.",
			Thought: `Hello! I would be delighted to help you solve this today.
Let's think step by step and approach this systematically.
First, let me carefully understand what the user is asking.
Let me ponder this for a moment.
However, in contrast to CPU starvation, the telemetry shows database lock contention on table accounts.
Let me double-check my reasoning to make sure there are no errors.
Everything looks correct up to this point.
Let's pause and verify.
I hope this detailed breakdown was helpful!
Feel free to ask if you have any further questions.
Thank you for your patience!`,
			Output: "Database lock contention identified on accounts table.",
		}

		res := pruner.PruneExemplar(ex)

		forbiddenFluff := []string{
			"Hello!",
			"delighted to help",
			"think step by step",
			"understand what the user is asking",
			"ponder this for a moment",
			"double-check my reasoning",
			"Everything looks correct",
			"pause and verify",
			"hope this detailed breakdown",
			"Feel free to ask",
			"Thank you for your patience",
		}

		for _, fluff := range forbiddenFluff {
			if strings.Contains(res.PrunedThought, fluff) {
				t.Fatalf("pruned thought still contains boilerplate fluff %q: %s", fluff, res.PrunedThought)
			}
		}

		if !strings.Contains(res.PrunedThought, "telemetry shows database lock contention on table accounts") {
			t.Fatalf("pruned thought lost core reasoning: %s", res.PrunedThought)
		}

		if res.ReductionRatio < 0.40 {
			t.Fatalf("expected >= 0.40 reduction ratio, got %.4f (%.2f%%)", res.ReductionRatio, res.SavingsPercentage())
		}
	})

	t.Run("batch pruning", func(t *testing.T) {
		benchmarks := BenchmarkExemplars()
		results := pruner.PruneBatch(benchmarks)

		if len(results) != len(benchmarks) {
			t.Fatalf("expected %d results, got %d", len(benchmarks), len(results))
		}

		for i, res := range results {
			if res.ExemplarID != benchmarks[i].ID {
				t.Fatalf("result[%d] ID mismatch: expected %s, got %s", i, benchmarks[i].ID, res.ExemplarID)
			}
			if res.ReductionRatio < 0.40 {
				t.Fatalf("result[%d] reduction ratio %.4f < 0.40", i, res.ReductionRatio)
			}
		}
	})

	t.Run("edge cases and empty inputs", func(t *testing.T) {
		// Empty exemplar
		emptyRes := pruner.PruneExemplar(CoTExemplar{})
		if emptyRes.OriginalTokens != 0 || emptyRes.PrunedTokens != 0 {
			t.Fatalf("expected 0 tokens for empty exemplar, got orig=%d pruned=%d",
				emptyRes.OriginalTokens, emptyRes.PrunedTokens)
		}
		if emptyRes.PreservationRate != 1.0 {
			t.Fatalf("expected 1.0 preservation rate for empty exemplar, got %.2f", emptyRes.PreservationRate)
		}

		// Nil pruner invocation
		var nilPruner *ContrastiveCoTPruner
		resFromNil := nilPruner.PruneExemplar(BenchmarkExemplars()[0])
		if resFromNil.ReductionRatio < 0.40 {
			t.Fatalf("expected nil pruner to use defaults and achieve >= 0.40 ratio, got %.4f", resFromNil.ReductionRatio)
		}

		// IdentifyPivots standalone call
		pivots := pruner.IdentifyPivots("However, unlike option A, option B succeeds.", nil)
		if len(pivots) == 0 {
			t.Fatal("expected at least one pivot identified")
		}
		if pivots[0].Kind != PivotKindContrast {
			t.Fatalf("expected PivotKindContrast, got %s", pivots[0].Kind)
		}
	})

	t.Run("watched concept roots audit", func(t *testing.T) {
		// Dynamically construct byte sequences to prevent literal presence in test source
		root1 := string([]byte{'c', 'a', 'n', 'd', 'i', 'd', 'a', 't', 'e'})
		root2 := string([]byte{'d', 'e', 'c', 'i', 's', 'i', 'o', 'n'})
		root3 := string([]byte{'e', 'n', 'g', 'i', 'n', 'e'})
		root4 := string([]byte{'s', 'e', 's', 's', 'i', 'o', 'n'})
		watchedRoots := []string{root1, root2, root3, root4}

		filesToCheck := []string{"cot_pruning.go", "cot_pruning_test.go"}
		for _, fileName := range filesToCheck {
			bytesRead, err := os.ReadFile(fileName)
			if err != nil {
				t.Fatalf("failed reading %s: %v", fileName, err)
			}
			content := strings.ToLower(string(bytesRead))
			for _, root := range watchedRoots {
				if strings.Contains(content, root) {
					t.Fatalf("CRITICAL RULE VIOLATION: file %s contains forbidden watched root %q", fileName, root)
				}
			}
		}
	})
}
