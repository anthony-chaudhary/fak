// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"errors"
	"math"
	"testing"
)

func intPtr(v int) *int { return &v }

func TestUsageDiff(t *testing.T) {
	t.Run("per-request signed percentage discrepancy and zero-client handling", func(t *testing.T) {
		pct, err := TokenDiffPct(110, 100)
		if err != nil || math.Abs(pct-10.0) > 1e-9 {
			t.Fatalf("TokenDiffPct(110, 100) = (%v, %v), want (10, nil)", pct, err)
		}
		pctNeg, err := TokenDiffPct(90, 100)
		if err != nil || math.Abs(pctNeg+10.0) > 1e-9 {
			t.Fatalf("TokenDiffPct(90, 100) = (%v, %v), want (-10, nil)", pctNeg, err)
		}
		pctZero, err := TokenDiffPct(0, 0)
		if err != nil || pctZero != 0 {
			t.Fatalf("TokenDiffPct(0, 0) = (%v, %v), want (0, nil)", pctZero, err)
		}
		if _, err := TokenDiffPct(5, 0); !errors.Is(err, ErrNoMetricValue) {
			t.Fatalf("TokenDiffPct(5, 0) err = %v, want ErrNoMetricValue", err)
		}
		if _, err := TokenDiffPct(-1, 10); err == nil {
			t.Fatal("expected error for negative token count")
		}
	})

	t.Run("run aggregate tracks discrepancy counts, overall diffs, and summary projection integration", func(t *testing.T) {
		inputs := []UsageDiffInput{
			{
				ClientPromptTokens:     intPtr(100),
				APIPromptTokens:        intPtr(105), // +5%
				ClientCompletionTokens: intPtr(50),
				APICompletionTokens:    intPtr(50), // exact match
				ClientReasoningTokens:  intPtr(20),
				APIReasoningTokens:     intPtr(24), // +20%
			},
			{
				ClientPromptTokens:     intPtr(200),
				APIPromptTokens:        intPtr(200), // exact match
				ClientCompletionTokens: intPtr(40),
				APICompletionTokens:    intPtr(36), // -10%
			},
			{
				ClientPromptTokens:     intPtr(50),
				APIPromptTokens:        intPtr(50),
				ClientCompletionTokens: intPtr(10),
				APICompletionTokens:    intPtr(10),
			},
		}

		sum, err := AggregateUsageDiff(inputs)
		if err != nil {
			t.Fatalf("AggregateUsageDiff: %v", err)
		}
		if sum.Requests != 3 || sum.DiscrepancyCount != 2 || sum.OSLMismatchCount != 1 {
			t.Fatalf("counts = %+v, want requests=3 discrepancy=2 oslMismatch=1", sum)
		}
		// Prompt totals: API=355, Client=350 -> (5/350)*100 = 1.428571428...%
		if sum.PromptOverallDiffPct == nil || math.Abs(*sum.PromptOverallDiffPct-(5.0/350.0*100.0)) > 1e-9 {
			t.Fatalf("PromptOverallDiffPct = %v", sum.PromptOverallDiffPct)
		}
		// Completion totals: API=96, Client=100 -> -4.0%
		if sum.CompletionOverallDiffPct == nil || math.Abs(*sum.CompletionOverallDiffPct-(-4.0)) > 1e-9 {
			t.Fatalf("CompletionOverallDiffPct = %v", sum.CompletionOverallDiffPct)
		}
		if sum.PromptMeanDiffPct == nil || math.Abs(*sum.PromptMeanDiffPct-(5.0/3.0)) > 1e-9 {
			t.Fatalf("PromptMeanDiffPct = %v, want 5/3", sum.PromptMeanDiffPct)
		}

		projected := ProjectSummary(sum.ToSummaryMetrics(), nil)
		for _, tag := range []string{
			TagUsageDiscrepancyCount,
			TagOSLMismatchCount,
			TagUsagePromptDiffPct,
			TagUsageCompletionDiffPct,
			TagUsageReasoningDiffPct,
			TagOSLMismatchDiffPct,
		} {
			if _, ok := projected[tag]; !ok {
				t.Errorf("projected summary missing curated usage-diff tag %q", tag)
			}
		}
	})
}
