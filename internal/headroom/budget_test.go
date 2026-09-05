package headroom

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestBudgetCalculationTiers verifies the discrete safe default tiers for Healthy, Low,
// Critical, and Unknown context headroom states.
func TestBudgetCalculationTiers(t *testing.T) {
	tests := []struct {
		name       string
		status     HeadroomStatus
		wantItems  int
		wantBytes  int
		wantTokens int
		wantTier   string
		wantClamp  bool
	}{
		{
			name:       "Healthy headroom allows standard budget",
			status:     HeadroomStatusHealthy,
			wantItems:  10,
			wantBytes:  16384,
			wantTokens: 4096,
			wantTier:   TierStandard,
			wantClamp:  false,
		},
		{
			name:       "Low headroom sets conservative budget",
			status:     HeadroomStatusLow,
			wantItems:  5,
			wantBytes:  8192,
			wantTokens: 2048,
			wantTier:   TierConservative,
			wantClamp:  false,
		},
		{
			name:       "Critical headroom clamps to strict minimal budget",
			status:     HeadroomStatusCritical,
			wantItems:  1,
			wantBytes:  2048,
			wantTokens: 512,
			wantTier:   TierMinimal,
			wantClamp:  true,
		},
		{
			name:       "Unknown headroom defaults to cautious tier",
			status:     HeadroomStatusUnknown,
			wantItems:  3,
			wantBytes:  4096,
			wantTokens: 1024,
			wantTier:   TierCautious,
			wantClamp:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			receipt := ComputeBudgetForStatus(tc.status)
			if receipt.Status != tc.status {
				t.Fatalf("Status = %q, want %q", receipt.Status, tc.status)
			}
			if receipt.Budget.MaxItems != tc.wantItems {
				t.Errorf("MaxItems = %d, want %d", receipt.Budget.MaxItems, tc.wantItems)
			}
			if receipt.Budget.ByteLimit != tc.wantBytes {
				t.Errorf("ByteLimit = %d, want %d", receipt.Budget.ByteLimit, tc.wantBytes)
			}
			if receipt.Budget.TokenLimit != tc.wantTokens {
				t.Errorf("TokenLimit = %d, want %d", receipt.Budget.TokenLimit, tc.wantTokens)
			}
			if receipt.Budget.Tier != tc.wantTier {
				t.Errorf("Tier = %q, want %q", receipt.Budget.Tier, tc.wantTier)
			}
			if receipt.Clamped != tc.wantClamp {
				t.Errorf("Clamped = %v, want %v", receipt.Clamped, tc.wantClamp)
			}
			if receipt.Schema != "fak-headroom-budget-receipt/1" {
				t.Errorf("Schema = %q, want %q", receipt.Schema, "fak-headroom-budget-receipt/1")
			}
			if receipt.Explanation == "" {
				t.Error("Explanation should not be empty")
			}
		})
	}
}

// TestCompressionRiskCoupling verifies that elevated compression risk clamps budgets appropriately.
func TestCompressionRiskCoupling(t *testing.T) {
	cases := []struct {
		name       string
		reserves   ReserveState
		wantTier   string
		wantItems  int
		wantClamp  bool
		wantReason string
	}{
		{
			name: "Healthy headroom with low risk remains standard",
			reserves: ReserveState{
				Status:          HeadroomStatusHealthy,
				CompressionRisk: RiskLow,
			},
			wantTier:  TierStandard,
			wantItems: 10,
			wantClamp: false,
		},
		{
			name: "Healthy headroom with high risk downgraded to conservative",
			reserves: ReserveState{
				Status:          HeadroomStatusHealthy,
				CompressionRisk: RiskHigh,
			},
			wantTier:  TierConservative,
			wantItems: 5,
			wantClamp: true,
		},
		{
			name: "Healthy headroom with critical risk clamped to minimal",
			reserves: ReserveState{
				Status:          HeadroomStatusHealthy,
				CompressionRisk: RiskCritical,
			},
			wantTier:  TierMinimal,
			wantItems: 1,
			wantClamp: true,
		},
		{
			name: "Low headroom with high risk clamped to minimal",
			reserves: ReserveState{
				Status:          HeadroomStatusLow,
				CompressionRisk: RiskHigh,
			},
			wantTier:  TierMinimal,
			wantItems: 1,
			wantClamp: true,
		},
		{
			name: "Unknown headroom with high risk clamped to minimal",
			reserves: ReserveState{
				Status:          HeadroomStatusUnknown,
				CompressionRisk: RiskHigh,
			},
			wantTier:  TierMinimal,
			wantItems: 1,
			wantClamp: true,
		},
		{
			name: "Critical headroom always minimal regardless of risk",
			reserves: ReserveState{
				Status:          HeadroomStatusCritical,
				CompressionRisk: RiskLow,
			},
			wantTier:  TierMinimal,
			wantItems: 1,
			wantClamp: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receipt := ComputeBudget(tc.reserves)
			if receipt.Budget.Tier != tc.wantTier {
				t.Errorf("Tier = %q, want %q", receipt.Budget.Tier, tc.wantTier)
			}
			if receipt.Budget.MaxItems != tc.wantItems {
				t.Errorf("MaxItems = %d, want %d", receipt.Budget.MaxItems, tc.wantItems)
			}
			if receipt.Clamped != tc.wantClamp {
				t.Errorf("Clamped = %v, want %v", receipt.Clamped, tc.wantClamp)
			}
			if tc.wantClamp && receipt.ClampReason == "" {
				t.Error("expected non-empty ClampReason when clamped")
			}
		})
	}
}

// TestLiveContextReservesClamping verifies that token reserves clamp TokenLimit and ByteLimit.
func TestLiveContextReservesClamping(t *testing.T) {
	t.Run("Remaining tokens clamp budget below tier default", func(t *testing.T) {
		res := ReserveState{
			Status:          HeadroomStatusHealthy,
			RemainingTokens: 800,
			ReserveTokens:   200, // available = 600 tokens (< 4096 default)
		}
		receipt := ComputeBudget(res)
		if !receipt.Clamped {
			t.Fatal("expected budget to be clamped")
		}
		if receipt.Budget.TokenLimit != 600 {
			t.Errorf("TokenLimit = %d, want 600", receipt.Budget.TokenLimit)
		}
		if receipt.Budget.ByteLimit != 2400 { // 600 * 4
			t.Errorf("ByteLimit = %d, want 2400", receipt.Budget.ByteLimit)
		}
		if receipt.Budget.MaxItems != 5 {
			t.Errorf("MaxItems = %d, want 5 for <2048 available tokens", receipt.Budget.MaxItems)
		}
	})

	t.Run("Severely constrained tokens clamp to minimal tier", func(t *testing.T) {
		res := ReserveState{
			Status:          HeadroomStatusHealthy,
			RemainingTokens: 350,
			ReserveTokens:   100, // available = 250 tokens (< 512)
		}
		receipt := ComputeBudget(res)
		if !receipt.Clamped {
			t.Fatal("expected budget to be clamped")
		}
		if receipt.Budget.TokenLimit != 250 {
			t.Errorf("TokenLimit = %d, want 250", receipt.Budget.TokenLimit)
		}
		if receipt.Budget.MaxItems != 1 {
			t.Errorf("MaxItems = %d, want 1 for <512 available tokens", receipt.Budget.MaxItems)
		}
		if receipt.Budget.Tier != TierMinimal {
			t.Errorf("Tier = %q, want %q", receipt.Budget.Tier, TierMinimal)
		}
	})

	t.Run("Exhausted reserve floor enforces minimal safety floor", func(t *testing.T) {
		res := ReserveState{
			Status:          HeadroomStatusLow,
			RemainingTokens: 50,
			ReserveTokens:   100, // available = -50 (exhausted)
		}
		receipt := ComputeBudget(res)
		if !receipt.Clamped {
			t.Fatal("expected budget to be clamped")
		}
		if receipt.Budget.TokenLimit != 128 {
			t.Errorf("TokenLimit = %d, want safety floor 128", receipt.Budget.TokenLimit)
		}
		if receipt.Budget.ByteLimit != 512 {
			t.Errorf("ByteLimit = %d, want safety floor 512", receipt.Budget.ByteLimit)
		}
		if receipt.Budget.MaxItems != 1 {
			t.Errorf("MaxItems = %d, want 1", receipt.Budget.MaxItems)
		}
		if receipt.Budget.Tier != TierMinimal {
			t.Errorf("Tier = %q, want %q", receipt.Budget.Tier, TierMinimal)
		}
	})
}

// TestStatusDerivationFromTokenCounts verifies that status is derived from metrics
// when not explicitly provided.
func TestStatusDerivationFromTokenCounts(t *testing.T) {
	cases := []struct {
		name       string
		reserves   ReserveState
		wantStatus HeadroomStatus
	}{
		{
			name: "Ample headroom derived as healthy",
			reserves: ReserveState{
				TotalTokens:     100000,
				RemainingTokens: 80000,
				ReserveTokens:   5000,
			},
			wantStatus: HeadroomStatusHealthy,
		},
		{
			name: "Remaining <= 25% derived as low",
			reserves: ReserveState{
				TotalTokens:     100000,
				RemainingTokens: 20000,
				ReserveTokens:   5000,
			},
			wantStatus: HeadroomStatusLow,
		},
		{
			name: "Remaining <= 10% derived as critical",
			reserves: ReserveState{
				TotalTokens:     100000,
				RemainingTokens: 8000,
				ReserveTokens:   5000,
			},
			wantStatus: HeadroomStatusCritical,
		},
		{
			name: "Available tokens <= 0 derived as critical",
			reserves: ReserveState{
				TotalTokens:     100000,
				RemainingTokens: 4000,
				ReserveTokens:   5000,
			},
			wantStatus: HeadroomStatusCritical,
		},
		{
			name: "Unmeasured context defaults to unknown",
			reserves: ReserveState{
				TotalTokens:     0,
				RemainingTokens: 0,
			},
			wantStatus: HeadroomStatusUnknown,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.reserves.EffectiveStatus(); got != tc.wantStatus {
				t.Errorf("EffectiveStatus() = %q, want %q", got, tc.wantStatus)
			}
		})
	}
}

// TestClampResultLines verifies line-level clamping for multi-item tool outputs.
func TestClampResultLines(t *testing.T) {
	lines := []string{"file1.go", "file2.go", "file3.go", "file4.go", "file5.go", "file6.go", "file7.go", "file8.go", "file9.go", "file10.go", "file11.go", "file12.go"}

	t.Run("Healthy budget allows 10 items", func(t *testing.T) {
		clamped, didClamp := ClampResultLines(lines, BudgetHealthy)
		if !didClamp {
			t.Fatal("expected 12 lines to be clamped under 10 max items")
		}
		// 10 items kept + 1 elision notice = 11 lines
		if len(clamped) != 11 {
			t.Fatalf("len(clamped) = %d, want 11", len(clamped))
		}
		if clamped[0] != "file1.go" || clamped[9] != "file10.go" {
			t.Errorf("unexpected kept lines: %v", clamped[:10])
		}
		if !strings.Contains(clamped[10], "clamped 2 item(s)") {
			t.Errorf("notice = %q, want mention of 2 elided items", clamped[10])
		}
	})

	t.Run("Critical budget clamps to 1 item", func(t *testing.T) {
		clamped, didClamp := ClampResultLines(lines, BudgetCritical)
		if !didClamp {
			t.Fatal("expected lines to be clamped under 1 max item")
		}
		// 1 item kept + 1 elision notice = 2 lines
		if len(clamped) != 2 {
			t.Fatalf("len(clamped) = %d, want 2", len(clamped))
		}
		if clamped[0] != "file1.go" {
			t.Errorf("kept line = %q, want file1.go", clamped[0])
		}
		if !strings.Contains(clamped[1], "clamped 11 item(s)") {
			t.Errorf("notice = %q, want mention of 11 elided items", clamped[1])
		}
	})

	t.Run("Unconstrained lines not clamped", func(t *testing.T) {
		short := []string{"file1.go", "file2.go"}
		clamped, didClamp := ClampResultLines(short, BudgetHealthy)
		if didClamp {
			t.Fatal("short lines should not be clamped")
		}
		if len(clamped) != 2 {
			t.Fatalf("len = %d, want 2", len(clamped))
		}
	})
}

// TestClampResultBytes verifies byte-level truncation.
func TestClampResultBytes(t *testing.T) {
	raw := []byte(strings.Repeat("line of test output data\n", 200)) // ~5000 bytes

	t.Run("Critical budget clamps to 2048 bytes", func(t *testing.T) {
		clamped, didClamp := ClampResultBytes(raw, BudgetCritical)
		if !didClamp {
			t.Fatal("expected 5000 bytes to be clamped under 2048 byte limit")
		}
		if len(clamped) > BudgetCritical.ByteLimit+64 { // slight leeway for notice alignment
			t.Fatalf("clamped len = %d, want <= %d", len(clamped), BudgetCritical.ByteLimit+64)
		}
		if !bytes.Contains(clamped, []byte("clamped")) {
			t.Errorf("notice missing in clamped output: %s", clamped[len(clamped)-128:])
		}
	})

	t.Run("Under-budget bytes not clamped", func(t *testing.T) {
		small := []byte("small output")
		clamped, didClamp := ClampResultBytes(small, BudgetHealthy)
		if didClamp {
			t.Fatal("small bytes should not be clamped")
		}
		if len(clamped) != len(small) {
			t.Errorf("len = %d, want %d", len(clamped), len(small))
		}
	})
}

// TestClampToolResult verifies combined multi-line and byte limits.
func TestClampToolResult(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 20; i++ {
		sb.WriteString(fmt.Sprintf("search match %02d: finding in package\n", i))
	}
	data := []byte(sb.String())

	t.Run("Critical headroom clamps to 1 line", func(t *testing.T) {
		clamped, didClamp := ClampToolResult(data, BudgetCritical)
		if !didClamp {
			t.Fatal("expected tool result to be clamped under critical budget")
		}
		s := string(clamped)
		if !strings.HasPrefix(s, "search match 01:") {
			t.Errorf("output does not start with first match: %s", s)
		}
		if strings.Contains(s, "search match 02:") {
			t.Errorf("output should not contain match 02: %s", s)
		}
		if !strings.Contains(s, "minimal budget") {
			t.Errorf("output missing minimal budget notice: %s", s)
		}
	})

	t.Run("Healthy headroom allows 10 items", func(t *testing.T) {
		clamped, didClamp := ClampToolResult(data, BudgetHealthy)
		if !didClamp {
			t.Fatal("expected tool result with 20 items to be clamped to 10")
		}
		s := string(clamped)
		if !strings.Contains(s, "search match 10:") {
			t.Error("match 10 should be present under healthy budget")
		}
		if strings.Contains(s, "search match 11:") {
			t.Error("match 11 should not be present under healthy budget")
		}
	})
}

// TestBudgetReceiptSerialization verifies JSON encoding and decoding of BudgetReceipt.
func TestBudgetReceiptSerialization(t *testing.T) {
	receipt := ComputeBudget(ReserveState{
		Status:          HeadroomStatusLow,
		RemainingTokens: 5000,
		TotalTokens:     20000,
		ReserveTokens:   1000,
		CompressionRisk: RiskMedium,
	})

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	parsed, err := ParseReceipt(data)
	if err != nil {
		t.Fatalf("ParseReceipt: %v", err)
	}

	if parsed.Schema != receipt.Schema {
		t.Errorf("Schema = %q, want %q", parsed.Schema, receipt.Schema)
	}
	if parsed.Status != receipt.Status {
		t.Errorf("Status = %q, want %q", parsed.Status, receipt.Status)
	}
	if parsed.Budget.Tier != receipt.Budget.Tier {
		t.Errorf("Tier = %q, want %q", parsed.Budget.Tier, receipt.Budget.Tier)
	}
	if parsed.Budget.MaxItems != receipt.Budget.MaxItems {
		t.Errorf("MaxItems = %d, want %d", parsed.Budget.MaxItems, receipt.Budget.MaxItems)
	}

	// Test invalid schema rejection
	badJSON := []byte(`{"schema":"wrong-schema","status":"healthy"}`)
	if _, err := ParseReceipt(badJSON); err == nil {
		t.Fatal("expected error for bad schema")
	}
}

// TestParseHeadroomStatusAndRisk verifies case-insensitive normalization.
func TestParseHeadroomStatusAndRisk(t *testing.T) {
	statusCases := []struct {
		in   string
		want HeadroomStatus
	}{
		{"healthy", HeadroomStatusHealthy},
		{"Healthy", HeadroomStatusHealthy},
		{"HEALTHY", HeadroomStatusHealthy},
		{"fresh", HeadroomStatusHealthy},
		{"ok", HeadroomStatusHealthy},
		{"low", HeadroomStatusLow},
		{"Low", HeadroomStatusLow},
		{"constrained", HeadroomStatusLow},
		{"warning", HeadroomStatusLow},
		{"critical", HeadroomStatusCritical},
		{"Critical", HeadroomStatusCritical},
		{"exhausted", HeadroomStatusCritical},
		{"danger", HeadroomStatusCritical},
		{"unknown", HeadroomStatusUnknown},
		{"invalid", HeadroomStatusUnknown},
		{"", HeadroomStatusUnknown},
	}
	for _, tc := range statusCases {
		if got := ParseHeadroomStatus(tc.in); got != tc.want {
			t.Errorf("ParseHeadroomStatus(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}

	riskCases := []struct {
		in   string
		want CompressionRisk
	}{
		{"low", RiskLow},
		{"Low", RiskLow},
		{"minimal", RiskLow},
		{"medium", RiskMedium},
		{"Medium", RiskMedium},
		{"moderate", RiskMedium},
		{"high", RiskHigh},
		{"High", RiskHigh},
		{"elevated", RiskHigh},
		{"critical", RiskCritical},
		{"Critical", RiskCritical},
		{"severe", RiskCritical},
		{"unknown", RiskUnknown},
		{"other", RiskUnknown},
		{"", RiskUnknown},
	}
	for _, tc := range riskCases {
		if got := ParseCompressionRisk(tc.in); got != tc.want {
			t.Errorf("ParseCompressionRisk(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestGateAdmitWithReservesCriticalClamps verifies end-to-end admission through Gate:
// critical context headroom clamps large tool results to the minimal budget,
// and healthy headroom allows standard budget.
func TestGateAdmitWithReservesCriticalClamps(t *testing.T) {
	withSelected(t, NativeName)
	gate := NewGate()

	var lines []string
	for i := 1; i <= 20; i++ {
		lines = append(lines, fmt.Sprintf("commit %02d: feat(component): update file %d", i, i))
	}
	orig := []byte(strings.Join(lines, "\n"))
	res := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: orig, Len: int64(len(orig))}}
	call := &abi.ToolCall{Tool: "git"}

	t.Run("Critical context headroom clamps rendering to minimal budget (1 item)", func(t *testing.T) {
		verdict, receipt := gate.AdmitWithReserves(context.Background(), call, res, ReserveState{
			Status: HeadroomStatusCritical,
		})

		if verdict.Kind != abi.VerdictTransform {
			t.Fatalf("verdict.Kind = %v, want VerdictTransform", verdict.Kind)
		}
		if receipt.Status != HeadroomStatusCritical {
			t.Errorf("receipt.Status = %q, want %q", receipt.Status, HeadroomStatusCritical)
		}
		if !receipt.Clamped {
			t.Errorf("receipt.Clamped = false, want true")
		}
		if verdict.Meta["headroom_status"] != "critical" {
			t.Errorf("meta[headroom_status] = %q, want critical", verdict.Meta["headroom_status"])
		}
		if verdict.Meta["budget_tier"] != TierMinimal {
			t.Errorf("meta[budget_tier] = %q, want %q", verdict.Meta["budget_tier"], TierMinimal)
		}

		payload, ok := verdict.Payload.(abi.TransformPayload)
		if !ok {
			t.Fatalf("verdict payload is not TransformPayload: %+v", verdict.Payload)
		}
		rendered := string(payload.NewArgs.Inline)
		if !strings.Contains(rendered, "commit 01:") {
			t.Errorf("expected first commit to survive: %s", rendered)
		}
		if strings.Contains(rendered, "commit 02:") {
			t.Errorf("commit 02 should have been clamped: %s", rendered)
		}
		if !strings.Contains(rendered, "minimal budget") {
			t.Errorf("expected minimal budget notice in rendered output: %s", rendered)
		}
	})

	t.Run("Healthy context headroom allows standard budget", func(t *testing.T) {
		verdict, receipt := gate.AdmitWithReserves(context.Background(), call, res, ReserveState{
			Status: HeadroomStatusHealthy,
		})

		if receipt.Status != HeadroomStatusHealthy {
			t.Errorf("receipt.Status = %q, want %q", receipt.Status, HeadroomStatusHealthy)
		}
		if receipt.Clamped {
			t.Errorf("receipt.Clamped = true, want false for unconstrained healthy headroom")
		}
		if receipt.Budget.Tier != TierStandard {
			t.Errorf("receipt.Budget.Tier = %q, want %q", receipt.Budget.Tier, TierStandard)
		}
		if receipt.Budget.MaxItems != 10 {
			t.Errorf("receipt.Budget.MaxItems = %d, want 10", receipt.Budget.MaxItems)
		}
		// Uncompressed output within standard budget is admitted as-is
		if verdict.Kind != abi.VerdictAllow {
			t.Fatalf("verdict.Kind = %v, want VerdictAllow", verdict.Kind)
		}
		if verdict.Meta["budget_tier"] != TierStandard {
			t.Errorf("meta[budget_tier] = %q, want %q", verdict.Meta["budget_tier"], TierStandard)
		}

		// Now verify that compressible output under healthy headroom transforms with standard tier
		pretty := prettyJSON()
		prettyRes := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: pretty, Len: int64(len(pretty))}}
		compVerdict, compReceipt := gate.AdmitWithReserves(context.Background(), call, prettyRes, ReserveState{
			Status: HeadroomStatusHealthy,
		})
		if compVerdict.Kind != abi.VerdictTransform {
			t.Fatalf("compVerdict.Kind = %v, want VerdictTransform", compVerdict.Kind)
		}
		if compReceipt.Budget.Tier != TierStandard {
			t.Errorf("compReceipt.Budget.Tier = %q, want %q", compReceipt.Budget.Tier, TierStandard)
		}
		if compVerdict.Meta["budget_tier"] != TierStandard {
			t.Errorf("compVerdict.Meta[budget_tier] = %q, want %q", compVerdict.Meta["budget_tier"], TierStandard)
		}
	})

	t.Run("Admit reads headroom metadata from ToolCall", func(t *testing.T) {
		callWithMeta := &abi.ToolCall{
			Tool: "git",
			Meta: map[string]string{
				"headroom_status": "critical",
			},
		}
		verdict := gate.Admit(context.Background(), callWithMeta, res)
		if verdict.Kind != abi.VerdictTransform {
			t.Fatalf("verdict.Kind = %v, want VerdictTransform", verdict.Kind)
		}
		if verdict.Meta["headroom_status"] != "critical" {
			t.Errorf("meta[headroom_status] = %q, want critical", verdict.Meta["headroom_status"])
		}
		if verdict.Meta["budget_tier"] != TierMinimal {
			t.Errorf("meta[budget_tier] = %q, want %q", verdict.Meta["budget_tier"], TierMinimal)
		}
	})
}
