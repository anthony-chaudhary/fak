package fleetaccounts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStatusReportRollsUpMixedProviderTierLimits(t *testing.T) {
	home, cfg, regPath := fixture(t)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(regPath))
	rep := BuildStatusReport(rows, []Lease{{Worker: "w-default", Tag: "default"}}, StatusOptions{
		GroupBy: []string{"provider", "tier"},
	})

	if rep.Schema != StatusReportSchema {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Totals.TotalSlots != 3 || rep.Totals.FreeSlots != 1 ||
		rep.Totals.LeasedSlots != 1 || rep.Totals.BlockedSlots != 1 {
		t.Fatalf("totals slots free/leased/blocked/total = %d/%d/%d/%d, want 1/1/1/3",
			rep.Totals.FreeSlots, rep.Totals.LeasedSlots, rep.Totals.BlockedSlots, rep.Totals.TotalSlots)
	}
	ru := findStatusRollup(rep, "provider=anthropic tier=t1")
	if ru == nil {
		t.Fatalf("missing anthropic tier rollup: %+v", rep.Rollups)
	}
	if ru.Mixed || ru.FreeSlots != 0 || ru.LeasedSlots != 1 || ru.BlockedSlots != 1 {
		t.Fatalf("anthropic t1 rollup = %+v, want uniform hard cap with 0 free, 1 leased, 1 blocked", *ru)
	}
	if strings.Contains(strings.Join(rep.Warnings, "\n"), "mixed limit posture") {
		t.Fatalf("uniform hard cap should not report mixed limit posture: %+v", rep.Warnings)
	}
	rendered := RenderStatusReport(rep, false)
	if !strings.Contains(rendered, "rollups by provider+tier") ||
		!strings.Contains(rendered, "provider=anthropic tier=t1") ||
		!strings.Contains(rendered, "slots=0/2 free leased=1 blocked=1") {
		t.Fatalf("render missing hard-cap rollup:\n%s", rendered)
	}
}

func TestStatusReportFiltersGroqTier(t *testing.T) {
	home, cfg, regPath := fixture(t)
	groq := filepath.Join(cfg, "opencode-groq-kimi")
	if err := os.MkdirAll(groq, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(groq, "opencode.json"),
		[]byte(`{"model":"groq/moonshotai/kimi-k2.6"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(regPath))
	rep := BuildStatusReport(rows, nil, StatusOptions{
		Filter:  StatusFilter{Provider: "groq", Tier: 1},
		GroupBy: []string{"provider"},
	})

	if rep.Totals.Accounts != 1 || rep.Totals.TotalSlots != 1 || rep.Totals.FreeSlots != 1 {
		t.Fatalf("groq filtered totals = %+v, want one free one-slot account", rep.Totals)
	}
	if len(rep.Accounts) != 1 {
		t.Fatalf("accounts = %+v, want one", rep.Accounts)
	}
	acct := rep.Accounts[0]
	if acct.Provider != "groq" || acct.Tag != "groq-kimi" || acct.ModelTier == nil || *acct.ModelTier != 1 {
		t.Fatalf("groq account = %+v, want provider groq tag groq-kimi tier 1", acct)
	}
	if len(rep.Rollups) != 1 || rep.Rollups[0].Key != "provider=groq" {
		t.Fatalf("rollups = %+v, want provider=groq", rep.Rollups)
	}
	rendered := RenderStatusReport(rep, true)
	if !strings.Contains(rendered, "provider=groq") || !strings.Contains(rendered, "opencode-groq-kimi") {
		t.Fatalf("filtered render should show groq rollup and account:\n%s", rendered)
	}
}

func TestStatusProviderFilterAliases(t *testing.T) {
	home, cfg, _ := fixture(t)
	nimDir := filepath.Join(cfg, "opencode-nim-glm52")
	if err := os.MkdirAll(nimDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nimDir, "opencode.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})

	anthropic := BuildStatusReport(rows, nil, StatusOptions{Filter: StatusFilter{Provider: "claude"}})
	if anthropic.Totals.Accounts == 0 {
		t.Fatalf("provider alias claude should match Anthropic/Claude rows")
	}

	nim := BuildStatusReport(rows, nil, StatusOptions{Filter: StatusFilter{Provider: "nvidia"}})
	if nim.Totals.Accounts != 1 || nim.Accounts[0].Provider != "nvidia-nim" {
		t.Fatalf("provider alias nvidia should match the NIM row: %+v", nim.Accounts)
	}
}

func findStatusRollup(rep StatusReport, key string) *StatusRollup {
	for i := range rep.Rollups {
		if rep.Rollups[i].Key == key {
			return &rep.Rollups[i]
		}
	}
	return nil
}
