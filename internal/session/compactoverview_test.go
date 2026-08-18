package session

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestRenderCompactOverviewLeadsWithWorkingVerdictAndDailyEvidence(t *testing.T) {
	res := CompactAuditResult{Aggregate: CompactAggregate{
		Sessions: 7, MeasuredFires: 3, ResidentTokensShed: 300000,
		MedianPreTokens: 120000, MedianPostTokens: 20000, MedianResidualRatio: 0.1667,
		CumulativeInputTokens: 900000,
		Daily:                 []DailyTokenStats{{Date: "2026-08-17", Sessions: 7, CumulativeInputTokens: 900000, Fires: 3, ResidentTokensShed: 300000}},
	}}
	var out bytes.Buffer
	RenderCompactOverview(&out, res, CompactOverviewOptions{Days: 4, Since: time.Date(2026, 8, 15, 0, 0, 0, 0, time.Local), Workspace: `C:\work\fak`, Roots: 2})
	got := out.String()
	for _, want := range []string{"Codex context — WORKING", "3 measured fires shed 300,000", "120,000 -> 20,000", "2026-08-17", "billable-token savings estimate", "compact-audit"} {
		if !strings.Contains(got, want) {
			t.Fatalf("overview missing %q:\n%s", want, got)
		}
	}
}

func TestRenderCompactOverviewDistinguishesBoundedAndNoData(t *testing.T) {
	for name, aggregate := range map[string]struct {
		aggregate CompactAggregate
		want      string
	}{
		"bounded": {CompactAggregate{Sessions: 2}, "Codex context — BOUNDED"},
		"empty":   {CompactAggregate{}, "Codex context — NO DATA"},
	} {
		t.Run(name, func(t *testing.T) {
			var out bytes.Buffer
			RenderCompactOverview(&out, CompactAuditResult{Aggregate: aggregate.aggregate}, CompactOverviewOptions{Days: 4, Since: time.Now(), Roots: 1})
			if !strings.Contains(out.String(), aggregate.want) {
				t.Fatalf("overview = %q, want %q", out.String(), aggregate.want)
			}
		})
	}
}
