package sessionaudit

// Witness for the shell-choice KPI (#3227). The load-bearing test is
// TestFoldShellChoiceReproducesAuditWindow: it feeds the exact counts the 7-day
// hand-read audit reported and asserts the KPI prints that same table, so the number
// the issue argued from and the number the tool emits are the same arithmetic.

import (
	"math"
	"strings"
	"testing"
	"time"
)

func approx(t *testing.T, label string, got *float64, want float64) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s = nil, want %.4f", label, want)
	}
	if math.Abs(*got-want) > 5e-4 {
		t.Errorf("%s = %.4f, want %.4f", label, *got, want)
	}
}

func shellRow(t *testing.T, sc ShellChoice, tool string) ShellStat {
	t.Helper()
	for _, s := range sc.Shells {
		if s.Tool == tool {
			return s
		}
	}
	t.Fatalf("no %s row in %+v", tool, sc.Shells)
	return ShellStat{}
}

// The table in #3227: Bash 194 calls / 5 errors / 2.6%, PowerShell 33 / 6 / 18.2%.
func TestFoldShellChoiceReproducesAuditWindow(t *testing.T) {
	sc := FoldShellChoice(
		map[string]int64{"Bash": 194, "PowerShell": 33, "Read": 900},
		map[string]int64{"Bash": 5, "PowerShell": 6, "Edit": 40},
	)
	bash, pwsh := shellRow(t, sc, "Bash"), shellRow(t, sc, "PowerShell")
	if bash.Calls != 194 || bash.Errors != 5 {
		t.Errorf("bash = %d calls / %d errors, want 194 / 5", bash.Calls, bash.Errors)
	}
	if pwsh.Calls != 33 || pwsh.Errors != 6 {
		t.Errorf("powershell = %d calls / %d errors, want 33 / 6", pwsh.Calls, pwsh.Errors)
	}
	approx(t, "bash error rate", bash.ErrorRate, 0.0258)    // 2.6%
	approx(t, "pwsh error rate", pwsh.ErrorRate, 0.1818)    // 18.2%
	approx(t, "bash call share", bash.CallShare, 194.0/227) // 85.5%
	approx(t, "pwsh call share", pwsh.CallShare, 33.0/227)  // 14.5%
	if sc.Calls != 227 || sc.Errors != 11 {
		t.Errorf("all shells = %d calls / %d errors, want 227 / 11", sc.Calls, sc.Errors)
	}
	approx(t, "all-shell error rate", sc.ErrorRate, 11.0/227)
	if sc.Preferred != "Bash" {
		t.Errorf("preferred = %q, want Bash", sc.Preferred)
	}
	// The denominator is SHELL calls: 900 unrelated Reads must not move the share, and
	// 40 Edit errors must not land on a shell.
	if got := bash.Calls + pwsh.Calls; int64(got) != sc.Calls {
		t.Errorf("shell totals leaked non-shell tools: %d vs %d", got, sc.Calls)
	}
}

// A shell nobody picked is still a row — that IS the choice signal.
func TestFoldShellChoiceKeepsUnusedShellRow(t *testing.T) {
	sc := FoldShellChoice(map[string]int64{"Bash": 10}, map[string]int64{"Bash": 1})
	if len(sc.Shells) != len(ShellTools) {
		t.Fatalf("got %d rows, want one per shell (%d)", len(sc.Shells), len(ShellTools))
	}
	pwsh := shellRow(t, sc, "PowerShell")
	if pwsh.Calls != 0 {
		t.Errorf("powershell calls = %d, want 0", pwsh.Calls)
	}
	// No calls means no rate to report, not a flawless 0%.
	if pwsh.ErrorRate != nil {
		t.Errorf("powershell error rate = %v, want nil (no calls)", *pwsh.ErrorRate)
	}
	approx(t, "powershell call share", pwsh.CallShare, 0)
}

// No shell calls at all: no answer, not 0%.
func TestFoldShellChoiceEmptyWindowHasNoRate(t *testing.T) {
	sc := FoldShellChoice(map[string]int64{"Read": 5}, nil)
	if sc.Calls != 0 || sc.Errors != 0 {
		t.Errorf("got %d calls / %d errors, want 0 / 0", sc.Calls, sc.Errors)
	}
	if sc.ErrorRate != nil {
		t.Errorf("error rate = %v, want nil", *sc.ErrorRate)
	}
	if sc.Preferred != "" {
		t.Errorf("preferred = %q, want empty", sc.Preferred)
	}
}

func TestSessionShellErrorRate(t *testing.T) {
	s := Session{
		Tools:    map[string]int64{"Bash": 6, "PowerShell": 4, "Read": 20},
		Behavior: Behavior{ToolErrors: map[string]int64{"PowerShell": 2, "Read": 3}},
	}
	approx(t, "session shell error rate", SessionShellErrorRate(s), 0.2) // 2 of 10 shell calls
	if r := SessionShellErrorRate(Session{Tools: map[string]int64{"Read": 3}}); r != nil {
		t.Errorf("shell-free session rate = %v, want nil", *r)
	}
}

// The KPI must reach the aggregate and the per-session distribution, not just the fold.
func TestAggregateSessionsCarriesShellChoiceKPI(t *testing.T) {
	sessions := []Session{
		{
			Path:     "ns/a.jsonl",
			Tools:    map[string]int64{"Bash": 100, "PowerShell": 10},
			Behavior: Behavior{ToolErrors: map[string]int64{"Bash": 1}},
		},
		{
			Path:     "ns/b.jsonl",
			Tools:    map[string]int64{"Bash": 94, "PowerShell": 23},
			Behavior: Behavior{ToolErrors: map[string]int64{"Bash": 4, "PowerShell": 6}},
		},
		// Errored sessions are skipped by the aggregate; so is their shell volume.
		{Path: "ns/c.jsonl", Error: "unreadable", Tools: map[string]int64{"PowerShell": 999}},
	}
	agg := AggregateSessions(sessions)
	sc := agg.ShellChoice
	if sc.Calls != 227 || sc.Errors != 11 {
		t.Fatalf("aggregate shell KPI = %d calls / %d errors, want 227 / 11", sc.Calls, sc.Errors)
	}
	approx(t, "bash error rate", shellRow(t, sc, "Bash").ErrorRate, 5.0/194)
	approx(t, "pwsh error rate", shellRow(t, sc, "PowerShell").ErrorRate, 6.0/33)
	// Per-session distribution: session a is 1/110, session b is 10/117 — the outlier.
	approx(t, "shell-error-rate median", agg.Distributions.ShellErrorRate.Median, (1.0/110+10.0/117)/2)
	approx(t, "shell-error-rate max", agg.Distributions.ShellErrorRate.Max, 10.0/117)
}

// The rendered report must SHOW it: in the scope rollup and in the distribution block.
func TestReportMarkdownRendersShellChoiceKPI(t *testing.T) {
	sessions := []Session{{
		Path:     "ns/a.jsonl",
		Session:  "a",
		Tools:    map[string]int64{"Bash": 194, "PowerShell": 33},
		Behavior: Behavior{ToolErrors: map[string]int64{"Bash": 5, "PowerShell": 6}},
	}}
	md := ReportMarkdown(sessions, AggregateSessions(sessions), "", nil, false, 0, 1, nil, time.Date(2026, 8, 9, 0, 0, 0, 0, time.UTC))
	for _, want := range []string{
		"## Shell choice (KPI)",
		"| Bash | 194 | 85.5% | 5 | 2.6% |",
		"| PowerShell | 33 | 14.5% | 6 | 18.2% |",
		"**Preferred shell:** Bash",
		"**Shell error rate/session",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("report is missing %q", want)
		}
	}
	// It belongs to the ROLLUP, not only to the tool-mix table at the bottom.
	kpi, mix := strings.Index(md, "## Shell choice (KPI)"), strings.Index(md, "## Global tool mix")
	if kpi < 0 || mix < 0 || kpi > mix {
		t.Errorf("shell KPI at %d should precede the global tool mix at %d", kpi, mix)
	}
}
