package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flowmetrics"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
	"github.com/anthony-chaudhary/fak/internal/wipreadiness"
)

func TestRunWipAdmitRealGitUntrackedRefusalAndCleanAdmission(t *testing.T) {
	repo := initWipAdmitRepo(t)
	if err := os.WriteFile(filepath.Join(repo, "new.txt"), []byte("work\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	code := runWip(&out, &errOut, []string{"admit", "-C", repo, "--session", "session-a", "--path", "new.txt", "--json"})
	if code != 3 {
		t.Fatalf("untracked code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	var refused wipattr.AdmitReport
	if err := json.Unmarshal(out.Bytes(), &refused); err != nil {
		t.Fatal(err)
	}
	if refused.Verdict != wipattr.AdmitHold || !admitHasReason(refused, wipattr.ReasonPathUntrackedWIP) {
		t.Fatalf("report=%+v", refused)
	}

	if err := os.Remove(filepath.Join(repo, "new.txt")); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	code = runWip(&out, &errOut, []string{"admit", "-C", repo, "--session", "session-a", "--json"})
	if code != 0 {
		t.Fatalf("clean code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	var clean wipattr.AdmitReport
	if err := json.Unmarshal(out.Bytes(), &clean); err != nil {
		t.Fatal(err)
	}
	if clean.Verdict != wipattr.AdmitOK {
		t.Fatalf("clean report=%+v", clean)
	}
}

func TestRunWipAdmitRequiresSession(t *testing.T) {
	t.Setenv("CLAUDE_CODE_SESSION_ID", "")
	t.Setenv("FAK_SESSION_ID", "")
	var out, errOut bytes.Buffer
	if code := runWip(&out, &errOut, []string{"admit", "-C", initWipAdmitRepo(t)}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func admitHasReason(rep wipattr.AdmitReport, reason wipattr.AdmitReason) bool {
	for _, finding := range rep.Findings {
		if finding.Reason == reason {
			return true
		}
	}
	return false
}

func initWipAdmitRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	for _, args := range [][]string{{"init", "-q"}, {"config", "user.email", "test@example.com"}, {"config", "user.name", "Test"}} {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"add", "base.txt"}, {"commit", "-qm", "base"}} {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	return repo
}

func TestRunWipAdmitRefusesFreshFlowOverloadWithTypedReceipt(t *testing.T) {
	repo := initWipAdmitRepo(t)
	issues := writeWipFlowIssues(t, repo, 3, 1)
	var out, errOut bytes.Buffer
	code := runWip(&out, &errOut, []string{
		"admit", "-C", repo, "--session", "session-a", "--json",
		"--flow-issues-file", issues, "--flow-window", "30",
	})
	if code != wipAdmitHoldExit {
		t.Fatalf("code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	var got struct {
		Verdict string `json:"verdict"`
		Flow    struct {
			Verdict    string  `json:"verdict"`
			ReasonCode string  `json:"reason_code"`
			Threshold  float64 `json:"threshold"`
			Observed   struct {
				Arrivals    int     `json:"arrivals"`
				Service     int     `json:"service"`
				ArrivalRate float64 `json:"arrival_rate_per_day"`
				ServiceRate float64 `json:"service_rate_per_day"`
				WindowDays  float64 `json:"window_days"`
			} `json:"observed"`
		} `json:"flow"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != string(wipattr.AdmitHold) || got.Flow.Verdict != "REFUSE" || got.Flow.ReasonCode != "FLOW_ARRIVAL_EXCEEDS_SERVICE" {
		t.Fatalf("receipt=%+v", got)
	}
	if got.Flow.Observed.Arrivals != 3 || got.Flow.Observed.Service != 1 || got.Flow.Observed.ArrivalRate <= got.Flow.Observed.ServiceRate || got.Flow.Observed.WindowDays < 29.9 || got.Flow.Threshold != 1.10 {
		t.Fatalf("flow=%+v", got.Flow)
	}
}

func TestRunWipAdmitExemptsRecoveryFromFlowOverload(t *testing.T) {
	repo := initWipAdmitRepo(t)
	issues := writeWipFlowIssues(t, repo, 3, 1)
	var out, errOut bytes.Buffer
	code := runWip(&out, &errOut, []string{
		"admit", "-C", repo, "--session", "session-a", "--json",
		"--work-intent", "recovery", "--flow-issues-file", issues,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	if !strings.Contains(out.String(), `"intent": "recovery"`) || !strings.Contains(out.String(), `"verdict": "ADMIT"`) {
		t.Fatalf("receipt=%s", out.String())
	}
}

func writeWipFlowIssues(t *testing.T, repo string, opened, closed int) string {
	t.Helper()
	now := time.Now().UTC()
	rows := make([]map[string]any, 0, opened)
	for i := 0; i < opened; i++ {
		row := map[string]any{
			"number":    i + 1,
			"title":     "flow fixture",
			"createdAt": now.Add(-time.Duration(i+1) * 24 * time.Hour),
			"closedAt":  nil,
			"labels":    []any{},
			"body":      "",
		}
		if i < closed {
			row["closedAt"] = now.Add(-time.Duration(i+1) * 12 * time.Hour)
		}
		rows = append(rows, row)
	}
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(repo, "flow-issues.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunWipAdmitAgingReceiptBlocksFreshAndScrubsUnit(t *testing.T) {
	repo := initWipAdmitRepo(t)
	now := time.Now().UTC()
	issues := writeWipAgingIssues(t, repo, now)
	commits := writeWipAgingCommits(t, repo, now)
	readiness := writeWipAgingReadiness(t, repo, now)
	var out, errOut bytes.Buffer
	code := runWip(&out, &errOut, []string{
		"admit", "-C", repo, "--session", "session-a", "--json",
		"--flow-issues-file", issues, "--aging-commits-file", commits,
		"--aging-readiness-file", readiness, "--aging-budget", "168h",
	})
	if code != wipAdmitHoldExit {
		t.Fatalf("code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	var got struct {
		Verdict string `json:"verdict"`
		Aging   struct {
			Verdict      string                `json:"verdict"`
			ReasonCode   string                `json:"reason_code"`
			BudgetDays   float64               `json:"budget_days"`
			BlockingUnit flowmetrics.AgingUnit `json:"blocking_unit"`
			SafeActions  []string              `json:"safe_actions"`
		} `json:"aging"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != string(wipattr.AdmitHold) || got.Aging.Verdict != "REFUSE" || got.Aging.ReasonCode != flowmetrics.AgingWIPReasonCode {
		t.Fatalf("receipt=%+v", got)
	}
	if got.Aging.BlockingUnit.Unit != "#10417" || got.Aging.BlockingUnit.Classification != flowmetrics.AgingActionable || got.Aging.BlockingUnit.AgeDays < 9.9 || got.Aging.BudgetDays != 7 {
		t.Fatalf("aging=%+v", got.Aging)
	}
	if len(got.Aging.SafeActions) < 5 || strings.Contains(out.String(), repo) || strings.Contains(out.String(), "private") {
		t.Fatalf("receipt leaked or omitted safe actions: %s", out.String())
	}
}

func TestRunWipAdmitRecoveryDoesNotRequireReadiness(t *testing.T) {
	repo := initWipAdmitRepo(t)
	now := time.Now().UTC()
	issues := writeWipAgingIssues(t, repo, now)
	commits := writeWipAgingCommits(t, repo, now)
	var out, errOut bytes.Buffer
	code := runWip(&out, &errOut, []string{
		"admit", "-C", repo, "--session", "session-a", "--json",
		"--work-intent", "recovery", "--flow-issues-file", issues,
		"--aging-commits-file", commits,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	if !strings.Contains(out.String(), `"aging"`) || !strings.Contains(out.String(), `"verdict": "ADMIT"`) {
		t.Fatalf("receipt=%s", out.String())
	}
}

func writeWipAgingIssues(t *testing.T, repo string, now time.Time) string {
	t.Helper()
	rows := []map[string]any{{
		"number": 10417, "title": "aging fixture", "createdAt": now.Add(-30 * 24 * time.Hour),
		"closedAt": nil, "labels": []any{}, "body": "",
	}}
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(repo, "aging-issues.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeWipAgingCommits(t *testing.T, repo string, now time.Time) string {
	t.Helper()
	rows := []flowmetrics.Commit{{SHA: "private-sha", When: now.Add(-10 * 24 * time.Hour), Subject: "start #10417", Issues: []int{10417}}}
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(repo, "aging-commits.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeWipAgingReadiness(t *testing.T, repo string, now time.Time) string {
	t.Helper()
	receipt := wipreadiness.Receipt{
		Schema: "fak-wip-readiness/1", ObservedAt: now.Add(-time.Minute), ExpiresAt: now.Add(time.Hour),
		Verdict: wipreadiness.VerdictCurrent, EvidenceOnly: true,
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(repo, "aging-readiness.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
