package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDogfoodIssuesDryRunSkipsUnscopedAggregateRows(t *testing.T) {
	report := filepath.Join(t.TempDir(), "report.json")
	const raw = `{
		"schema": "fak.recent-feature-dogfood.v1",
		"ok": true,
		"probes": [{
			"key": "code-slop-scorecard",
			"ok": true,
			"payload": {
				"schema": "fleet-code-slop-scorecard/1",
				"verdict": "ACTION",
				"finding": "code_slop",
				"corpus": {"score": 71.5, "grade": "C", "slop_debt": 12},
				"next_action": "retire slop-debt worst-first; re-run to prove the drop"
			}
		}]
	}`
	if err := os.WriteFile(report, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--json", report})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Planned []struct {
			Key string `json:"key"`
		} `json:"planned"`
		Skipped []struct {
			Key    string `json:"key"`
			Reason string `json:"reason"`
		} `json:"skipped"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if len(got.Planned) != 0 {
		t.Fatalf("planned = %+v, want no dispatchable aggregate rows", got.Planned)
	}
	if len(got.Skipped) != 1 || got.Skipped[0].Key != "recent-feature-dogfood/code-slop-scorecard/code_slop" {
		t.Fatalf("skipped = %+v, want code-slop aggregate row", got.Skipped)
	}
	if got.Skipped[0].Reason != "ISSUE_SCOPE_INCOMPLETE,ISSUE_UNROUTED" {
		t.Fatalf("skip reason = %q", got.Skipped[0].Reason)
	}
}

func TestDogfoodIssuesDryRunReportsFreshnessInJSON(t *testing.T) {
	report := writeDogfoodIssuesReport(t, 5*time.Minute)

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--json", report})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		ReportFreshness struct {
			Timestamp     string `json:"timestamp"`
			Source        string `json:"source"`
			AgeSeconds    int64  `json:"age_seconds"`
			MaxAgeSeconds int64  `json:"max_age_seconds"`
			Stale         bool   `json:"stale"`
			StaleAllowed  bool   `json:"stale_allowed"`
		} `json:"report_freshness"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.ReportFreshness.Source != "mtime" || got.ReportFreshness.Timestamp == "" {
		t.Fatalf("freshness missing timestamp/source: %+v", got.ReportFreshness)
	}
	if got.ReportFreshness.Stale {
		t.Fatalf("freshness = %+v, want fresh", got.ReportFreshness)
	}
	if got.ReportFreshness.AgeSeconds >= got.ReportFreshness.MaxAgeSeconds {
		t.Fatalf("freshness = %+v, want age below max", got.ReportFreshness)
	}
}

func TestDogfoodIssuesDryRunFlagsStaleReportInText(t *testing.T) {
	report := writeDogfoodIssuesReport(t, 2*time.Hour)

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--max-report-age=1h", report})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	body := out.String()
	for _, want := range []string{"report timestamp:", "report age:", "stale=yes", "STALE report:"} {
		if !strings.Contains(body, want) {
			t.Fatalf("output missing %q:\n%s", want, body)
		}
	}
}

func TestDogfoodIssuesLiveRefusesStaleReportBeforeGithub(t *testing.T) {
	report := writeDogfoodIssuesReport(t, 2*time.Hour)

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--live", "--json", "--max-report-age=1h", report})
	if code != 2 {
		t.Fatalf("exit = %d, want stale refusal 2\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	var got struct {
		Error           string `json:"error"`
		Refused         bool   `json:"refused"`
		ReportFreshness struct {
			Stale bool `json:"stale"`
		} `json:"report_freshness"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Error != "stale_report" || !got.Refused || !got.ReportFreshness.Stale {
		t.Fatalf("refusal json = %+v, want stale_report refusal", got)
	}
	if !strings.Contains(errb.String(), "--allow-stale-report") {
		t.Fatalf("stderr missing override hint:\n%s", errb.String())
	}
}

func TestDogfoodIssuesLiveOverrideAllowsStaleReportWithoutGithub(t *testing.T) {
	dir := t.TempDir()
	report := writeDogfoodIssuesReportIn(t, dir, 2*time.Hour)
	existing := filepath.Join(dir, "existing.json")
	if err := os.WriteFile(existing, []byte(`[]`), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{
		"--live", "--allow-stale-report", "--existing-json", existing,
		"--json", "--max-report-age=1h", report,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	var got struct {
		Mode            string `json:"mode"`
		Refused         bool   `json:"refused"`
		ReportFreshness struct {
			Stale        bool `json:"stale"`
			StaleAllowed bool `json:"stale_allowed"`
		} `json:"report_freshness"`
		Synced []struct{} `json:"synced"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Mode != "live" || got.Refused || !got.ReportFreshness.Stale || !got.ReportFreshness.StaleAllowed {
		t.Fatalf("override json = %+v, want allowed stale live result", got)
	}
	if len(got.Synced) != 0 {
		t.Fatalf("synced = %+v, want no GitHub sync for unscoped fixture", got.Synced)
	}
}

// With no report argument, the bridge resolves the newest report under the
// workspace's .fak/recent-feature-dogfood/ — nobody has to hand-locate an
// evidence stamp to run the chain.
func TestDogfoodIssuesDefaultsToNewestReport(t *testing.T) {
	ws := t.TempDir()
	older := filepath.Join(ws, ".fak", "recent-feature-dogfood", "20260629T000000Z")
	newer := filepath.Join(ws, ".fak", "recent-feature-dogfood", "20260630T000000Z")
	for _, dir := range []string{older, newer} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	writeDogfoodIssuesReportIn(t, older, 30*time.Minute)
	want := writeDogfoodIssuesReportIn(t, newer, 5*time.Minute)

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--workspace", ws, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Report string `json:"report"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	wantAbs, err := filepath.Abs(want)
	if err != nil {
		t.Fatal(err)
	}
	if got.Report != wantAbs {
		t.Fatalf("report = %q, want newest stamp %q", got.Report, wantAbs)
	}
}

// A workspace with no packet evidence refuses with the producing command, so the
// stumble ends in a one-line remedy instead of a path hunt.
func TestDogfoodIssuesNoEvidenceNamesTheProducingCommand(t *testing.T) {
	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--workspace", t.TempDir(), "--json"})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errb.String(), "make dogfood-recent") {
		t.Fatalf("stderr must name the producing command:\n%s", errb.String())
	}
}

// A --live run leaves the bridge receipt beside the report — the witness the
// dogfood-score chain axis reads. (The unscoped fixture plans nothing, so no gh
// is touched; the receipt still witnesses that the bridge ran on this report.)
func TestDogfoodIssuesLiveWritesBridgeReceipt(t *testing.T) {
	dir := t.TempDir()
	report := writeDogfoodIssuesReportIn(t, dir, 5*time.Minute)
	existing := filepath.Join(dir, "existing.json")
	if err := os.WriteFile(existing, []byte(`[]`), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDogfoodIssues(&out, &errb, []string{"--live", "--existing-json", existing, "--json", report})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	var got struct {
		Receipt string `json:"receipt"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Receipt == "" {
		t.Fatalf("live result must carry the receipt path:\n%s", out.String())
	}
	raw, err := os.ReadFile(filepath.Join(dir, "issues-sync.json"))
	if err != nil {
		t.Fatalf("receipt must sit beside the report: %v", err)
	}
	var rec struct {
		Mode string `json:"mode"`
	}
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatalf("receipt json: %v\n%s", err, raw)
	}
	if rec.Mode != "live" {
		t.Fatalf("receipt mode = %q, want live", rec.Mode)
	}
}

// A pure dry-run consults nothing and must leave NO receipt — unverified must
// not masquerade as verified.
func TestDogfoodIssuesDryRunLeavesNoReceipt(t *testing.T) {
	dir := t.TempDir()
	report := writeDogfoodIssuesReportIn(t, dir, 5*time.Minute)
	var out, errb bytes.Buffer
	if code := runDogfoodIssues(&out, &errb, []string{"--json", report}); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "issues-sync.json")); !os.IsNotExist(err) {
		t.Fatalf("dry-run must not write a receipt (stat err = %v)", err)
	}
}

func writeDogfoodIssuesReport(t *testing.T, age time.Duration) string {
	t.Helper()
	return writeDogfoodIssuesReportIn(t, t.TempDir(), age)
}

func writeDogfoodIssuesReportIn(t *testing.T, dir string, age time.Duration) string {
	t.Helper()
	report := filepath.Join(dir, "report.json")
	const raw = `{
		"schema": "fak.recent-feature-dogfood.v1",
		"ok": true,
		"probes": [{
			"key": "code-slop-scorecard",
			"ok": true,
			"payload": {
				"schema": "fleet-code-slop-scorecard/1",
				"verdict": "ACTION",
				"finding": "code_slop",
				"corpus": {"score": 71.5, "grade": "C", "slop_debt": 12},
				"next_action": "retire slop-debt worst-first; re-run to prove the drop"
			}
		}]
	}`
	if err := os.WriteFile(report, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	mtime := time.Now().Add(-age)
	if err := os.Chtimes(report, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	return report
}
