package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

func TestBenchmarksListOfflineIncludesVCache(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchmarks(&out, &errb, []string{"list", "--offline"}); code != 0 {
		t.Fatalf("benchmarks list exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "vcache") || !strings.Contains(s, "offline") ||
		!strings.Contains(s, "vCache 2x readiness scorecard") {
		t.Fatalf("offline list missing vcache scorecard:\n%s", s)
	}
}

func TestBenchmarksDescribeVCacheShowsScorecardInputs(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchmarks(&out, &errb, []string{"describe", "vcache"}); code != 0 {
		t.Fatalf("benchmarks describe exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"fak vcache bench --json",
		"--telemetry",
		"--anchors-file",
		"--index-out",
		"--plan-out",
		"--two-x",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("describe missing %q:\n%s", want, s)
		}
	}
}

func TestBenchmarksRunVCacheExecutesScoreGate(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchmarks(&out, &errb, []string{"run", "vcache"}); code != 0 {
		t.Fatalf("benchmarks run exit=%d stderr=%s", code, errb.String())
	}
	var rep struct {
		Schema     string `json:"schema"`
		Status     string `json:"status"`
		TwoXBetter bool   `json:"two_x_better"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("run output is not score JSON: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.vcache.score.v1" || rep.Status != "2x_ready" || !rep.TwoXBetter {
		t.Fatalf("score = %+v, want default vCache 2x gate pass", rep)
	}
}

func TestBenchmarksPreflightReadsEnvironmentFixtures(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchmarks(&out, &errb, []string{
		"preflight",
		"--requirement", environmentFixture("requirement.json"),
		"--receipt", environmentFixture("receipt-pass.json"),
		"--json",
	})
	if code != 0 {
		t.Fatalf("preflight exit=%d stderr=%s", code, errb.String())
	}
	var admission benchcatalog.EnvironmentAdmission
	if err := json.Unmarshal(out.Bytes(), &admission); err != nil {
		t.Fatalf("preflight JSON: %v\n%s", err, out.String())
	}
	if admission.Status != benchcatalog.AdmissionAccepted || admission.RequirementHash == "" || admission.ReceiptHash == "" {
		t.Fatalf("admission = %+v, want accepted hash-bound read-back", admission)
	}
}

func TestBenchmarksPreflightRefusalIsTyped(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchmarks(&out, &errb, []string{
		"preflight",
		"--requirement", environmentFixture("requirement.json"),
		"--receipt", environmentFixture("receipt-forbidden.json"),
		"--json",
	})
	if code != 1 {
		t.Fatalf("preflight exit=%d stderr=%s, want typed refusal exit 1", code, errb.String())
	}
	var admission benchcatalog.EnvironmentAdmission
	if err := json.Unmarshal(out.Bytes(), &admission); err != nil {
		t.Fatalf("preflight JSON: %v\n%s", err, out.String())
	}
	if len(admission.Refusals) != 1 || admission.Refusals[0].Code != benchcatalog.CodeForbidden || admission.Refusals[0].Axis != "network" {
		t.Fatalf("refusals = %+v, want forbidden network", admission.Refusals)
	}
}

func TestBenchmarksPreflightLegacyCatalogEntryFailsClosed(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchmarks(&out, &errb, []string{
		"preflight", "vcache",
		"--receipt", environmentFixture("receipt-pass.json"),
		"--json",
	})
	if code != 1 {
		t.Fatalf("legacy preflight exit=%d stderr=%s, want refusal", code, errb.String())
	}
	var admission benchcatalog.EnvironmentAdmission
	if err := json.Unmarshal(out.Bytes(), &admission); err != nil {
		t.Fatal(err)
	}
	if len(admission.Refusals) != 1 || admission.Refusals[0].Code != benchcatalog.CodeRequirementUnknown || admission.Refusals[0].Axis != "environment" {
		t.Fatalf("legacy admission = %+v, want stable unknown-environment refusal", admission)
	}
}

func environmentFixture(name string) string {
	return filepath.Join("..", "..", "internal", "benchcatalog", "testdata", "environment-admission", name)
}
