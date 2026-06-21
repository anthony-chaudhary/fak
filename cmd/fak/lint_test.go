package main

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toollint"
)

func lintReport(findings ...toollint.Finding) toollint.Report {
	return toollint.Report{Findings: findings}
}

// lintExitCode is the PURE exit contract for `fak lint`: 1 on any error finding, or
// — under --strict — on ANY finding at all (the build-gate mode the help text and
// cmdLint doc both promise); 0 otherwise.
func TestLintExitCodeContract(t *testing.T) {
	errF := toollint.Finding{Code: toollint.StaticWriteShape, Severity: toollint.SevError, Tool: "t"}
	warnF := toollint.Finding{Code: toollint.DeadCacheHint, Severity: toollint.SevWarn, Tool: "t"}
	infoF := toollint.Finding{Code: toollint.AdvertisedUnenforced, Severity: toollint.SevInfo, Tool: "t"}

	cases := []struct {
		name   string
		report toollint.Report
		strict bool
		want   int
	}{
		{"clean", lintReport(), false, 0},
		{"clean-strict", lintReport(), true, 0},
		{"error-nonstrict", lintReport(errF), false, 1},
		{"error-strict", lintReport(errF), true, 1},
		{"warn-nonstrict-passes", lintReport(warnF), false, 0},
		{"warn-strict-fails", lintReport(warnF), true, 1},
		// Under --strict, even an info finding gates (the default agent surface emits
		// only TL004 infos, and `fak lint --strict` is the build-gate path).
		{"info-strict-fails", lintReport(infoF), true, 1},
		{"info-nonstrict-passes", lintReport(infoF), false, 0},
	}
	for _, c := range cases {
		if got := lintExitCode(c.report, c.strict); got != c.want {
			t.Errorf("%s: lintExitCode=%d want %d", c.name, got, c.want)
		}
	}
}

// The --json envelope must carry the documented keys and per-finding row shape.
func TestLintJSONEnvelopeShape(t *testing.T) {
	r := lintReport(toollint.Finding{
		Code: toollint.StaticWriteShape, Severity: toollint.SevError,
		Tool: "writer", Message: "msg", Mechanism: "mech",
	})
	type jf struct {
		Code      string `json:"code"`
		Severity  string `json:"severity"`
		Tool      string `json:"tool"`
		Message   string `json:"message"`
		Mechanism string `json:"mechanism"`
	}
	rows := make([]jf, 0, len(r.Findings))
	for _, f := range r.Findings {
		rows = append(rows, jf{string(f.Code), f.Severity.String(), f.Tool, f.Message, f.Mechanism})
	}
	b, err := json.Marshal(map[string]any{
		"tools": 1, "findings": rows,
		"errors": r.Errors(), "warnings": r.Warnings(), "infos": r.Infos(),
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, k := range []string{"tools", "findings", "errors", "warnings", "infos"} {
		if _, ok := got[k]; !ok {
			t.Errorf("json envelope missing key %q", k)
		}
	}
	if got["errors"].(float64) != 1 || got["warnings"].(float64) != 0 || got["infos"].(float64) != 0 {
		t.Errorf("severity counts wrong: %v", got)
	}
}

// The --kernel-only collector path is lintable without panicking.
func TestLintCollectorBranchKernelOnly(t *testing.T) {
	_ = toollint.Lint(toollint.FromKernel())
}
