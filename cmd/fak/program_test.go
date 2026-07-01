package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/internal/worktype"
)

func TestProgramReportIncludesHumanOperatorProgram(t *testing.T) {
	root := writeOperatorHeavinessWorkspace(t)
	var out, errb bytes.Buffer
	code := runProgramReport(&out, &errb, []string{
		"--workspace", root,
		"--cache-ledger", filepath.Join(root, "missing-cache-ledger.jsonl"),
		"--ledger", filepath.Join(root, "program-ledger.jsonl"),
		"--date", "2026-06-30",
		"--json",
	})
	if code != 0 {
		t.Fatalf("program report exit = %d, stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var report programreport.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("program report JSON did not parse: %v\n%s", err, out.String())
	}
	var found bool
	for _, s := range report.Programs.Signals {
		if s.Class == worktype.HumanOperatorEffectiveness {
			found = true
			if s.Label != worktype.HumanOperatorEffectiveness.Label() || s.Metric != 1 || s.Direction != "advancing" {
				t.Fatalf("human operator signal = %+v", s)
			}
		}
	}
	if !found {
		t.Fatalf("program report missing %s signal: %+v", worktype.HumanOperatorEffectiveness, report.Programs.Signals)
	}
}
