package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessStudyCrossoverCLI(t *testing.T) {
	p := filepath.Join(t.TempDir(), "study.json")
	raw := `{"schema":"fak.harness-crossover-study/v1alpha1","id":"x","tasks":[{"id":"c","domain":"coding"},{"id":"l","domain":"legal"},{"id":"i","domain":"integrated"}],"weights":{"switch_action_seconds":10},"alternatives":[{"id":"native","kind":"native-profile","documentation":[{"url":"https://example.test","retrieved":"2026-08-15"}],"setup":{"value":1,"provenance":"modeled"},"maintenance":{"value":1,"provenance":"modeled"},"runs":[{"task_id":"c","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"l","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"i","explanation":{"provenance":"modeled"},"provenance":"modeled"}]},{"id":"fak","kind":"contextual-harness","documentation":[{"url":"https://example.test","retrieved":"2026-08-15"}],"setup":{"value":2,"provenance":"modeled"},"maintenance":{"value":2,"provenance":"modeled"},"runs":[{"task_id":"c","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"l","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"i","explanation":{"provenance":"modeled"},"provenance":"modeled"}]}]}`
	if err := os.WriteFile(p, []byte(raw), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "crossover", "--input", p})
	if code != 0 || !strings.Contains(out.String(), `"winner": "native"`) {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errb.String())
	}
}

func TestHarnessStudyCreationCLIKeepsFailedBuildersInDenominator(t *testing.T) {
	p := filepath.Join(t.TempDir(), "study.json")
	raw := `{"schema":"fak.harness-creation-study/v1alpha1","id":"creation-1","protocol":{"frozen":true,"ten_minute_limit_seconds":600,"assistance_policy":"task-card-and-help-only","failures_in_denominator":true,"parity":{"frozen":true,"minimum_pairs":2,"max_median_elapsed_ratio":1.25}},"baseline":{"id":"tuned-alt","runnable":true,"tuned":true,"frozen":true,"evidence":"receipts/baseline.json"},"runs":[{"id":"maintainer","participant_id":"maintainer","track":"ten-minute","participant_class":"maintainer-calibration","independent":false,"outcome":"success","elapsed_seconds":10,"receipt":"receipts/m.json"},{"id":"builder-a","participant_id":"builder-a","track":"ten-minute","participant_class":"unfamiliar-builder","independent":true,"outcome":"timeout","elapsed_seconds":600,"receipt":"receipts/a.json"}]}`
	if err := os.WriteFile(p, []byte(raw), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "creation", "--input", p})
	if code != 0 || !strings.Contains(out.String(), `"calibration_runs": 1`) || !strings.Contains(out.String(), `"failures": 1`) || !strings.Contains(out.String(), `"claim_status": "not_yet"`) || !strings.Contains(out.String(), `"complete_pairs": 0`) {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errb.String())
	}
}
