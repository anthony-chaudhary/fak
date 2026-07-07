package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecost"
)

// TestTierCalibrateDemoRender drives the runnable spine: `fak tier-calibrate
// --demo` folds the embedded fixture and renders the advisory readout with all
// three recommendation branches present.
func TestTierCalibrateDemoRender(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runTierCalibrate(&stdout, &stderr, []string{"--demo"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"tier calibration", "joined=4", "unjoined=0",
		"rec T0 -> expand-cheaper", "rec T1 -> raise-floor", "rec T2 -> hold",
		"auto_apply=false",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q in:\n%s", want, out)
		}
	}
}

// TestTierCalibrateDemoJSON is the acceptance-gate artifact: the command emits a
// captured JSON calibration report that round-trips and carries the advisory bit
// and a named witness source.
func TestTierCalibrateDemoJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runTierCalibrate(&stdout, &stderr, []string{"--demo", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	raw := stdout.Bytes()
	t.Logf("captured calibration command report:\n%s", raw)

	var rep issuecost.CalibrationReport
	if err := json.Unmarshal(raw, &rep); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if rep.Decisions != 4 || rep.Joined != 4 || rep.Unjoined != 0 {
		t.Errorf("join counts: got decisions=%d joined=%d unjoined=%d, want 4/4/0", rep.Decisions, rep.Joined, rep.Unjoined)
	}
	if len(rep.Recommendations) != 3 {
		t.Errorf("recommendations: got %d, want 3", len(rep.Recommendations))
	}
	for _, rec := range rep.Recommendations {
		if rec.AutoApply {
			t.Errorf("recommendation %s has auto_apply=true; the command must stay advisory", rec.Tier)
		}
	}
	// the witness source for success must rest on a non-forgeable commit witness.
	if !strings.Contains(rep.WitnessSources[issuecost.BucketSuccess], "commit-audit") {
		t.Errorf("success witness source should cite commit-audit: %q", rep.WitnessSources[issuecost.BucketSuccess])
	}
}

// TestTierCalibrateFromFiles proves the operator path: rows read from JSON files
// fold identically to the same rows passed in-process.
func TestTierCalibrateFromFiles(t *testing.T) {
	dir := t.TempDir()
	decPath := filepath.Join(dir, "decisions.json")
	outPath := filepath.Join(dir, "outcomes.json")
	writeJSONTestFile(t, decPath, []issuecost.TierDecision{
		{Issue: 1, Chosen: issuecost.TierT2, Required: issuecost.TierT2, Optimal: issuecost.TierT2},
	})
	writeJSONTestFile(t, outPath, []issuecost.WitnessedOutcome{
		{Issue: 1, CommitWitnessed: true, TestsGreen: true, Closed: true},
	})

	var stdout, stderr bytes.Buffer
	if rc := runTierCalibrate(&stdout, &stderr, []string{"--decisions", decPath, "--outcomes", outPath, "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var rep issuecost.CalibrationReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if rep.Joined != 1 || rep.Buckets[issuecost.BucketSuccess] != 1 {
		t.Errorf("from-files fold: got joined=%d success=%d, want 1/1", rep.Joined, rep.Buckets[issuecost.BucketSuccess])
	}
}

// TestTierCalibrateErrors pins the exit conventions: no source -> usage (2), a
// missing file -> read error (1), and --demo mixed with a file -> usage (2).
func TestTierCalibrateErrors(t *testing.T) {
	cases := []struct {
		name string
		argv []string
		want int
	}{
		{"no-source", nil, 2},
		{"missing-file", []string{"--decisions", filepath.Join(t.TempDir(), "nope.json")}, 1},
		{"demo-with-file", []string{"--demo", "--decisions", "x.json"}, 2},
		{"stray-arg", []string{"--demo", "extra"}, 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if rc := runTierCalibrate(&stdout, &stderr, c.argv); rc != c.want {
				t.Errorf("rc=%d, want %d (stderr=%s)", rc, c.want, stderr.String())
			}
		})
	}
}

func writeJSONTestFile(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal %s: %v", path, err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
