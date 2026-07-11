package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// writeAblationFixture runs a real vdso sweep and captures the AblationReport to a file,
// returning the path — the artifact the --report read path re-analyzes. It is the same
// thing --out writes into experiments/ablate/.
func writeAblationFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "ablation.json")
	if code, _, errb := runAB("--sweep", "vdso", "--out", path); code != 0 {
		t.Fatalf("seed sweep exit=%d stderr=%s", code, errb)
	}
	return path
}

// --report re-renders a saved sweep's table + deltas with NO --trace, engine, or replay:
// the whole value of "ablate from experiments" is reading an ablation back from its file.
func TestAblateReportRerendersSavedRun(t *testing.T) {
	path := writeAblationFixture(t)

	var out, errb bytes.Buffer
	if code := runAblate(&out, &errb, []string{"--report", path}); code != 0 {
		t.Fatalf("--report exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"fak ablate", "workload hash", "all-off", "vdso", "deltas vs all-off"} {
		if !strings.Contains(got, want) {
			t.Fatalf("re-rendered table missing %q:\n%s", want, got)
		}
	}
}

// --report --json re-emits the loaded AblationReport verbatim enough to parse back into
// arms bound to one workload hash.
func TestAblateReportJSONReemitsArms(t *testing.T) {
	path := writeAblationFixture(t)

	var out, errb bytes.Buffer
	if code := runAblate(&out, &errb, []string{"--report", path, "--json"}); code != 0 {
		t.Fatalf("--report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep struct {
		WorkloadHash string `json:"workload_hash"`
		Baseline     string `json:"baseline_arm"`
		Runs         []struct {
			ArmID        string `json:"arm_id"`
			WorkloadHash string `json:"workload_hash"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("re-emitted JSON did not parse: %v\n%s", err, out.String())
	}
	if len(rep.Runs) != 2 {
		t.Fatalf("re-emitted report has %d arms, want 2", len(rep.Runs))
	}
	for _, r := range rep.Runs {
		if r.WorkloadHash != rep.WorkloadHash {
			t.Errorf("arm %q workload hash %q != report %q", r.ArmID, r.WorkloadHash, rep.WorkloadHash)
		}
	}
}

// --report --baseline re-selects the delta reference arm on the loaded report without
// re-running the sweep — the same experiment read against a different reference.
func TestAblateReportRebaseline(t *testing.T) {
	path := writeAblationFixture(t)

	var out, errb bytes.Buffer
	if code := runAblate(&out, &errb, []string{"--report", path, "--baseline", "vdso"}); code != 0 {
		t.Fatalf("--report --baseline exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "deltas vs vdso") {
		t.Fatalf("rebaselined table did not switch reference to vdso:\n%s", got)
	}
	if strings.Contains(got, "deltas vs all-off") {
		t.Fatalf("rebaselined table still references the stored baseline all-off:\n%s", got)
	}
}

// A --baseline that names no arm in the loaded report fails loud via the report guard.
func TestAblateReportRebaselineUnknownArmFailsLoud(t *testing.T) {
	path := writeAblationFixture(t)

	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--report", path, "--baseline", "nonesuch"})
	if code != 1 {
		t.Fatalf("exit=%d, want 1 for an unknown rebaseline arm; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "baseline arm") {
		t.Fatalf("stderr missing the baseline-membership reason:\n%s", errb.String())
	}
}

// A missing report path is a clean load error (exit 1), not a panic or empty table.
func TestAblateReportMissingPath(t *testing.T) {
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--report", filepath.Join(t.TempDir(), "nope.json")})
	if code != 1 {
		t.Fatalf("exit=%d, want 1 for a missing report file; stderr=%s", code, errb.String())
	}
}

// A committed cross-agent artifact (a different schema, no runs[]) is rejected rather
// than rendered as an empty table.
func TestAblateReportRejectsNonArmArtifact(t *testing.T) {
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--report", "../../experiments/ablate/cross-agent-pong-opus.json"})
	if code != 1 {
		t.Fatalf("exit=%d, want 1 for a non-arm artifact; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "AblationReport") {
		t.Fatalf("stderr missing the not-an-AblationReport reason:\n%s", errb.String())
	}
}
