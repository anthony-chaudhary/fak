package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunFleetcapPlansWithoutLiveSideEffects is the #6022 parity row for the
// fleet-capacity planner: `fak-dev fleetcap` must produce the same Little's-law
// plan the runtime binary used to print, while touching no worker, no ledger, and
// no network. The witness runs from inside a scratch workspace so any stray write
// (a spawned worker's marker, a capacity ledger, a fleet roster) lands in the
// snapshotted tree and fails the test.
func TestRunFleetcapPlansWithoutLiveSideEffects(t *testing.T) {
	workspace := t.TempDir()
	if err := os.WriteFile(filepath.Join(workspace, "keep.txt"), []byte("untouched"), 0o644); err != nil {
		t.Fatal(err)
	}
	chdir(t, workspace)
	witness := beginPlanningWitness(t, workspace)

	var out, errOut bytes.Buffer
	if code := RunFleetcap(&out, &errOut, []string{"--rate", "400", "--session", "10", "--available", "40"}); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	witness.assertNoLiveSideEffects()

	// The plan itself must still be produced: the capacity table plus the
	// availability verdict line Little's law implies (400/hr * 10min => 66.67 workers,
	// so 40 available workers is UNDER_CAPACITY).
	if !strings.Contains(out.String(), "assessment") {
		t.Fatalf("no assessment line in plan:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "UNDER_CAPACITY") {
		t.Fatalf("40 workers against 400 issues/hr must read UNDER_CAPACITY:\n%s", out.String())
	}
}

// TestRunFleetcapJSONPlanIsMachineReadable pins the --json planning contract that
// callers (dispatch preflight, capacity dashboards) parse.
func TestRunFleetcapJSONPlanIsMachineReadable(t *testing.T) {
	workspace := t.TempDir()
	chdir(t, workspace)
	witness := beginPlanningWitness(t, workspace)

	var out, errOut bytes.Buffer
	if code := RunFleetcap(&out, &errOut, []string{"--json", "--rate", "120", "--seats", "200", "--cap", "64"}); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	witness.assertNoLiveSideEffects()

	var got struct {
		TargetRate float64 `json:"target_rate_per_hour"`
		Table      []struct {
			MedianSessionMinutes float64
			RequiredWorkers      int
		} `json:"table"`
		Assessment *struct {
			Verdict          string
			AvailableWorkers int
		} `json:"assessment"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("plan is not JSON: %v\n%s", err, out.String())
	}
	if got.TargetRate != 120 {
		t.Errorf("target_rate_per_hour = %v, want 120", got.TargetRate)
	}
	if len(got.Table) == 0 {
		t.Error("plan carries no capacity table")
	}
	if got.Assessment == nil {
		t.Fatal("supplying --cap/--seats must yield an assessment")
	}
	// The tightest supplied ceiling wins: --cap 64 below --seats 200.
	if got.Assessment.AvailableWorkers != 64 {
		t.Errorf("available_workers = %d, want the tightest ceiling 64", got.Assessment.AvailableWorkers)
	}
}

// TestRunFleetcapRejectsPositionalArguments keeps the dev-artifact front door as
// strict as the runtime arm it replaced.
func TestRunFleetcapRejectsPositionalArguments(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunFleetcap(&out, &errOut, []string{"400"}); code != 2 {
		t.Fatalf("exit = %d, want 2\nstdout: %s", code, out.String())
	}
	if !strings.Contains(errOut.String(), "unexpected positional arguments") {
		t.Fatalf("stderr = %s", errOut.String())
	}
}

// chdir moves the test into dir for its duration.
func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(prev); err != nil {
			t.Fatal(err)
		}
	})
}
