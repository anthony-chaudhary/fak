package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/naivecontrol"
)

// naiveControlFixture is a throwaway git repository standing in for the trunk
// checkout, with the verb's clock and arm seams pinned.
type naiveControlFixture struct {
	t      *testing.T
	ws     string
	ledger string
}

func newNaiveControlFixture(t *testing.T, planJSON string, armErr error) *naiveControlFixture {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	f := &naiveControlFixture{t: t, ws: t.TempDir()}
	f.ledger = filepath.Join(f.ws, naivecontrol.DefaultLedgerRel)
	f.git("init", "-q", "-b", "main")
	f.git("config", "user.email", "fixture@example.invalid")
	f.git("config", "user.name", "fixture")
	f.git("config", "commit.gpgsign", "false")
	f.git("commit", "-q", "--allow-empty", "-m", "chore: base")

	oldGit, oldNow, oldArm := naiveControlGit, naiveControlNow, naiveControlRunArm
	t.Cleanup(func() { naiveControlGit, naiveControlNow, naiveControlRunArm = oldGit, oldNow, oldArm })
	naiveControlGit = naivecontrol.ExecGit
	clock := time.Date(2026, 9, 28, 12, 0, 0, 0, time.UTC)
	naiveControlNow = func() time.Time { clock = clock.Add(250 * time.Millisecond); return clock }
	naiveControlRunArm = func(context.Context, string) ([]byte, error) { return []byte(planJSON), armErr }
	return f
}

func (f *naiveControlFixture) git(args ...string) string {
	f.t.Helper()
	out, code, err := naivecontrol.ExecGit(context.Background(), f.ws, args...)
	if err != nil || code != 0 {
		f.t.Fatalf("git %v: code %d err %v out %q", args, code, err, out)
	}
	return strings.TrimSpace(out)
}

func (f *naiveControlFixture) run(want int, argv ...string) (string, string) {
	f.t.Helper()
	var stdout, stderr bytes.Buffer
	argv = append(argv, "--workspace", f.ws)
	if rc := runNaiveControl(&stdout, &stderr, argv); rc != want {
		f.t.Fatalf("fak naive-control %v = rc %d, want %d\nstdout:\n%s\nstderr:\n%s", argv, rc, want, stdout.String(), stderr.String())
	}
	return stdout.String(), stderr.String()
}

func (f *naiveControlFixture) row(want int, argv ...string) naivecontrol.Row {
	f.t.Helper()
	out, _ := f.run(want, append(argv, "--json")...)
	var r naivecontrol.Row
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		f.t.Fatalf("decode row: %v\n%s", err, out)
	}
	return r
}

const naiveControlPlan = `{"schema":"fak.issue-orchestrator-wave-plan.v1","workspace":".","total_issues":10,
"waves":[{"index":1,"id":"wave-1","issue_numbers":[7,8]}]}`

// TestNaiveControlRunHarvestCompare drives the verb end to end on a real git
// fixture: the arm offers #7 and #8, a fix citing #7 lands, the fanout also claims a
// fabricated sha for #8, and harvest counts exactly one shipped pick. The compare
// screen then shows the naive numbers beside an orchestrated column with no data.
func TestNaiveControlRunHarvestCompare(t *testing.T) {
	f := newNaiveControlFixture(t, naiveControlPlan, nil)
	planned := f.row(0, "run", "--run-id", "n1")
	if planned.Stage != naivecontrol.StagePlanned || planned.Arm != naivecontrol.ArmNaive {
		t.Fatalf("run row = %s/%s, want PLANNED naive", planned.Stage, planned.Arm)
	}
	if v, ok := planned.PicksOffered.Get(); !ok || v != 2 {
		t.Fatalf("picks_offered = %s, want 2", planned.PicksOffered)
	}
	if planned.PicksShipped.IsMeasured() {
		t.Fatalf("a PLANNED row reports shipped = %s; it must be UNKNOWN until harvested", planned.PicksShipped)
	}
	if v, ok := planned.ControllerMS.Get(); !ok || v != 250 {
		t.Fatalf("controller_ms = %s, want the arm's 250ms", planned.ControllerMS)
	}

	f.git("commit", "-q", "--allow-empty", "-m", "fix(x): resolve the thing (#7)")
	claims := filepath.Join(t.TempDir(), "claims.json")
	if err := os.WriteFile(claims, []byte(`[{"issue":8,"sha":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	h := f.row(0, "harvest", "--run-id", "n1", "--claims", claims, "--cost-usd", "4.5")
	if v, ok := h.PicksShipped.Get(); !ok || v != 1 {
		t.Fatalf("picks_shipped = %s (%s), want 1", h.PicksShipped, h.PicksShipped.WhyMissing())
	}
	if v, ok := h.CommitsLanded.Get(); !ok || v != 1 {
		t.Fatalf("commits_landed = %s (%s), want 1", h.CommitsLanded, h.CommitsLanded.WhyMissing())
	}
	if h.WallMS.IsMeasured() {
		t.Fatalf("wall_ms = %s though no --wall-ms was given", h.WallMS)
	}

	out, _ := f.run(0, "compare")
	flat := strings.Join(strings.Fields(out), " ") // column widths are tabwriter's business
	for _, want := range []string{
		"picks shipped (git) 1 UNKNOWN",
		"runs 1 (1 harvested) 0 (no data)",
		"cost / shipped pick $4.50 UNKNOWN",
		"naive.wall_ms: 1 of 1 run(s) unmeasured",
		"orchestrated: no runs recorded, so every field is UNKNOWN",
		"verdict: INCONCLUSIVE: orchestrated ship rate UNKNOWN",
	} {
		if !strings.Contains(flat, want) {
			t.Fatalf("compare output missing %q:\n%s", want, out)
		}
	}
}

// TestNaiveControlRecordsAFailedArm: an arm that fails is still a run. The row is
// written with picks UNKNOWN and a reason, and the verb exits 1 so a scheduler
// notices, rather than dropping the evidence of the failure.
func TestNaiveControlRecordsAFailedArm(t *testing.T) {
	f := newNaiveControlFixture(t, "", errors.New("exit status 2: no issues.json"))
	f.run(1, "run", "--run-id", "bad")
	rows, _, err := naivecontrol.ReadLedger(f.ledger)
	if err != nil || len(rows) != 1 {
		t.Fatalf("ledger rows = %d, err %v; the failed run must be recorded", len(rows), err)
	}
	if rows[0].PicksOffered.IsMeasured() || !strings.Contains(rows[0].PicksOffered.WhyMissing(), "arm failed") {
		t.Fatalf("picks_offered = %s (%s), want UNKNOWN naming the failure", rows[0].PicksOffered, rows[0].PicksOffered.WhyMissing())
	}
}

// TestNaiveControlOrchestratedArm folds a witness sidecar fixture: the pipeline's
// CLAIM_WITNESSED sha is re-verified by git, and the missing loop ledger makes
// controller time UNKNOWN rather than 0.
func TestNaiveControlOrchestratedArm(t *testing.T) {
	f := newNaiveControlFixture(t, naiveControlPlan, nil)
	base := f.git("rev-parse", "HEAD")
	f.git("commit", "-q", "--allow-empty", "-m", "fix(y): land it (#42)")
	landed := f.git("rev-parse", "HEAD")
	runs := filepath.Join(f.ws, ".dispatch-runs")
	if err := os.MkdirAll(runs, 0o755); err != nil {
		t.Fatal(err)
	}
	stem := filepath.Join(runs, "resolve-42-20260928-110000")
	sidecar := `{"claim":"CLAIM_WITNESSED","issue":42,"log":"resolve-42-20260928-110000.log","sha":"` + landed + `","verdict":"OK","witness":"diff-witnessed"}`
	for path, body := range map[string]string{
		stem + ".witness": sidecar,
		stem + ".basesha": base + "\n",
		filepath.Join(runs, "resolve-43-20260101-000000.witness"): `{"claim":"CLAIM_NO_COMMIT","issue":43,"sha":null}`,
	} {
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	r := f.row(0, "orchestrated", "--since", "24h", "--loop-ledger", filepath.Join(f.ws, "absent-loops.jsonl"))
	if r.Arm != naivecontrol.ArmOrchestrated || r.Stage != naivecontrol.StageHarvested {
		t.Fatalf("row = %s/%s", r.Arm, r.Stage)
	}
	if v, ok := r.PicksOffered.Get(); !ok || v != 1 {
		t.Fatalf("picks_offered = %s, want 1 (the out-of-window sidecar is excluded)", r.PicksOffered)
	}
	if v, ok := r.PicksShipped.Get(); !ok || v != 1 {
		t.Fatalf("picks_shipped = %s (%s), want 1", r.PicksShipped, r.PicksShipped.WhyMissing())
	}
	for name, m := range map[string]naivecontrol.Metric[int64]{"controller_ms": r.ControllerMS, "wall_ms": r.WallMS} {
		if m.IsMeasured() {
			t.Fatalf("%s = %s, want UNKNOWN", name, m)
		}
	}
	if r.CostUSD.IsMeasured() {
		t.Fatalf("cost_usd = %s, want UNKNOWN (no orchestrated cost source)", r.CostUSD)
	}
}

// TestNaiveControlUsage pins the exit-code convention.
func TestNaiveControlUsage(t *testing.T) {
	for _, tc := range []struct {
		argv []string
		want int
	}{
		{nil, 2},
		{[]string{"--help"}, 0},
		{[]string{"nope"}, 2},
		{[]string{"compare", "-h"}, 0},
		{[]string{"compare", "stray"}, 2},
		{[]string{"harvest"}, 2},
		{[]string{"harvest", "--run-id", "x", "--arm", "human-driven"}, 2},
	} {
		var stdout, stderr bytes.Buffer
		if rc := runNaiveControl(&stdout, &stderr, tc.argv); rc != tc.want {
			t.Fatalf("fak naive-control %v = %d, want %d (stderr %q)", tc.argv, rc, tc.want, stderr.String())
		}
	}
}
