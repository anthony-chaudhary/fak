package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// TestSuperloopDriveExecuteRunsRunnableMemberAndWitnesses is the DoD witness for the
// execution rung (#2224 follow-on): a `--execute` drive whose worst-first member is a
// RUNNABLE front door actually runs that command behind the held lease, records its exit
// code as the member's witness, and lands a running→witnessed_done pair on the loop
// ledger. The runner is injected so nothing real spawns — the test asserts the drive
// hands the member's OWN declared command to the executor and folds exit 0 to a witness.
func TestSuperloopDriveExecuteRequiresRepositoryTargetBeforeEffects(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer
	code := runSuperloopDrive(&out, &errb, []string{"drain-throughput", "--workspace", root, "--ledger", ledger, "--execute", "--json"})
	if code != 2 || !strings.Contains(errb.String(), "REPO_TARGET_REQUIRED") {
		t.Fatalf("code=%d stderr=%q, want REPO_TARGET_REQUIRED", code, errb.String())
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("target refusal must precede ledger side effects, stat err=%v", err)
	}
}

func TestSuperloopDriveExecuteRejectsRepositoryMismatchBeforeEffects(t *testing.T) {
	root := t.TempDir()
	initSuperloopTargetRepo(t, root, "https://github.com/anthony-chaudhary/fleet-public.git")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer
	code := runSuperloopDrive(&out, &errb, []string{"drain-throughput", "--workspace", root, "--repo", "anthony-chaudhary/fak", "--ledger", ledger, "--execute", "--json"})
	if code != 2 || !strings.Contains(errb.String(), "REPO_TARGET_MISMATCH") || !strings.Contains(errb.String(), "anthony-chaudhary/fleet-public") {
		t.Fatalf("code=%d stderr=%q, want typed mismatch with actual repo", code, errb.String())
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("target mismatch must precede ledger side effects, stat err=%v", err)
	}
}

func TestSuperloopRepositoryTargetNormalizesHTTPSAndSSH(t *testing.T) {
	for _, remote := range []string{
		"https://github.com/anthony-chaudhary/fak.git",
		"git@github.com:anthony-chaudhary/fak.git",
	} {
		t.Run(remote, func(t *testing.T) {
			root := t.TempDir()
			initSuperloopTargetRepo(t, root, remote)
			if err := verifyDriveRepositoryTarget(root, "anthony-chaudhary/fak"); err != nil {
				t.Fatalf("verify target for %q: %v", remote, err)
			}
		})
	}
}

func initSuperloopTargetRepo(t *testing.T, root, remote string) {
	t.Helper()
	for _, args := range [][]string{{"init", root}, {"-C", root, "remote", "add", "origin", remote}} {
		cmd := exec.Command("git", args...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
}

func TestSuperloopDriveExecuteRunsRunnableMemberAndWitnesses(t *testing.T) {
	orig := superloopFrontDoorRunner
	t.Cleanup(func() { superloopFrontDoorRunner = orig })
	var gotCmd string
	var gotTimeout time.Duration
	superloopFrontDoorRunner = func(command string, timeout time.Duration, out io.Writer) (int, error) {
		gotCmd = command
		gotTimeout = timeout
		return 0, nil // clean run
	}

	root := t.TempDir()
	initSuperloopTargetRepo(t, root, "https://github.com/anthony-chaudhary/fak.git")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer
	// drain-throughput's worst-first member is the runnable `go run ./cmd/fak dispatch
	// auto --goal throughput`; an empty workspace leaves it unmeasured so it is selected.
	code := runSuperloopDrive(&out, &errb, []string{"drain-throughput", "--workspace", root, "--repo", "anthony-chaudhary/fak", "--ledger", ledger, "--execute", "--json"})

	var rep superloopDriveReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("drive json: %v\n%s\nstderr=%s", err, out.String(), errb.String())
	}
	if rep.Outcome != "entered" {
		t.Fatalf("outcome = %q, want entered (code=%d, stderr=%s)", rep.Outcome, code, errb.String())
	}
	if rep.Exec == nil {
		t.Fatal("--execute must attach an exec record")
	}
	if rep.Exec.Kind != superloop.FrontRunnable {
		t.Errorf("front door kind = %q, want runnable", rep.Exec.Kind)
	}
	if !rep.Exec.Ran || !rep.Exec.Witnessed || rep.Exec.ExitCode != 0 {
		t.Errorf("a clean run must be ran+witnessed exit 0, got %+v", rep.Exec)
	}
	if !strings.Contains(gotCmd, "dispatch auto") || gotCmd != rep.Exec.Command {
		t.Errorf("executor got %q, want the member's own front door %q", gotCmd, rep.Exec.Command)
	}

	// BIND (#2224): the member's declared Time allocation caps the run. drain-throughput
	// declares MaxMinutes 15 over its single worklist member, so the per-member share (15)
	// is tighter than the flat 30-minute default and MUST bind the deadline the executor
	// received — the reservation the walk divided down is now a real ceiling, not decor.
	allocMin := rep.Decision.Allocation.MaxMinutes
	if allocMin <= 0 {
		t.Fatalf("drain-throughput must carry a positive Time allocation, got %d", allocMin)
	}
	wantTimeout := defaultSuperloopExecTimeout
	wantSource := "operator"
	if a := time.Duration(allocMin) * time.Minute; a < wantTimeout {
		wantTimeout, wantSource = a, "allocation"
	}
	if gotTimeout != wantTimeout {
		t.Errorf("exec deadline not bound to the member's allocation: got %s want %s (allocMin=%d)", gotTimeout, wantTimeout, allocMin)
	}
	if rep.Exec.TimeoutMinutes != int(wantTimeout/time.Minute) || rep.Exec.TimeoutSource != wantSource {
		t.Errorf("exec record must witness the applied deadline: got %dm/%q want %dm/%q",
			rep.Exec.TimeoutMinutes, rep.Exec.TimeoutSource, int(wantTimeout/time.Minute), wantSource)
	}

	// The member's own lifecycle lands on the loop ledger: a running witness before the
	// run and a witnessed_done witness after — beyond the admitted admission row.
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	row := string(data)
	for _, want := range []string{`"status":"admitted"`, `"status":"running"`, `"status":"witnessed_done"`} {
		if !strings.Contains(row, want) {
			t.Errorf("ledger missing %s witness:\n%s", want, row)
		}
	}
}

// TestSuperloopDriveExecuteSurfacesSkillNeverFakes is the honesty witness: when the
// worst-first member's front door is a Claude SKILL (needs an agent, not a shell), a
// `--execute` drive SURFACES it and runs NOTHING — no running/witnessed_done rows are
// forged on the ledger. The drive can never claim to have run a skill it cannot run
// headless. sweep-surfaces' worst-first member is `/quality-score`.
func TestSuperloopDriveExecuteSurfacesSkillNeverFakes(t *testing.T) {
	orig := superloopFrontDoorRunner
	t.Cleanup(func() { superloopFrontDoorRunner = orig })
	ran := false
	superloopFrontDoorRunner = func(command string, timeout time.Duration, out io.Writer) (int, error) {
		ran = true
		return 0, nil
	}

	root := t.TempDir()
	initSuperloopTargetRepo(t, root, "git@github.com:anthony-chaudhary/fak.git")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer
	runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--repo", "anthony-chaudhary/fak", "--ledger", ledger, "--execute", "--json"})

	var rep superloopDriveReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if rep.Exec == nil {
		t.Fatal("--execute must attach an exec record even for a non-runnable front door")
	}
	if rep.Exec.Kind != superloop.FrontSkill {
		t.Errorf("front door kind = %q, want skill", rep.Exec.Kind)
	}
	if rep.Exec.Ran || rep.Exec.Witnessed {
		t.Errorf("a skill front door must NOT run headless, got %+v", rep.Exec)
	}
	if ran {
		t.Error("the shell runner must never be called for a skill front door")
	}
	data, _ := os.ReadFile(ledger)
	row := string(data)
	if !strings.Contains(row, `"status":"admitted"`) {
		t.Errorf("the admission witness must still land:\n%s", row)
	}
	for _, forbidden := range []string{`"status":"running"`, `"status":"witnessed_done"`} {
		if strings.Contains(row, forbidden) {
			t.Errorf("a surfaced (not run) skill must NOT forge a %s witness:\n%s", forbidden, row)
		}
	}
}

// TestSuperloopExecuteMemberFoldsFailureAndLaunchError pins the two non-clean runnable
// outcomes directly on the execution rung: a non-zero exit is a member FAILURE (ran, not
// witnessed, exit preserved, StatusFailed on the ledger), and a launch error is exit -1
// with an EXEC_ERROR witness — neither is ever mistaken for a witnessed_done.
func TestSuperloopExecuteMemberFoldsFailureAndLaunchError(t *testing.T) {
	runnable := superloop.DriveDecision{
		Enter:  true,
		Member: superloop.Member{Kind: superloop.KindLoop, Ref: "throughput", Enter: "go run ./cmd/fak dispatch auto"},
		Action: "drive it",
	}

	t.Run("non-zero exit is a member failure", func(t *testing.T) {
		orig := superloopFrontDoorRunner
		t.Cleanup(func() { superloopFrontDoorRunner = orig })
		superloopFrontDoorRunner = func(string, time.Duration, io.Writer) (int, error) { return 2, nil }
		ledger := filepath.Join(t.TempDir(), "l.jsonl")
		ex := superloopExecuteMember(io.Discard, ledger, "drain-throughput", runnable, "lease-x", time.Minute)
		if !ex.Ran || ex.Witnessed || ex.ExitCode != 2 {
			t.Fatalf("want ran, unwitnessed, exit 2, got %+v", ex)
		}
		row, _ := os.ReadFile(ledger)
		if !strings.Contains(string(row), `"status":"failed"`) {
			t.Errorf("a failed member run must land a failed witness:\n%s", row)
		}
	})

	t.Run("launch error is exit -1", func(t *testing.T) {
		orig := superloopFrontDoorRunner
		t.Cleanup(func() { superloopFrontDoorRunner = orig })
		superloopFrontDoorRunner = func(string, time.Duration, io.Writer) (int, error) {
			return -1, io.ErrUnexpectedEOF
		}
		ledger := filepath.Join(t.TempDir(), "l.jsonl")
		ex := superloopExecuteMember(io.Discard, ledger, "drain-throughput", runnable, "lease-x", time.Minute)
		if !ex.Ran || ex.Witnessed || ex.ExitCode != -1 {
			t.Fatalf("want ran, unwitnessed, exit -1, got %+v", ex)
		}
		if !strings.Contains(ex.Note, "could not be launched") {
			t.Errorf("launch-error note should explain, got %q", ex.Note)
		}
	})
}

// TestSuperloopEffectiveTimeoutBindsAllocation pins the #2224 fold rule directly: the
// front-door deadline is the tighter of the operator's --exec-timeout and the member's
// declared Time allocation, and the source names which bound won. A bind can only ever
// TIGHTEN the run — an unbudgeted Time dimension (allocation 0) is a HOLD that leaves the
// operator ceiling in force, and both-zero is a deliberately unbounded run.
func TestSuperloopEffectiveTimeoutBindsAllocation(t *testing.T) {
	cases := []struct {
		name       string
		operator   time.Duration
		allocMin   int
		wantDur    time.Duration
		wantSource string
	}{
		{"allocation tighter binds", 30 * time.Minute, 15, 15 * time.Minute, "allocation"},
		{"operator tighter binds", 10 * time.Minute, 15, 10 * time.Minute, "operator"},
		{"equal folds to allocation", 15 * time.Minute, 15, 15 * time.Minute, "allocation"},
		{"unbounded operator, allocation binds", 0, 15, 15 * time.Minute, "allocation"},
		{"held time dimension leaves operator", 30 * time.Minute, 0, 30 * time.Minute, "operator"},
		{"both zero is unbounded", 0, 0, 0, "none"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotDur, gotSource := superloopEffectiveTimeout(tc.operator, tc.allocMin)
			if gotDur != tc.wantDur || gotSource != tc.wantSource {
				t.Errorf("superloopEffectiveTimeout(%s, %d) = %s/%q, want %s/%q",
					tc.operator, tc.allocMin, gotDur, gotSource, tc.wantDur, tc.wantSource)
			}
		})
	}
}

// TestSuperloopRunFrontDoorLive proves the live shell seam captures a real exit code
// cross-platform: a harmless success (`go version`, no side effects) returns 0, and a
// bogus command returns a non-zero exit with a nil exec-layer error (a non-zero exit is a
// member failure, not a launch error).
func TestSuperloopRunFrontDoorLive(t *testing.T) {
	code, err := superloopRunFrontDoorLive("go version", time.Minute, io.Discard)
	if err != nil || code != 0 {
		t.Fatalf("`go version` should exit 0 cleanly, got code=%d err=%v", code, err)
	}

	code, err = superloopRunFrontDoorLive("go definitely-not-a-real-subcommand-xyz", time.Minute, io.Discard)
	if err != nil {
		t.Fatalf("a clean non-zero exit must not surface an exec-layer error, got %v", err)
	}
	if code == 0 {
		t.Fatalf("a bogus command must report a non-zero exit, got %d", code)
	}
}
