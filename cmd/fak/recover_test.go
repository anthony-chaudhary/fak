package main

import (
	"bytes"
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestRecoverOffTrunkDryRunPrintsCommands(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"OFF_TRUNK", "--dry-run", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"recover OFF_TRUNK", "git fetch origin main", "git merge --no-edit origin/main", "never force-push"} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
}

func TestRecoverOffTrunkExecuteRunsSafeSteps(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"OFF_TRUNK", "--execute", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	want := [][]string{
		{"git", "fetch", "origin", "main"},
		{"git", "merge", "--no-edit", "origin/main"},
	}
	if !reflect.DeepEqual(ran, want) {
		t.Fatalf("ran = %v, want %v", ran, want)
	}
}

func TestRecoverMergeInProgressExecuteRestoresStaged(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"MERGE_IN_PROGRESS", "--execute"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	want := [][]string{{"git", "restore", "--staged"}}
	if !reflect.DeepEqual(ran, want) {
		t.Fatalf("ran = %v, want %v", ran, want)
	}
}

func TestRecoverManualPlanRefusesExecute(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"STALE_RECALL", "--execute"}); rc != 3 {
		t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no safe executable recovery") {
		t.Fatalf("stderr missing refusal: %s", errb.String())
	}
}

// TestRecoverStaleUntrackedRoutesToTheContentComparison binds the STALE_UNTRACKED refusal
// (#5408) to an actionable playbook. The two things the printed plan must carry are the ones
// the refusal itself was written to correct: compare the trunk copy content-to-content rather
// than by diff, and there is a deliberate one-shot escape for a supersede you actually mean.
func TestRecoverStaleUntrackedRoutesToTheContentComparison(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"STALE_UNTRACKED", "--dry-run", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"recover STALE_UNTRACKED",
		"git fetch origin main",
		"git show origin/main:<path>",
		"FAK_STALE_BASE_GUARD=warn",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
}

// TestRecoverStaleUntrackedIsNotAutoExecutable: merging while the path is still untracked can
// stop on an overwrite refusal, so this playbook is read-only by design and --execute must say
// so rather than run half of it.
func TestRecoverStaleUntrackedIsNotAutoExecutable(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"STALE_UNTRACKED", "--execute"}); rc != 3 {
		t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no safe executable recovery") {
		t.Fatalf("stderr missing refusal: %s", errb.String())
	}
}

// TestRecoverCollisionRiskOffersARouteThatPreservesFinishedWork binds the COLLISION_RISK
// playbook (#5481) to the case its original two routes both dropped. "wait for the live lease"
// and "choose a disjoint lane/region" each assume the change is not written yet; when it is
// already written, built and green, waiting leaves it dirty in a shared checkout (exactly the
// hazard `fak wip sweep-guard` exists to warn about) and it cannot be re-aimed at another lane.
// The plan must therefore name a route that PRESERVES the finished delta — the checkpoint verb —
// and it must be offered first, because it is the only one of the three that is safe to do
// immediately and loses nothing if the operator then also waits or repartitions.
func TestRecoverCollisionRiskOffersARouteThatPreservesFinishedWork(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"COLLISION_RISK", "--dry-run"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"recover COLLISION_RISK",
		"fak wip checkpoint",
		"dos top",
		"dos arbitrate",
		"fak wip sweep-guard",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
	if i, j := strings.Index(got, "fak wip checkpoint"), strings.Index(got, "dos top"); i > j {
		t.Fatalf("checkpoint route must be offered before the wait route (%d > %d):\n%s", i, j, got)
	}

	plan, ok := recoveryPlans("main")["COLLISION_RISK"]
	if !ok {
		t.Fatal("COLLISION_RISK plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"fak", "wip", "checkpoint"}) {
		t.Fatalf("first step = %+v, want the preserve-the-work checkpoint route", plan.Steps)
	}
	if plan.Steps[0].Safe {
		t.Fatal("the checkpoint step must not be marked Safe: --execute would then run it on every collision, which is a behaviour change, not a routing fix")
	}
}

// TestRecoverCollisionRiskStaysManual: adding a concrete command to the plan must not silently
// turn it into an auto-run playbook — a recovery that checkpoints on every collision is a
// behaviour change. --execute still refuses and points at the dry-run notes.
func TestRecoverCollisionRiskStaysManual(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"COLLISION_RISK", "--execute"}); rc != 3 {
		t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no safe executable recovery") {
		t.Fatalf("stderr missing refusal: %s", errb.String())
	}
}

func TestRecoverUnknownFailsClosed(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"NOT_A_REASON"}); rc != 2 {
		t.Fatalf("rc = %d, want 2; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "unknown recovery reason") {
		t.Fatalf("stderr = %s", errb.String())
	}
	if !strings.Contains(errb.String(), "NOT_A_REASON") {
		t.Fatalf("stderr should name the refused token: %s", errb.String())
	}
}
