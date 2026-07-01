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
