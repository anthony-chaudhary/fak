package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunDoneCheckRedOnDirtyPaths(t *testing.T) {
	runner := fakeDoneRunner(map[string]doneRunResult{
		"git status --porcelain -- cmd/fak/done.go": {Stdout: []byte(" M cmd/fak/done.go\n")},
	})
	oldSweepNext := doneSweepNextActionForDirty
	doneSweepNextActionForDirty = func(ctx context.Context, root string) string {
		return "remove 3 junk path(s) with `fak sweep --clean-junk` if you own them, then rerun `fak sweep --json`"
	}
	t.Cleanup(func() { doneSweepNextActionForDirty = oldSweepNext })

	report := runDoneCheck(context.Background(), doneOptions{
		Dir:        ".",
		Paths:      []string{"cmd/fak/done.go"},
		TestTarget: "none",
		Witness:    "commit-audit HEAD",
	}, runner)
	if report.OK || report.Verdict != doneVerdictRed {
		t.Fatalf("report = %+v, want red", report)
	}
	if report.MissingWitness != "clean_path_state" {
		t.Fatalf("missing witness = %q, want clean_path_state", report.MissingWitness)
	}
	for _, want := range []string{"next from sweep", "remove 3 junk path(s)", "fak sweep --clean-junk", "fak sweep --json", "fak sweep --apply --lane <lane>", "fak commit --path"} {
		if !strings.Contains(report.NextStep, want) {
			t.Fatalf("dirty next step missing %q: %q", want, report.NextStep)
		}
	}
}

func TestRunDoneCheckGreenOnLoopgateWitnessedCommit(t *testing.T) {
	runner := fakeDoneRunner(map[string]doneRunResult{
		"git status --porcelain --":    {},
		"fak-test test fast":           {Stdout: []byte("tests pass\n")},
		"make claims-lint":             {Stdout: []byte("claims-lint: 10 lines, 0 violations\n")},
		"dos commit-audit --json HEAD": {Stdout: []byte(`[{"verdict":"OK","witness":"diff-witnessed","sha":"abc123"}]`)},
	})
	report := runDoneCheck(context.Background(), doneOptions{
		Dir:        ".",
		TestTarget: "fast",
		Witness:    "commit-audit HEAD",
	}, runner)
	if !report.OK || report.Verdict != doneVerdictGreen {
		t.Fatalf("report = %+v, want green", report)
	}
	if got := report.Checks[len(report.Checks)-1].Detail; !strings.Contains(got, "commit-audit diff witnessed") {
		t.Fatalf("witness detail = %q", got)
	}
}

func TestRunDoneCheckUsesLoopgateForUnwitnessedCommit(t *testing.T) {
	runner := fakeDoneRunner(map[string]doneRunResult{
		"git status --porcelain --":    {},
		"make claims-lint":             {},
		"dos commit-audit --json HEAD": {Stdout: []byte(`[{"verdict":"CLAIM_UNWITNESSED","witness":"diff-witnessed","reason":"no test files"}]`)},
	})
	report := runDoneCheck(context.Background(), doneOptions{
		Dir:        ".",
		TestTarget: "none",
		Witness:    "commit-audit HEAD",
	}, runner)
	if report.OK {
		t.Fatalf("report = %+v, want red", report)
	}
	if report.MissingWitness != "LOOP_DONE_UNWITNESSED" {
		t.Fatalf("missing witness = %q, want loopgate reason", report.MissingWitness)
	}
	if !strings.Contains(report.NextStep, "dos commit-audit --json HEAD") {
		t.Fatalf("next step = %q", report.NextStep)
	}
}

func TestRunDoneJSON(t *testing.T) {
	runner := fakeDoneRunner(map[string]doneRunResult{
		"git status --porcelain --":    {},
		"make claims-lint":             {},
		"dos commit-audit --json HEAD": {Stdout: []byte(`[{"verdict":"OK","witness":"diff-witnessed"}]`)},
	})

	var out, errb bytes.Buffer
	code := runDoneWithRunner(&out, &errb, []string{"--dir", ".", "--test", "none", "--json"}, runner)
	if code != 0 {
		t.Fatalf("exit = %d; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var report doneReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if !report.OK || report.Verdict != doneVerdictGreen {
		t.Fatalf("report = %+v, want green", report)
	}
}

func fakeDoneRunner(results map[string]doneRunResult) doneRunner {
	return func(ctx context.Context, dir, name string, args ...string) doneRunResult {
		displayName := name
		if len(args) > 0 && args[0] == "test" {
			displayName = "fak-test"
		}
		parts := append([]string{displayName}, args...)
		key := strings.Join(parts, " ")
		if res, ok := results[key]; ok {
			return res
		}
		return doneRunResult{Code: 127, Stderr: []byte("unexpected command: " + key)}
	}
}
