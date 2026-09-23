package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestWorktreeWorkerPreparedBindingNormalizesSymptomTestsAndPreservesDefault(t *testing.T) {
	legacy, err := worktreeWorkerPreparedVerificationBindingForPolicy("go-build", "red-parent-green-candidate", []string{"vulkan"})
	if err != nil {
		t.Fatal(err)
	}
	withoutSelectors, err := worktreeWorkerPreparedVerificationBindingForPolicyAndTests("go-build", "red-parent-green-candidate", []string{"vulkan"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(legacy, withoutSelectors) {
		t.Fatalf("empty --symptom-test changed legacy binding:\nlegacy=%+v\nnew=%+v", legacy, withoutSelectors)
	}

	selected, err := worktreeWorkerPreparedVerificationBindingForPolicyAndTests(
		"go-build", "red-parent-green-candidate", []string{"vulkan"},
		[]string{" ^TestBeta$ ", "^TestAlpha$", "^TestBeta$"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if selected.Command == legacy.Command {
		t.Fatal("selected symptom scope did not change prepared verification digest input")
	}
	var recipe worktreeWorkerPreparedVerificationRecipe
	if err := json.Unmarshal([]byte(selected.Command), &recipe); err != nil {
		t.Fatal(err)
	}
	want := []string{"^TestAlpha$", "^TestBeta$"}
	if !reflect.DeepEqual(recipe.SymptomTests, want) {
		t.Fatalf("symptom_tests=%v, want normalized %v", recipe.SymptomTests, want)
	}
}

func TestWorktreeWorkerLandAcceptRejectsChangedSymptomSelector(t *testing.T) {
	repo, wt, base, paths := newPreparedCLIWorkerFixture(t, false)
	prepare := append(preparedCLIArgs("prepare", repo, wt, base, paths), "--symptom-test", "^TestAlpha$")
	prep, receipt, code, stderr := runPreparedCLI(t, prepare)
	if code != 0 || !prep.OK || receipt == nil {
		t.Fatalf("prepare result=%+v receipt=%+v code=%d stderr=%s", prep, receipt, code, stderr)
	}

	before := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main"))
	accept := append(preparedCLIArgs("accept", repo, wt, base, paths),
		"--symptom-test", "^TestBeta$", "--receipt-id", receipt.ReceiptID)
	got, _, code, stderr := runPreparedCLI(t, accept)
	if code == 0 || got.OK {
		t.Fatalf("changed symptom selector accepted: result=%+v code=%d stderr=%s", got, code, stderr)
	}
	if got.Code != workerworktree.LandResultPreparedMismatch {
		t.Fatalf("changed selector code=%q, want %q", got.Code, workerworktree.LandResultPreparedMismatch)
	}
	if after := strings.TrimSpace(worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main")); after != before {
		t.Fatalf("changed selector moved trunk: %s -> %s", before, after)
	}
}
