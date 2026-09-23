package witness

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestRunnableGoTestNamesSelectsOnlyTopLevelRunnables(t *testing.T) {
	body := `package sample

import "testing"

func TestAlpha(t *testing.T) {}
func Testhelper(t *testing.T) {}
func FuzzBytes(f *testing.F) {}
func helper(t *testing.T) {}

type suite struct{}
func (suite) TestMethod(t *testing.T) {}

func ExampleWidget() {
	println("widget")
	// Output: widget
}
`
	got, ok := runnableGoTestNames("sample_test.go", body)
	want := []string{"ExampleWidget", "FuzzBytes", "TestAlpha"}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("runnableGoTestNames() = (%v, %v), want (%v, true)", got, ok, want)
	}
}

func TestResolveGoTestSelectionsUsesCommitBlob(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	writeRepoFile(t, dir, "selection_test.go", "package m\n\nimport \"testing\"\n\nfunc TestCommitted(t *testing.T) {}\n")
	gitIn(t, dir, "add", "selection_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "test: committed selection")

	// Deliberately diverge the worktree. Selection must remain bound to HEAD's blob.
	writeRepoFile(t, dir, "selection_test.go", "package m\n\nimport \"testing\"\n\nfunc TestWorkingTree(t *testing.T) {}\n")
	got, ok := resolveGoTestSelections(context.Background(), gitRunner, dir, "HEAD", []string{"selection_test.go"})
	want := []goTestSelection{{pkg: "./.", names: []string{"TestCommitted"}}}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("resolveGoTestSelections() = (%v, %v), want (%v, true)", got, ok, want)
	}
}

func TestResolveGoTestSelectionsFallsBackOnHelperOnlyOrUncertainBlob(t *testing.T) {
	tests := []struct {
		name string
		body string
		err  error
	}{
		{name: "helper only", body: "package p\n\nfunc helper() {}\n"},
		{name: "unparseable", body: "package p\nfunc TestBroken("},
		{name: "unreadable", err: errors.New("git show failed")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			run := func(context.Context, string, ...string) (string, int, error) {
				if tc.err != nil {
					return "", 128, tc.err
				}
				return tc.body, 0, nil
			}
			got, ok := resolveGoTestSelections(context.Background(), run, ".", "deadbeef", []string{"internal/p/p_test.go"})
			if ok || got != nil {
				t.Fatalf("resolveGoTestSelections() = (%v, %v), want (nil, false) full-package fallback", got, ok)
			}
		})
	}
}

func TestSelectedGoTestArgvAnchorsEscapesAndPreservesTags(t *testing.T) {
	got, ok := selectedGoTestArgv("./internal/p", []string{"metal", "vulkan"}, []string{"TestA.B", "TestA+B", "TestA.B"})
	want := []string{"go", "test", "-count=1", "./internal/p", "-tags", "metal,vulkan", "-run", `^(?:TestA\+B|TestA\.B)$`}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("selectedGoTestArgv() = (%q, %v), want (%q, true)", got, ok, want)
	}
	if legacy := goTestArgv([]string{"./internal/p"}, nil); !reflect.DeepEqual(legacy, []string{"go", "test", "-count=1", "./internal/p"}) {
		t.Fatalf("legacy goTestArgv changed: %q", legacy)
	}
}

func TestSelectedGoTestsPassNoTestFilesFallsBack(t *testing.T) {
	run := func(context.Context, string, ...string) (string, int, error) {
		return "?\tm/internal/p\t[no test files]\n", 0, nil
	}
	passed, definitive := selectedGoTestsPass(context.Background(), run, ".", []goTestSelection{{pkg: "./internal/p", names: []string{"TestChanged"}}}, nil)
	if passed || definitive {
		t.Fatalf("selectedGoTestsPass([no test files]) = (%v, %v), want (false, false) full-package fallback", passed, definitive)
	}
}

func TestRunSelectedGoTestsNoTestFilesAbstains(t *testing.T) {
	run := func(context.Context, string, ...string) (string, int, error) {
		return "?\tm/internal/p\t[no test files]\n", 0, nil
	}
	passed, buildErr := runSelectedGoTests(context.Background(), run, ".", []goTestSelection{{pkg: "./internal/p", names: []string{"TestChanged"}}}, nil)
	if passed || !buildErr {
		t.Fatalf("runSelectedGoTests([no test files]) = (%v, %v), want (false, true) uncertain parent result", passed, buildErr)
	}
}

func TestSymptomSelectionKeepsCandidateAndParentOnSameChangedTests(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int { return 0 }\n")
	writeRepoFile(t, dir, "unrelated_test.go", "package m\n\nimport \"testing\"\n\nfunc TestUnrelatedFailure(t *testing.T) { t.Fatal(\"pre-existing failure\") }\n")
	gitIn(t, dir, "add", "sign.go", "unrelated_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: buggy sign and unrelated failing test")

	writeRepoFile(t, dir, "sign.go", "package m\n\nfunc Sign(n int) int { if n < 0 { return -1 }; return 0 }\n")
	writeRepoFile(t, dir, "sign_test.go", "package m\n\nimport \"testing\"\n\nfunc TestSignNegative(t *testing.T) { if Sign(-1) != -1 { t.Fatal(\"negative sign\") } }\n")
	gitIn(t, dir, "add", "sign.go", "sign_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix: sign with focused regression")

	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("selected red-green symptom = %v, want confirmed; full-package execution would hit the unrelated failure", got)
	}
}

func TestSymptomSelectionFallsBackWhenChangedHelperFeedsAnotherTestFile(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	ctx := context.Background()
	t.Setenv(SymptomFlagEnv, "1")

	writeRepoFile(t, dir, "value.go", "package m\n\nfunc Value() int { return 0 }\n")
	writeRepoFile(t, dir, "shared_test.go", "package m\n\nimport \"testing\"\n\nfunc expectedValue() int { return 0 }\nfunc TestSmoke(t *testing.T) {}\n")
	writeRepoFile(t, dir, "dependent_test.go", "package m\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) { if Value() != expectedValue() { t.Fatal(\"wrong value\") } }\n")
	gitIn(t, dir, "add", "value.go", "shared_test.go", "dependent_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: value zero")

	writeRepoFile(t, dir, "value.go", "package m\n\nfunc Value() int { return 1 }\n")
	writeRepoFile(t, dir, "shared_test.go", "package m\n\nimport \"testing\"\n\nfunc expectedValue() int { return 1 }\nfunc TestSmoke(t *testing.T) {}\n")
	gitIn(t, dir, "add", "value.go", "shared_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix: value and shared test expectation")

	if selections, ok := resolveGoTestSelections(ctx, gitRunner, dir, "HEAD", []string{"shared_test.go"}); ok || selections != nil {
		t.Fatalf("modified helper-bearing file selection = (%v, %v), want (nil, false) full-package fallback", selections, ok)
	}
	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "symptom:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("helper-dependent symptom = %v, want confirmed; selecting only unchanged TestSmoke would falsely refute", got)
	}
}
