package witness

import "testing"

// #13213: the go.work load-failure needle classifies ONLY workspace-load errors. Build
// failures, red tests, and missing-package diagnostics stay outside it, so those receipts
// keep their own verdicts instead of collapsing into an ABSTAIN.
func TestIsGoWorkspaceLoadFailureClassifiesOnlyWorkspaceLoads(t *testing.T) {
	positives := []string{
		"go: cannot load module ../companion listed in go.work file: open ../companion/go.mod: no such file or directory\n",
		"go: cannot load module ..\\missing-sibling listed in go.work file: open ..\\missing-sibling\\go.mod: The system cannot find the path specified.\n",
	}
	for _, out := range positives {
		if !isGoWorkspaceLoadFailure(out) {
			t.Errorf("isGoWorkspaceLoadFailure(%q) = false, want true", out)
		}
	}
	negatives := []string{
		"# example.invalid/kernel [test]\nkernel\\value_test.go:4:2: undefined: companion\nFAIL\texample.invalid/kernel [build failed]\n",
		"--- FAIL: TestNegativeValue (0.00s)\n    value_test.go:9: Value()=0, want -1\nFAIL\texample.invalid/kernel\t0.012s\n",
		"go: cannot find package \"example.invalid/companion\" in any folder:\n\tno required module provides package example.invalid/companion\n",
		"",
	}
	for _, out := range negatives {
		if isGoWorkspaceLoadFailure(out) {
			t.Errorf("isGoWorkspaceLoadFailure(%q) = true, want false", out)
		}
	}
}
