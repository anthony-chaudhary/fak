package dependencyquarantine

import (
	"testing"
)

// Invariant: Dependency quarantine checks must discover nested modules and verify root dependencies.
// Guard: NestedModules discovers nested go.mod directories across the workspace.

func TestDependencyQuarantineLifecycle(t *testing.T) {
	t.Parallel()

	root := repoRoot(t)
	modules, err := NestedModules(root)
	if err != nil {
		t.Fatalf("NestedModules failed: %v", err)
	}
	if len(modules) == 0 {
		t.Fatal("expected at least one nested module discovered")
	}
}
