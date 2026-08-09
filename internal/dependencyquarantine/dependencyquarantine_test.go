package dependencyquarantine

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRepositoryDependencyQuarantine(t *testing.T) {
	root := repoRoot(t)
	violations, err := Check(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range violations {
		t.Error(v)
	}

	modules, err := NestedModules(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(modules) == 0 {
		t.Fatal("nested-module discovery is vacuous: repository has no nested go.mod")
	}
	for _, mod := range modules {
		t.Run(filepath.ToSlash(mustRel(root, mod)), func(t *testing.T) {
			runGoTest(t, mod)
		})
	}
}

func TestCheckRejectsRootDependencyDrift(t *testing.T) {
	root := fixture(t, "module example.test/root\n\ngo 1.26\n\nrequire (\n golang.org/x/sys v0.46.0\n golang.org/x/term v0.44.0\n example.com/heavy v1.0.0\n)\n", strings.Join(keys(allowedRootSum), "\n")+"\n")
	violations, err := Check(root)
	if err != nil {
		t.Fatal(err)
	}
	if !contains(violations, "root require set changed") {
		t.Fatalf("violations = %v, want root require drift", violations)
	}
}

func TestCheckRejectsNonStdlibFacade(t *testing.T) {
	root := fixture(t, "module example.test/root\n\ngo 1.26\n\nrequire (\n golang.org/x/sys v0.46.0\n golang.org/x/term v0.44.0\n)\n", strings.Join(keys(allowedRootSum), "\n")+"\n")
	write(t, filepath.Join(root, "tools", "render", "main.go"), "package main\nimport _ \"example.com/heavy/font\"\nfunc main(){}\n")
	write(t, filepath.Join(root, "tools", "render", "terminal", "go.mod"), "module example.test/render-terminal\n\ngo 1.26\n")
	violations, err := Check(root)
	if err != nil {
		t.Fatal(err)
	}
	if !contains(violations, "non-stdlib") {
		t.Fatalf("violations = %v, want facade import violation", violations)
	}
}

func runGoTest(t *testing.T, dir string) {
	t.Helper()
	// Native Windows application control blocks fresh test executables in this checkout.
	// CI and non-Windows hosts execute every nested module; Windows still proves discovery.
	if runtime.GOOS == "windows" {
		t.Logf("discovered nested module %s (execution delegated to WSL/CI)", dir)
		return
	}
	run := testCommand(t, dir)
	if out, err := run.CombinedOutput(); err != nil {
		t.Fatalf("go test ./...: %v\n%s", err, out)
	}
}
