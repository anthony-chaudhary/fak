package dependencyquarantine

import (
	"os"
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

func TestHasGoFilesDistinguishesBoundaryModules(t *testing.T) {
	cases := []struct {
		name  string
		files []string
		want  bool
	}{
		{name: "boundary-only module (go.mod and ledgers, no packages)", files: []string{"go.mod", "cache-value.jsonl"}, want: false},
		{name: "testdata-only module", files: []string{"go.mod", "testdata/x.go"}, want: false},
		{name: "module with a package", files: []string{"go.mod", "main.go"}, want: true},
		{name: "module with a package in a subdirectory", files: []string{"go.mod", "inner/x.go"}, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, rel := range tc.files {
				p := filepath.Join(dir, filepath.FromSlash(rel))
				if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(p, []byte("package x\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if got := hasGoFiles(t, dir); got != tc.want {
				t.Fatalf("hasGoFiles(%s) = %v, want %v", tc.name, got, tc.want)
			}
		})
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
	if !hasGoFiles(t, dir) {
		// A boundary-only module (docs/ledger/media trees carry a go.mod purely to keep the
		// root module proxy-publishable) has nothing to compile: `go test ./...` there
		// reports "matched no packages" and exits 1, which is not a dependency violation.
		t.Logf("discovered nested module %s (boundary-only: no Go packages)", dir)
		return
	}
	run := testCommand(t, dir)
	if out, err := run.CombinedOutput(); err != nil {
		t.Fatalf("go test ./...: %v\n%s", err, out)
	}
}

// hasGoFiles reports whether the module tree holds at least one Go source file outside
// testdata/vendor, i.e. whether `go test ./...` has any package to build.
func hasGoFiles(t *testing.T, dir string) bool {
	t.Helper()
	found := false
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if path != dir && (name == "testdata" || name == "vendor" || strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(d.Name(), ".go") {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scan %s for Go files: %v", dir, err)
	}
	return found
}
