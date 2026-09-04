package maturity

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPackageAnatomyEdgeAndAdversarialInputs(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/safe/safe.go", "package safe\nfunc Run() error { return nil }\n")
	write("internal/testonly/safe_test.go", "package testonly\nfunc helper() {}\n")

	cases := []struct{ name, target, want string }{
		{name: "empty target defaults safely", target: "", want: ""},
		{name: "missing package", target: "internal/missing", want: "no such file"},
		{name: "hostile traversal", target: "../escape", want: "inside root"},
		{name: "malformed go", target: "internal/bad", want: "expected"},
		{name: "test only package", target: "internal/testonly", want: ""},
		{name: "oversized target", target: strings.Repeat("x", 4096), want: ""},
	}
	write("internal/bad/bad.go", "package bad\nfunc broken(")
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			anatomy, err := AnalyzeAnatomy(root, tc.target)
			if tc.target == "" {
				if err == nil || !strings.Contains(err.Error(), "maturity") {
					t.Fatalf("default error = %v", err)
				}
				return
			}
			if tc.name == "test only package" {
				if err != nil || anatomy.Package != tc.target || anatomy.Shape.Files != 1 {
					t.Fatalf("anatomy=%+v err=%v", anatomy, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("AnalyzeAnatomy accepted %q: %+v", tc.target, anatomy)
			}
			if tc.want != "" && !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(tc.want)) && !os.IsNotExist(err) {
				t.Fatalf("error=%q want %q", err, tc.want)
			}
		})
	}
}
