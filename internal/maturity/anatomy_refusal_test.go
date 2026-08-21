package maturity

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPackageAnatomyRefusalsNameRecovery(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	cases := []struct {
		name, root, target string
		want               []string
	}{
		{name: "target escapes root", root: root, target: outside, want: []string{"target", "inside root"}},
		{name: "missing directory", root: root, target: "internal/missing", want: []string{"missing"}},
		{name: "directory has no go package", root: root, target: "internal/empty", want: []string{"no Go package", "internal/empty"}},
		{name: "malformed source", root: root, target: "internal/bad", want: []string{"bad.go", "expected"}},
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "empty"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "bad"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "bad", "bad.go"), []byte("package bad\nfunc broken("), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := AnalyzeAnatomy(tc.root, tc.target)
			if err == nil {
				t.Fatal("AnalyzeAnatomy accepted refusal input")
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q omits recovery text %q", err, want)
				}
			}
		})
	}
}
