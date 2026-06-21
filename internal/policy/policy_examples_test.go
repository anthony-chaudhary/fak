package policy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExamplePoliciesParse(t *testing.T) {
	paths, err := filepath.Glob("../../examples/*.json")
	if err != nil {
		t.Fatalf("glob examples: %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("no example policies found")
	}
	for _, path := range paths {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			if _, err := ParseRuntime(b); err != nil {
				t.Fatalf("parse %s: %v", path, err)
			}
		})
	}
}
