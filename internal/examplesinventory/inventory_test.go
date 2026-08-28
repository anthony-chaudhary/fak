package examplesinventory

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCheckREADMETracksCorpus(t *testing.T) {
	root := t.TempDir()
	must := func(path, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	must(filepath.Join(root, "a", "README.md"), "a")
	must(filepath.Join(root, "data", "fixture.txt"), "x")
	must(filepath.Join(root, "policy.json"), `{"schema":"fak-policy.v1"}`)
	must(filepath.Join(root, "config.json"), `{"schema":"other"}`)
	must(filepath.Join(root, "README.md"), "| **Runnable demos** | x | 2 directories, 1 with their own README |\n")
	if err := CheckREADME(root); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "new-demo"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := CheckREADME(root); err == nil {
		t.Fatal("added directory did not make summary stale")
	}
}
func TestRepositoryREADMEInventory(t *testing.T) {
	root := filepath.Join("..", "..", "examples")
	if err := CheckREADME(root); err != nil {
		t.Fatal(err)
	}
}
