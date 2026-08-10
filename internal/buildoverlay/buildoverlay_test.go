package buildoverlay

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestSelectMaskedFiles(t *testing.T) {
	masked, kept, stale := SelectMaskedFiles(
		[]string{"peer/a.go", "mine/b.go", "edited/c.go"},
		[]string{"mine/b.go", "gone.go"},
		map[string]bool{"edited": true},
	)
	if !reflect.DeepEqual(masked, []string{"peer/a.go"}) || !reflect.DeepEqual(kept, []string{"edited/c.go"}) || !reflect.DeepEqual(stale, []string{"gone.go"}) {
		t.Fatalf("masked=%v kept=%v stale=%v", masked, kept, stale)
	}
}

func TestBuildMakesFilesAbsent(t *testing.T) {
	root := t.TempDir()
	got := Build(root, []string{"peer/a.go"})
	want := filepath.Join(root, "peer", "a.go")
	if len(got.Replace) != 1 || got.Replace[want] != "" {
		t.Fatalf("replace=%v want key %s", got.Replace, want)
	}
}

func TestModulePath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "go.mod")
	if err := os.WriteFile(path, []byte("module example.test/m\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ModulePath(path)
	if err != nil || got != "example.test/m" {
		t.Fatalf("got=%q err=%v", got, err)
	}
}
