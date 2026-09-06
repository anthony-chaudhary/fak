package buildoverlay

import (
	"fmt"
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

func TestUntrackedGoFilesAllowsStandaloneModule(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "peer.go"), []byte("package peer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := UntrackedGoFiles(root)
	if err != nil {
		t.Fatalf("UntrackedGoFiles() error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("UntrackedGoFiles() = %v, want no Git-managed files", got)
	}
}

func BenchmarkSelectMaskedFiles(b *testing.B) {
	untracked := make([]string, 0, 120)
	for i := 0; i < 100; i++ {
		pkg := fmt.Sprintf("internal/pkg%d", i%10)
		ext := ".go"
		if i%7 == 0 {
			ext = ".txt"
		}
		untracked = append(untracked, fmt.Sprintf("%s/file_%d%s", pkg, i, ext))
	}
	mine := []string{
		"internal/pkg1/file_1.go",
		"internal/pkg2/file_2.go",
		"internal/pkg99/nonexistent.go",
	}
	modifiedDirs := map[string]bool{
		"internal/pkg0": true,
		"internal/pkg3": true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		masked, kept, stale := SelectMaskedFiles(untracked, mine, modifiedDirs)
		if len(masked) == 0 && len(untracked) > 0 {
			b.Fatal("unexpected empty masked")
		}
		_ = kept
		_ = stale
	}
}

func BenchmarkBuildMakesFilesAbsent(b *testing.B) {
	root := "/path/to/repository/root"
	masked := make([]string, 0, 100)
	for i := 0; i < 100; i++ {
		masked = append(masked, fmt.Sprintf("internal/subpkg%d/peer_%d.go", i%10, i))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ov := Build(root, masked)
		if len(ov.Replace) != len(masked) {
			b.Fatalf("expected %d entries, got %d", len(masked), len(ov.Replace))
		}
	}
}

func BenchmarkModulePath(b *testing.B) {
	dir := b.TempDir()
	modPath := filepath.Join(dir, "go.mod")
	content := []byte("module github.com/anthony-chaudhary/fak\n\ngo 1.26\n\nrequire (\n\t// zero external deps\n)\n")
	if err := os.WriteFile(modPath, content, 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mod, err := ModulePath(modPath)
		if err != nil || mod != "github.com/anthony-chaudhary/fak" {
			b.Fatalf("unexpected result mod=%q err=%v", mod, err)
		}
	}
}

func BenchmarkSlash(b *testing.B) {
	paths := []string{
		`internal\buildoverlay\buildoverlay.go`,
		`cmd/fak/main.go`,
		`internal\windowgate\windowgate_windows.go`,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := Slash(paths[i%len(paths)])
		if s == "" {
			b.Fatal("empty slash path")
		}
	}
}
