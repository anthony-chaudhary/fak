package walkfiles

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func BenchmarkFilesFlat(b *testing.B) {
	root := b.TempDir()
	for i := 0; i < 100; i++ {
		p := filepath.Join(root, fmt.Sprintf("file_%03d.txt", i))
		if err := os.WriteFile(p, []byte("benchmark"), 0o644); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		err := Files(root, func(_ string, _ fs.DirEntry) error {
			count++
			return nil
		})
		if err != nil {
			b.Fatalf("Files: %v", err)
		}
		if count != 100 {
			b.Fatalf("expected 100 files, visited %d", count)
		}
	}
}

func BenchmarkFilesNested(b *testing.B) {
	root := b.TempDir()
	for d := 0; d < 5; d++ {
		dir := filepath.Join(root, fmt.Sprintf("dir_%d", d), "nested")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			b.Fatal(err)
		}
		for f := 0; f < 20; f++ {
			p := filepath.Join(dir, fmt.Sprintf("leaf_%02d.go", f))
			if err := os.WriteFile(p, []byte("package bench"), 0o644); err != nil {
				b.Fatal(err)
			}
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		err := Files(root, func(_ string, _ fs.DirEntry) error {
			count++
			return nil
		})
		if err != nil {
			b.Fatalf("Files: %v", err)
		}
		if count != 100 {
			b.Fatalf("expected 100 files, visited %d", count)
		}
	}
}

func BenchmarkFilesFilter(b *testing.B) {
	root := b.TempDir()
	for i := 0; i < 100; i++ {
		ext := ".txt"
		if i%2 == 0 {
			ext = ".go"
		}
		p := filepath.Join(root, fmt.Sprintf("item_%03d%s", i, ext))
		if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matched := 0
		err := Files(root, func(p string, _ fs.DirEntry) error {
			if strings.HasSuffix(p, ".go") {
				matched++
			}
			return nil
		})
		if err != nil {
			b.Fatalf("Files: %v", err)
		}
		if matched != 50 {
			b.Fatalf("expected 50 matched files, got %d", matched)
		}
	}
}

func BenchmarkFilesEarlyAbort(b *testing.B) {
	root := b.TempDir()
	for i := 0; i < 100; i++ {
		p := filepath.Join(root, fmt.Sprintf("file_%03d.txt", i))
		if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
			b.Fatal(err)
		}
	}
	stopErr := errors.New("stop")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		visited := 0
		err := Files(root, func(_ string, _ fs.DirEntry) error {
			visited++
			if visited >= 5 {
				return stopErr
			}
			return nil
		})
		if !errors.Is(err, stopErr) {
			b.Fatalf("expected stopErr, got %v", err)
		}
	}
}

func BenchmarkFilesMissingRoot(b *testing.B) {
	missing := filepath.Join(b.TempDir(), "nonexistent")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		err := Files(missing, func(_ string, _ fs.DirEntry) error {
			count++
			return nil
		})
		if err != nil {
			b.Fatalf("Files: %v", err)
		}
		if count != 0 {
			b.Fatalf("expected 0 files, visited %d", count)
		}
	}
}
