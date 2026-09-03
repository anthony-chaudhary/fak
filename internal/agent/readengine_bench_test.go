package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkReadEngine(b *testing.B) {
	root := b.TempDir()
	path := filepath.Join(root, "fixture.bin")
	data := make([]byte, 64<<10)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		b.Fatal(err)
	}
	e := readEngine{root: root}
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, isErr := e.read(path); isErr {
			b.Fatal("read failed")
		}
	}
}
