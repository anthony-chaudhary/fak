package modelsrc

import (
	"io"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkModelSrc(b *testing.B) {
	payload := []byte("benchmark-model-weights-payload")
	path := filepath.Join(b.TempDir(), "bench-model.gguf")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		b.Fatal(err)
	}
	fileURL := (&url.URL{Scheme: "file", Path: "/" + filepath.ToSlash(path)}).String()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r, size, err := Open(fileURL)
		if err != nil {
			b.Fatalf("Open failed: %v", err)
		}
		if size != int64(len(payload)) {
			b.Fatalf("unexpected size: got %d, want %d", size, len(payload))
		}
		if closer, ok := r.(io.Closer); ok {
			_ = closer.Close()
		}
	}
}
