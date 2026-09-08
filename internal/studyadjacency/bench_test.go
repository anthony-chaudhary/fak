package studyadjacency

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func loadCheckedInManifest(tb testing.TB) (Manifest, []byte) {
	tb.Helper()
	root := filepath.Join("..", "..")
	manifestPath := filepath.Join(root, "docs", "research", "inventory", "vllm-related-system-adjacency-v1.json")
	rawManifest, err := os.ReadFile(manifestPath)
	if err != nil {
		tb.Fatalf("ReadFile(%q) error = %v", manifestPath, err)
	}
	manifest, err := Read(bytes.NewReader(rawManifest))
	if err != nil {
		tb.Fatalf("Read(checked-in manifest) error = %v", err)
	}
	return manifest, rawManifest
}

func BenchmarkValidate(b *testing.B) {
	manifest := validManifest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(manifest); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

func BenchmarkValidateCheckedIn(b *testing.B) {
	manifest, _ := loadCheckedInManifest(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(manifest); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

func BenchmarkRenderMarkdown(b *testing.B) {
	manifest := validManifest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := RenderMarkdown(manifest)
		if err != nil {
			b.Fatalf("RenderMarkdown failed: %v", err)
		}
		if len(data) == 0 {
			b.Fatal("unexpected empty rendered output")
		}
	}
}

func BenchmarkRenderMarkdownCheckedIn(b *testing.B) {
	manifest, _ := loadCheckedInManifest(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := RenderMarkdown(manifest)
		if err != nil {
			b.Fatalf("RenderMarkdown failed: %v", err)
		}
		if len(data) == 0 {
			b.Fatal("unexpected empty rendered output")
		}
	}
}

func BenchmarkWriteMarkdown(b *testing.B) {
	manifest := validManifest()
	var buf bytes.Buffer
	buf.Grow(4096)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		if err := WriteMarkdown(&buf, manifest); err != nil {
			b.Fatalf("WriteMarkdown failed: %v", err)
		}
	}
}

func BenchmarkCanonicalJSON(b *testing.B) {
	manifest := validManifest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := CanonicalJSON(manifest)
		if err != nil {
			b.Fatalf("CanonicalJSON failed: %v", err)
		}
		if len(data) == 0 {
			b.Fatal("unexpected empty canonical JSON")
		}
	}
}

func BenchmarkCanonicalJSONCheckedIn(b *testing.B) {
	manifest, _ := loadCheckedInManifest(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := CanonicalJSON(manifest)
		if err != nil {
			b.Fatalf("CanonicalJSON failed: %v", err)
		}
		if len(data) == 0 {
			b.Fatal("unexpected empty canonical JSON")
		}
	}
}

func BenchmarkWriteCanonicalJSON(b *testing.B) {
	manifest := validManifest()
	var buf bytes.Buffer
	buf.Grow(4096)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		if err := WriteCanonicalJSON(&buf, manifest); err != nil {
			b.Fatalf("WriteCanonicalJSON failed: %v", err)
		}
	}
}

func BenchmarkReadJSON(b *testing.B) {
	manifest := validManifest()
	raw, err := CanonicalJSON(manifest)
	if err != nil {
		b.Fatalf("CanonicalJSON failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(raw)
		m, err := Read(r)
		if err != nil {
			b.Fatalf("Read failed: %v", err)
		}
		if m.ID == "" {
			b.Fatal("unexpected empty manifest ID")
		}
	}
}

func BenchmarkReadCheckedInJSON(b *testing.B) {
	_, raw := loadCheckedInManifest(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(raw)
		m, err := Read(r)
		if err != nil {
			b.Fatalf("Read failed: %v", err)
		}
		if m.ID == "" {
			b.Fatal("unexpected empty manifest ID")
		}
	}
}
