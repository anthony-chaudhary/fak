package managedocs

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// BenchmarkManageDocs measures documentation auditing and document set validation throughput.
func BenchmarkManageDocs(b *testing.B) {
	root := b.TempDir()
	docsDir := filepath.Join(root, "docs")
	examplesDir := filepath.Join(root, "examples")
	if err := os.MkdirAll(docsDir, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(examplesDir, 0o755); err != nil {
		b.Fatal(err)
	}

	readme := "# FAK\n\nManaged agent documentation.\n"
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte(readme), 0o644); err != nil {
		b.Fatal(err)
	}

	docContent := strings.Repeat("Line of documentation for managed agent architecture.\n", 50)
	for i := 0; i < 20; i++ {
		p := filepath.Join(docsDir, "guide_"+strconv.Itoa(i)+".md")
		if err := os.WriteFile(p, []byte(docContent), 0o644); err != nil {
			b.Fatal(err)
		}
	}

	indexedDoc := DocumentSetMarker + "\n# Index\n" + strings.Repeat("- [page](pages/page.md)\n", 200)
	if err := os.WriteFile(filepath.Join(docsDir, "index.md"), []byte(indexedDoc), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AuditDocumentSets(root, "docs"); err != nil {
			b.Fatalf("AuditDocumentSets failed: %v", err)
		}
	}
}

// BenchmarkAudit measures documentation audit throughput across repository surfaces.
func BenchmarkAudit(b *testing.B) {
	root := filepath.Clean(filepath.Join("..", ".."))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Audit(root); err != nil {
			b.Fatalf("Audit failed: %v", err)
		}
	}
}

// BenchmarkAuditDocumentSets measures document set validation on varied directory depths.
func BenchmarkAuditDocumentSets(b *testing.B) {
	root := b.TempDir()
	for depth := 0; depth < 5; depth++ {
		dir := filepath.Join(root, "docs", "section_"+strconv.Itoa(depth))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			b.Fatal(err)
		}
		for f := 0; f < 10; f++ {
			p := filepath.Join(dir, "doc_"+strconv.Itoa(f)+".md")
			body := strings.Repeat("Bounded content line for doc audit benchmark.\n", 40)
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				b.Fatal(err)
			}
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AuditDocumentSets(root, "docs"); err != nil {
			b.Fatalf("AuditDocumentSets failed: %v", err)
		}
	}
}
