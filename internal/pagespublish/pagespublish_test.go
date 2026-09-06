package pagespublish

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuditSourceRejectsInvalidUTF8(t *testing.T) {
	d := t.TempDir()
	if err := os.WriteFile(filepath.Join(d, "bad.md"), []byte{'#', ' ', 0x97}, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := AuditSource(d)
	if err == nil || !strings.Contains(err.Error(), "not valid UTF-8") {
		t.Fatalf("error = %v", err)
	}
}

func TestAuditArtifactWritesExactManifestAndChecksSEO(t *testing.T) {
	d := t.TempDir()
	page := `<html><head><title>Token efficiency</title><meta name="description" content="catalog"><link rel="canonical" href="https://example.test/fak/awesome-token-efficiency.html"></head></html>`
	if err := os.WriteFile(filepath.Join(d, "awesome-token-efficiency.html"), []byte(page), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(d, "sitemap.xml"), []byte(`<url><loc>https://example.test/fak/awesome-token-efficiency.html</loc></url>`), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := AuditArtifact(d, "https://example.test/fak/", 1, []string{"awesome-token-efficiency.html"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if r.Pages != 1 {
		t.Fatalf("pages = %d", r.Pages)
	}
	b, err := os.ReadFile(filepath.Join(d, ".pages-manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "awesome-token-efficiency.html") || strings.Contains(string(b), ".pages-manifest.json") {
		t.Fatalf("manifest = %s", b)
	}
}

func TestAuditSourceRejectsUnterminatedLiquidInMarkdownTable(t *testing.T) {
	d := t.TempDir()
	content := "| * | crystal | symbol | `Sources: []SourceWitness{{Kind: Foo` |\n"
	if err := os.WriteFile(filepath.Join(d, "bad_table.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := AuditSource(d)
	if err == nil || !strings.Contains(err.Error(), "unterminated Liquid tag") {
		t.Fatalf("expected unterminated Liquid error, got: %v", err)
	}

	// Terminated Liquid {{ ... }} is allowed
	contentValid := "| * | crystal | symbol | `Sources: []SourceWitness{{Kind: Foo}}` |\n"
	if err := os.WriteFile(filepath.Join(d, "bad_table.md"), []byte(contentValid), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = AuditSource(d)
	if err != nil {
		t.Fatalf("expected nil error for terminated Liquid tag, got: %v", err)
	}
}
