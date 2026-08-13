package amoprofpub

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateLoadsHTMLFirstAndLinksEveryFile(t *testing.T) {
	in := t.TempDir()
	out := t.TempDir()
	must := func(p, s string) {
		t.Helper()
		p = filepath.Join(in, p)
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(s), 0644); err != nil {
			t.Fatal(err)
		}
	}
	must("report.html", "<html><head><script>bad()</script></head><body><h1>AMOProf Native Report</h1><h2>CPU</h2><p>Observed load.</p></body></html>")
	must("raw/metrics.csv", "name,value\ncpu,1\n")
	must("raw/data.json", `{"ok":true}`)
	m, err := Generate(Options{Input: in, Out: out, Title: "Run"})
	if err != nil {
		t.Fatal(err)
	}
	if m.DefaultHTML != "report.html" {
		t.Fatalf("default=%q", m.DefaultHTML)
	}
	if len(m.Files) != 4 {
		t.Fatalf("files=%d", len(m.Files))
	}
	parent, _ := os.ReadFile(filepath.Join(out, "index.confluence.xhtml"))
	p := string(parent)
	for _, f := range m.Files {
		if !strings.Contains(p, f.Attachment) || !strings.Contains(p, f.Path) {
			t.Errorf("missing link for %s", f.Path)
		}
	}
	source, _ := os.ReadFile(filepath.Join(out, "source-report.confluence.xhtml"))
	s := string(source)
	if !strings.Contains(s, "AMOProf Native Report") || !strings.Contains(s, "Observed load.") {
		t.Fatalf("source content missing: %s", s)
	}
	if strings.Contains(s, "bad()") {
		t.Fatal("script content leaked")
	}
	useful, _ := os.ReadFile(filepath.Join(out, "usefulness.confluence.xhtml"))
	if !strings.Contains(string(useful), "not sufficient by itself") {
		t.Fatal("missing usefulness boundary")
	}
}

func TestPickHTMLPrefersIndexThenReport(t *testing.T) {
	got := pickHTML([]File{{Path: "deep/amoprof.html", MediaType: "text/html"}, {Path: "index.html", MediaType: "text/html"}, {Path: "report.html", MediaType: "text/html"}})
	if got != "index.html" {
		t.Fatalf("got %q", got)
	}
}
