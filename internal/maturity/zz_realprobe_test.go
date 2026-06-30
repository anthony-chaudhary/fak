package maturity

import (
	"os"
	"path/filepath"
	"testing"
)

// TestGenerateDoc is a throwaway generator (NOT committed): it writes the maturity
// scorecard doc from the same Markdown() the `fak maturity --markdown` verb uses,
// run against the real repo root. Gated on an env var so a normal `go test` does
// not write files. Run once with MATURITY_WRITE_DOC=1 to (re)generate the doc.
func TestGenerateDoc(t *testing.T) {
	if os.Getenv("MATURITY_WRITE_DOC") == "" {
		t.Skip("set MATURITY_WRITE_DOC=1 to regenerate docs/MATURITY-SCORECARD.md")
	}
	root := "../.."
	p := Build(Options{Root: root})
	out := filepath.Join(root, "docs", "MATURITY-SCORECARD.md")
	if err := os.WriteFile(out, []byte(Markdown(p)), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s (caps=%v debt=%v score=%v)", out,
		p.Corpus["capabilities"], p.Corpus["maturity_debt"], p.Corpus["score"])
}
