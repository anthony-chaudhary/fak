package maturity

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestAnalyzeAnatomyPortfolioRanksRelativeStructure(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("dos.toml", `[lanes.trees]
alpha = ["internal/alpha/**"]
beta = ["internal/beta/**"]
cmd = ["cmd/**"]
`)
	write("internal/alpha/alpha.go", `// Package alpha is central.
package alpha

// Run expects a non-negative value.
func Run(n int) error {
	if n < 0 { return errNegative }
	if n == 0 || n == 1 { return nil }
	return nil
}
var errNegative error
`)
	write("internal/beta/beta.go", `package beta
import "github.com/anthony-chaudhary/fak/internal/alpha"
func Use() error { return alpha.Run(1) }
`)
	write("cmd/fak/main.go", `package main
import "github.com/anthony-chaudhary/fak/internal/beta"
var _ = beta.Use
`)

	got, err := AnalyzeAnatomyPortfolio(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.Summary.Packages != 2 || got.Summary.CLIReachablePackages != 2 {
		t.Fatalf("summary = %+v", got.Summary)
	}
	if len(got.Rankings.Complexity) != 1 || got.Rankings.Complexity[0].Package != "internal/alpha" {
		t.Fatalf("complexity ranking = %+v", got.Rankings.Complexity)
	}
	if len(got.Rankings.Dependencies) != 1 || got.Rankings.Dependencies[0].Package != "internal/beta" {
		t.Fatalf("dependency ranking = %+v", got.Rankings.Dependencies)
	}
	if len(got.Rankings.Dependents) != 1 || got.Rankings.Dependents[0].Package != "internal/alpha" {
		t.Fatalf("dependent ranking = %+v", got.Rankings.Dependents)
	}
	if len(got.Packages) != 2 || got.Packages[0].Package != "internal/alpha" || got.Packages[1].Package != "internal/beta" {
		t.Fatalf("packages = %+v", got.Packages)
	}
	var out bytes.Buffer
	RenderAnatomyPortfolioText(&out, got)
	for _, want := range [][]byte{[]byte("MATURITY ANATOMY PORTFOLIO"), []byte("TOP AGGREGATE COMPLEXITY"), []byte("not quality grades")} {
		if !bytes.Contains(out.Bytes(), want) {
			t.Fatalf("missing %q in %s", want, out.String())
		}
	}
}

func TestAnalyzeAnatomyPortfolioRejectsMissingRoster(t *testing.T) {
	_, err := AnalyzeAnatomyPortfolio(t.TempDir(), 10)
	if err == nil {
		t.Fatal("expected missing roster error")
	}
}
