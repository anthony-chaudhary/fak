package maturity

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestAnalyzeAnatomyCapturesFlowContractsDocsAndPosition(t *testing.T) {
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
	write("internal/sample/sample.go", `// Package sample demonstrates anatomy.
package sample

import "errors"

// Run expects n to be non-negative.
func Run(n int) error {
	if n < 0 { return errors.New("negative") }
	if n == 0 || n == 1 { return nil }
	return nil
}
`)
	write("internal/consumer/consumer.go", `package consumer
import "github.com/anthony-chaudhary/fak/internal/sample"
func use() { _ = sample.Run(1) }
`)
	write("cmd/fak/main.go", `package main
import "github.com/anthony-chaudhary/fak/internal/consumer"
var _ = consumer.Use
`)

	got, err := AnalyzeAnatomy(root, "internal/sample")
	if err != nil {
		t.Fatal(err)
	}
	if got.Flow.DecisionPoints != 3 || got.Flow.CyclomaticComplexity != 4 {
		t.Fatalf("flow = %+v", got.Flow)
	}
	if got.Outcomes.ReturnSites != 3 || got.Outcomes.ErrorExits != 1 || got.Outcomes.SuccessExits != 2 {
		t.Fatalf("outcomes = %+v", got.Outcomes)
	}
	if got.Contracts.GuardClauses != 2 || got.Contracts.AssumptionComments == 0 {
		t.Fatalf("contracts = %+v", got.Contracts)
	}
	if !got.Documentation.PackageDoc || got.Documentation.DocumentedExports != 1 {
		t.Fatalf("documentation = %+v", got.Documentation)
	}
	if len(got.Position.InternalDependents) != 1 || got.Position.InternalDependents[0] != "internal/consumer" || !got.Position.CLIReachable {
		t.Fatalf("position = %+v", got.Position)
	}
	var out bytes.Buffer
	RenderAnatomyText(&out, got)
	if !bytes.Contains(out.Bytes(), []byte("outcomes")) || !bytes.Contains(out.Bytes(), []byte("static production-code counts")) {
		t.Fatalf("render = %q", out.String())
	}
}

func TestAnalyzeAnatomyKeepsUnclassifiedReturnsVisible(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sample.go"), []byte("package sample\nfunc Value() int { return 42 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := AnalyzeAnatomy(root, "internal/sample")
	if err != nil {
		t.Fatal(err)
	}
	if got.Outcomes.AmbiguousExits != 1 || got.Outcomes.SuccessExits != 0 {
		t.Fatalf("outcomes = %+v", got.Outcomes)
	}
}
