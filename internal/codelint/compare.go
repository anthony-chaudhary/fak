package codelint

import (
	"context"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"time"
)

type ComparisonArm struct {
	Name               string
	Kind               string
	Available          bool
	Correct            bool
	Latency            time.Duration
	Files              int
	CorrectFiles       int
	FalseSyntaxErrors  int
	MissedSyntaxErrors int
	LocationErrors     int
	ReportedErrors     int
	CPUSeconds         float64
	PeakRSSBytes       int64
	InputBytes         int64
	NetworkBytes       int64
	OperatorSeconds    float64
	CostUSD            float64
	Note               string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}
type syntaxCase struct {
	name, src string
	wantError bool
}

var syntaxCases = []syntaxCase{{"valid.go", "package p\nfunc Good(){ println(1) }\n", false}, {"single.go", "package p\nfunc Bad( {\n", true}, {"multiple.go", "package p\nfunc A( {\nfunc B( {\n", true}, {"unresolved.go", "package p\nfunc Fine(){ missingSymbol() }\n", false}}

func writeSyntaxCases(dir string) (int64, error) {
	var n int64
	for _, tc := range syntaxCases {
		if err := os.WriteFile(filepath.Join(dir, tc.name), []byte(tc.src), 0600); err != nil {
			return 0, err
		}
		n += int64(len(tc.src))
	}
	return n, nil
}
func runNativeSyntax(dir string, bytes int64) ComparisonArm {
	a := ComparisonArm{Name: "fak native Go syntax pack", Kind: "native", Available: true, Files: len(syntaxCases), InputBytes: bytes, Note: "real codelint go pack parses all errors and intentionally avoids semantic/type diagnostics"}
	start := time.Now()
	for _, tc := range syntaxCases {
		fs, err := goCheck(context.Background(), filepath.Join(dir, tc.name))
		got := err != nil || len(fs) > 0
		a.ReportedErrors += len(fs)
		if got == tc.wantError {
			a.CorrectFiles++
		} else if got {
			a.FalseSyntaxErrors++
		} else {
			a.MissedSyntaxErrors++
		}
		for _, f := range fs {
			if f.Line <= 0 || f.Col <= 0 {
				a.LocationErrors++
			}
		}
	}
	a.Latency = time.Since(start)
	a.Correct = a.CorrectFiles == len(syntaxCases) && a.FalseSyntaxErrors == 0 && a.MissedSyntaxErrors == 0 && a.LocationErrors == 0
	return a
}
func runFirstError(dir string, bytes int64) ComparisonArm {
	a := ComparisonArm{Name: "go/parser first-error-only", Kind: "baseline", Available: true, Files: len(syntaxCases), InputBytes: bytes, Note: "tuned syntax-only baseline parses each standalone file without parser.AllErrors"}
	start := time.Now()
	for _, tc := range syntaxCases {
		src, _ := os.ReadFile(filepath.Join(dir, tc.name))
		_, err := parser.ParseFile(token.NewFileSet(), tc.name, src, 0)
		got := err != nil
		if got {
			a.ReportedErrors++
		}
		if got == tc.wantError {
			a.CorrectFiles++
		} else if got {
			a.FalseSyntaxErrors++
		} else {
			a.MissedSyntaxErrors++
		}
	}
	a.Latency = time.Since(start)
	a.Correct = a.CorrectFiles == len(syntaxCases) && a.FalseSyntaxErrors == 0 && a.MissedSyntaxErrors == 0
	return a
}
func CompareGoSyntaxLocal() ComparisonResult {
	dir, err := os.MkdirTemp("", "fak-codelint-go-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	n, err := writeSyntaxCases(dir)
	if err != nil {
		panic(err)
	}
	return ComparisonResult{Workload: "classify four standalone Go files including valid, single-error, multi-error, and unresolved-identifier syntax-valid cases", Arms: []ComparisonArm{runNativeSyntax(dir, n), runFirstError(dir, n), {Name: "go test compile", Kind: "external", Note: "requires real isolated module compile"}, {Name: "gofmt", Kind: "external", Note: "requires real gofmt process"}, {Name: "go vet", Kind: "external", Note: "requires real package-context vet"}, {Name: "staticcheck", Kind: "external", Note: "requires pinned staticcheck"}, {Name: "golangci-lint", Kind: "external", Note: "requires pinned golangci-lint"}, {Name: "gopls diagnostics", Kind: "external", Note: "requires real gopls/LSP diagnostics"}}}
}
