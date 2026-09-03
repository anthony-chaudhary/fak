package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

var codeQualityCheckerPaths = []string{
	"tools/code_quality_scorecard.py",
	"tools/scorecard_since.py",
}

// codeQualityRunner is injectable so the drift witness can mutate a checker
// between pin-time and grade-time without a shell or timing race.
type codeQualityRunner func(context.Context, string, []string) ([]byte, []byte, int, error)

func cmdCodeQualityScore(argv []string) {
	os.Exit(runCodeQualityScore(os.Stdout, os.Stderr, argv, execCodeQuality))
}

func runCodeQualityScore(stdout, stderr io.Writer, argv []string, run codeQualityRunner) int {
	fs := flag.NewFlagSet("fak score code-quality", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	python := fs.String("python", "python", "Python executable for the grandfathered checker")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the scorecard markdown body")
	stamp := fs.String("stamp", "", "date stamp for the markdown header")
	noToolchain := fs.Bool("no-toolchain", false, "skip go build/vet + gofmt")
	noDOS := fs.Bool("no-dos", false, "skip the dos review ship-integrity probe")
	revRange := fs.String("range", "", "git range for ship-integrity")
	since := fs.String("since", "", "skip-gate reference")
	deterministic := fs.Bool("deterministic", false, "tree-deterministic mode (ignore sliding commit history)")
	kpi := fs.String("kpi", "", "filter code debt by KPI")
	category := fs.String("category", "", "filter code debt by structural category")
	pathFilter := fs.String("path", "", "filter code debt by file or package path")
	search := fs.String("search", "", "filter code debt by substring in defect text")
	limit := fs.Int("limit", 0, "limit max defects returned")
	countOnly := fs.Bool("count", false, "print matching defect count only")
	summaryOnly := fs.Bool("summary", false, "print structural category & KPI debt summary")
	native := fs.Bool("native", false, "run native Go code-debt engine instead of Python checker")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *native {
		return runCodeDebt(stdout, stderr, argv)
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	baseline, err := safecommit.PinCheckers(root, codeQualityCheckerPaths)
	if err != nil {
		fmt.Fprintf(stderr, "fak score code-quality: pin checker bytes: %v\n", err)
		return 1
	}
	checkerArgs := []string{filepath.FromSlash("tools/code_quality_scorecard.py"), "--workspace", root}
	if *asJSON {
		checkerArgs = append(checkerArgs, "--json")
	}
	if *asMarkdown {
		checkerArgs = append(checkerArgs, "--markdown")
	}
	if *stamp != "" {
		checkerArgs = append(checkerArgs, "--stamp", *stamp)
	}
	if *noToolchain {
		checkerArgs = append(checkerArgs, "--no-toolchain")
	}
	if *noDOS {
		checkerArgs = append(checkerArgs, "--no-dos")
	}
	if *revRange != "" {
		checkerArgs = append(checkerArgs, "--range", *revRange)
	}
	if *since != "" {
		checkerArgs = append(checkerArgs, "--since", *since)
	}
	if *deterministic {
		checkerArgs = append(checkerArgs, "--deterministic")
	}
	if *kpi != "" {
		checkerArgs = append(checkerArgs, "--kpi", *kpi)
	}
	if *category != "" {
		checkerArgs = append(checkerArgs, "--category", *category)
	}
	if *pathFilter != "" {
		checkerArgs = append(checkerArgs, "--path", *pathFilter)
	}
	if *search != "" {
		checkerArgs = append(checkerArgs, "--search", *search)
	}
	if *limit > 0 {
		checkerArgs = append(checkerArgs, "--limit", fmt.Sprintf("%d", *limit))
	}
	if *countOnly {
		checkerArgs = append(checkerArgs, "--count")
	}
	if *summaryOnly {
		checkerArgs = append(checkerArgs, "--summary")
	}
	checkerArgs = append(checkerArgs, fs.Args()...)
	out, errOut, code, runErr := run(context.Background(), root, append([]string{*python}, checkerArgs...))
	if runErr != nil {
		fmt.Fprintf(stderr, "fak score code-quality: run checker: %v\n", runErr)
		return 1
	}
	if reason, refused := safecommit.GuardCheckerPin(root, baseline); refused {
		fmt.Fprintf(stderr, "fak score code-quality: %s: checker bytes drifted during grade\n", reason)
		return 1
	}
	// Output is held in memory until after the pin verifies: no untrusted grade
	// reaches a caller before the watcher has been watched.
	_, _ = stdout.Write(out)
	_, _ = stderr.Write(errOut)
	return code
}

func execCodeQuality(ctx context.Context, root string, argv []string) ([]byte, []byte, int, error) {
	if len(argv) == 0 {
		return nil, nil, 2, fmt.Errorf("empty command")
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = root
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err == nil {
		return stdout.Bytes(), stderr.Bytes(), 0, nil
	}
	if exit, ok := err.(*exec.ExitError); ok {
		return stdout.Bytes(), stderr.Bytes(), exit.ExitCode(), nil
	}
	return stdout.Bytes(), stderr.Bytes(), 1, err
}
