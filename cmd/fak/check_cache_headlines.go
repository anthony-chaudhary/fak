package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/cacheheadlines"
)

func cmdCheckCacheHeadlines(argv []string) {
	os.Exit(runCheckCacheHeadlines(os.Stdout, os.Stderr, argv))
}

func runCheckCacheHeadlines(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("check-cache-headlines", flag.ContinueOnError)
	fs.SetOutput(stderr)

	auditTree := fs.Bool("audit-tree", false, "scan the whole tracked tree (CI / hygiene)")
	auditStaged := fs.Bool("audit-staged", false, "scan staged additions (pre-commit hook)")
	check := fs.Bool("check", false, "gate mode: run cache headline check and exit non-zero on violation")
	root := fs.String("root", ".", "repo root")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *auditTree && *auditStaged {
		fmt.Fprintln(stderr, "cache-headlines: cannot specify both --audit-tree and --audit-staged")
		return 2
	}

	if !*auditTree && !*auditStaged && !*check {
		fmt.Fprintln(stderr, "usage: fak check-cache-headlines (--audit-tree | --audit-staged | --check) [--root DIR] [FILES...]")
		return 2
	}

	useStaged := *auditStaged

	rootPath, err := filepath.Abs(*root)
	if err != nil {
		fmt.Fprintf(stderr, "cache-headlines: could not resolve root: %v\n", err)
		return 2
	}

	var findings []cacheheadlines.Finding
	if len(fs.Args()) > 0 {
		for _, arg := range fs.Args() {
			fullPath := filepath.Join(rootPath, filepath.FromSlash(arg))
			data, err := os.ReadFile(fullPath)
			if err != nil {
				fmt.Fprintf(stderr, "cache-headlines: could not read file %s: %v\n", arg, err)
				return 2
			}
			fileFindings := cacheheadlines.ScanText(string(data), arg)
			findings = append(findings, fileFindings...)
		}
	} else if useStaged {
		findings, err = cacheheadlines.AuditStaged(rootPath)
	} else {
		findings, err = cacheheadlines.AuditTree(rootPath)
	}

	if err != nil {
		fmt.Fprintf(stderr, "cache-headlines: could not run git: %v\n", err)
		return 2
	}

	if len(findings) == 0 {
		scope := "tracked tree"
		if useStaged {
			scope = "staged"
		}
		fmt.Fprintf(stdout, "cache-headlines: clean (%s); every cache headline names its plane + provenance.\n", scope)
		return 0
	}

	if useStaged && os.Getenv("ALLOW_CACHE_HEADLINE_DRIFT") == "1" {
		fmt.Fprintf(stderr, "cache-headlines: ALLOW_CACHE_HEADLINE_DRIFT=1 set — overriding %d hit(s) once.\n", len(findings))
		return 0
	}

	fmt.Fprintf(stderr, "CACHE_HEADLINE: %d cache headline(s) without a plane/provenance label:\n", len(findings))
	for _, h := range findings {
		fmt.Fprintf(stderr, "  %s:%d: %s\n", h.File, h.Line, h.Text)
		fmt.Fprintf(stderr, "    fix: %s\n", h.Fix)
	}
	fmt.Fprintln(stderr, "  override once (staged): ALLOW_CACHE_HEADLINE_DRIFT=1 <git cmd>.")
	return 1
}
