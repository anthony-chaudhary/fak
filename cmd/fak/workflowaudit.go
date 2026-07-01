package main

// fak workflow-audit -- classify every git-branch / tag reference in .github/workflows
// against the branch-role contract (#1697 / #1701), so the dev->main front-door migration
// has a checkable map and a gate that reds on a new unclassified development-path ref.
//
//	fak workflow-audit              # print the classification table; exit 1 if anything is unclassified
//	fak workflow-audit --json       # emit the full workflowaudit.Report as JSON
//	fak workflow-audit --write-doc  # regenerate the committed report block in docs/ci/workflow-branch-audit.md
//	fak workflow-audit --check-doc  # CI gate: red when the committed block is stale vs the live workflows
//
// Thin I/O shell over internal/workflowaudit (the pure auditor) + internal/branchrole (the
// contract reader). The default form is itself a gate: exit 1 when a development-path
// reference names no configured role and is not an intentional allowlisted legacy arm.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/workflowaudit"
)

func cmdWorkflowAudit(argv []string) {
	os.Exit(runWorkflowAudit(os.Stdout, os.Stderr, argv))
}

func runWorkflowAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak workflow-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the full audit as JSON (workflowaudit.Report)")
	writeDoc := fs.Bool("write-doc", false, "regenerate the audit block in "+workflowaudit.DocRel+" in place")
	checkDoc := fs.Bool("check-doc", false, "CI gate: red when the committed "+workflowaudit.DocRel+" block is stale")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak workflow-audit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	roles, err := branchrole.Load(root)
	if err != nil {
		// branchrole.Load returns usable Defaults even on error; surface the reason but
		// keep auditing against the no-cutover defaults.
		fmt.Fprintf(stderr, "fak workflow-audit: branch roles: %v (using defaults)\n", err)
	}
	dir := filepath.Join(root, ".github", "workflows")
	rep, err := workflowaudit.Audit(dir, roles, workflowaudit.DefaultAllowlist())
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow-audit: %v\n", err)
		return 1
	}

	if *writeDoc {
		return workflowAuditWriteDoc(stdout, stderr, root, rep)
	}
	if *checkDoc {
		return workflowAuditCheckDoc(stdout, root, rep, *asJSON)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak workflow-audit: %v\n", err)
			return 1
		}
		if !rep.Clean() {
			return 1
		}
		return 0
	}

	// Human form: the class distribution, then the verdict.
	verdict := "ALL CLASSIFIED"
	if !rep.Clean() {
		verdict = fmt.Sprintf("%d UNCLASSIFIED", len(rep.Unclassified))
	}
	fmt.Fprintf(stdout, "%s -- %d ref(s) across %d workflow file(s)\n", verdict, len(rep.Refs), rep.Files)
	for _, c := range []workflowaudit.RefClass{
		workflowaudit.ClassDevelopment, workflowaudit.ClassReleaseFrontDoor,
		workflowaudit.ClassTag, workflowaudit.ClassLegacy, workflowaudit.ClassUnclassified,
	} {
		fmt.Fprintf(stdout, "  %-20s %d\n", c, rep.ByClass[c])
	}
	if !rep.Clean() {
		fmt.Fprintln(stdout, "unclassified development-path references (classify in dos.toml [branch_roles] or internal/workflowaudit/allow.txt):")
		for _, r := range rep.Unclassified {
			fmt.Fprintf(stdout, "  %s:%d %s %s\n", r.File, r.Line, r.Kind, r.Raw)
		}
		return 1
	}
	return 0
}

func workflowAuditWriteDoc(stdout, stderr io.Writer, root string, rep workflowaudit.Report) int {
	docPath := filepath.Join(root, filepath.FromSlash(workflowaudit.DocRel))
	raw, err := os.ReadFile(docPath)
	doc := string(raw)
	if err != nil {
		if !os.IsNotExist(err) {
			fmt.Fprintf(stderr, "fak workflow-audit --write-doc: read %s: %v\n", docPath, err)
			return 1
		}
		doc = workflowaudit.Scaffold()
		if mkErr := os.MkdirAll(filepath.Dir(docPath), 0o755); mkErr != nil {
			fmt.Fprintf(stderr, "fak workflow-audit --write-doc: mkdir: %v\n", mkErr)
			return 1
		}
	}
	next, err := workflowaudit.Splice(doc, rep)
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow-audit --write-doc: %v\n", err)
		return 1
	}
	if next == doc {
		fmt.Fprintf(stdout, "%s audit block already fresh; no change\n", workflowaudit.DocRel)
		return 0
	}
	if err := os.WriteFile(docPath, []byte(next), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak workflow-audit --write-doc: write %s: %v\n", docPath, err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s audit block\n", workflowaudit.DocRel)
	return 0
}

func workflowAuditCheckDoc(stdout io.Writer, root string, rep workflowaudit.Report, asJSON bool) int {
	docPath := filepath.Join(root, filepath.FromSlash(workflowaudit.DocRel))
	raw, err := os.ReadFile(docPath)
	emit := func(fresh bool, msg string) int {
		if asJSON {
			fmt.Fprintf(stdout, "{\"doc\":%q,\"fresh\":%t,\"message\":%q}\n", workflowaudit.DocRel, fresh, msg)
		} else {
			fmt.Fprintln(stdout, msg)
		}
		if fresh {
			return 0
		}
		return 1
	}
	if err != nil {
		return emit(false, fmt.Sprintf("STALE  %s: not found; run `fak workflow-audit --write-doc`", workflowaudit.DocRel))
	}
	doc := string(raw)
	if _, ok := workflowaudit.Extract(doc); !ok {
		return emit(false, fmt.Sprintf("STALE  %s: audit markers not found; run `fak workflow-audit --write-doc`", workflowaudit.DocRel))
	}
	if !workflowaudit.Fresh(doc, rep) {
		return emit(false, fmt.Sprintf("STALE  %s: audit block drifted from the live workflows; run `fak workflow-audit --write-doc`", workflowaudit.DocRel))
	}
	return emit(true, fmt.Sprintf("OK  %s audit block is fresh", workflowaudit.DocRel))
}
