package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/docfreshrsi"
)

// docFreshnessPayload is the machine-readable output for `fak doc-freshness --json`.
type docFreshnessPayload struct {
	Schema          string                           `json:"schema"`
	Workspace       string                           `json:"workspace"`
	TargetVersion   string                           `json:"target_version"`
	Fresh           bool                             `json:"fresh"`
	Mode            string                           `json:"mode"`
	Generated       []docfreshrsi.GeneratedDocStatus `json:"generated,omitempty"`
	CorpusAudit     *docfreshrsi.CorpusAuditReport   `json:"corpus_audit,omitempty"`
	CorpusRefreshed *docfreshrsi.CorpusRefreshReport `json:"corpus_refreshed,omitempty"`
	Message         string                           `json:"message"`
}

func cmdDocFreshness(args []string) {
	code := runDocFreshness(os.Stdout, os.Stderr, args)
	if code != 0 {
		os.Exit(code)
	}
}

func runDocFreshness(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak doc-freshness", flag.ContinueOnError)
	fs.SetOutput(stderr)

	check := fs.Bool("check", false, "audit doc freshness without mutating; exit 0 if fresh, 1 if stale")
	refresh := fs.Bool("refresh", false, "regenerate/refresh stale generated docs and apply safe mechanical doc fixes")
	scopeGen := fs.Bool("generated", false, "scope operation to registered generated docs/marker blocks only")
	scopeCorp := fs.Bool("corpus", false, "scope operation to markdown prose corpus only")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON payload")
	dryRun := fs.Bool("dry-run", false, "simulate refresh actions without mutating files on disk")
	targetVersion := fs.String("target-version", "", "override target release version pin (default: read from VERSION)")
	workspace := fs.String("workspace", "", "workspace root directory (default: repo root)")
	globsFlag := fs.String("globs", "", "comma-separated glob patterns to limit corpus scan (e.g. docs/**/*.md)")

	if !parseFlags(fs, args) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doc-freshness: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	// Default mode is check if refresh not passed
	isRefresh := *refresh
	isCheck := *check || !isRefresh

	// If neither scope is set, run both
	runGenerated := *scopeGen || (!*scopeGen && !*scopeCorp)
	runCorpus := *scopeCorp || (!*scopeGen && !*scopeCorp)

	targetVer := *targetVersion
	if targetVer == "" {
		targetVer = docfreshrsi.DefaultTargetVersion(root)
	}

	payload := docFreshnessPayload{
		Schema:        "fak.doc-freshness.v1",
		Workspace:     root,
		TargetVersion: targetVer,
		Mode:          "check",
		Fresh:         true,
	}
	if isRefresh {
		payload.Mode = "refresh"
	}

	var globs []string
	if *globsFlag != "" {
		for _, g := range strings.Split(*globsFlag, ",") {
			g = strings.TrimSpace(g)
			if g != "" {
				globs = append(globs, g)
			}
		}
	}

	var errList []string
	staleGenCount := 0

	// 1. Generated docs handling
	if runGenerated {
		if isRefresh {
			statuses, err := docfreshrsi.RefreshGeneratedDocs(root)
			if err != nil {
				errList = append(errList, fmt.Sprintf("refresh generated: %v", err))
			}
			payload.Generated = statuses
			for _, s := range statuses {
				if !s.Fresh {
					staleGenCount++
				}
			}
		} else {
			statuses, allFresh := docfreshrsi.AuditGeneratedDocs(root)
			payload.Generated = statuses
			if !allFresh {
				payload.Fresh = false
				for _, s := range statuses {
					if !s.Fresh {
						staleGenCount++
					}
				}
			}
		}
	}

	// 2. Prose corpus handling
	if runCorpus {
		corpus, err := docfreshrsi.ScanCorpus(root, globs)
		if err != nil {
			errList = append(errList, fmt.Sprintf("scan corpus: %v", err))
		}

		tgt := docfreshrsi.Target{Version: targetVer}
		auditOpts := docfreshrsi.CorpusAuditOptions{
			Root:          root,
			TargetVersion: targetVer,
			CheckLinks:    true,
			CheckClaims:   true,
			CheckModver:   true,
		}

		if isRefresh {
			refreshRep, err := docfreshrsi.RefreshCorpus(root, corpus, tgt, *dryRun)
			if err != nil {
				errList = append(errList, fmt.Sprintf("refresh corpus: %v", err))
			}
			payload.CorpusRefreshed = &refreshRep

			// Re-scan refreshed corpus to report post-refresh state
			reCorpus, _ := docfreshrsi.ScanCorpus(root, globs)
			postAudit := docfreshrsi.AuditCorpus(root, reCorpus, tgt, auditOpts)
			payload.CorpusAudit = &postAudit
			if !postAudit.Fresh {
				payload.Fresh = false
			}
		} else {
			audit := docfreshrsi.AuditCorpus(root, corpus, tgt, auditOpts)
			payload.CorpusAudit = &audit
			if !audit.Fresh {
				payload.Fresh = false
			}
		}
	}

	if staleGenCount > 0 {
		payload.Fresh = false
	}

	if len(errList) > 0 {
		payload.Message = strings.Join(errList, "; ")
	} else if payload.Fresh {
		payload.Message = "all documentation is fresh"
	} else {
		payload.Message = fmt.Sprintf("stale documentation detected (generated drifted: %d); run `fak doc-freshness --refresh`", staleGenCount)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(payload)
		if isCheck && !payload.Fresh {
			return 1
		}
		return 0
	}

	// Human readable output
	fmt.Fprintf(stdout, "fak doc-freshness (%s, target: %s)\n", payload.Mode, targetVer)
	fmt.Fprintf(stdout, "Workspace: %s\n\n", root)

	if runGenerated && len(payload.Generated) > 0 {
		fmt.Fprintln(stdout, "=== Generated Documents ===")
		for _, s := range payload.Generated {
			statusStr := "OK   "
			if !s.Fresh {
				statusStr = "STALE"
			}
			if s.Refreshed {
				statusStr = "FIXED"
			}
			detail := s.Detail
			if s.Error != "" {
				detail = "error: " + s.Error
			}
			fmt.Fprintf(stdout, "  [%s] %-25s %s (%s)\n", statusStr, s.ID, s.Path, detail)
		}
		fmt.Fprintln(stdout)
	}

	if runCorpus && payload.CorpusAudit != nil {
		ca := payload.CorpusAudit
		fmt.Fprintln(stdout, "=== Prose Corpus Audit ===")
		fmt.Fprintf(stdout, "  Total docs scanned : %d\n", ca.TotalDocs)
		fmt.Fprintf(stdout, "  Mechanical defects : %d\n", ca.TotalDefects)
		if len(ca.StalePins) > 0 {
			fmt.Fprintf(stdout, "  Stale version pins : %d\n", len(ca.StalePins))
			for i, p := range ca.StalePins {
				if i < 5 {
					fmt.Fprintf(stdout, "    - %s:%d: %s -> want %s\n", p.DocPath, p.Line, p.Found, p.Target)
				}
			}
			if len(ca.StalePins) > 5 {
				fmt.Fprintf(stdout, "    ... and %d more\n", len(ca.StalePins)-5)
			}
		}
		if len(ca.DanglingLinks) > 0 {
			fmt.Fprintf(stdout, "  Dangling links     : %d\n", len(ca.DanglingLinks))
			for i, l := range ca.DanglingLinks {
				if i < 5 {
					fmt.Fprintf(stdout, "    - %s\n", l)
				}
			}
			if len(ca.DanglingLinks) > 5 {
				fmt.Fprintf(stdout, "    ... and %d more\n", len(ca.DanglingLinks)-5)
			}
		}
		if len(ca.VersionClaims) > 0 {
			fmt.Fprintf(stdout, "  Unpointed claims   : %d\n", len(ca.VersionClaims))
			for i, c := range ca.VersionClaims {
				if i < 5 {
					fmt.Fprintf(stdout, "    - %s:%d [%s]: %s\n", c.Path, c.Line, c.Signature, c.Text)
				}
			}
			if len(ca.VersionClaims) > 5 {
				fmt.Fprintf(stdout, "    ... and %d more\n", len(ca.VersionClaims)-5)
			}
		}
		if len(ca.ModverFindings) > 0 {
			fmt.Fprintf(stdout, "  Modver doc revs    : %d mapped\n", len(ca.ModverFindings))
		}
		fmt.Fprintln(stdout)
	}

	if payload.CorpusRefreshed != nil {
		cr := payload.CorpusRefreshed
		fmt.Fprintln(stdout, "=== Corpus Refresh Actions ===")
		fmt.Fprintf(stdout, "  Updated documents  : %d\n", len(cr.UpdatedDocs))
		for _, u := range cr.UpdatedDocs {
			fmt.Fprintf(stdout, "    - %s\n", u)
		}
		fmt.Fprintf(stdout, "  Debt paydown       : %d -> %d\n\n", cr.DebtBefore, cr.DebtAfter)
	}

	if payload.Fresh {
		fmt.Fprintln(stdout, "VERDICT: PASS (all documentation is fresh)")
		return 0
	}

	if isRefresh {
		fmt.Fprintln(stdout, "VERDICT: REFRESHED (some manual reviews or unpointed claims may remain)")
		return 0
	}

	fmt.Fprintln(stdout, "VERDICT: STALE (run `fak doc-freshness --refresh` to auto-repair)")
	return 1
}
