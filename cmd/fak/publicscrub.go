package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

func cmdPublicScrub(argv []string) { os.Exit(runPublicScrub(os.Stdout, os.Stderr, argv)) }

func runPublicScrub(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		publicScrubUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "audit-staged":
		return runPublicScrubStaged(stdout, stderr, argv[1:])
	case "audit-range":
		return runPublicScrubRange(stdout, stderr, argv[1:])
	case "audit-tree":
		return runPublicScrubTree(stdout, stderr, argv[1:])
	case "audit-message":
		return runPublicScrubMessage(stdout, stderr, argv[1:])
	case "help", "-h", "--help":
		publicScrubUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak public-scrub: unknown subcommand %q\n", argv[0])
		publicScrubUsage(stderr)
		return 2
	}
}

func runPublicScrubStaged(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("public-scrub audit-staged", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2
	}
	findings, err := hooks.ScanStagedPublicLeak(r)
	if err != nil {
		return 2
	}
	return emitPublicScrubFindings(stdout, stderr, findings, "staged content", *asJSON, "")
}

func runPublicScrubRange(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("public-scrub audit-range", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	rest := fs.Args()
	if len(rest) != 1 {
		fmt.Fprintln(stderr, "fak public-scrub audit-range: pass BASE..HEAD")
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2
	}
	findings, err := hooks.ScanRangePublicLeak(r, rest[0])
	if err != nil {
		return 2
	}
	clean := fmt.Sprintf("leak-scan: %s clean (no AUDIT_NEEDLES in added lines)", rest[0])
	return emitPublicScrubFindings(stdout, stderr, findings, "range "+rest[0], *asJSON, clean)
}

func runPublicScrubMessage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("public-scrub audit-message", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	rest := fs.Args()
	if len(rest) != 1 {
		fmt.Fprintln(stderr, "fak public-scrub audit-message: message file required")
		return 2
	}
	raw, err := os.ReadFile(rest[0])
	if err != nil {
		return 2
	}
	findings := hooks.ScanMessageNeedles(string(raw), resolveRoot(*root))
	return emitPublicScrubFindings(stdout, stderr, findings, "the commit message", *asJSON, "")
}

func runPublicScrubTree(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("public-scrub audit-tree", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit scan report as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2
	}
	report, err := hooks.AuditPublicLeakTree(r)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if encErr := enc.Encode(report); encErr != nil {
			fmt.Fprintf(stderr, "fak public-scrub audit-tree: %v\n", encErr)
			return 2
		}
	} else {
		fmt.Fprint(stdout, hooks.PublicLeakTreeText(report))
	}
	if err != nil {
		return 2
	}
	if report.OK {
		return 0
	}
	return 1
}

func emitPublicScrubFindings(stdout, stderr io.Writer, findings []hooks.Finding, where string, asJSON bool, cleanLine string) int {
	if asJSON {
		if findings == nil {
			findings = []hooks.Finding{}
		}
		if err := writeIndentedJSON(stdout, map[string]any{"findings": findings, "count": len(findings), "ok": len(findings) == 0}); err != nil {
			fmt.Fprintf(stderr, "fak public-scrub: %v\n", err)
			return 2
		}
	} else if len(findings) > 0 {
		for _, line := range hooks.PublicLeakFindingLines(findings, where) {
			fmt.Fprintln(stdout, line)
		}
	} else if cleanLine != "" {
		fmt.Fprintln(stdout, cleanLine)
	}
	if len(findings) > 0 {
		return 1
	}
	return 0
}

func publicScrubUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak public-scrub audit-staged [--root DIR] [--json]
  fak public-scrub audit-range BASE..HEAD [--root DIR] [--json]
  fak public-scrub audit-tree [--root DIR] [--json]
  fak public-scrub audit-message MSGFILE [--root DIR] [--json]
`)
}
