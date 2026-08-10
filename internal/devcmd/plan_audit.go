package devcmd

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/planaudit"
)

func RunPlanAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("plan-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	pattern := fs.String("glob", planaudit.DefaultGlob, "plan-doc glob (brace ok)")
	asJSON := fs.Bool("json", false, "emit JSON")
	markdown := fs.Bool("markdown", false, "emit markdown")
	out := fs.String("out", "", "write rendered output to this path")
	check := fs.Bool("check", false, "exit 1 on drift")
	asOf := fs.String("as-of", "", "date stamp for report (default: today UTC)")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "plan-audit: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	report, err := planaudit.Collect(root, *pattern)
	if err != nil {
		fmt.Fprintf(stderr, "ERROR: audit failed: %v\n", err)
		return 2
	}
	if *check {
		if len(report.Drift) > 0 {
			return 1
		}
		return 0
	}
	if report.Counts["total_plans"] == 0 {
		if *asJSON {
			data, _ := planaudit.MarshalJSON(report)
			fmt.Fprintln(stdout, string(data))
		} else {
			fmt.Fprintln(stdout, "no plan docs found - nothing to audit")
		}
		return 0
	}
	stamp := *asOf
	if stamp == "" {
		stamp = time.Now().UTC().Format("2006-01-02")
	}
	var rendered string
	if *markdown {
		rendered = planaudit.RenderMarkdown(report, stamp)
	} else {
		data, err := planaudit.MarshalJSON(report)
		if err != nil {
			fmt.Fprintf(stderr, "plan-audit: %v\n", err)
			return 2
		}
		rendered = string(data)
	}
	if *out != "" {
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			fmt.Fprintf(stderr, "plan-audit: %v\n", err)
			return 2
		}
		if err := os.WriteFile(*out, []byte(rendered+"\n"), 0o644); err != nil {
			fmt.Fprintf(stderr, "plan-audit: %v\n", err)
			return 2
		}
		fmt.Fprintf(stdout, "wrote %s\n", *out)
	} else {
		fmt.Fprintln(stdout, rendered)
	}
	_ = asJSON // accepted for compatibility; JSON is the default unless --markdown is set.
	return 0
}
