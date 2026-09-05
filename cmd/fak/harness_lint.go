package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnesslint"
)

func runHarnessLint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness lint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "path to harness product lock file")
	manifestPath := fs.String("manifest", "", "path to harness manifest or lock file")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON report")
	allowSinglePlatform := fs.Bool("allow-single-platform", false, "allow single-platform target without warning")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	target := *lockPath
	if target == "" {
		target = *manifestPath
	}
	if target == "" {
		fmt.Fprintln(stderr, "fak harness lint: --lock or --manifest is required")
		return 2
	}

	raw, err := os.ReadFile(target)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness lint: read file: %v\n", err)
		return 1
	}

	report := harnesslint.LintLock(raw, harnesslint.WithAllowSinglePlatform(*allowSinglePlatform), harnesslint.WithLockPath(target))

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak harness lint: encode JSON: %v\n", err)
			return 1
		}
	} else {
		if report.Valid {
			fmt.Fprintf(stdout, "fak harness lint: OK (%s)\n", target)
			for _, d := range report.Diagnostics {
				if d.Severity == harnesslint.SeverityWarn {
					fmt.Fprintf(stdout, "  [WARNING] %s: %s (field=%s)\n", d.Rule, d.Message, d.Field)
				}
			}
		} else {
			fmt.Fprintf(stderr, "fak harness lint: FAILED (%s)\n", target)
			for _, d := range report.Diagnostics {
				if d.Severity == harnesslint.SeverityError {
					fmt.Fprintf(stderr, "  [ERROR] %s: %s (field=%s)\n", d.Rule, d.Message, d.Field)
				} else {
					fmt.Fprintf(stderr, "  [WARNING] %s: %s (field=%s)\n", d.Rule, d.Message, d.Field)
				}
			}
		}
	}

	if !report.Valid {
		return 1
	}
	return 0
}
