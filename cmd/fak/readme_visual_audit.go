package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/readmevisualaudit"
)

func cmdReadmeVisualAudit(argv []string) {
	os.Exit(runReadmeVisualAudit(os.Stdout, os.Stderr, argv))
}

func runReadmeVisualAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("readme-visual-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := *workspace
	if root == "" {
		root = resolveRoot("")
		if root == "" {
			root = "."
		}
	}
	report := readmevisualaudit.Collect(root)
	if *asJSON {
		data, err := readmevisualaudit.MarshalJSON(report)
		if err != nil {
			fmt.Fprintf(stderr, "readme-visual-audit: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, readmevisualaudit.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
}
