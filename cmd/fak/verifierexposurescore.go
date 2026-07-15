package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/verifierexposure"
)

func cmdVerifierExposureScore(argv []string) {
	os.Exit(runVerifierExposureScore(os.Stdout, os.Stderr, argv))
}

func runVerifierExposureScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score verifier-exposure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the committed-baseline markdown body")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score verifier-exposure: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	report := verifierexposure.Gather(root)
	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak score verifier-exposure: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, verifierexposure.Markdown(report))
	default:
		fmt.Fprintln(stdout, verifierexposure.Render(report))
	}
	if len(report.InventoryErrors) != 0 {
		return 1
	}
	return 0
}
