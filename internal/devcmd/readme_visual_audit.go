package devcmd

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/readmevisualaudit"
)

func RunReadmeVisualAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("readme-visual-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "readme-visual-audit: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
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
