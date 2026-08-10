package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/commitsubject"
)

func RunCommitSubjectCoverage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit-subject-coverage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	last := fs.Int("last", commitsubject.DefaultLast, "commits to scan")
	minCoverage := fs.Float64("min-coverage", -1, "advisory floor pct")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "commit-subject-coverage: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	var floor *float64
	if *minCoverage >= 0 {
		v := *minCoverage
		floor = &v
	}
	payload := commitsubject.Collect(root, *last, floor, nil)
	if *asJSON {
		data, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "commit-subject-coverage: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, commitsubject.Render(payload))
	}
	if !payload.OK {
		return 1
	}
	return 0
}
