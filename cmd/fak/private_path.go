package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/privatepath"
)

func runPrivatePath(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("private-path", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "private repository root (default: $FAK_PRIVATE_ROOT or ../fak-private)")
	create := fs.Bool("create", false, "create the private run directory with owner-only permissions")
	jsonOut := fs.Bool("json", false, "emit a machine-readable result")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak private-path [--create] [--json] [--root DIR]")
		return 2
	}
	result, err := privatepath.ResolveRun(privatepath.Options{RepoRoot: repoRoot(), Root: *root, Create: *create})
	if err != nil {
		fmt.Fprintf(stderr, "private-path: %v\n", err)
		return 1
	}
	if *jsonOut {
		if err := json.NewEncoder(stdout).Encode(result); err != nil {
			fmt.Fprintf(stderr, "private-path: encode result: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, result.Path)
	return 0
}
