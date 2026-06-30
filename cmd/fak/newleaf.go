package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/newleaf"
)

func cmdNewLeaf(argv []string) { os.Exit(runNewLeaf(os.Stdout, os.Stderr, argv)) }

func runNewLeaf(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("new-leaf", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tier := fs.String("tier", "", "layering tier: foundation, mechanism, composer, or integrator")
	register := fs.Bool("register", false, "add the leaf to the defconfig blank-import list")
	summary := fs.String("summary", "", "one-line package summary for doc.go")
	dryRun := fs.Bool("dry-run", false, "print the planned edits without writing files")
	workspace := fs.String("workspace", "", "workspace root (defaults to current directory)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak new-leaf: pass exactly one package name")
		return 2
	}
	if *tier == "" {
		fmt.Fprintln(stderr, "fak new-leaf: --tier is required")
		return 2
	}
	report, err := newleaf.Apply(newleaf.Options{
		Root:     *workspace,
		Name:     fs.Arg(0),
		Tier:     *tier,
		Register: *register,
		Summary:  *summary,
		DryRun:   *dryRun,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak new-leaf: %v\n", err)
		return 2
	}
	raw, err := report.JSON()
	if err != nil {
		fmt.Fprintf(stderr, "fak new-leaf: %v\n", err)
		return 1
	}
	if _, err := stdout.Write(append(raw, '\n')); err != nil {
		fmt.Fprintf(stderr, "fak new-leaf: %v\n", err)
		return 1
	}
	return 0
}
