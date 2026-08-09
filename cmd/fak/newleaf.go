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
	tier := fs.String("tier", "", "layering tier: primitive, foundation-composite, mechanism, composer, or integrator")
	suggestTier := fs.String("suggest-tier", "", "read-only: print the minimum-correct tier for an existing internal leaf derived from its imports, then exit")
	register := fs.Bool("register", false, "add the leaf to the defconfig blank-import list")
	summary := fs.String("summary", "", "one-line package summary for doc.go")
	dryRun := fs.Bool("dry-run", false, "print the planned edits without writing files")
	workspace := fs.String("workspace", "", "workspace root (defaults to current directory)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *suggestTier != "" {
		// Read-only suggestion mode: the leaf is named by the flag, so no positional
		// package argument. An optional --tier adds the over/under-declaration advisory.
		if fs.NArg() != 0 {
			fmt.Fprintln(stderr, "fak new-leaf: --suggest-tier names the leaf itself; pass no package argument")
			return 2
		}
		s, err := newleaf.Suggest(*workspace, *suggestTier, *tier)
		if err != nil {
			fmt.Fprintf(stderr, "fak new-leaf: %v\n", err)
			return 2
		}
		return writeNewLeafJSON(stdout, stderr, s)
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
	return writeNewLeafJSON(stdout, stderr, report)
}

// writeNewLeafJSON marshals one `fak new-leaf` result and writes it as a single line on
// stdout. Both the suggestion and the apply path end this way, and both report a marshal
// or write failure as exit 1 with the reason on stderr.
func writeNewLeafJSON(stdout, stderr io.Writer, doc interface{ JSON() ([]byte, error) }) int {
	raw, err := doc.JSON()
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
