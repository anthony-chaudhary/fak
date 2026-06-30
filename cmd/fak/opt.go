package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/opttarget"
)

func cmdOpt(argv []string) { os.Exit(runOpt(os.Stdout, os.Stderr, argv)) }

func runOpt(stdout, stderr io.Writer, argv []string) int {
	cmd := "discover"
	if len(argv) > 0 {
		cmd = argv[0]
		argv = argv[1:]
	}
	switch cmd {
	case "discover":
		return runOptDiscover(stdout, stderr, argv)
	case "-h", "--help", "help":
		optUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak opt: unknown command %q\n", cmd)
		optUsage(stderr)
		return 2
	}
}

func runOptDiscover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak opt discover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the inventory as JSON")
	check := fs.String("check", "", "coverage ratchet: comma-separated target names that must still be present")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak opt discover: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	targets, err := opttarget.DiscoverDir(root)
	if err != nil {
		// A malformed annotation is a WARNING: DiscoverDir still returns the
		// well-formed targets, so we report the defect and continue.
		fmt.Fprintf(stderr, "fak opt discover: %v\n", err)
	}

	if *check != "" {
		var required []string
		for _, name := range strings.Split(*check, ",") {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			required = append(required, name)
		}
		if cerr := opttarget.Check(targets, required); cerr != nil {
			fmt.Fprintln(stdout, cerr)
			return 1
		}
		fmt.Fprintf(stdout, "fak opt discover --check: OK (%d targets, all %d required present)\n",
			len(targets), len(required))
		return 0
	}

	if *asJSON {
		b, merr := opttarget.MarshalInventory(targets)
		if merr != nil {
			fmt.Fprintf(stderr, "fak opt discover: marshal inventory: %v\n", merr)
			return 1
		}
		stdout.Write(b)
		return 0
	}

	fmt.Fprintf(stdout, "fak opt discover: %d target(s)\n", len(targets))
	for _, t := range targets {
		fmt.Fprintf(stdout, "  - %s  metric=%s measurer=%s site=%s:%s\n",
			t.Name, t.Metric, t.Measurer, t.Site.Path, t.Site.Const)
	}
	return 0
}

func optUsage(w io.Writer) {
	fmt.Fprint(w, `fak opt - inventory the repo's annotated optimization targets

  fak opt discover [--workspace DIR] [--json] [--check name,...]

  discover walks the workspace for // fak:opttarget annotated consts and reports
  the OptTargets they declare. --json emits the stable JSON inventory; --check
  NAMES,... is the coverage ratchet (exit 1 if any named target is gone).
`)
}
