package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/archcheck"
)

func cmdArch(argv []string) {
	os.Exit(runArch(os.Stdout, os.Stderr, argv))
}

func runArch(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		archUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "check":
		return runArchCheck(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		archUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak arch: unknown subcommand %q (want check)\n", argv[0])
		archUsage(stderr)
		return 2
	}
}

func archUsage(w io.Writer) {
	fmt.Fprint(w, `fak arch — shift-left architecture import DAG and tier validation

Usage:
  fak arch check [--package <pkg>] [--mine] [--json]

Examples:
  fak arch check --package internal/agentquery
  fak arch check --mine
  fak arch check --json
`)
}

func runArchCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arch check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	pkgFlag := fs.String("package", "", "check a specific repo-relative package, e.g. internal/compute")
	mineFlag := fs.Bool("mine", false, "check only packages touched by uncommitted or staged changes")
	dirFlag := fs.String("dir", "", "repo root directory (default: discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as structured JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	root := resolveRoot(*dirFlag)

	var res *archcheck.CheckResult
	var err error

	switch {
	case *pkgFlag != "":
		res, err = archcheck.CheckPackage(root, *pkgFlag)
	case *mineFlag:
		res, err = archcheck.CheckMine(root)
	default:
		res, err = archcheck.CheckAll(root)
	}

	if err != nil {
		fmt.Fprintf(stderr, "fak arch check: %v\n", err)
		return 1
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak arch check: encode json: %v\n", err)
			return 1
		}
	} else {
		if res.OK {
			fmt.Fprintf(stdout, "PASS arch check: %d package(s) verified clean in %dms\n", res.CheckedPackages, res.ElapsedMS)
		} else {
			fmt.Fprintf(stderr, "FAIL arch check: %d violation(s) across %d package(s) (%dms):\n", len(res.Violations), res.CheckedPackages, res.ElapsedMS)
			for _, v := range res.Violations {
				fmt.Fprintf(stderr, "  [%s] %s:%d: %s\n", v.Rule, v.File, v.Line, v.Detail)
			}
		}
	}

	if !res.OK {
		return 1
	}
	return 0
}
