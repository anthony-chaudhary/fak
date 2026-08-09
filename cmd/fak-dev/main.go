package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime/debug"

	"github.com/anthony-chaudhary/fak/internal/devcmd"
	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		writeHelp(stdout)
		return 0
	}
	switch argv[0] {
	case "version", "--version":
		return runVersion(stdout, argv[1:])
	case "index":
		return runIndex(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak-dev: unknown command %q\n", argv[0])
		fmt.Fprintln(stderr, "run 'fak-dev help' for repository-development commands")
		return 2
	}
}

func writeHelp(w io.Writer) {
	fmt.Fprintln(w, "fak-dev — repository-development tooling for fak maintainers")
	fmt.Fprintln(w, "usage: fak-dev <command> [args...]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "commands:")
	fmt.Fprintln(w, "  index ownership [--json] [--root PATH]  audit runtime/dev command ownership and dependency leaks")
	fmt.Fprintln(w, "  version                               print fak-dev build identity")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "The serving/guard product surface is the separately buildable 'fak' artifact.")
}

func runVersion(w io.Writer, argv []string) int {
	if len(argv) != 0 {
		return 2
	}
	version := "(devel)"
	revision := "unknown"
	if info, ok := debug.ReadBuildInfo(); ok {
		if info.Main.Version != "" {
			version = info.Main.Version
		}
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" && setting.Value != "" {
				revision = setting.Value
			}
		}
	}
	fmt.Fprintf(w, "fak-dev %s (%s)\n", version, revision)
	return 0
}

func runIndex(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "ownership" {
		fmt.Fprintln(stderr, "usage: fak-dev index ownership [--json] [--root PATH]")
		return 2
	}
	root := ""
	asJSON := false
	for i := 1; i < len(argv); i++ {
		switch argv[i] {
		case "--json":
			asJSON = true
		case "--root":
			i++
			if i >= len(argv) {
				fmt.Fprintln(stderr, "fak-dev index ownership: --root requires a path")
				return 2
			}
			root = argv[i]
		default:
			fmt.Fprintf(stderr, "fak-dev index ownership: unknown argument %q\n", argv[i])
			return 2
		}
	}
	if root == "" {
		root = devindex.FindRoot(".")
	}
	return devcmd.RunOwnership(stdout, stderr, root, asJSON)
}

var _ = json.Valid // keep encoding/json available to source scanners validating JSON surfaces
