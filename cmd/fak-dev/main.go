package main

import (
	"fmt"
	"io"
	"os"
	"runtime/debug"

	"github.com/anthony-chaudhary/fak/internal/devcmd"
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
	case "index", "devindex":
		return devcmd.RunIndex(stdout, stderr, argv[1:])
	case "wiki":
		return devcmd.RunWiki(stdout, stderr, argv[1:])
	case "orient":
		return devcmd.RunOrient(stdout, stderr, argv[1:])
	case "backend":
		return devcmd.RunBackend(stdout, stderr, argv[1:])
	case "catchup":
		return devcmd.RunCatchUpScore(stdout, stderr, argv[1:])
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
	fmt.Fprintln(w, "  index policy [--json] [--root PATH]     enforce dev ownership at the repository boundary")
	fmt.Fprintln(w, "  wiki <structure|verify|fresh|score>    audit the repository documentation wiki")
	fmt.Fprintln(w, "  orient [env] --paths GLOB             show repository conventions and live ownership")
	fmt.Fprintln(w, "  backend scaffold NAME --lane LANE     generate a repository compute backend")
	fmt.Fprintln(w, "  catchup [flags]                       measure repository development catch-up debt")
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
