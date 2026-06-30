package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/codexmemory"
)

// cmdCodexMemory wires `fak codex-memory doctor` — a READ-ONLY diagnostic over
// an OpenAI Codex home that reports the operator-visible memory posture of a
// guarded Codex session (what is enabled, what can be injected later, what
// generated state lives on disk). It never writes Codex state and never prints
// raw memory contents.
func cmdCodexMemory(argv []string) { os.Exit(runCodexMemory(os.Stdout, os.Stderr, argv)) }

func runCodexMemory(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak codex-memory doctor [--codex-home DIR] [--repo DIR] [--json]")
		return 2
	}
	switch argv[0] {
	case "doctor":
		return runCodexMemoryDoctor(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stderr, "usage: fak codex-memory doctor [--codex-home DIR] [--repo DIR] [--json]")
		return 0
	default:
		fmt.Fprintf(stderr, "fak codex-memory: unknown subcommand %q\n", argv[0])
		return 2
	}
}

// runCodexMemoryDoctor reads the Codex home and reports posture. Exit codes:
// 0 healthy posture, 1 risky posture (advisory), 2 usage/IO error.
func runCodexMemoryDoctor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak codex-memory doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	repo := fs.String("repo", "", "repo root to check for the AGENTS.md guidance boundary")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak codex-memory doctor: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	posture := codexmemory.Doctor(codexmemory.Options{
		CodexHome: *codexHome,
		RepoRoot:  *repo,
	})

	if *asJSON {
		if err := writeIndentedJSON(stdout, posture); err != nil {
			fmt.Fprintf(stderr, "fak codex-memory doctor: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, codexmemory.Render(posture))
	}

	if posture.OK {
		return 0
	}
	return 1
}
