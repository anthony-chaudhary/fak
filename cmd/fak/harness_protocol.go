package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnessprotocol"
)

func runHarnessProtocol(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "project" {
		fmt.Fprintln(stderr, "usage: fak harness protocol project --input EVENTS.jsonl --view cli|tui|json")
		return 2
	}
	fs := flag.NewFlagSet("harness protocol project", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "JSONL event fixture")
	view := fs.String("view", "json", "cli, tui, or json")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "--input is required")
		return 2
	}
	f, err := os.Open(*input)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	defer f.Close()
	events, err := harnessprotocol.ReadJSONL(f)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	p, err := harnessprotocol.Project(events)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	switch *view {
	case "cli":
		fmt.Fprint(stdout, harnessprotocol.CLIText(p))
	case "tui":
		fmt.Fprint(stdout, harnessprotocol.TUIText(p))
	case "json":
		if err = json.NewEncoder(stdout).Encode(p); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	default:
		fmt.Fprintln(stderr, "--view must be cli, tui, or json")
		return 2
	}
	return 0
}
