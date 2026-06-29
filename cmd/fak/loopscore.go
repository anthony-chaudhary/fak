package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/loopscore"
)

func cmdLoopScore(argv []string) { os.Exit(runLoopScore(os.Stdout, os.Stderr, argv)) }

// runLoopScore scores fak's own agentic background loops — are they durable
// (auto-restart on system restart), self-reporting, and run THROUGH fak's tooling? —
// re-derived from the loop ledger + job registry fak writes. Exit codes: 0 healthy
// (no debt), 1 carries debt, 2 usage/IO error.
func runLoopScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak loop-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	ledger := fs.String("ledger", "", "loop ledger path (default: <root>/.fak/loops.jsonl, $FAK_LOOP_LEDGER)")
	registry := fs.String("registry", "", "loop registry path (default: <root>/tools/loop-registry.json, $FAK_LOOP_REGISTRY)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop-score: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	// Honor the same env defaults `fak loop` uses, so loop-score reads the same ledger
	// and registry the live loops write to.
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = os.Getenv("FAK_LOOP_LEDGER")
	}
	registryPath := *registry
	if registryPath == "" {
		registryPath = os.Getenv("FAK_LOOP_REGISTRY")
	}
	payload := loopscore.Build(loopscore.Options{
		Root:         root,
		LedgerPath:   ledgerPath,
		RegistryPath: registryPath,
	})
	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak loop-score", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, loopscore.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak loop-score: encode json: %v\n", err)
			return 1
		}
	} else if *asMarkdown {
		fmt.Fprint(stdout, loopscore.Markdown(payload))
	} else {
		fmt.Fprintln(stdout, loopscore.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}
