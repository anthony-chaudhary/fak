package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnessinit"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdHarness(argv []string) { os.Exit(runHarness(os.Stdout, os.Stderr, argv)) }

func runHarness(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "classify" {
		return runHarnessClassify(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "discover" {
		return runHarnessDiscover(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "select" {
		return runHarnessSelect(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "protocol" {
		return runHarnessProtocol(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 || argv[0] != "init" {
		fmt.Fprintln(stderr, "usage: fak harness <init|classify|discover|select|protocol>")
		return 2
	}
	fs := flag.NewFlagSet("harness init", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "external product directory")
	module := fs.String("module", "", "Go module path for the product")
	version := fs.String("fak-version", harnessinit.DefaultFAKVersion, "pinned fak module version")
	jsonOut := fs.Bool("json", false, "emit machine-readable result")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	result, err := harnessinit.Init(harnessinit.Options{Dir: pathutil.ExpandTilde(*dir), Module: *module, FAKVersion: *version})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness init: %v\n", err)
		return 1
	}
	if *jsonOut {
		if err := json.NewEncoder(stdout).Encode(result); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "created external product at %s\nrun: cd %s && go run ./cmd/product --selfcheck\n", result.Directory, result.Directory)
	}
	return 0
}
