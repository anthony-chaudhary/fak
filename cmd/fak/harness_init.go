package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnesshost"
	"github.com/anthony-chaudhary/fak/internal/harnessinit"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runHarness(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "web" {
		return runHarnessWeb(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "gallery" {
		return runHarnessGallery(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "release" {
		return runHarnessRelease(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "study" {
		return runHarnessStudy(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "preview" {
		return runHarnessPreview(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "inspect" {
		return runHarnessInspect(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "derive" {
		return runHarnessDerive(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "mix" {
		return runHarnessMix(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "override" {
		return runHarnessOverride(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "verify-run" {
		return runHarnessVerifyRun(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "cross-dogfood" {
		return runHarnessCrossDogfood(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "resolve" {
		return runHarnessResolve(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "compose" {
		return runHarnessCompose(stdout, stderr, argv[1:])
	}
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
		fmt.Fprintln(stderr, "usage: fak harness <init|classify|compose|cross-dogfood|derive|discover|gallery|inspect|mix|override|preview|release|resolve|select|study|protocol|verify-run|web>")
		return 2
	}
	fs := flag.NewFlagSet("harness init", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "external product directory")
	module := fs.String("module", "", "Go module path for the product")
	version := fs.String("fak-version", harnessinit.DefaultFAKVersion, "pinned fak module version")
	host := fs.String("host", "", "seed a versioned first-party host component (codex|claude)")
	jsonOut := fs.Bool("json", false, "emit machine-readable result")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	hostArtifacts, err := harnesshost.Build(*host, harnessinit.ContractVersion)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness init: %v\n", err)
		return 1
	}
	result, err := harnessinit.Init(harnessinit.Options{
		Dir: pathutil.ExpandTilde(*dir), Module: *module, FAKVersion: *version,
		Host: hostArtifacts.Host, HostManifest: hostArtifacts.Manifest, HostLock: hostArtifacts.Lock,
	})
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
		fmt.Fprintf(stdout, "created external product at %s\nrun: cd %s && go run ./cmd/product --launch --agent-id local-agent\n", result.Directory, result.Directory)
	}
	return 0
}
