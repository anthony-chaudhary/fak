package main

// fak garden -- the garden bundle: one default-on, read-only fold over the
// repo's self-maintenance passes (the scorecard control pane + fresh status), so
// "run the gardening" is one command instead of three. It runs each member
// (grandfathered Python tools), reads its control-pane JSON, and folds one
// schema/ok/verdict/finding/reason/next_action envelope. It mutates nothing.
// --check is the CI gate (exit non-zero only when a gating member regressed or a
// pass failed to run); --deep adds the slower fleet loop-audit member. Skipped
// when FAK_GARDEN is off (the env-side governor brake).

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
)

func cmdGarden(argv []string) { os.Exit(runGarden(os.Stdout, os.Stderr, argv)) }

func runGarden(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("garden", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "CI gate: exit non-zero if a gating member regressed or failed to run")
	deep := fs.Bool("deep", false, "also run the fleet loop-audit member (slower; non-gating advisory)")
	timeout := fs.Int("timeout", 240, "per-member timeout seconds")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	commit := gardenbundle.HeadCommit(root)

	var payload gardenbundle.Payload
	if gardenbundle.GardenOff() {
		payload = gardenbundle.SkippedPayload(root, commit)
	} else {
		results := gardenbundle.Collect(root, "", time.Duration(*timeout)*time.Second, *deep)
		payload = gardenbundle.Fold(results, root, commit)
	}

	if *check {
		code, message := gardenbundle.CheckGate(payload)
		if *asJSON {
			gated := payload.WithGate(code, message)
			emitGardenJSON(stdout, gated)
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
	}

	if *asJSON {
		emitGardenJSON(stdout, payload)
	} else {
		fmt.Fprintln(stdout, gardenbundle.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}

func emitGardenJSON(w io.Writer, p gardenbundle.Payload) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	_ = enc.Encode(p)
}

// repoRoot resolves the repo root the way the Python tool did: the parent of
// tools/. It walks up from the cwd looking for the go.mod / tools marker, and
// falls back to the cwd.
func repoRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return wd
		}
		dir = parent
	}
}
