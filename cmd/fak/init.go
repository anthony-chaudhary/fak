package main

// fak init — emit a minimal, valid `fak.toml` deployment manifest (#3421,
// Workstream E of epic #3256). `fak.toml` is the single declarative artifact for
// the whole all-in-one deployment; `fak init` writes the smallest reviewable
// starting point, `fak up` (E1) reads it, and the deployment `fak doctor` (E4)
// validates it. This verb owns only the emit half of that round-trip.
//
// It refuses to clobber an existing manifest unless --force, so an operator does
// not silently lose a hand-edited deployment descriptor.
//
//	fak init [--path fak.toml] [--force] [--stdout]
//
// The emitted bytes round-trip: deploymanifest.Parse(deploymanifest.Minimal())
// parses clean (asserted in internal/deploymanifest).

import (
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

func cmdInit(args []string) {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	path := fs.String("path", "fak.toml", "path to write the manifest")
	force := fs.Bool("force", false, "overwrite an existing manifest")
	stdout := fs.Bool("stdout", false, "write to stdout instead of a file")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}

	body := deploymanifest.Minimal()

	if *stdout {
		os.Stdout.Write(body)
		return
	}

	if !*force {
		if _, err := os.Stat(*path); err == nil {
			fmt.Fprintf(os.Stderr, "fak init: %s already exists (use --force to overwrite)\n", *path)
			os.Exit(1)
		}
	}
	if err := os.WriteFile(*path, body, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "fak init: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stdout, "wrote %s\n", *path)
}
