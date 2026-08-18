package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/harnessgallery"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runHarnessGallery(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak harness gallery <list|show|init|selfcheck>")
		return 2
	}
	switch argv[0] {
	case "list":
		fs := flag.NewFlagSet("harness gallery list", flag.ContinueOnError)
		fs.SetOutput(stderr)
		jsonOut := fs.Bool("json", false, "emit JSON")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		items := harnessgallery.Builtins()
		if *jsonOut {
			if err := json.NewEncoder(stdout).Encode(items); err != nil {
				fmt.Fprintln(stderr, err)
				return 1
			}
			return 0
		}
		fmt.Fprintln(stdout, "Harness starter packs")
		for _, b := range items {
			fmt.Fprintf(stdout, "\n%s - %s\n  Use when: %s.\n  Outcome: %s.\n  Start from: %s.\n", b.ID, b.Name, b.For, b.BetterBecause, b.Seam)
		}
		fmt.Fprintln(stdout, "\nNext: inspect one pack with fak harness gallery show --id <pack>")
		return 0
	case "show":
		fs := flag.NewFlagSet("harness gallery show", flag.ContinueOnError)
		fs.SetOutput(stderr)
		id := fs.String("id", "", "blueprint ID")
		jsonOut := fs.Bool("json", false, "emit JSON")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		b, ok := harnessgallery.Find(*id)
		if !ok {
			fmt.Fprintf(stderr, "unknown blueprint %q\n", *id)
			return 1
		}
		if *jsonOut {
			return encodeGallery(stdout, stderr, b)
		}
		printGalleryBlueprint(stdout, b)
		return 0
	case "init":
		fs := flag.NewFlagSet("harness gallery init", flag.ContinueOnError)
		fs.SetOutput(stderr)
		id := fs.String("id", "", "blueprint ID")
		dir := fs.String("dir", "", "starter directory")
		jsonOut := fs.Bool("json", false, "emit JSON")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		r, err := harnessgallery.Init(*id, pathutil.ExpandTilde(*dir))
		if err != nil {
			fmt.Fprintf(stderr, "fak harness gallery init: %v\n", err)
			return 1
		}
		if *jsonOut {
			return encodeGallery(stdout, stderr, r)
		}
		fmt.Fprintf(stdout, "Initialized %s in %s\n", *id, r.Directory)
		printGalleryFiles(stdout, "Created", r.Created)
		printGalleryFiles(stdout, "Preserved", r.Preserved)
		fmt.Fprintf(stdout, "\nNext:\n  1. Read %s\n  2. Edit %s to fit your tools and boundaries\n  3. Check all built-ins with fak harness gallery selfcheck\n", filepath.Join(r.Directory, "README.md"), filepath.Join(r.Directory, "harness.pack.json"))
		return 0
	case "selfcheck":
		if err := harnessgallery.Validate(harnessgallery.Builtins()); err != nil {
			fmt.Fprintf(stderr, "fak harness gallery selfcheck: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "harness-gallery-selfcheck OK blueprints=%d\n", len(harnessgallery.Builtins()))
		return 0
	default:
		fmt.Fprintln(stderr, "usage: fak harness gallery <list|show|init|selfcheck>")
		return 2
	}
}
func encodeGallery(stdout, stderr io.Writer, v any) int {
	if err := json.NewEncoder(stdout).Encode(v); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func printGalleryBlueprint(w io.Writer, b harnessgallery.Blueprint) {
	fmt.Fprintf(w, "%s (%s)\n\n", b.Name, b.ID)
	fmt.Fprintf(w, "For: %s\nProblem: %s\nToday: %s\nBetter because: %s\n\n", b.For, b.Problem, b.Today, b.BetterBecause)
	fmt.Fprintf(w, "Build from: %s\nProof to capture: %s\n", b.Seam, b.Witness)
	printGalleryList(w, "Needs", b.RequiredCapabilities)
	printGalleryList(w, "Does not get", b.ExcludedCapabilities)
	printGalleryList(w, "Ten-minute path", b.TenMinuteSpine)
	printGalleryList(w, "Then extend it", b.WeekendExtension)
	fmt.Fprintf(w, "\nNext: fak harness gallery init --id %s --dir ./%s-pack\n", b.ID, b.ID)
}

func printGalleryList(w io.Writer, heading string, items []string) {
	fmt.Fprintf(w, "\n%s:\n", heading)
	for i, item := range items {
		fmt.Fprintf(w, "  %d. %s\n", i+1, item)
	}
}

func printGalleryFiles(w io.Writer, heading string, files []string) {
	if len(files) == 0 {
		return
	}
	fmt.Fprintf(w, "%s:\n", heading)
	for _, file := range files {
		fmt.Fprintf(w, "  - %s\n", file)
	}
}
