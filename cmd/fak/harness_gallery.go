package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"text/tabwriter"

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
		w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(w, "ID\tNAME\tPUBLIC SEAM")
		for _, b := range items {
			fmt.Fprintf(w, "%s\t%s\t%s\n", b.ID, b.Name, b.Seam)
		}
		_ = w.Flush()
		return 0
	case "show":
		fs := flag.NewFlagSet("harness gallery show", flag.ContinueOnError)
		fs.SetOutput(stderr)
		id := fs.String("id", "", "blueprint ID")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		b, ok := harnessgallery.Find(*id)
		if !ok {
			fmt.Fprintf(stderr, "unknown blueprint %q\n", *id)
			return 1
		}
		if err := json.NewEncoder(stdout).Encode(b); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
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
		fmt.Fprintf(stdout, "initialized %s in %s (created=%d preserved=%d)\n", *id, r.Directory, len(r.Created), len(r.Preserved))
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
