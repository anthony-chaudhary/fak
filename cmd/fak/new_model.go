package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func runNewModel(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("new-model", flag.ContinueOnError)
	fs.SetOutput(stderr)
	family := fs.String("family", "", "family name, lowercase (e.g. myfamily)")
	topology := fs.String("topology", "identity", "topology: prenorm, postnorm, parallel, or identity")
	dryRun := fs.Bool("dry-run", false, "print scaffold without writing files")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	fromManifest := fs.String("from-manifest", "", "compile a pinned release manifest into an onboarding packet")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak new-model: positional arguments are not supported")
		return 2
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })

	if *fromManifest != "" {
		if *family != "" || set["topology"] || set["dry-run"] || !*asJSON {
			fmt.Fprintln(stderr, "fak new-model: --from-manifest requires --json and cannot be combined with scaffold flags")
			return 2
		}
		data, err := os.ReadFile(*fromManifest)
		if err != nil {
			fmt.Fprintf(stderr, "fak new-model: read manifest: %v\n", err)
			return 1
		}
		packet, err := newmodel.CompileManifestJSON(data)
		if err != nil {
			var refusal *newmodel.Refusal
			if errors.As(err, &refusal) {
				enc := json.NewEncoder(stderr)
				enc.SetIndent("", "  ")
				_ = enc.Encode(refusal)
				return 3
			}
			fmt.Fprintf(stderr, "fak new-model: compile manifest: %v\n", err)
			return 1
		}
		if _, err := stdout.Write(packet); err != nil {
			fmt.Fprintf(stderr, "fak new-model: write packet: %v\n", err)
			return 1
		}
		return 0
	}

	if *family == "" {
		fmt.Fprintln(stderr, "fak new-model: --family or --from-manifest is required")
		fmt.Fprintln(stderr, "usage: fak new-model (--family <name> [--topology <topology>] [--dry-run] | --from-manifest <file>) [--json]")
		return 2
	}

	res, err := newmodel.Run(newmodel.Scaffold{Family: *family, Topology: *topology, DryRun: *dryRun})
	if err != nil {
		fmt.Fprintf(stderr, "fak new-model: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak new-model: encode scaffold: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "=== Scaffolding model family '%s' (topology: %s) ===\n\n", res.Family, res.Topology)
	fmt.Fprintln(stdout, "Files to edit:")
	for _, edit := range res.Edits {
		fmt.Fprintf(stdout, "  - %s\n", edit)
	}
	fmt.Fprintln(stdout, "\nNext steps:")
	for _, step := range res.NextSteps {
		fmt.Fprintln(stdout, step)
	}
	return 0
}
