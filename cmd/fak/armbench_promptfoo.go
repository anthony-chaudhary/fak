package main

// promptfoo CLI wiring is kept separate so the comparator reproduction remains
// an additive armbench leaf.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/armbench"
)

func armbenchPromptfoo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench ponytail-promptfoo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	source := fs.String("source", "", "pinned Ponytail checkout (required)")
	out := fs.String("out", "", "witness output directory (required)")
	execute := fs.Bool("execute", false, "attempt every declared provider/model cell against real endpoints")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *source == "" || *out == "" {
		fmt.Fprintln(stderr, "fak armbench ponytail-promptfoo: --source and --out are required")
		return 2
	}
	r, err := armbench.RunPonytailPromptfoo(*source, *out, *execute)
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench ponytail-promptfoo:", err)
		return 1
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	b = append(b, '\n')
	summary := *out + string(os.PathSeparator) + "summary.json"
	if err := os.WriteFile(summary, b, 0644); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	_, _ = stdout.Write(b)
	if *execute && !r.Complete {
		return 3
	}
	return 0
}
