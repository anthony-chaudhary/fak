package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

func runModelDefault(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model-default", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit machine-readable default identity")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak model-default [--json]")
		return 2
	}
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(map[string]any{
			"schema":       "fak-model-default/v1",
			"alias":        modelreg.DefaultAlias,
			"ref":          modelreg.DefaultRef(),
			"coding":       true,
			"tool_capable": true,
		})
		return 0
	}
	fmt.Fprintf(stdout, "%s\t%s\n", modelreg.DefaultAlias, modelreg.DefaultRef())
	return 0
}
