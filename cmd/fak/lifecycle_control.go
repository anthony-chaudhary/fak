package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"io"
	"os"
)

func cmdLifecycle(argv []string) { os.Exit(runLifecycle(os.Stdout, os.Stderr, argv)) }
func runLifecycle(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak lifecycle <inspect|cancel|rollback|retry|resume> --tx FILE [--apply]")
		return 2
	}
	control := argv[0]
	fs := flag.NewFlagSet("fak lifecycle "+control, flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("tx", "", "durable transaction JSON path")
	apply := fs.Bool("apply", false, "persist the planned state transition")
	if !parseFlags(fs, argv[1:]) || *path == "" || fs.NArg() != 0 {
		if *path == "" {
			fmt.Fprintln(stderr, "fak lifecycle: --tx is required")
		}
		return 2
	}
	b, err := os.ReadFile(*path)
	if err != nil {
		fmt.Fprintf(stderr, "fak lifecycle: read transaction: %v\n", err)
		return 1
	}
	var tx lifecycle.Transaction
	if err = json.Unmarshal(b, &tx); err != nil {
		fmt.Fprintf(stderr, "fak lifecycle: parse transaction: %v\n", err)
		return 1
	}
	preview, err := lifecycle.Control(tx, control)
	if err != nil {
		fmt.Fprintf(stderr, "fak lifecycle: %v\n", err)
		return 1
	}
	if *apply && control != "inspect" {
		next := lifecycle.Apply(tx, preview)
		out, _ := json.MarshalIndent(next, "", "  ")
		out = append(out, '\n')
		if err = os.WriteFile(*path, out, 0600); err != nil {
			fmt.Fprintf(stderr, "fak lifecycle: persist transaction: %v\n", err)
			return 1
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err = enc.Encode(preview); err != nil {
		return 1
	}
	return 0
}
