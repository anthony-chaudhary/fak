package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/portabilitylab"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("portability-lab", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("out", "", "write authoritative JSON report")
	work := fs.String("work", "", "clean-room directory (default: temporary)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	fail := func(err error) int {
		fmt.Fprintln(stderr, "portability-lab:", err)
		return 1
	}
	root := *work
	if root == "" {
		var err error
		root, err = os.MkdirTemp("", "fak-portability-lab-")
		if err != nil {
			return fail(err)
		}
		defer os.RemoveAll(root)
	}
	r, err := portabilitylab.Run(root)
	if *out != "" {
		if e := os.MkdirAll(filepath.Dir(*out), 0755); e != nil {
			return fail(e)
		}
		if e := portabilitylab.WriteReport(*out, r); e != nil {
			return fail(e)
		}
	}
	b, _ := json.MarshalIndent(r, "", "  ")
	fmt.Fprintln(stdout, string(b))
	if err != nil {
		return fail(err)
	}
	return 0
}
