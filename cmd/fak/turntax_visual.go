package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/turntaxvisual"
)

func runTurnTaxVisual(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("turntax visual", flag.ContinueOnError)
	fs.SetOutput(stderr)
	data := fs.String("data", turntaxvisual.DefaultData, "turn-tax visual JSON source of truth")
	check := fs.Bool("check", false, "don't write; fail if the checked-in SVG drifted from the data")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "turntax visual: unexpected arguments: %v\n", fs.Args())
		return 2
	}
	path, err := turntaxvisual.Generate(*data, *check)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *check {
		fmt.Fprintf(stdout, "check: %s is up to date with %s.\n", filepath.ToSlash(path), filepath.ToSlash(*data))
	} else {
		fmt.Fprintf(stdout, "wrote %s\n", filepath.ToSlash(path))
	}
	return 0
}

func cmdTurnTaxVisual(argv []string) {
	if code := runTurnTaxVisual(os.Stdout, os.Stderr, argv); code != 0 {
		os.Exit(code)
	}
}
