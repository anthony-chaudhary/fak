package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/launchvisual"
)

// runLaunchVisual writes the launch SVGs from the figure data and the headline index, or
// with --check verifies them and the launch docs. Exit 0 clean, 1 drift or violations, 2 usage.
func runLaunchVisual(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch-visual", flag.ContinueOnError)
	fs.SetOutput(stderr)
	data := fs.String("data", "", "figure data JSON (default "+launchvisual.DefaultData+" under the repo root)")
	index := fs.String("index", "", "headline index JSON (default "+launchvisual.DefaultIndex+" under the repo root)")
	check := fs.Bool("check", false, "don't write; fail if the SVGs drifted from the index or the launch docs cite a number that does not trace to it")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "launch-visual: unexpected arguments: %v\n", fs.Args())
		return 2
	}
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	root, err := launchvisual.FindRoot(cwd)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *data == "" {
		*data = filepath.Join(root, filepath.FromSlash(launchvisual.DefaultData))
	}
	if *index == "" {
		*index = filepath.Join(root, filepath.FromSlash(launchvisual.DefaultIndex))
	}
	paths, err := launchvisual.Generate(*data, *index, *check)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, p := range paths {
		if *check {
			fmt.Fprintf(stdout, "check: %s is up to date\n", filepath.ToSlash(p))
		} else {
			fmt.Fprintf(stdout, "wrote %s\n", filepath.ToSlash(p))
		}
	}
	if *check {
		fmt.Fprintln(stdout, "check: launch docs cite only index-traced numbers")
	}
	return 0
}

func cmdLaunchVisual(argv []string) {
	if code := runLaunchVisual(os.Stdout, os.Stderr, argv); code != 0 {
		os.Exit(code)
	}
}
