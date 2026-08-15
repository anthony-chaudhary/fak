package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessselect"
)

func runHarnessSelect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness select", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "harness selection manifest")
	path := fs.String("path", "", "current project or task path")
	var tags harnessTagFlag
	fs.Var(&tags, "tag", "context tag (repeatable or comma-separated)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *manifestPath == "" {
		fmt.Fprintln(stderr, "fak harness select: --manifest is required")
		return 2
	}
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness select: %v\n", err)
		return 1
	}
	manifest, err := harnessselect.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness select: %v\n", err)
		return 1
	}
	result, err := harnessselect.Resolve(manifest, harnessselect.Context{Path: *path, Tags: tags.values()})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness select: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness select: %v\n", err)
		return 1
	}
	return 0
}

type harnessTagFlag []string

func (f *harnessTagFlag) String() string { return strings.Join(*f, ",") }
func (f *harnessTagFlag) Set(value string) error {
	*f = append(*f, strings.Split(value, ",")...)
	return nil
}
func (f harnessTagFlag) values() []string { return []string(f) }
