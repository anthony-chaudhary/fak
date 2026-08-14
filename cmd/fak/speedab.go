package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/speedab"
)

func runSpeedAB(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("speed-ab", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", "captured speed A/B manifest JSON")
	if fs.Parse(argv) != nil {
		return 2
	}
	if *manifest == "" {
		fmt.Fprintln(stderr, "speed-ab: --manifest is required")
		return 2
	}
	b, err := os.ReadFile(*manifest)
	if err != nil {
		fmt.Fprintln(stderr, "speed-ab:", err)
		return 1
	}
	var m speedab.Manifest
	if json.Unmarshal(b, &m) != nil {
		fmt.Fprintln(stderr, "speed-ab: invalid manifest")
		return 2
	}
	r := speedab.Grade(m)
	out, _ := json.MarshalIndent(r, "", "  ")
	fmt.Fprintln(stdout, string(out))
	if r.Verdict != "NET_TRUE" {
		return 3
	}
	return 0
}
func cmdSpeedAB(argv []string) { os.Exit(runSpeedAB(os.Stdout, os.Stderr, argv)) }
