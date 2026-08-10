package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/bench"
)

func cmdScheduleHeld(argv []string) { os.Exit(runScheduleHeld(os.Stdout, os.Stderr, argv)) }

func runScheduleHeld(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("schedule-held", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "held hardware-service JSON file")
	iterations := fs.Int("overhead-iterations", 10000, "admission-overhead iterations")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *in == "" {
		fmt.Fprintln(stderr, "fak schedule-held: --in is required")
		return 2
	}
	data, err := os.ReadFile(*in)
	if err != nil {
		fmt.Fprintf(stderr, "fak schedule-held: %v\n", err)
		return 1
	}
	var req struct {
		Schema      string                   `json:"schema"`
		Jobs        []bench.HeldScheduleJob  `json:"jobs"`
		Calibrated  bench.HeldSchedulePolicy `json:"calibrated"`
		ScalarTotal bench.HeldSchedulePolicy `json:"scalar_total"`
	}
	if err := json.Unmarshal(data, &req); err != nil {
		fmt.Fprintf(stderr, "fak schedule-held: decode: %v\n", err)
		return 1
	}
	if req.Schema != "fak-held-schedule-input/1" {
		fmt.Fprintf(stderr, "fak schedule-held: unsupported schema %q\n", req.Schema)
		return 1
	}
	cal, scalar, err := bench.MeasureHeldDecisionOverhead(req.Jobs, req.Calibrated, req.ScalarTotal, *iterations)
	if err != nil {
		fmt.Fprintf(stderr, "fak schedule-held: %v\n", err)
		return 1
	}
	rep, err := bench.EvaluateHeldSchedule(req.Jobs, cal, scalar)
	if err != nil {
		fmt.Fprintf(stderr, "fak schedule-held: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		fmt.Fprintf(stderr, "fak schedule-held: encode: %v\n", err)
		return 1
	}
	return 0
}
