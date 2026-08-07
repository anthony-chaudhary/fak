package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func cmdToolWidth(argv []string) { os.Exit(runToolWidth(os.Stdout, os.Stderr, argv)) }

func runToolWidth(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tool-width", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "JSONL width observations")
	baseline := fs.Float64("baseline", -1, "baseline batched_turn_rate for single-series ratchet")
	minDrop := fs.Float64("min-drop", 0.1, "minimum downward step that alarms")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "fak tool-width: --input is required")
		return 2
	}
	f, err := os.Open(*input)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	defer f.Close()
	var obs []agent.WidthObservation
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 4096), 4<<20)
	for scan.Scan() {
		var o agent.WidthObservation
		if err := json.Unmarshal(scan.Bytes(), &o); err != nil {
			fmt.Fprintln(stderr, "fak tool-width:", err)
			return 1
		}
		obs = append(obs, o)
	}
	if err := scan.Err(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	report := agent.FoldWidth(obs)
	payload := struct {
		agent.WidthReport
		Ratchet *agent.WidthRegression `json:"ratchet,omitempty"`
	}{WidthReport: report}
	if *baseline >= 0 {
		if len(report.Series) != 1 {
			fmt.Fprintln(stderr, "fak tool-width: --baseline requires exactly one lane/engine/model series")
			return 2
		}
		r := agent.DetectWidthRegression(*baseline, report.Series[0].BatchedTurnRate, *minDrop)
		payload.Ratchet = &r
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(payload); err != nil {
		return 1
	}
	if payload.Ratchet != nil && payload.Ratchet.Regressed {
		return 3
	}
	return 0
}
