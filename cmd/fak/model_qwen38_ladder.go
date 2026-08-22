package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/qwen38ladder"
)

func runModelQwen38Ladder(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model qwen38-ladder", flag.ContinueOnError)
	fs.SetOutput(stderr)
	evidencePath := fs.String("evidence", "", "evaluate a fak.qwen38-ladder-evidence/1 JSON file")
	selfcheck := fs.Bool("selfcheck", false, "validate and render the built-in ladder")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || (*evidencePath == "" && !*selfcheck) || (*evidencePath != "" && *selfcheck) {
		fmt.Fprintln(stderr, "usage: fak model qwen38-ladder (--selfcheck | --evidence EVIDENCE.json)")
		return 2
	}
	if err := qwen38ladder.ValidateDefinition(); err != nil {
		fmt.Fprintf(stderr, "fak model qwen38-ladder: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if *selfcheck {
		if err := enc.Encode(struct {
			Schema string               `json:"schema"`
			Stages []qwen38ladder.Stage `json:"stages"`
		}{qwen38ladder.Schema, qwen38ladder.Stages}); err != nil {
			fmt.Fprintf(stderr, "fak model qwen38-ladder: %v\n", err)
			return 1
		}
		return 0
	}
	f, err := os.Open(*evidencePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak model qwen38-ladder: %v\n", err)
		return 1
	}
	defer f.Close()
	evidence, err := qwen38ladder.Decode(f)
	if err != nil {
		fmt.Fprintf(stderr, "fak model qwen38-ladder: %v\n", err)
		return 1
	}
	decision, err := qwen38ladder.Evaluate(evidence)
	if err != nil {
		fmt.Fprintf(stderr, "fak model qwen38-ladder: %v\n", err)
		return 1
	}
	if err := enc.Encode(decision); err != nil {
		fmt.Fprintf(stderr, "fak model qwen38-ladder: %v\n", err)
		return 1
	}
	if decision.Verdict == "HOLD" {
		return 1
	}
	return 0
}
