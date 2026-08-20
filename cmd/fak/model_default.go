package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

func runModelDefault(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model-default", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit machine-readable default identity")
	evidencePath := fs.String("evidence", "", "fold a Qwen3.8 default evidence JSON file")
	at := fs.String("at", "", "evaluate freshness at RFC3339 time (default: now)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak model-default [--json] [--evidence EVIDENCE.json] [--at RFC3339]")
		return 2
	}
	now := time.Now().UTC()
	if *at != "" {
		parsed, err := time.Parse(time.RFC3339, *at)
		if err != nil {
			fmt.Fprintf(stderr, "fak model-default: --at must be RFC3339: %v\n", err)
			return 2
		}
		now = parsed
	}
	evidence := modelreg.EmptyDefaultEvidence()
	if *evidencePath != "" {
		body, err := os.ReadFile(*evidencePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak model-default: read evidence: %v\n", err)
			return 1
		}
		if err := json.Unmarshal(body, &evidence); err != nil {
			fmt.Fprintf(stderr, "fak model-default: decode evidence: %v\n", err)
			return 1
		}
	}
	decision := modelreg.EvaluateDefaultEvidence(evidence, now)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(map[string]any{
			"schema":        "fak-model-default/v1",
			"alias":         modelreg.DefaultAlias,
			"ref":           modelreg.DefaultRef(),
			"coding":        true,
			"tool_capable":  true,
			"verdict":       decision.Verdict,
			"evidence_refs": decision.EvidenceRefs,
			"decision":      decision,
		})
		return 0
	}
	fmt.Fprintf(stdout, "%s\t%s\n", modelreg.DefaultAlias, modelreg.DefaultRef())
	fmt.Fprintf(stdout, "VERDICT\t%s\n", decision.Verdict)
	for _, reason := range decision.Reasons {
		fmt.Fprintf(stdout, "REASON\t%s\t%s\t%s\n", reason.Family, reason.Code, reason.Detail)
	}
	return 0
}
