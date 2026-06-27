package swebench

import (
	"fmt"
	"strings"
)

// twoArmNames carries the per-contract labels that distinguish the raw and fak
// arms of a raw-vs-fak smoke; the arm SHAPE (model, command, output dir,
// predictions path, eval command hint) is identical across contracts.
type twoArmNames struct {
	RawName    string
	RawHarness string
	RawEvalID  string
	FakName    string
	FakHarness string
	FakEvalID  string
}

// buildTwoArms constructs the standard raw + fak SmokeArm pair shared by the
// Opus and DeepSWE contracts. Only the names/harness/eval-run-id labels differ
// between contracts; everything else (predictions path, eval command hint) is
// derived identically here.
func buildTwoArms(model, rawCommand, fakCommand, rawOutputDir, fakOutputDir, rawPreds, fakPreds string, maxWorkers int, n twoArmNames) []SmokeArm {
	return []SmokeArm{
		{
			Name:            n.RawName,
			Harness:         n.RawHarness,
			Model:           model,
			Command:         rawCommand,
			OutputDir:       rawOutputDir,
			PredictionsPath: rawPreds,
			EvalRunID:       n.RawEvalID,
			EvalCommand:     EvalCommandHint(rawPreds, n.RawEvalID, maxWorkers),
		},
		{
			Name:            n.FakName,
			Harness:         n.FakHarness,
			Model:           model,
			Command:         fakCommand,
			OutputDir:       fakOutputDir,
			PredictionsPath: fakPreds,
			EvalRunID:       n.FakEvalID,
			EvalCommand:     EvalCommandHint(fakPreds, n.FakEvalID, maxWorkers),
		},
	}
}

// renderSmokeArmsTable writes the shared "Arm | Harness | Model | Predictions |
// Eval run id" markdown table for a slice of SmokeArm.
func renderSmokeArmsTable(b *strings.Builder, arms []SmokeArm) {
	fmt.Fprintf(b, "| Arm | Harness | Model | Predictions | Eval run id |\n")
	fmt.Fprintf(b, "|---|---|---|---|---|\n")
	for _, arm := range arms {
		fmt.Fprintf(b, "| `%s` | `%s` | `%s` | `%s` | `%s` |\n",
			mdCell(arm.Name), mdCell(arm.Harness), mdCell(arm.Model), mdCell(arm.PredictionsPath), mdCell(arm.EvalRunID))
	}
}

// renderSmokeGatesTable writes the shared "Gate | OK | Detail" markdown table
// for a slice of SmokeGate.
func renderSmokeGatesTable(b *strings.Builder, gates []SmokeGate) {
	fmt.Fprintf(b, "## Gates\n\n")
	fmt.Fprintf(b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(b, "|---|:---:|---|\n")
	for _, gate := range gates {
		mark := "no"
		if gate.OK {
			mark = "yes"
		}
		fmt.Fprintf(b, "| `%s` | %s | %s |\n", mdCell(gate.Name), mark, mdCell(gate.Detail))
	}
}

// renderRequiredBeforeClaim writes the shared "Required Before Any Result Claim"
// bullet list.
func renderRequiredBeforeClaim(b *strings.Builder, reqs []string) {
	fmt.Fprintf(b, "\n## Required Before Any Result Claim\n\n")
	for _, req := range reqs {
		fmt.Fprintf(b, "- %s\n", req)
	}
}
