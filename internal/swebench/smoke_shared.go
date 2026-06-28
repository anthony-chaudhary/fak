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

// smokeEvidenceLink is the canonical raw-vs-fak compare-evidence-link shape
// shared by every smoke contract. The DeepSWE and Opus contracts expose it
// under their own exported names via type aliases; the field set, JSON tags,
// and serialized output are identical across both.
type smokeEvidenceLink struct {
	Required     bool     `json:"required"`
	Predictions  []string `json:"predictions"`
	Metadata     []string `json:"metadata"`
	OfficialEval []string `json:"official_eval"`
	FakEvidence  []string `json:"fak_evidence"`
	JoinKeys     []string `json:"join_keys"`
	Detail       string   `json:"detail"`
}

// buildSmokeEvidenceLink constructs the shared compare-evidence-link for a
// raw-vs-fak smoke. Only the second fak-evidence artifact (the compare leaf
// filename), the join keys, and the detail prose differ between contracts;
// everything else (predictions, metadata, official-eval paths, the first
// fak-evidence artifact) is derived identically here.
func buildSmokeEvidenceLink(rawOutputDir, fakOutputDir, rawPreds, fakPreds, compareLeaf string, joinKeys []string, detail string) smokeEvidenceLink {
	return smokeEvidenceLink{
		Required: true,
		Predictions: []string{
			rawPreds,
			fakPreds,
		},
		Metadata: []string{
			joinPath(rawOutputDir, "meta.json"),
			joinPath(fakOutputDir, "meta.json"),
		},
		OfficialEval: []string{
			joinPath(rawOutputDir, "eval.json"),
			joinPath(fakOutputDir, "eval.json"),
		},
		FakEvidence: []string{
			joinPath(fakOutputDir, "fak-adjudication-evidence.jsonl"),
			joinPath(fakOutputDir, compareLeaf),
		},
		JoinKeys: joinKeys,
		Detail:   detail,
	}
}

// renderSmokeEvidenceLink writes the shared "Compare Evidence Link" markdown
// section for a smoke contract.
func renderSmokeEvidenceLink(b *strings.Builder, link smokeEvidenceLink) {
	fmt.Fprintf(b, "\n## Compare Evidence Link\n\n")
	fmt.Fprintf(b, "- Required: `%t`\n", link.Required)
	fmt.Fprintf(b, "- Predictions: `%s`\n", strings.Join(link.Predictions, "`, `"))
	fmt.Fprintf(b, "- Metadata: `%s`\n", strings.Join(link.Metadata, "`, `"))
	fmt.Fprintf(b, "- Official eval: `%s`\n", strings.Join(link.OfficialEval, "`, `"))
	fmt.Fprintf(b, "- fak evidence: `%s`\n", strings.Join(link.FakEvidence, "`, `"))
	fmt.Fprintf(b, "- Join keys: `%s`\n", strings.Join(link.JoinKeys, "`, `"))
	fmt.Fprintf(b, "- Detail: %s\n", link.Detail)
	fmt.Fprintf(b, "\n")
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
