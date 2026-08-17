// toolcallcontroldemo is a no-model spine for deterministic tool-call control.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/toolcallcontrol"
)

type output struct {
	Schema      string                       `json:"schema"`
	Instruction string                       `json:"instruction"`
	Decisions   []toolcallcontrol.Verdict    `json:"decisions"`
	Ablation    []toolcallcontrol.ArmMetrics `json:"ablation"`
	SelfCheck   string                       `json:"self_check,omitempty"`
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the captured no-model spine")
	pretty := flag.Bool("pretty", false, "indent JSON")
	flag.Parse()
	if !*selfcheck {
		fmt.Fprintln(os.Stderr, "usage: toolcallcontroldemo -selfcheck [-pretty]")
		os.Exit(2)
	}

	contextTokens := int64(128000)
	proposals := []toolcallcontrol.Proposal{
		{ID: "repeat", Tool: "read_file", Args: `{"path":"README.md"}`, StateEpoch: "git:8457bbd", PromptTokens: contextTokens, ReadOnly: true, EvidenceGap: "confirm unchanged introduction", EffectIfNew: "revise explanation"},
		{ID: "search-a", Tool: "search_code", Args: `{"q":"tool call"}`, StateEpoch: "git:8457bbd", PromptTokens: contextTokens, ReadOnly: true, BatchKey: "repo-inspection", EvidenceGap: "find call sites", EffectIfNew: "select package"},
		{ID: "search-b", Tool: "search_code", Args: `{"q":"ablation"}`, StateEpoch: "git:8457bbd", PromptTokens: contextTokens, ReadOnly: true, BatchKey: "repo-inspection", EvidenceGap: "find metrics", EffectIfNew: "reuse existing report"},
		{ID: "browse", Tool: "search_web", Args: `{"q":"maybe more ideas"}`, PromptTokens: contextTokens, ReadOnly: true, ExpectedInfoGainBP: 100},
		{ID: "write", Tool: "write_file", Args: `{"path":"result"}`, PromptTokens: contextTokens, EffectIfNew: "complete task"},
	}
	prior := []toolcallcontrol.Observation{{Tool: "read_file", Args: `{"path":"README.md"}`, StateEpoch: "git:8457bbd", ResultRef: "turn-previous"}}
	decisions := make([]toolcallcontrol.Verdict, 0, len(proposals))
	byID := map[string]toolcallcontrol.Verdict{}
	for _, p := range proposals {
		d := toolcallcontrol.Evaluate(toolcallcontrol.Config{}, p, prior, proposals)
		decisions = append(decisions, d)
		byID[p.ID] = d
	}
	trace := make([]toolcallcontrol.LabeledProposal, 0, len(proposals))
	needed := map[string]bool{"search-a": true, "search-b": true, "write": true}
	for _, p := range proposals {
		trace = append(trace, toolcallcontrol.LabeledProposal{Proposal: p, Needed: needed[p.ID]})
	}
	control := map[string]toolcallcontrol.Verdict{}
	for _, p := range proposals {
		control[p.ID] = toolcallcontrol.Verdict{ID: p.ID, Action: toolcallcontrol.Allow}
	}

	result := output{
		Schema:      "fak-tool-call-control-demo/1",
		Instruction: toolcallcontrol.Instruction(contextTokens),
		Decisions:   decisions,
		Ablation:    toolcallcontrol.Ablate(trace, []toolcallcontrol.Arm{{Name: "control", Decisions: control}, {Name: "prefilter", Decisions: byID}}),
		SelfCheck:   "PASS",
	}
	enc := json.NewEncoder(os.Stdout)
	if *pretty {
		enc.SetIndent("", "  ")
	}
	if err := enc.Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
