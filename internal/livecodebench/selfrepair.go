package livecodebench

import (
	"fmt"
	"strings"
)

// selfrepair.go is the self-repair scenario (epic #2085): a second pass built ON
// TOP of a graded codegen run. For each source generation that FAILED grading, the
// model is shown its own wrong code plus the failing-test feedback and asked to fix
// it; the repaired generations are re-graded and scored against the source. Two
// pieces are pure here so they are unit-tested without a model call: BuildRepairPrompt
// (the "prior wrong generation + failing-test feedback" request the CLI sends to the
// gateway) and BuildSelfRepairDelta (the source-vs-repaired pass@1 delta). Like the
// rest of the package the delta is evidence-gated — ResultClaimAllowed stays false;
// the same generations must be graded by the official lcb_runner before any
// pass-rate is claimable.

// SelfRepairDeltaSchema tags a self-repair delta artifact.
const SelfRepairDeltaSchema = "fak.livecodebench.selfrepair.v1"

// SelfRepairRepairN is the fixed fan-out of the repair pass. Upstream LCB self-repair
// repairs each source codegen generation exactly ONCE — the scenario is defined at
// n=1, and a wider repair fan-out is a different experiment, not a self-repair run.
// It is recorded on every delta so a reader can see the repair pass ran at n=1.
const SelfRepairRepairN = 1

// RepairSource is the graded codegen result self-repair builds on, for one problem:
// SourceSamples (= upstream --codegen_n) generations of which SourceCorrect passed
// grading, and Fixed of the (SourceSamples-SourceCorrect) failing generations that
// the n=1 repair pass turned into passes. Self-repair cannot run without this source
// — the repair prompt feeds a prior wrong generation and its failing-test feedback,
// so a codegen run must have produced and graded those generations first.
type RepairSource struct {
	QuestionID    string `json:"question_id"`
	SourceSamples int    `json:"source_samples"` // n: source codegen generations for this problem
	SourceCorrect int    `json:"source_correct"` // c: source generations that passed grading
	Fixed         int    `json:"fixed"`          // of the (n-c) failing, how many the n=1 repair fixed
}

// SelfRepairDelta is the machine-readable result of a self-repair run: the source
// codegen pass@1, the post-repair pass@1, and their difference, plus the run
// identity the scenario demands (the source codegen_n and the fixed repair_n=1).
type SelfRepairDelta struct {
	Schema             string   `json:"schema"`
	Scenario           Scenario `json:"scenario"`
	Model              string   `json:"model,omitempty"`
	CodegenN           int      `json:"codegen_n"`
	RepairN            int      `json:"repair_n"`
	Problems           int      `json:"problems"`
	SourcePassAt1      float64  `json:"source_pass_at_1"`
	RepairedPassAt1    float64  `json:"repaired_pass_at_1"`
	Delta              float64  `json:"delta"`
	EvidenceClass      string   `json:"evidence_class"`
	ResultClaimAllowed bool     `json:"result_claim_allowed"`
}

// BuildRepairPrompt renders the self-repair request for one failing generation: the
// original problem, the prior WRONG solution, and the failing-test feedback, asking
// the model for a corrected solution. This is the upstream self-repair prompt shape
// (problem + wrong attempt + observed failure) rendered purely so it is unit-tested
// without a model call. It refuses an empty prior generation or empty feedback — a
// repair with nothing to repair from, or no observed failure, is not a self-repair
// request.
func BuildRepairPrompt(p Problem, wrong, feedback string) (string, error) {
	if strings.TrimSpace(wrong) == "" {
		return "", fmt.Errorf("livecodebench self-repair: prior generation is empty (nothing to repair)")
	}
	if strings.TrimSpace(feedback) == "" {
		return "", fmt.Errorf("livecodebench self-repair: failing-test feedback is empty (a repair needs the observed failure)")
	}
	var b strings.Builder
	b.WriteString("You previously wrote a solution that failed some tests. ")
	b.WriteString("Fix it and return a corrected, complete solution.\n\n")
	b.WriteString("## Problem\n")
	b.WriteString(strings.TrimSpace(p.Prompt))
	b.WriteString("\n\n## Your previous (incorrect) solution\n")
	b.WriteString(strings.TrimSpace(wrong))
	b.WriteString("\n\n## Failing test feedback\n")
	b.WriteString(strings.TrimSpace(feedback))
	b.WriteString("\n\n## Corrected solution\n")
	return b.String(), nil
}

// BuildSelfRepairDelta scores a self-repair run against its source codegen run. It
// REFUSES (clear error) without a codegen source — an empty source set, or a problem
// with no source generations, means there is nothing to repair. The repair pass is
// defined at n=1 (SelfRepairRepairN) and recorded as such. The source pass@1 is the
// mean pass@1 over the source (SourceSamples, SourceCorrect) tallies; the repaired
// pass@1 is the mean pass@1 over the SAME sample counts with correct = SourceCorrect
// + Fixed; the delta is repaired minus source.
func BuildSelfRepairDelta(model string, codegenN int, sources []RepairSource) (SelfRepairDelta, error) {
	if len(sources) == 0 {
		return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: a graded codegen source run is required (no source generations to repair)")
	}
	if codegenN < 1 {
		return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: codegen_n must be >= 1, got %d", codegenN)
	}
	srcTallies := make([]SampleTally, len(sources))
	repTallies := make([]SampleTally, len(sources))
	for i, s := range sources {
		if strings.TrimSpace(s.QuestionID) == "" {
			return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: source %d question_id is required", i)
		}
		if s.SourceSamples < 1 {
			return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: problem %q has no source generations to repair (source_samples=%d); run codegen first", s.QuestionID, s.SourceSamples)
		}
		if s.SourceSamples != codegenN {
			return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: problem %q source_samples %d does not match codegen_n %d", s.QuestionID, s.SourceSamples, codegenN)
		}
		if s.SourceCorrect < 0 || s.SourceCorrect > s.SourceSamples {
			return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: problem %q source_correct %d out of range [0,%d]", s.QuestionID, s.SourceCorrect, s.SourceSamples)
		}
		failing := s.SourceSamples - s.SourceCorrect
		if s.Fixed < 0 || s.Fixed > failing {
			return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: problem %q fixed %d out of range [0,%d] (only failing generations can be repaired)", s.QuestionID, s.Fixed, failing)
		}
		srcTallies[i] = SampleTally{QuestionID: s.QuestionID, Samples: s.SourceSamples, Correct: s.SourceCorrect}
		repTallies[i] = SampleTally{QuestionID: s.QuestionID, Samples: s.SourceSamples, Correct: s.SourceCorrect + s.Fixed}
	}
	srcPass, err := MeanPassAtK(srcTallies, 1)
	if err != nil {
		return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: source pass@1: %w", err)
	}
	repPass, err := MeanPassAtK(repTallies, 1)
	if err != nil {
		return SelfRepairDelta{}, fmt.Errorf("livecodebench self-repair: repaired pass@1: %w", err)
	}
	return SelfRepairDelta{
		Schema:             SelfRepairDeltaSchema,
		Scenario:           ScenarioSelfRepair,
		Model:              model,
		CodegenN:           codegenN,
		RepairN:            SelfRepairRepairN,
		Problems:           len(sources),
		SourcePassAt1:      srcPass,
		RepairedPassAt1:    repPass,
		Delta:              repPass - srcPass,
		EvidenceClass:      EvidenceLocalUngraded,
		ResultClaimAllowed: false,
	}, nil
}
