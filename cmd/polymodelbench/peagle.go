package main

import (
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

const (
	peagleArmName     = "p-eagle-parallel-depth-shape"
	peagleEngine      = "fak-native/internal/model"
	peagleTarget      = "model.NewSynthetic deterministic CPU target"
	peagleDraftSource = "synthetic parallel-depth oracle (shape/cost witness; no trained P-EAGLE weights)"
)

type PEagleShapeBench struct {
	Name                         string                         `json:"name"`
	Engine                       string                         `json:"engine"`
	TargetProvenance             string                         `json:"target_provenance"`
	DraftSourceProvenance        string                         `json:"draft_source_provenance"`
	NumDepths                    int                            `json:"num_depths"`
	LogicalDraftCalls            int                            `json:"logical_draft_calls"`
	TargetVerifyRounds           int                            `json:"target_verify_rounds"`
	SequentialTargetVerifyRounds int                            `json:"sequential_target_verify_rounds"`
	SequentialDraftSteps         int                            `json:"sequential_draft_steps"`
	SequentialDraftStepsAvoided  int                            `json:"sequential_draft_steps_avoided"`
	ProposedTokens               int                            `json:"proposed_tokens"`
	AcceptedTokens               int                            `json:"accepted_tokens"`
	AcceptanceProfile            []polymodel.AcceptancePosition `json:"acceptance_profile"`
	MeanAcceptanceLength         float64                        `json:"mean_acceptance_length"`
	TokenIdenticalToGreedy       bool                           `json:"token_identical_to_greedy"`
	ClaimFence                   string                         `json:"claim_fence"`
}

// measurePEagleShape compares one logical parallel-depth proposal per round with
// today's sequential co-resident drafter. Its known-good synthetic predictions
// isolate call shape and accounting; they do not represent a trained P-EAGLE head.
func measurePEagleShape(target, sequentialDraft *model.Model, prompt []int, n, depths int) PEagleShapeBench {
	want := greedyDecode(target, prompt, n)
	seq := specDecodeModelRun(target, sequentialDraft, prompt, n, depths)
	logicalCalls := 0
	tv := newTargetVerify(target, prompt)
	parallel, err := polymodel.SpecDecode(prompt, func(committed []int) []int {
		logicalCalls++
		start := len(committed) - len(prompt)
		remaining := len(want) - start
		if remaining > depths {
			remaining = depths
		}
		if remaining <= 0 {
			return nil
		}
		return append([]int(nil), want[start:start+remaining]...)
	}, tv.verify, polymodel.SpecDecodeConfig{MaxNewTokens: n, MaxDraft: depths, Rollback: tv.rollback})
	identical := err == nil && losslessEqual(parallel.Output, want, n)
	return PEagleShapeBench{
		Name: peagleArmName, Engine: peagleEngine, TargetProvenance: peagleTarget,
		DraftSourceProvenance: peagleDraftSource, NumDepths: depths,
		LogicalDraftCalls: logicalCalls, TargetVerifyRounds: parallel.Rounds,
		SequentialTargetVerifyRounds: seq.Rounds, SequentialDraftSteps: seq.DraftedTokens,
		SequentialDraftStepsAvoided: parallel.DraftedTokens, ProposedTokens: parallel.DraftedTokens,
		AcceptedTokens: parallel.AcceptedDrafts, AcceptanceProfile: parallel.AcceptanceProfile,
		MeanAcceptanceLength: parallel.MeanAcceptanceLength, TokenIdenticalToGreedy: identical,
		ClaimFence: "CPU-only deterministic shape/cost witness; optional P-EAGLE draft-source candidate under #3197, not a trained-head, Qwen3.8, CUDA, latency, or speedup result",
	}
}
