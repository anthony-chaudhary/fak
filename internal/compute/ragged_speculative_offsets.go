package compute

import (
	"fmt"
)

// RaggedPromptDraftOffsets records the host-computed boundaries for merged prompt and draft tokens.
type RaggedPromptDraftOffsets struct {
	PromptLengths []int    `json:"prompt_lengths"`
	DraftK        int      `json:"draft_k"`
	NumRequests   int      `json:"num_requests"`
	TotalTokens   int      `json:"total_tokens"`
	MergedOffsets []int    `json:"merged_offsets"`
	PromptSlices  [][2]int `json:"prompt_slices"`
	DraftSlices   [][2]int `json:"draft_slices"`
	D2HEvents     int      `json:"d2h_events"`
}

// ComputeRaggedPromptDraftOffsets calculates segment offsets on the host prior to device dispatch,
// guaranteeing zero device-to-host synchronizations or shape derivations during CUDA graph capture.
func ComputeRaggedPromptDraftOffsets(promptLengths []int, draftK int) (RaggedPromptDraftOffsets, error) {
	if len(promptLengths) == 0 {
		return RaggedPromptDraftOffsets{}, fmt.Errorf("prompt lengths must not be empty")
	}
	if draftK < 0 {
		return RaggedPromptDraftOffsets{}, fmt.Errorf("draftK must be non-negative, got %d", draftK)
	}

	numRequests := len(promptLengths)
	mergedOffsets := make([]int, numRequests)
	promptSlices := make([][2]int, numRequests)
	draftSlices := make([][2]int, numRequests)

	cursor := 0
	for i, pLen := range promptLengths {
		if pLen < 0 {
			return RaggedPromptDraftOffsets{}, fmt.Errorf("request %d has negative prompt length %d", i, pLen)
		}

		mergedOffsets[i] = cursor
		pEnd := cursor + pLen
		promptSlices[i] = [2]int{cursor, pEnd}

		dEnd := pEnd + draftK
		draftSlices[i] = [2]int{pEnd, dEnd}

		cursor = dEnd
	}

	return RaggedPromptDraftOffsets{
		PromptLengths: append([]int(nil), promptLengths...),
		DraftK:        draftK,
		NumRequests:   numRequests,
		TotalTokens:   cursor,
		MergedOffsets: mergedOffsets,
		PromptSlices:  promptSlices,
		DraftSlices:   draftSlices,
		D2HEvents:     0,
	}, nil
}

// MergeRaggedPromptAndDraft combines ragged prompts and fixed-width draft panels into a contiguous
// host array according to host-computed offsets, keeping graph replay sync-free.
func MergeRaggedPromptAndDraft(
	prompts [][]int32,
	drafts [][]int32,
	offsets RaggedPromptDraftOffsets,
) ([]int32, error) {
	if len(prompts) != offsets.NumRequests {
		return nil, fmt.Errorf("prompts count %d != offsets.NumRequests %d", len(prompts), offsets.NumRequests)
	}
	if offsets.DraftK > 0 && len(drafts) != offsets.NumRequests {
		return nil, fmt.Errorf("drafts count %d != offsets.NumRequests %d", len(drafts), offsets.NumRequests)
	}

	merged := make([]int32, offsets.TotalTokens)

	for i := 0; i < offsets.NumRequests; i++ {
		pSlice := offsets.PromptSlices[i]
		pLen := pSlice[1] - pSlice[0]
		if len(prompts[i]) != pLen {
			return nil, fmt.Errorf("request %d prompt length %d != expected %d", i, len(prompts[i]), pLen)
		}
		copy(merged[pSlice[0]:pSlice[1]], prompts[i])

		if offsets.DraftK > 0 {
			dSlice := offsets.DraftSlices[i]
			dLen := dSlice[1] - dSlice[0]
			if len(drafts[i]) != dLen {
				return nil, fmt.Errorf("request %d draft length %d != expected %d", i, len(drafts[i]), dLen)
			}
			copy(merged[dSlice[0]:dSlice[1]], drafts[i])
		}
	}

	return merged, nil
}
