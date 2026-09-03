package compute

import (
	"reflect"
	"testing"
)

func TestRaggedPromptDraftOffsetsWitness(t *testing.T) {
	// First witness requirements (#9959):
	// 1. Ragged prompt lengths [1, 5, 2] crossed with K={0, 1, 4}
	// 2. Exact merged rows and offsets
	// 3. Execution receipt proves strictly 0 D2H events
	// 4. Exact token identity and ordering preservation in merged buffer

	promptLengths := []int{1, 5, 2}
	ks := []int{0, 1, 4}

	expectedMergedOffsets := map[int][]int{
		0: {0, 1, 6},
		1: {0, 2, 8},
		4: {0, 5, 14},
	}

	expectedTotals := map[int]int{
		0: 8,
		1: 11,
		4: 20,
	}

	for _, k := range ks {
		offsets, err := ComputeRaggedPromptDraftOffsets(promptLengths, k)
		if err != nil {
			t.Fatalf("k=%d: ComputeRaggedPromptDraftOffsets failed: %v", k, err)
		}

		// 3. Verify exactly 0 D2H events
		if offsets.D2HEvents != 0 {
			t.Fatalf("k=%d: expected 0 D2H events, got %d", k, offsets.D2HEvents)
		}

		// 1 & 2. Verify exact offsets and totals
		wantOffsets := expectedMergedOffsets[k]
		if !reflect.DeepEqual(offsets.MergedOffsets, wantOffsets) {
			t.Fatalf("k=%d: merged offsets = %v, want %v", k, offsets.MergedOffsets, wantOffsets)
		}
		wantTotal := expectedTotals[k]
		if offsets.TotalTokens != wantTotal {
			t.Fatalf("k=%d: total tokens = %d, want %d", k, offsets.TotalTokens, wantTotal)
		}

		// Generate sample prompts and drafts
		prompts := [][]int32{
			{101},
			{201, 202, 203, 204, 205},
			{301, 302},
		}

		var drafts [][]int32
		if k > 0 {
			drafts = make([][]int32, 3)
			for req := 0; req < 3; req++ {
				drafts[req] = make([]int32, k)
				for d := 0; d < k; d++ {
					drafts[req][d] = int32((req+1)*1000 + d)
				}
			}
		}

		merged, err := MergeRaggedPromptAndDraft(prompts, drafts, offsets)
		if err != nil {
			t.Fatalf("k=%d: MergeRaggedPromptAndDraft failed: %v", k, err)
		}
		if len(merged) != wantTotal {
			t.Fatalf("k=%d: merged length %d != expected %d", k, len(merged), wantTotal)
		}

		// 4. Verify token ordering and boundaries for each request
		for req := 0; req < 3; req++ {
			pSlice := offsets.PromptSlices[req]
			gotP := merged[pSlice[0]:pSlice[1]]
			if !reflect.DeepEqual(gotP, prompts[req]) {
				t.Fatalf("k=%d req=%d prompt tokens mismatch: got %v, want %v", k, req, gotP, prompts[req])
			}

			if k > 0 {
				dSlice := offsets.DraftSlices[req]
				gotD := merged[dSlice[0]:dSlice[1]]
				if !reflect.DeepEqual(gotD, drafts[req]) {
					t.Fatalf("k=%d req=%d draft tokens mismatch: got %v, want %v", k, req, gotD, drafts[req])
				}
			}
		}
	}
}

func TestRaggedPromptDraftOffsetsFailClosed(t *testing.T) {
	// Empty prompt lengths
	if _, err := ComputeRaggedPromptDraftOffsets(nil, 2); err == nil {
		t.Fatal("expected error on empty prompt lengths")
	}

	// Negative K
	if _, err := ComputeRaggedPromptDraftOffsets([]int{1, 2}, -1); err == nil {
		t.Fatal("expected error on negative K")
	}

	// Negative prompt length
	if _, err := ComputeRaggedPromptDraftOffsets([]int{1, -3}, 2); err == nil {
		t.Fatal("expected error on negative prompt length")
	}
}
