package model

import (
	"errors"
	"reflect"
	"testing"
)

// referenceQwenPositionOracle independently computes the expected M-RoPE 3-axis positions.
func referenceQwenPositionOracle(prefixText int, grid VisionImageGrid, suffixText int, decodeSteps int) ([]MRoPEAxisPosition, []MRoPEAxisPosition) {
	var prefillPositions []MRoPEAxisPosition

	// Prefix text: T=t, H=t, W=t
	t := 0
	for i := 0; i < prefixText; i++ {
		prefillPositions = append(prefillPositions, MRoPEAxisPosition{Temporal: t, Height: t, Width: t})
		t++
	}

	// Image: T=startT, H=startH+r, W=startW+c
	imgStartT := t
	imgStartH := t
	imgStartW := t
	for r := 0; r < grid.GridHeight; r++ {
		for c := 0; c < grid.GridWidth; c++ {
			prefillPositions = append(prefillPositions, MRoPEAxisPosition{
				Temporal: imgStartT,
				Height:   imgStartH + r,
				Width:    imgStartW + c,
			})
		}
	}

	// Suffix text
	postT := imgStartT + 1
	postH := imgStartH + grid.GridHeight
	postW := imgStartW + grid.GridWidth

	for i := 0; i < suffixText; i++ {
		prefillPositions = append(prefillPositions, MRoPEAxisPosition{
			Temporal: postT + i,
			Height:   postH + i,
			Width:    postW + i,
		})
	}

	decodeStartT := postT + suffixText
	decodeStartH := postH + suffixText
	decodeStartW := postW + suffixText

	var decodePositions []MRoPEAxisPosition
	for i := 0; i < decodeSteps; i++ {
		decodePositions = append(decodePositions, MRoPEAxisPosition{
			Temporal: decodeStartT + i,
			Height:   decodeStartH + i,
			Width:    decodeStartW + i,
		})
	}

	return prefillPositions, decodePositions
}

func TestMRoPECursorWitness(t *testing.T) {
	// First witness requirements (#9965):
	// 1. Image+text prefill and two-request decode match independent Qwen position oracle
	// 2. Disabling M-RoPE produces a typed refusal, never silently flat positions
	// 3. Per-request position delta carries into every decode row

	// Two requests with different image shapes and prompt lengths
	reqs := []struct {
		prefixText  int
		grid        VisionImageGrid
		suffixText  int
		decodeSteps int
	}{
		{prefixText: 5, grid: VisionImageGrid{GridHeight: 4, GridWidth: 4}, suffixText: 3, decodeSteps: 4},
		{prefixText: 10, grid: VisionImageGrid{GridHeight: 2, GridWidth: 8}, suffixText: 2, decodeSteps: 5},
	}

	for reqIdx, req := range reqs {
		cursor := NewMRoPECursor(true)

		wantPrefill, wantDecode := referenceQwenPositionOracle(req.prefixText, req.grid, req.suffixText, req.decodeSteps)

		var gotPrefill []MRoPEAxisPosition
		p1, err := cursor.AdvanceText(req.prefixText)
		if err != nil {
			t.Fatalf("req %d advance text: %v", reqIdx, err)
		}
		gotPrefill = append(gotPrefill, p1...)

		p2, err := cursor.AdvanceImage(req.grid)
		if err != nil {
			t.Fatalf("req %d advance image: %v", reqIdx, err)
		}
		gotPrefill = append(gotPrefill, p2...)

		p3, err := cursor.AdvanceText(req.suffixText)
		if err != nil {
			t.Fatalf("req %d advance suffix: %v", reqIdx, err)
		}
		gotPrefill = append(gotPrefill, p3...)

		if !reflect.DeepEqual(gotPrefill, wantPrefill) {
			t.Fatalf("req %d prefill positions mismatch:\ngot =%v\nwant=%v", reqIdx, gotPrefill, wantPrefill)
		}

		// Decode steps
		var gotDecode []MRoPEAxisPosition
		for step := 0; step < req.decodeSteps; step++ {
			dPos, err := cursor.DecodeStep()
			if err != nil {
				t.Fatalf("req %d decode step %d failed: %v", reqIdx, step, err)
			}
			gotDecode = append(gotDecode, dPos)
		}

		if !reflect.DeepEqual(gotDecode, wantDecode) {
			t.Fatalf("req %d decode positions mismatch:\ngot =%v\nwant=%v", reqIdx, gotDecode, wantDecode)
		}

		// Verify 3-axis position delta is non-zero (diverged from flat token count)
		if cursor.DeltaT == 0 && cursor.DeltaH == 0 && cursor.DeltaW == 0 {
			t.Fatalf("req %d delta is zero; image did not shift rotary cursor", reqIdx)
		}
	}

	// 2. Typed refusal when M-RoPE is disabled
	t.Run("typed_refusal_when_disabled", func(t *testing.T) {
		disabledCursor := NewMRoPECursor(false)
		_, _ = disabledCursor.AdvanceText(4)

		_, err := disabledCursor.AdvanceImage(VisionImageGrid{GridHeight: 2, GridWidth: 2})
		if !errors.Is(err, ErrMRoPEDisabled) {
			t.Fatalf("expected ErrMRoPEDisabled, got: %v", err)
		}
	})
}
