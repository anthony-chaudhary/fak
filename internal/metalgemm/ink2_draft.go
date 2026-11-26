package metalgemm

// ink2DFShards is the vocabulary chunks the selector's first pass splits a row into.
const ink2DFShards = 32

// DraftAttendSpan is the most keys one DraftAttend query sees.
func DraftAttendSpan(rows, window, cap int) int {
	ctx := cap
	if window > 0 {
		ctx = min(ctx, window-1)
	}
	return ctx + rows
}

// DraftSelectScratch is the floats of scratch DraftSelect needs.
func DraftSelectScratch(rows, k int) int { return 2 * rows * ink2DFShards * k }
