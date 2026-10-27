package model

// qwen35PrefillGateDisjoint applies the ordinary full-attention output gate.
// The builtin caller owns a fresh attention output and an independent gate
// allocation populated by the earlier joined Q/gate split. No callback can
// retain either panel, and attention workers have joined before this call.
// This private helper requires disjoint inputs; it is not a generic aliasing
// operation. Every element retains the original sigmoidf/math.Exp expression.
func qwen35PrefillGateDisjoint(out, gate []float32, rows, width int) {
	if rows <= 0 || width <= 0 || len(out)%width != 0 || len(out)/width != rows || len(gate) != len(out) {
		// Preserve the original order for malformed geometry, including all
		// prefix writes before a short gate eventually causes a bounds panic.
		for i := range out {
			out[i] *= sigmoidf(gate[i])
		}
		return
	}
	// Rows have no reduction or recurrence dependency. Use the existing work
	// threshold and worker budget; no tuned threshold or numeric scratch.
	parForRange(rows, len(out), func(lo, hi int) {
		for row := lo; row < hi; row++ {
			start := row * width // admitted by the division-first shape check
			for i := start; i < start+width; i++ {
				out[i] *= sigmoidf(gate[i])
			}
		}
	})
}
