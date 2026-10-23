package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// v41Q2KTokenParityLeaf names the dense projections whose multi-row device
// panel must reproduce the per-token MatMul bits. On Vulkan a one-row Q2_K
// MatMul runs q2k_matvec (16 lanes per superblock, LDS tree reduction) while
// BatchedMatMul runs q2k_matmul (one thread per row, sequential sum); the f32
// order differs and changed greedy text when attn.wkv moved onto the panel
// (Ref #12640). attn.wq_a/attn.wq_b were already batched before that change
// and the greedy goldens are pinned on their batched order, so they are not
// listed; widening this set is a deliberate re-baseline.
func v41Q2KTokenParityLeaf(leaf string) bool {
	return leaf == "attn.wkv.weight"
}

// v41Q2KTokenParityRoute selects the bit-parity route for a multi-row Q2_K
// panel of a parity leaf: the token-panel kernel, which keeps q2k_matvec's
// per-row arithmetic and reduction tree, or a per-row MatMul fallback when the
// backend does not admit the panel. Both false means the ordinary route.
func v41Q2KTokenParityRoute(be compute.Backend, weight compute.Tensor, leaf string, rows int) (tile compute.V41Q2KTokenPanelBackend, perRow bool) {
	if rows <= 1 || weight.Dtype != compute.Q2_K || !v41Q2KTokenParityLeaf(leaf) {
		return nil, false
	}
	if op, ok := be.(compute.V41Q2KTokenPanelBackend); ok && op.SupportsV41Q2KTokenPanel(weight, rows) {
		return op, false
	}
	return nil, true
}

// v41Q2KMatMulRows projects each row through the one-row MatMul and returns the
// stitched [rows*out] panel as a fresh device tensor owned by the caller.
func (s *Session) v41Q2KMatMulRows(weight compute.Tensor, input []float32, rows, in, out int) compute.Tensor {
	values := make([]float32, 0, rows*out)
	for row := 0; row < rows; row++ {
		x := s.uploadHostF32([]int{in}, input[row*in:(row+1)*in], compute.MemoryActivation, "V4.1 dense projection parity row")
		y := s.Backend.MatMul(weight, x)
		values = append(values, s.Backend.Read(y)...)
		s.Backend.Free(y)
		s.Backend.Free(x)
	}
	return s.uploadHostF32([]int{len(values)}, values, compute.MemoryActivation, "V4.1 dense projection parity panel")
}
