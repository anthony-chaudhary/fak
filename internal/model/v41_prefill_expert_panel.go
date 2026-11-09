package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// v41_prefill_expert_panel.go -- the DRC-K17 batched routed-expert prefill
// panel seam.
//
// A multi-token prefill makes seq*top-k routed picks; today each pick dispatches
// its own gate/up and down GEMM through the expert ring (the measured 586-token
// H200 prefill issues 431,040 gate/up + 431,040 down dispatches, ~45% of prefill
// wall). The llama.cpp mul_mat_vec_q_moe shape collapses that: group a chunk's
// picks by expert and contract every row assigned to an expert through ONE
// batched GEMM per projection. This file lands the CPU-witnessable half of that
// collapse:
//
//   - the deterministic chunk plan over the existing expert-major groups
//     (v41PlanExpertGroups), so a group's rows are offered in bounded panels of
//     2..compute.V41PrefillExpertPanelMaxRows rows and a trailing singleton keeps
//     the historical per-row stream;
//   - the scalar oracle (v41PrefillExpertPanelOracle) that proves the grouped
//     panel is bit-exact to the token-major per-pick stream on representative
//     shapes;
//   - the typed seam (v41ExpertPanelFunc + compute.V41PrefillExpertPanelBackend)
//     that admits the CUDA kernel. It declines cleanly on CUDA until its own
//     qualification build supplies V41PrefillExpertPanelQualified, mirroring how
//     v41RoutedLayerQualified gates the decode layer. No backend implements the
//     contract yet, so the bind is inert and the grouped contraction keeps its
//     per-row stream byte-for-byte.

// v41ExpertPanelOutcome is the closed result vocabulary of one
// v41ExpertPanelFunc call. handled-success returns the chunk's weighted H-wide
// rows in the offer order; declined leaves the chunk to the historical per-row
// stream; handled-error is a SELECTED execution failure and must surface, never
// be swallowed as a decline.
type v41ExpertPanelOutcome uint8

const (
	// v41PanelDeclined: the backend did not admit this expert panel (no
	// qualified kernel for the resolved encodings or extent), so the caller
	// falls back to the per-row stream byte-for-byte.
	v41PanelDeclined v41ExpertPanelOutcome = iota
	// v41PanelHandled: the three batched projections ran on the backend and the
	// returned slice holds the weighted H-wide expert output per row.
	v41PanelHandled
	// v41PanelError: a selected device execution failed; it must remain visible.
	v41PanelError
)

// v41ExpertPanelFunc is the optional batched prefill-expert panel operation. It
// receives the routed expert's layer stem, the chunk's rows in original
// (token, slot) order, each row's normalized input and route weight, and reports
// one of the closed v41ExpertPanelOutcome values. On v41PanelHandled it returns
// one weighted H-wide expert output per row, sized to the hidden width: the
// full-mode contract, matching the cohort and id-panel consumers that store the
// result into the unweighted table whose replay sums it without re-weighting.
type v41ExpertPanelFunc func(layer int, stem string, rows []v41ExpertRow, inputs [][]float32, weights []float32) ([][]float32, v41ExpertPanelOutcome, error)

// v41PrefillExpertPanelQualified separates method-set availability from
// qualification for serving. Existing backends retain their admission
// contract; CUDA must explicitly qualify the batched prefill panel path, not
// merely acquire the panel methods during kernel development. No environment
// override can substitute for that qualification. Backends may explicitly
// decline too.
func v41PrefillExpertPanelQualified(backend compute.Backend) bool {
	if backend == nil {
		return false
	}
	if qualified, ok := backend.(interface{ V41PrefillExpertPanelQualified() bool }); ok {
		return qualified.V41PrefillExpertPanelQualified()
	}
	return backend.Name() != "cuda"
}

// v41NextPrefillExpertPanelChunk returns the next bounded panel window
// [first,last) over one expert group's rows under the prefill panel rule: at
// most maxRows rows, and a trailing singleton is pulled into the window rather
// than dispatched alone, because a one-row panel pays the batched dispatch for
// a GEMV. The pull-back only fires when the window still holds at least two
// rows after it. It reports done when fewer than two rows remain: those keep
// the historical per-row stream.
func v41NextPrefillExpertPanelChunk(rows []v41ExpertRow, first, maxRows int) (last int, done bool) {
	remaining := len(rows) - first
	if remaining < 2 {
		return len(rows), true
	}
	last = first + maxRows
	if last > len(rows) {
		last = len(rows)
	}
	if len(rows)-last == 1 && last-first >= 3 {
		last--
	}
	return last, false
}

// v41PlanPrefillExpertPanelChunks splits one expert group's rows into the
// bounded panels the batched prefill seam offers them in. It is the pure
// planner the oracle and the production offer point share: every offered row
// appears in exactly one chunk, in original (token, slot) order, each chunk
// holds 2..maxRows rows, and a group with fewer than two rows plans nothing so
// its picks stay on the per-row stream. Rows left behind by the trailing-singleton
// rule are reported absent, never duplicated.
func v41PlanPrefillExpertPanelChunks(rows []v41ExpertRow, maxRows int) [][]v41ExpertRow {
	if len(rows) < 2 || maxRows < 2 {
		return nil
	}
	var chunks [][]v41ExpertRow
	for first := 0; first < len(rows); {
		last, done := v41NextPrefillExpertPanelChunk(rows, first, maxRows)
		if done {
			break
		}
		chunks = append(chunks, rows[first:last])
		first = last
	}
	return chunks
}

// v41PrefillExpertPanelOracle is the scalar Go reference for the batched
// prefill expert panel: it plans the expert-major groups (v41PlanExpertGroups),
// chunks each group's rows under the panel bound (v41PlanPrefillExpertPanelChunks),
// contracts every row through the same host SwiGLU twin the grouped arm uses
// (v41SwiGLUParallel), and replays the per-token weighted sums in ORIGINAL slot
// order through the shared accumulation primitive (ffn.AddScaled). It performs
// no device operation, so it is the CPU oracle a batched panel must match
// bit-for-bit: the panel reorders the READS, never a token's arithmetic.
func v41PrefillExpertPanelOracle(xnByToken [][]float32, perTokenPicks [][]routePick, cfg Config, triple func(expert int) ([]float32, []float32, []float32, error)) ([][]float32, error) {
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	seq := len(perTokenPicks)
	out := make([][]float32, seq)
	for t := range out {
		out[t] = make([]float32, H)
	}
	slots := 0
	for _, picks := range perTokenPicks {
		slots += len(picks)
	}
	unweighted := make([][][]float32, seq)
	outputs := make([][]float32, slots)
	at := 0
	for t, picks := range perTokenPicks {
		end := at + len(picks)
		unweighted[t] = outputs[at:end:end]
		at = end
	}
	contract := func(r v41ExpertRow, w1, w3, w2 []float32) {
		y := v41SwiGLUParallel(w1, w3, w2, xnByToken[r.Token], I, H, cfg)
		unweighted[r.Token][r.Slot] = y
	}
	for _, g := range v41PlanExpertGroups(perTokenPicks) {
		w1, w3, w2, err := triple(g.Expert)
		if err != nil {
			return nil, err
		}
		chunks := v41PlanPrefillExpertPanelChunks(g.Rows, compute.V41PrefillExpertPanelMaxRows)
		for _, chunk := range chunks {
			for _, r := range chunk {
				contract(r, w1, w3, w2)
			}
		}
		// Rows the chunker left behind (a group of one) keep the per-row stream.
		if len(chunks) == 0 {
			for _, r := range g.Rows {
				contract(r, w1, w3, w2)
			}
		}
	}
	for t, picks := range perTokenPicks {
		dst := out[t]
		for slot, pick := range picks {
			y := unweighted[t][slot]
			if y == nil {
				return nil, fmt.Errorf("v41 prefill expert panel oracle: pick (token=%d,slot=%d,expert=%d) was not materialized", t, slot, pick.expert)
			}
			if err := ffn.AddScaled(dst, y, pick.weight); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// v41ExpertPanelFunc binds the optional batched prefill-expert panel onto the
// session backend, or returns nil when no backend admits it. The bind requires
// the backend to implement compute.V41PrefillExpertPanelBackend AND to qualify
// the panel for serving (v41PrefillExpertPanelQualified): a CUDA build without
// the panel archive lacks the qualifier method and declines, exactly as
// v41RoutedLayerQualified gates the decode layer. Today no backend implements
// the contract, so the bind is inert and the grouped contraction keeps its
// per-row stream byte-for-byte; the CUDA adapter joins behind its own
// qualification build (DRC-K17 leaf 2).
func (s *Session) v41ExpertPanelFunc() v41ExpertPanelFunc {
	if s == nil || s.Backend == nil || s.M == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	panel, ok := s.Backend.(compute.V41PrefillExpertPanelBackend)
	if !ok || !v41PrefillExpertPanelQualified(s.Backend) {
		return nil
	}
	cfg := s.M.Cfg
	full, geometryErr := v41ForwardGeometry(cfg)
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	limit := float32(cfg.SwigluLimit)
	return func(layer int, stem string, rows []v41ExpertRow, inputs [][]float32, weights []float32) (result [][]float32, outcome v41ExpertPanelOutcome, cause error) {
		s.ensureOpenBackendSession()
		stage := "admission"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-expert-panel", Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		defer func() {
			if p := recover(); p != nil {
				if err, ok := p.(error); ok {
					var closed *BackendForwardOperationError
					var backend *compute.BackendError
					if errors.As(err, &closed) {
						panic(p)
					}
					if errors.As(err, &backend) {
						result, outcome, cause = nil, v41PanelError, closeFailure(err)
						return
					}
				}
				if converted, ok := compute.ConvertCUDAPanic(p, "", ""); ok {
					if original, ok := p.(error); ok {
						converted = original
					}
					result, outcome, cause = nil, v41PanelError, closeFailure(converted)
					return
				}
				err, ok := p.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", p)
				}
				closeFailure(err)
				panic(p)
			}
		}()
		if geometryErr != nil {
			return nil, v41PanelError, geometryErr
		}
		n := len(rows)
		if !full || n < 2 || n > compute.V41PrefillExpertPanelMaxRows || len(inputs) != n || len(weights) != n {
			return nil, v41PanelDeclined, nil
		}
		for r := range rows {
			if len(inputs[r]) != H || !finite32(weights[r]) || !finite32(limit) {
				return nil, v41PanelError, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
			}
		}
		gateName := layerName(layer, stem+".w1.weight")
		upName := layerName(layer, stem+".w3.weight")
		downName := layerName(layer, stem+".w2.weight")
		if s.M.has(downName[:len(downName)-len("weight")] + "bias") {
			return nil, v41PanelDeclined, nil
		}
		gateW, releaseGate, upW, releaseUp, _, admitted := expertInputDeviceAdmitted(s, gateName, upName, inputs[0], H)
		if !admitted {
			return nil, v41PanelDeclined, nil
		}
		defer releaseGate()
		defer releaseUp()
		downW, releaseDown, admitted := s.q4kExpertDownDeviceWeight(downName)
		if !admitted {
			return nil, v41PanelDeclined, nil
		}
		defer releaseDown()
		if !panel.SupportsV41PrefillExpertPanel(gateW, n) || !panel.SupportsV41PrefillExpertPanel(upW, n) || !panel.SupportsV41PrefillExpertPanel(downW, n) {
			return nil, v41PanelDeclined, nil
		}
		publish := s.v41Forward.gatherOps.bf16PublishFunc()
		// Pack the activation panel: each row is published exactly as the
		// per-pick path publishes its gate/up input, so the batched GEMMs
		// consume the same operands the per-pick GEMVs consumed.
		packed := make([]float32, n*H)
		for r, in := range inputs {
			published, err := v41RoutedBF16Publish(layer, "gate/up", in, H, publish)
			if err != nil {
				return nil, v41PanelError, closeFailure(err)
			}
			copy(packed[r*H:(r+1)*H], published)
		}
		xp := s.uploadHostF32([]int{n * H}, packed, compute.MemoryActivation, "V4.1 prefill expert panel activations")
		defer s.Backend.Free(xp)
		stage = "gate/up"
		gateT, err := panel.V41PrefillExpertPanel(gateW, xp, n)
		if err != nil {
			return nil, v41PanelError, closeFailure(err)
		}
		defer s.Backend.Free(gateT)
		upT, err := panel.V41PrefillExpertPanel(upW, xp, n)
		if err != nil {
			return nil, v41PanelError, closeFailure(err)
		}
		defer s.Backend.Free(upT)
		gateRaw := s.Backend.Read(gateT)
		if len(gateRaw) != n*I {
			return nil, v41PanelError, closeFailure(v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult))
		}
		upRaw := s.Backend.Read(upT)
		if len(upRaw) != n*I {
			return nil, v41PanelError, closeFailure(v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult))
		}
		// Per row, in offer order: BF16-published projections, the configured
		// clamp, SwiGLU, the route weight, and the down-input publication --
		// byte-for-byte the per-pick sequence, batched only in the GEMMs.
		operands := make([]float32, n*I)
		for r := 0; r < n; r++ {
			grow, urow := gateRaw[r*I:(r+1)*I], upRaw[r*I:(r+1)*I]
			gate, err := v41RoutedBF16PublishInput(layer, "gate/up", grow, I, publish, true)
			if err != nil {
				return nil, v41PanelError, closeFailure(err)
			}
			up, err := v41RoutedBF16PublishInput(layer, "gate/up", urow, I, publish, true)
			if err != nil {
				return nil, v41PanelError, closeFailure(err)
			}
			clampSwiGLUProjections(gate, up, limit)
			weighted := v41PrefillPanelWeightedRow(operands, r, I, publish)
			for i := range weighted {
				fused := silu(gate[i]) * up[i]
				if !finite32(fused) {
					return nil, v41PanelError, closeFailure(v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult))
				}
				weighted[i] = float32(fused * weights[r])
			}
			operand, err := v41RoutedBF16PublishInput(layer, "down", weighted, I, publish, true)
			if err != nil {
				return nil, v41PanelError, closeFailure(err)
			}
			copy(operands[r*I:(r+1)*I], operand)
		}
		op := s.uploadHostF32([]int{n * I}, operands, compute.MemoryActivation, "V4.1 prefill expert panel down operands")
		defer s.Backend.Free(op)
		stage = "down"
		downT, err := panel.V41PrefillExpertPanel(downW, op, n)
		if err != nil {
			return nil, v41PanelError, closeFailure(err)
		}
		defer s.Backend.Free(downT)
		downRaw := s.Backend.Read(downT)
		if len(downRaw) != n*H {
			return nil, v41PanelError, closeFailure(v41ExpertOperationErr(layer, "down", errV41ExpertResult))
		}
		result = make([][]float32, n)
		for r := 0; r < n; r++ {
			row, err := v41RoutedBF16Publish(layer, "down", downRaw[r*H:(r+1)*H], H, publish)
			if err != nil {
				return nil, v41PanelError, closeFailure(err)
			}
			result[r] = row
		}
		return result, v41PanelHandled, nil
	}
}

// Host-only publication may round a private packed row in place. Selected
// publication keeps its historical independent input allocation and ownership.
func v41PrefillPanelWeightedRow(operands []float32, row, width int, publish v41BF16BoundaryFunc) []float32 {
	if publish != nil {
		return make([]float32, width)
	}
	first, last := row*width, (row+1)*width
	return operands[first:last:last]
}
