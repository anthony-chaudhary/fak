package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedPanelWeights struct {
	a      []compute.Tensor
	b      compute.Tensor
	aDtype compute.Dtype
	bDtype compute.Dtype
}

type v41GroupedPanelStatus struct {
	status compute.V41DeferredStatus
	stage  int
	group  int
	width  int
}

// v41GroupedOutputPanel contracts a whole prompt's attention outputs in bounded
// row tiles. Capability refusal happens before activation work; after selection
// every failure retires the owner and is never replayed by the scalar arm.
func (s *Session) v41GroupedOutputPanel(layer int, panel []float32, rows, heads, headDim, groups, rank, dim int) ([][]float32, v41DenseProjectionOutcome, error) {
	if s == nil || s.M == nil || s.Backend == nil || rows <= 1 || heads <= 0 || headDim <= 0 || groups <= 0 || rank <= 0 || dim <= 0 || heads%groups != 0 {
		return nil, v41GroupedPanelDecline(layer, rows, "shape"), nil
	}
	totalIn, ok := checkedMulInt(heads, headDim)
	panelN, panelOK := checkedMulInt(rows, totalIn)
	joined, joinedOK := checkedMulInt(groups, rank)
	if !ok || !panelOK || !joinedOK || len(panel) != panelN {
		return nil, v41GroupedPanelDecline(layer, rows, "panel_size"), nil
	}
	for _, v := range panel {
		if !finite32(v) {
			return nil, v41GroupedPanelDecline(layer, rows, "non_finite"), nil
		}
	}
	rowOps, rowOK := s.Backend.(compute.V41RowLogBackend)
	pub, pubOK := s.Backend.(compute.V41BF16PublishBackend)
	deferred, deferredOK := s.Backend.(compute.V41DeferredStatusBackend)
	if !rowOK || !rowOps.V41RowLogAvailable() || !pubOK || !deferredOK {
		return nil, v41GroupedPanelDecline(layer, rows, "backend_caps"), nil
	}
	groupIn := totalIn / groups
	aName, bName := layerName(layer, "attn.wo_a.weight"), layerName(layer, "attn.wo_b.weight")
	ar, ac, ap := s.M.residentShape(aName)
	br, bc, bp := s.M.residentShape(bName)
	if !ap || !bp || !((ar == joined && ac == groupIn) || (ar == rank && ac == totalIn)) || br != dim || bc != joined {
		return nil, v41GroupedPanelDecline(layer, rows, "resident_shape"), nil
	}
	aSource, aOK, aErr := s.v41GroupedWeight(aName, groupIn)
	bSource, bOK, bErr := s.v41GroupedWeight(bName, joined)
	if aErr != nil || bErr != nil {
		return nil, v41ProjectionError, errors.Join(aErr, bErr)
	}
	if !aOK || !bOK {
		return nil, v41GroupedPanelDecline(layer, rows, "weight_unsupported"), nil
	}
	tiles := v41GroupedPanelTiles(rows)
	for _, n := range tiles {
		for _, count := range []int{n * groupIn, n * rank, n * dim} {
			if !pub.SupportsV41BF16Publish(count) {
				return nil, v41GroupedPanelDecline(layer, rows, "publish_count"), nil
			}
			if _, supported := v41DeferredBF16Publisher(s.Backend, count); !supported {
				return nil, v41GroupedPanelDecline(layer, rows, "deferred_count"), nil
			}
		}
	}

	weights, err := v41SelectedDeviceCall(s, layer, "v41-grouped-output-panel-weights", func(stage *string) (v41GroupedPanelWeights, error) {
		*stage = "weight upload"
		w := v41GroupedPanelWeights{a: make([]compute.Tensor, groups), aDtype: aSource.dtype, bDtype: bSource.dtype}
		groupElems := rank * groupIn
		for g := range w.a {
			key := fmt.Sprintf("v41-grouped:%s:%v:%d:%d:%d", aName, aSource.dtype, g, rank, groupIn)
			var cause error
			w.a[g], cause = s.v41GroupedPanelWeight(key, aSource, g*groupElems, rank, groupIn)
			if cause != nil {
				return v41GroupedPanelWeights{}, cause
			}
		}
		key := fmt.Sprintf("v41-grouped:%s:%v:%d:%d", bName, bSource.dtype, dim, joined)
		var cause error
		w.b, cause = s.v41GroupedPanelWeight(key, bSource, 0, dim, joined)
		if cause != nil {
			return v41GroupedPanelWeights{}, cause
		}
		return w, nil
	})
	if err != nil {
		return nil, v41ProjectionError, err
	}
	for _, n := range tiles {
		if n == 1 {
			continue
		}
		for _, w := range weights.a {
			if !v41GroupedPanelWeightSupports(s.Backend, w, n) {
				return nil, v41GroupedPanelDecline(layer, rows, "tile_weight_a"), nil
			}
		}
		if !v41GroupedPanelWeightSupports(s.Backend, weights.b, n) {
			return nil, v41GroupedPanelDecline(layer, rows, "tile_weight_b"), nil
		}
	}
	// Widen only after the historical tiles have passed admission. Each larger
	// chunk must be supported by both weight families and every publication;
	// older Q2/Q3 modules therefore keep their original 8/4/2/1 path. The
	// backend owns inner kernel tiling and per-row reduction order.
	tiles = v41GroupedPanelChunks(rows, func(n int) bool {
		for _, width := range []int{groupIn, rank, dim, joined} {
			count, valid := checkedMulInt(n, width)
			if !valid || count > math.MaxInt32/4 || !pub.SupportsV41BF16Publish(count) {
				return false
			}
			if _, supported := v41DeferredBF16Publisher(s.Backend, count); !supported {
				return false
			}
		}
		spanCount, valid := checkedMulInt(n, groups)
		if !valid || spanCount > math.MaxInt32 || compute.ValidateV41RowLogAlloc(spanCount, rank) != nil {
			return false
		}
		for _, w := range weights.a {
			if !v41GroupedPanelWeightSupports(s.Backend, w, n) {
				return false
			}
		}
		return v41GroupedPanelWeightSupports(s.Backend, weights.b, n)
	})

	s.ensureOpenBackendSession()
	opened := s.M.v41NowNanos()
	completed := 0
	var stats v41GroupedResidentStats
	defer func() {
		s.M.v41NoteGroupedOutput(1, 0, completed, 0, stats.calls, stats.upload, stats.readback, 0, opened)
	}()
	result, err := v41SelectedDeviceCall(s, layer, "v41-grouped-output-panel", func(stage *string) ([][]float32, error) {
		out := make([][]float32, 0, rows)
		at := 0
		for _, n := range tiles {
			part, cause := s.v41GroupedOutputTile(rowOps, deferred, panel[at*totalIn:(at+n)*totalIn], n, groupIn, groups, rank, dim, weights, stage, &stats)
			if cause != nil {
				return nil, cause
			}
			out = append(out, part...)
			at += n
		}
		return out, nil
	})
	if err != nil {
		return nil, v41ProjectionError, err
	}
	completed = rows
	return result, v41ProjectionHandled, nil
}

// Resolve the immutable handle before materializing or validating its payload.
// Both session hits and model-owned borrowed hits have already staged these bytes.
func (s *Session) v41GroupedPanelWeight(key string, source v41GroupedWeight, offset, out, in int) (weight compute.Tensor, err error) {
	var payload error
	var recovered any
	completed := false
	func() {
		defer func() { recovered = recover() }()
		weight = s.cachedImmutableWeight(key, key, func() compute.Tensor {
			host, cause := source.host(offset, out, in)
			if cause != nil {
				// The staging callback cannot return an error. Unwind its cache
				// lock without publishing a failed handle, then return the original
				// error to the selected-call boundary. Backend panics pass through.
				payload = cause
				panic(&payload)
			}
			return s.Backend.Upload(host, source.dtype)
		})
		completed = true
	}()
	if !completed {
		if recovered == &payload {
			return compute.Tensor{}, payload
		}
		// Only the private error sentinel is converted. A legacy nil panic
		// must reach the selected boundary; Goexit never returns from capture.
		panic(recovered)
	}
	return weight, nil
}

func v41GroupedPanelTiles(rows int) []int {
	var out []int
	for rows > 0 {
		n := 8
		if rows < n {
			switch {
			case rows >= 4:
				n = 4
			case rows >= 2:
				n = 2
			default:
				n = 1
			}
		}
		out = append(out, n)
		rows -= n
	}
	return out
}

func v41GroupedPanelWeightSupports(be compute.Backend, w compute.Tensor, rows int) bool {
	switch w.Dtype {
	case compute.Q2_K:
		op, ok := be.(compute.V41Q2KTokenPanelBackend)
		return ok && op.SupportsV41Q2KTokenPanel(w, rows)
	case compute.Q3_K:
		op, ok := be.(compute.V41Q3KTokenPanelBackend)
		return ok && op.SupportsV41Q3KTokenPanel(w, rows)
	case compute.F32:
		// The row-log device backend's F32 kernel assigns one workgroup per
		// token; P changes its grid, not the per-row serial reduction. Other
		// quantized GEMMs stay bounded until their wider dispatch is qualified.
		return true
	default:
		return rows <= 8
	}
}

func (s *Session) v41GroupedOutputTile(rows compute.V41RowLogBackend, deferred compute.V41DeferredStatusBackend, input []float32, count, groupIn, groups, rank, dim int, weights v41GroupedPanelWeights, stage *string, stats *v41GroupedResidentStats) (result [][]float32, err error) {
	var owned []compute.Tensor
	var pending []v41GroupedPanelStatus
	borrowed := append([]compute.Tensor(nil), weights.a...)
	borrowed = append(borrowed, weights.b)
	own := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != s.Backend {
			return false
		}
		for _, prior := range owned {
			if prior.Buf() == t.Buf() {
				return false
			}
		}
		for _, status := range pending {
			if status.status.Record.Buf() == t.Buf() {
				return false
			}
		}
		for _, prior := range borrowed {
			if prior.Buf() == t.Buf() {
				return false
			}
		}
		owned = append(owned, t)
		return true
	}
	valid := func(t compute.Tensor, n int) bool {
		return t.Backend() == s.Backend && t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && t.Numel() == n
	}
	read := func(outputs []compute.Tensor) ([][]float32, error) {
		status := make([]compute.V41DeferredStatus, len(pending))
		for i := range pending {
			status[i] = pending[i].status
		}
		meta := pending
		pending = nil
		verdicts, values, cause := deferred.V41ReadDeferred(status, outputs)
		if cause != nil {
			return nil, cause
		}
		if len(verdicts) != len(meta) || len(values) != len(outputs) {
			return nil, errV41BF16BoundaryResult
		}
		if fault := v41GroupedPanelFirstFault(verdicts, meta, count, groups); fault != nil {
			return nil, fault
		}
		return values, nil
	}
	var primary any
	var primaryPresent, bodyReturned bool
	defer func() {
		p := primary
		present := bodyReturned && primaryPresent
		if len(pending) != 0 {
			_, drain := read(nil)
			if drain != nil {
				err = errors.Join(drain, err)
				if pe, ok := p.(error); ok {
					err = errors.Join(err, pe)
					present = false
				}
			}
		}
		for i := len(owned) - 1; i >= 0; i-- {
			s.Backend.Free(owned[i])
		}
		if present {
			panic(p)
		}
	}()
	func() {
		complete := false
		defer func() {
			if !complete {
				primary, primaryPresent = recover(), true
			}
		}()
		result, err = func() ([][]float32, error) {
			publish := func(x compute.Tensor, n, logicalStage, group int) (compute.Tensor, error) {
				*stage = "BF16 publication"
				x.Shape = []int{n}
				p, status, cause := deferred.V41BF16PublishDeferred(x, compute.V41BF16PublishGroupedOutputProjection)
				fresh := own(p)
				recordFresh := own(status.Record)
				recordValid := recordFresh && status.Kind == compute.V41DeferredBF16Publish && status.Count == n && status.Site == compute.V41BF16PublishGroupedOutputProjection && status.Records <= 1 && valid(status.Record, 4)
				if recordValid {
					owned = owned[:len(owned)-1]
					pending = append(pending, v41GroupedPanelStatus{status: status, stage: logicalStage, group: group, width: n / count})
				}
				if cause != nil {
					return compute.Tensor{}, cause
				}
				if !fresh || !valid(p, n) || !recordValid {
					return compute.Tensor{}, errV41BF16BoundaryResult
				}
				return p, nil
			}
			project := func(w, x compute.Tensor, n int) compute.Tensor {
				stats.calls++
				if count == 1 {
					x.Shape = []int{n}
					return s.Backend.MatMul(w, x)
				}
				x.Shape = []int{count, n}
				return s.Backend.BatchedMatMul(w, x, count)
			}

			aPanels := make([]compute.Tensor, groups)
			for g := 0; g < groups; g++ {
				grouped := make([]float32, count*groupIn)
				for t := 0; t < count; t++ {
					copy(grouped[t*groupIn:(t+1)*groupIn], input[t*groups*groupIn+g*groupIn:t*groups*groupIn+(g+1)*groupIn])
				}
				*stage = "activation upload"
				x := s.uploadHostF32([]int{count, groupIn}, grouped, compute.MemoryActivation, "V4.1 grouped output panel")
				stats.upload += int64(len(grouped)) * 4
				if !own(x) || !valid(x, count*groupIn) {
					return nil, errV41ProjectionResult
				}
				p, cause := publish(x, count*groupIn, 0, g)
				if cause != nil {
					return nil, cause
				}
				*stage = "wo_a"
				y := project(weights.a[g], p, groupIn)
				if !own(y) || !valid(y, count*rank) {
					return nil, errV41ProjectionResult
				}
				p, cause = publish(y, count*rank, 1+g, g)
				if cause != nil {
					return nil, cause
				}
				p.Shape = []int{count, rank}
				aPanels[g] = p
			}
			spans := make([]compute.V41RowSpan, 0, count*groups)
			for t := 0; t < count; t++ {
				for g := 0; g < groups; g++ {
					spans = append(spans, compute.V41RowSpan{Src: aPanels[g], Row: t, Rows: 1})
				}
			}
			*stage = "device concat"
			joined, cause := rows.V41RowLogGather(spans, rank)
			fresh := own(joined)
			if cause != nil {
				return nil, cause
			}
			if !fresh || !valid(joined, count*groups*rank) {
				return nil, errV41ProjectionResult
			}
			joined.Shape = []int{count, groups * rank}
			*stage = "wo_b"
			y := project(weights.b, joined, groups*rank)
			if !own(y) || !valid(y, count*dim) {
				return nil, errV41ProjectionResult
			}
			p, cause := publish(y, count*dim, 1+groups, -1)
			if cause != nil {
				return nil, cause
			}
			*stage = "readback"
			values, cause := read([]compute.Tensor{p})
			if cause != nil {
				return nil, cause
			}
			if len(values[0]) != count*dim || !v41BF16Widened(values[0]) {
				return nil, errV41ProjectionResult
			}
			stats.readback += int64(count*dim) * 4
			result = make([][]float32, count)
			for t := range result {
				result[t] = append([]float32(nil), values[0][t*dim:(t+1)*dim]...)
			}
			return result, nil
		}()
		complete = true
	}()
	bodyReturned = true
	return
}

func v41GroupedPanelFirstFault(verdicts []error, meta []v41GroupedPanelStatus, rows, groups int) error {
	bestToken, bestStage, bestElement := math.MaxInt, math.MaxInt, math.MaxInt
	var best error
	for i, verdict := range verdicts {
		if verdict == nil {
			continue
		}
		var arithmetic *compute.V41BF16CheckedArithmeticError
		if !errors.As(verdict, &arithmetic) || meta[i].width <= 0 {
			if best == nil {
				best = verdict
			}
			continue
		}
		token, element := arithmetic.Index/meta[i].width, arithmetic.Index%meta[i].width
		stage := meta[i].stage
		if stage == 0 {
			element += meta[i].group * meta[i].width
		}
		if token < 0 || token >= rows {
			if best == nil {
				best = verdict
			}
			continue
		}
		if token < bestToken || token == bestToken && (stage < bestStage || stage == bestStage && element < bestElement) {
			bestToken, bestStage, bestElement, best = token, stage, element, verdict
		}
	}
	_ = groups
	return best
}
