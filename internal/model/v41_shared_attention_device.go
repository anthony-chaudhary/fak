package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Binding is session-local. Nil means the original host function; a selected
// callback has no decline outcome and must be rebound after clone or restore.
type v41SharedAttentionFunc func(layer int, request v41SharedAttentionRequest) ([]float32, error)

// Both entries share one selected-operation lifecycle. Ordinary calls always
// validate every row; only the synchronous composer can supply validated rows.
type v41SharedAttentionCore struct {
	run       v41SharedAttentionFunc
	validated func(int, *v41SharedAttentionRequest, *v41SharedAttentionValidation) ([]float32, error)
}

// A continuation captures ownership before projections and other callbacks.
// It is local to one role step, including each serial incremental suffix token.
// Cold/full prefill and ordinary callback reconstruction never receive it.
type v41SharedAttentionContinuation struct {
	binding                                 *v41SharedAttentionBinding
	rowOwner                                *V41AttentionState
	layer, pos, sourceRatio, heads, headDim int
	ringRows, firstKey                      int
	scale                                   float32
	live                                    func() bool // private metadata checks only
	used                                    bool
}

func newV41SharedAttentionContinuation(m *Model, layer, pos int, plan V41AttentionPlan, layerState, registry *V41AttentionState, scratch *v41ProjScratch) *v41SharedAttentionContinuation {
	if m == nil || scratch == nil || layerState == nil || registry == nil {
		return nil
	}
	binding := scratch.sharedAttentionBinding
	if binding == nil || binding.native.core == nil || binding.native.core.validated == nil {
		return nil
	}
	s := binding.native.owner
	if s == nil || !backendHasComparableIdentity(s.Backend) || s.M != m || binding.native.model != m || binding.native.backend != s.Backend {
		return nil
	}
	state, scope, epoch, backend := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch, s.Backend
	if state == nil || scope == nil || scope.owner == nil {
		return nil
	}
	cfg, owner, ringRows := m.Cfg, scope.owner, layerState.windowSize
	window := cfg.windowForLayer(layer)
	c := &v41SharedAttentionContinuation{binding: binding, layer: layer, pos: pos,
		rowOwner: layerState, ringRows: ringRows,
		sourceRatio: plan.kvGroupSize(cfg), heads: cfg.NumHeads, headDim: cfg.HeadDim, scale: cfg.attnScale()}
	if window > 0 {
		c.firstKey = max(0, pos-window+1)
	}
	c.live = func() bool {
		if s.M != m || !backendHasComparableIdentity(s.Backend) || s.Backend != backend ||
			s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.v41Forward != state || state.callbackOwner != s ||
			state.sharedAttentionBinding != binding || scratch.sharedAttentionBinding != binding || state.attn != registry || state.layerState(layer) != layerState ||
			s.cacheGeometryEpoch != epoch || s.backendExecutionScope != scope || !scope.live || scope.owner != owner ||
			len(scope.sessions) != 1 || scope.sessions[0] != s || len(state.history) != pos || layerState.nextWindowPos != pos ||
			layerState.windowSize != ringRows || layerState.headDim != c.headDim {
			return false
		}
		current := m.Cfg
		if layer < 0 || layer >= current.NumLayers || current.DeepSeekV41 != cfg.DeepSeekV41 || current.NumLayers != cfg.NumLayers || current.NumHeads != c.heads ||
			current.HeadDim != c.headDim || current.windowForLayer(layer) != window || current.attnScale() != c.scale {
			return false
		}
		// For an in-range nonzero layer, nearest-source identity gives exactly
		// v41AttentionRoles' role without allocating a whole-layer map. Recheck
		// current metadata rather than trusting the memoized plan after callbacks.
		resolved, err := v41AttentionPlanFor(current, layer, nil)
		if err != nil || resolved.Ratio <= 0 {
			return false
		}
		if resolved.KVSourceLayer == layer {
			resolved.Role = V41AttentionRoleKVSource
		} else if resolved.KVSourceLayer >= 0 {
			resolved.Role = V41AttentionRoleReader
		}
		return resolved == plan && resolved.kvGroupSize(current) == c.sourceRatio
	}
	if !c.live() {
		return nil
	}
	return c
}

// The proof never travels in the ordinary request. It names the exact local
// request and borrowed stream descriptors, and is consumed even on mismatch.
// Between the composer's original-row validation and consumption there are no
// arbitrary callbacks; session ownership excludes concurrent row mutation.
type v41SharedAttentionValidation struct {
	continuation       *v41SharedAttentionContinuation
	request            *v41SharedAttentionRequest
	rows               *v41SharedAttentionRows
	window, compressed [][]float32
	used               bool
	accepted           bool
}

func (v *v41SharedAttentionValidation) consume(core *v41SharedAttentionCore, layer int, request *v41SharedAttentionRequest) bool {
	if v == nil || v.used {
		return false
	}
	v.used = true
	c := v.continuation
	if c == nil || c.used {
		return false
	}
	c.used = true
	if c.binding == nil || c.live == nil || request == nil || c.binding.native.core != core || layer != c.layer || v.request != request || request.rowKV != v.rows || v.rows == nil ||
		request.mode != compute.V41SharedAttentionPlain || request.kv != nil || request.plain.Heads != c.heads || request.plain.HeadDim != c.headDim || request.plain.Softmax != c.scale {
		return false
	}
	view := request.resident
	if view == nil || view.owner != c.rowOwner || view.pos != c.pos || view.ringRows != c.ringRows || view.firstKey != c.firstKey ||
		view.windowRows != len(v.window) || view.compressedRows != len(v.compressed) {
		return false
	}
	same := func(a, b [][]float32) bool {
		return len(a) == len(b) && cap(a) == cap(b) && (len(a) == 0 || &a[0] == &b[0])
	}
	v.accepted = same(v.window, v.rows.window) && same(v.compressed, v.rows.compressed) && c.live()
	return v.accepted
}

// Rows borrow the host window and append-only compressed log only for the
// synchronous callback. Neither the callback nor the mirror retains or mutates
// these slices; device uploads own their storage before the callback returns.
type v41SharedAttentionRows struct {
	window, compressed [][]float32
}

func (r *v41SharedAttentionRows) len() int { return len(r.window) + len(r.compressed) }

func (r *v41SharedAttentionRows) row(i int) []float32 {
	if i < len(r.window) {
		return r.window[i]
	}
	return r.compressed[i-len(r.window)]
}

// Keep original inputs intact until validation completes. Gathering first would
// incorrectly hide invalid padding indices or non-finite invisible source rows.
type v41SharedAttentionRequest struct {
	mode        compute.V41SharedAttentionMode
	q, kv, sink []float32
	idx         []int32
	plain       V41SparseAttentionSinkOptions
	values      [][]float32
	compressed  V41AttentionSharedKVOptions
	// rowKV is an alternative to flat kv for plain combined attention. It
	// survives loss of resident capability so fallback gathers selected rows.
	rowKV *v41SharedAttentionRows
	// resident names the decode step's flat KV rows in the session mirror; nil
	// keeps the per-call KV upload. sinkKey names the checkpoint tensor the sink
	// was read from; residency is never inferred from a layer index.
	resident *v41ResidentKVView
	sinkKey  string
}

// V41SharedAttentionOperationError identifies a selected contraction failure.
// Continuation must exclude this type from host replay even when its cause wraps
// ErrV41ForwardStage. Teardown preserves the existing per-token suffix rollback
// boundary; this operation does not make a whole prefill call atomic.
type V41SharedAttentionOperationError struct {
	Layer int
	Cause error
}

func (e *V41SharedAttentionOperationError) Error() string {
	return fmt.Sprintf("model: selected V4.1 shared attention failed at layer %d: %v", e.Layer, e.Cause)
}

func (e *V41SharedAttentionOperationError) Unwrap() error { return e.Cause }

var errV41SharedAttentionResult = errors.New("model: invalid selected V4.1 shared attention result")

func v41SparseAttentionSinkWithDevice(layer int, q, kv, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions, attend v41SharedAttentionFunc) ([]float32, error) {
	return v41SparseAttentionSinkResident(layer, q, kv, sink, idx, opt, nil, attend)
}

func v41SparseAttentionSinkResident(layer int, q, kv, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions, resident *v41ResidentKVView, attend v41SharedAttentionFunc) ([]float32, error) {
	if attend == nil {
		return V41SparseAttentionSink(q, kv, sink, idx, opt)
	}
	request := v41SharedAttentionRequest{mode: compute.V41SharedAttentionPlain, q: q, kv: kv, sink: sink, idx: idx, plain: opt}
	if resident != nil && !v41ResidentKVDisabled {
		request.resident, request.sinkKey = resident, resident.sinkKey
	}
	return v41SelectedSharedAttention(layer, request, attend)
}

func v41AttentionCompressedForwardWithDevice(q []float32, values [][]float32, opt V41AttentionSharedKVOptions, attend v41SharedAttentionFunc) ([]float32, error) {
	if attend == nil {
		return V41AttentionCompressedForward(q, values, opt)
	}
	return v41SelectedSharedAttention(opt.Layer, v41SharedAttentionRequest{
		mode: compute.V41SharedAttentionCompressed, q: q, values: values, sink: opt.Sink, idx: opt.Idx, compressed: opt,
	}, attend)
}

func v41SelectedSharedAttention(layer int, request v41SharedAttentionRequest, attend v41SharedAttentionFunc) ([]float32, error) {
	out, err := attend(layer, request)
	if err != nil {
		return nil, &V41SharedAttentionOperationError{Layer: layer, Cause: v41StageErr(v41StageAttention, layer, err)}
	}
	return out, nil
}

type v41SharedAttentionPayload struct {
	q, kv, sink    []float32
	sourceRows     []int // selected slot to original row/group; duplicates are distinct
	heads, headDim int
	scale          float32
	mode           compute.V41SharedAttentionMode
}

func v41SharedAttentionGeometry(heads, headDim, rows int) (int, int, error) {
	qSize, qOK := checkedMulInt(heads, headDim)
	kvSize, kvOK := checkedMulInt(rows, headDim)
	_, qBytesOK := checkedMulInt(qSize, 4)
	_, kvBytesOK := checkedMulInt(kvSize, 4)
	_, statusBytesOK := checkedMulInt(heads, 16)
	if heads <= 0 || headDim <= 0 || rows < 0 || heads > math.MaxInt32 || headDim > math.MaxInt32 || rows > math.MaxInt32 ||
		!qOK || !kvOK || qSize > math.MaxInt32 || kvSize > math.MaxInt32 || !qBytesOK || !kvBytesOK || !statusBytesOK || uint64(heads)*4 > math.MaxUint32 {
		return 0, 0, fmt.Errorf("%w: shared attention geometry heads=%d dim=%d rows=%d exceeds native/index limits", ErrV41ForwardStage, heads, headDim, rows)
	}
	return qSize, kvSize, nil
}

func v41PrepareSharedAttention(request v41SharedAttentionRequest) (v41SharedAttentionPayload, error) {
	return v41PrepareSharedAttentionWithRows(request, nil)
}

// Only selection descriptors borrow scratch. The caller must finish consuming
// the payload, including fault attribution, before reusing these descriptors.
func v41PrepareSharedAttentionWithRows(request v41SharedAttentionRequest, sourceRows []int) (v41SharedAttentionPayload, error) {
	return v41PrepareSharedAttentionRows(request, sourceRows, false)
}

func v41PrepareSharedAttentionRows(request v41SharedAttentionRequest, sourceRows []int, rowsValidated bool) (v41SharedAttentionPayload, error) {
	p := v41SharedAttentionPayload{mode: request.mode, sourceRows: sourceRows[:0]}
	bad := func(format string, args ...any) (v41SharedAttentionPayload, error) {
		return p, fmt.Errorf("%w: shared attention %s", ErrV41ForwardStage, fmt.Sprintf(format, args...))
	}
	var rows, slots int
	switch request.mode {
	case compute.V41SharedAttentionPlain:
		o := request.plain
		p.heads, p.headDim, p.scale, rows, slots = o.Heads, o.HeadDim, o.Softmax, o.N, o.TopK
		if o.B != 1 || o.M != 1 || o.TopK < 0 || o.TopK > math.MaxInt32 || o.Inverse != nil {
			return bad("selected plain contraction requires B=M=1, native top-k, and no inverse output rotation")
		}
		if o.TopKLength != nil && (len(o.TopKLength) != 1 || o.TopKLength[0] < 0 || int(o.TopKLength[0]) > o.TopK) {
			return bad("invalid plain top-k length")
		}
	case compute.V41SharedAttentionCompressed:
		o := request.compressed
		p.heads, p.headDim, p.scale, rows = o.Heads, o.HeadDim, o.Softmax, o.Groups
		if o.Ratio < 1 || o.TopK < 0 || o.QueryOffset < 0 || o.QueryOffset > math.MaxInt-1 || o.Inverse != nil {
			return bad("invalid compressed ratio, top-k, query offset, or inverse output rotation")
		}
		if len(request.values) != rows || (o.SourceRows != nil && len(o.SourceRows) != rows) {
			return bad("compressed stream length does not match groups=%d", rows)
		}
		if request.idx != nil {
			slots = o.IndexTopK
			if slots <= 0 || slots > math.MaxInt32 {
				return bad("compressed index top-k must fit positive int32")
			}
		}
	default:
		return bad("unknown mode %d", request.mode)
	}
	qSize, kvSize, err := v41SharedAttentionGeometry(p.heads, p.headDim, rows)
	if err != nil {
		return p, err
	}
	if len(request.q) != qSize || !finite32(p.scale) || p.scale == 0 {
		return bad("query length or finite nonzero scale is invalid")
	}
	if request.rowKV != nil {
		if request.mode != compute.V41SharedAttentionPlain || request.kv != nil || request.rowKV.len() != rows {
			return bad("row KV requires plain mode, no flat KV, and %d rows", rows)
		}
		for _, stream := range [][][]float32{request.rowKV.window, request.rowKV.compressed} {
			for _, row := range stream {
				if len(row) != p.headDim {
					return bad("row KV width %d, want %d", len(row), p.headDim)
				}
				if !rowsValidated {
					if err := finiteRow32(row, "shared attention row KV"); err != nil {
						return bad("%v", err)
					}
				}
			}
		}
	} else if request.mode == compute.V41SharedAttentionPlain && len(request.kv) != kvSize {
		return bad("plain KV length %d, want %d", len(request.kv), kvSize)
	}
	if request.sink != nil && len(request.sink) != p.heads {
		return bad("sink length %d, want %d", len(request.sink), p.heads)
	}
	if len(request.idx) != slots {
		return bad("index length %d, want %d", len(request.idx), slots)
	}
	for _, input := range []struct {
		name   string
		values []float32
	}{{"query", request.q}, {"KV", request.kv}, {"sink", request.sink}} {
		for i, v := range input.values {
			if !finite32(v) {
				return bad("non-finite %s element %d", input.name, i)
			}
		}
	}
	for g, row := range request.values {
		if len(row) != p.headDim {
			return bad("compressed row %d width %d, want %d", g, len(row), p.headDim)
		}
		for d, v := range row {
			if !finite32(v) {
				return bad("non-finite compressed row %d element %d", g, d)
			}
		}
	}
	for i, row := range request.idx {
		if row < -1 || int(row) >= rows {
			return bad("row index %d out of range at original slot %d", row, i)
		}
	}
	// Only now apply length/causal masks. No sorting, deduplication, selection
	// recomputation, BF16 conversion, or publication/cache mutation occurs here.
	if request.mode == compute.V41SharedAttentionPlain {
		if request.plain.TopKLength != nil {
			slots = int(request.plain.TopKLength[0])
		}
		for _, row := range request.idx[:slots] {
			if row >= 0 {
				p.sourceRows = append(p.sourceRows, int(row))
			}
		}
	} else {
		o := request.compressed
		lastVisible := -1
		if o.QueryOffset >= o.Ratio-1 {
			lastVisible = (o.QueryOffset - (o.Ratio - 1)) / o.Ratio
		}
		if request.idx != nil {
			for _, row := range request.idx {
				if row >= 0 && int(row) <= lastVisible {
					p.sourceRows = append(p.sourceRows, int(row))
				}
			}
		} else {
			for row := 0; row < rows && row <= lastVisible; row++ {
				p.sourceRows = append(p.sourceRows, row)
			}
		}
	}
	_, selectedSize, err := v41SharedAttentionGeometry(p.heads, p.headDim, len(p.sourceRows))
	if err != nil {
		return p, err
	}
	if len(p.sourceRows) == 0 {
		return p, nil
	}
	p.q = append([]float32(nil), request.q...)
	if request.sink != nil {
		p.sink = append([]float32(nil), request.sink...)
	}
	if request.resident != nil {
		return p, nil // the device mirror gathers the selected rows
	}
	p.kv = make([]float32, 0, selectedSize)
	for _, row := range p.sourceRows {
		if request.rowKV != nil {
			p.kv = append(p.kv, request.rowKV.row(row)...)
		} else if request.mode == compute.V41SharedAttentionPlain {
			p.kv = append(p.kv, request.kv[row*p.headDim:(row+1)*p.headDim]...)
		} else {
			p.kv = append(p.kv, request.values[row]...)
		}
	}
	return p, nil
}

func v41SharedAttentionCause(layer int, p v41SharedAttentionPayload, cause error) error {
	var protocol *compute.V41SharedAttentionProtocolError
	if errors.As(cause, &protocol) {
		return cause // a protocol failure cannot establish an arithmetic producer
	}
	var arithmetic *compute.V41SharedAttentionArithmeticError
	if !errors.As(cause, &arithmetic) {
		return cause
	}
	valid := arithmetic.Head >= 0 && arithmetic.Head < p.heads &&
		arithmetic.SelectedSlot >= -1 && arithmetic.SelectedSlot < len(p.sourceRows) &&
		arithmetic.Element >= -1 && arithmetic.Element < p.headDim && !finite32(math.Float32frombits(arithmetic.ValueBits))
	slot, element := arithmetic.SelectedSlot, arithmetic.Element
	var producer v41CompressedStage
	switch arithmetic.Stage {
	case compute.V41SharedAttentionStageScore:
		valid = valid && slot >= 0 && element == -1
		producer = v41CompressedStageScore
	case compute.V41SharedAttentionStageExp:
		valid = valid && element == -1 && (slot >= 0 || p.sink != nil)
		producer = v41CompressedStageSoftmax
	case compute.V41SharedAttentionStageDenominator:
		valid = valid && slot == -1 && element == -1
		producer = v41CompressedStageSoftmax
	case compute.V41SharedAttentionStageWeight:
		valid = valid && slot >= 0 && element == -1
		producer = v41CompressedStageValue
	case compute.V41SharedAttentionStageValue:
		valid = valid && slot >= 0 && element >= 0
		producer = v41CompressedStageValue
	case compute.V41SharedAttentionStageOutput:
		valid = valid && slot == -1 && element >= 0
	default:
		valid = false
	}
	if !valid {
		return errors.Join(&compute.V41SharedAttentionProtocolError{Head: arithmetic.Head, Message: "invalid arithmetic fault metadata"}, cause)
	}
	if p.mode == compute.V41SharedAttentionCompressed && producer != "" {
		group := -1
		if slot >= 0 {
			group = p.sourceRows[slot]
		}
		return errors.Join(v41CompressedNonFinite(layer, producer, 0, arithmetic.Head, group, element, math.Float32frombits(arithmetic.ValueBits)), cause)
	}
	// Retain the selected-operation wrapper and original arithmetic cause.
	// Valid plain sparse-sink arithmetic shares the host refusal contract;
	// malformed status was rejected above and cannot acquire this sentinel.
	// Output failures still do not invent a producer.
	selected := fmt.Errorf("%w: selected attention layer=%d mode=%d head=%d slot=%d stage=%d: %w", ErrV41ForwardStage, layer, p.mode, arithmetic.Head, slot, arithmetic.Stage, cause)
	if p.mode == compute.V41SharedAttentionPlain {
		return errors.Join(ErrV41SparseSinkNonFinite, selected)
	}
	return selected
}

// One nonempty call uploads Q, selected shared KV, and a transient sink. Upload
// bytes are 4*(H*D+R*D+H) with a sink, or 4*(H*D+R*D+1) without one; result
// readback is 4*H*D, plus the primitive's 16*H status readback. A decode step
// that names its rows in the session KV mirror (#13821) instead uploads Q plus
// only the rows the mirror lacks, and reads its sink from the tensor named by
// sinkKey; sink residency is never inferred from a layer/cache key.
func (s *Session) v41SharedAttentionFunc() v41SharedAttentionFunc {
	return s.v41BindSharedAttention().attentionFunc()
}

func (s *Session) v41BindSharedAttention() *v41SharedAttentionBinding {
	if s == nil {
		return nil
	}
	model, backend := s.M, s.Backend
	if model == nil || backend == nil || !backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(backend, compute.F32) {
		return nil
	}
	operation, ok := backend.(compute.V41SharedAttentionBackend)
	if !ok || !operation.SupportsV41SharedAttention() {
		return nil
	}
	resident := s.newV41ResidentKV()
	// Session ownership remains serial. Lease this metadata through cleanup and
	// fault attribution; reentrant calls get independent descriptors instead.
	var sourceRows []int
	selectionInUse := false
	run := func(layer int, request v41SharedAttentionRequest, rowsValidated bool) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		reuseSelection := !selectionInUse
		if reuseSelection {
			selectionInUse = true
			defer func() { selectionInUse = false }()
		}
		if resident == nil {
			request.resident, request.sinkKey = nil, ""
		}
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-shared-attention", Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, cause = nil, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, cause = nil, closeFailure(err)
					return
				}
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		var p v41SharedAttentionPayload
		var err error
		if reuseSelection {
			p, err = v41PrepareSharedAttentionRows(request, sourceRows, rowsValidated)
			sourceRows = p.sourceRows[:0]
		} else {
			p, err = v41PrepareSharedAttention(request)
		}
		if err != nil {
			return nil, closeFailure(err)
		}
		if len(p.sourceRows) == 0 {
			return make([]float32, p.heads*p.headDim), nil
		}
		run := func() (result []float32, cause error) {
			var owned []compute.Tensor
			defer func() {
				// Cleanup must not replace the selected operation's error or
				// original panic. Still attempt every owned release.
				primaryPanic := recover()
				var cleanupPanic any
				for i := len(owned) - 1; i >= 0; i-- {
					// Try every release even if one backend Free panics. The
					// outer selected-operation guard still retires the session.
					func() {
						defer func() {
							if r := recover(); r != nil && cleanupPanic == nil {
								cleanupPanic = r
							}
						}()
						s.Backend.Free(owned[i])
					}()
				}
				if primaryPanic != nil {
					panic(primaryPanic)
				}
				if cause == nil && cleanupPanic != nil {
					panic(cleanupPanic)
				}
			}()
			fresh := func(x compute.Tensor) bool {
				if x.Buf() == nil || x.Backend() != s.Backend {
					return false
				}
				for _, previous := range owned {
					if previous.Buf() == x.Buf() {
						return false
					}
				}
				owned = append(owned, x)
				return true
			}
			shapeOK := func(x compute.Tensor, shape []int) bool {
				if x.Dtype != compute.F32 || x.Layout != compute.RowMajor || x.Quant != nil || len(x.Shape) != len(shape) {
					return false
				}
				for i, d := range shape {
					if x.Shape[i] != d {
						return false
					}
				}
				return true
			}
			upload := func(name string, shape []int, values []float32) (compute.Tensor, error) {
				stage = name + " upload"
				x := s.uploadHostF32(shape, values, compute.MemoryActivation, "V4.1 shared attention "+name)
				if !fresh(x) || !shapeOK(x, shape) {
					return compute.Tensor{}, errV41SharedAttentionResult
				}
				return x, nil
			}
			qt, err := upload("query", []int{p.heads, p.headDim}, p.q)
			if err != nil {
				return nil, err
			}
			var kt compute.Tensor
			if request.resident != nil {
				stage = "resident KV"
				kt, err = resident.panel(layer, request.resident, request.kv, request.rowKV, p.headDim, p.sourceRows)
				if err != nil {
					return nil, err
				}
				if !fresh(kt) || !shapeOK(kt, []int{len(p.sourceRows), p.headDim}) {
					return nil, errV41SharedAttentionResult
				}
			} else if kt, err = upload("KV", []int{len(p.sourceRows), p.headDim}, p.kv); err != nil {
				return nil, err
			}
			var st compute.Tensor
			if request.sinkKey != "" && p.sink != nil {
				stage = "resident sink"
				st = resident.sink(request.sinkKey, p.sink)
				if st.Backend() != s.Backend || !shapeOK(st, []int{len(p.sink)}) {
					return nil, errV41SharedAttentionResult
				}
			} else {
				sink := p.sink
				if sink == nil {
					sink = []float32{0}
				}
				if st, err = upload("sink", []int{len(sink)}, sink); err != nil {
					return nil, err
				}
			}
			stage = "shared attention"
			out, err := operation.V41SharedAttention(qt, kt, st, len(p.sourceRows), p.heads, p.headDim, p.scale, p.mode, p.sink != nil)
			ownedOutput := fresh(out) // also retain any fresh partial output on error
			if err != nil {
				return nil, v41SharedAttentionCause(layer, p, err)
			}
			stage = "output validation"
			if !ownedOutput || !shapeOK(out, []int{p.heads, p.headDim}) {
				return nil, errV41SharedAttentionResult
			}
			stage = "readback"
			values := s.Backend.Read(out)
			if len(values) != p.heads*p.headDim {
				return nil, errV41SharedAttentionResult
			}
			for i, value := range values {
				if !finite32(value) {
					return nil, v41SharedAttentionCause(layer, p, &compute.V41SharedAttentionArithmeticError{
						Stage: compute.V41SharedAttentionStageOutput, Head: i / p.headDim, SelectedSlot: -1, Element: i % p.headDim, ValueBits: math.Float32bits(value),
					})
				}
			}
			return append([]float32(nil), values...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
	core := &v41SharedAttentionCore{}
	core.run = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
		return run(layer, request, false)
	}
	core.validated = func(layer int, request *v41SharedAttentionRequest, validation *v41SharedAttentionValidation) ([]float32, error) {
		rowsValidated := validation.consume(core, layer, request)
		if selectionInUse && validation != nil {
			rowsValidated, validation.accepted = false, false
		}
		return run(layer, *request, rowsValidated)
	}
	ordinary := func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
		return core.run(layer, request)
	}
	return &v41SharedAttentionBinding{ordinary: ordinary, native: v41BoundSharedAttention{core: core, owner: s, model: model, backend: backend}}
}
