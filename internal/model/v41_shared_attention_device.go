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

// Keep original inputs intact until validation completes. Gathering first would
// incorrectly hide invalid padding indices or non-finite invisible source rows.
type v41SharedAttentionRequest struct {
	mode        compute.V41SharedAttentionMode
	q, kv, sink []float32
	idx         []int32
	plain       V41SparseAttentionSinkOptions
	values      [][]float32
	compressed  V41AttentionSharedKVOptions
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
	if attend == nil {
		return V41SparseAttentionSink(q, kv, sink, idx, opt)
	}
	return v41SelectedSharedAttention(layer, v41SharedAttentionRequest{
		mode: compute.V41SharedAttentionPlain, q: q, kv: kv, sink: sink, idx: idx, plain: opt,
	}, attend)
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
	p := v41SharedAttentionPayload{mode: request.mode}
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
	if request.mode == compute.V41SharedAttentionPlain && len(request.kv) != kvSize {
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
	p.kv = make([]float32, 0, selectedSize)
	for _, row := range p.sourceRows {
		if request.mode == compute.V41SharedAttentionPlain {
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
// readback is 4*H*D, plus the primitive's 16*H status readback. Sink residency is
// deliberately not inferred from a layer/cache key. No speedup is claimed.
func (s *Session) v41SharedAttentionFunc() v41SharedAttentionFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41SharedAttentionBackend)
	if !ok || !operation.SupportsV41SharedAttention() {
		return nil
	}
	return func(layer int, request v41SharedAttentionRequest) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-shared-attention", Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.Close()
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
		p, err := v41PrepareSharedAttention(request)
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
			kt, err := upload("KV", []int{len(p.sourceRows), p.headDim}, p.kv)
			if err != nil {
				return nil, err
			}
			sink := p.sink
			if sink == nil {
				sink = []float32{0}
			}
			st, err := upload("sink", []int{len(sink)}, sink)
			if err != nil {
				return nil, err
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
}
