package model

import (
	"fmt"
	"math"
	"reflect"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/v41kv"
)

// v41PackFlashMLAQualification snapshots one ordinary, fully validated attention
// request. There is deliberately no retained cache: host quantization and full
// H2D staging are qualification costs, not a device-residency performance claim.
func v41PackFlashMLAQualification(request v41SharedAttentionRequest) (compute.FlashMLADecodeQualificationInput, error) {
	var out compute.FlashMLADecodeQualificationInput
	bad := func() (compute.FlashMLADecodeQualificationInput, error) {
		return out, fmt.Errorf("model: FlashMLA qualification requires bounded plain 64x512 operands")
	}
	o := request.plain
	if request.mode != compute.V41SharedAttentionPlain || o.Heads != 64 || o.HeadDim != 512 || o.N < 0 || o.TopK < 0 || o.TopK > 1<<20 || o.Softmax <= 0 {
		return bad()
	}
	// Native operand-staging admission precedes maps, packed storage and
	// validation's snapshots. This is not a total Go-memory bound: selected-index
	// metadata, the deduplication map and validation snapshots need separate
	// executor resource admission. Packed capacity itself is allocated once.
	// Conservative unique-row bound may refuse an unusually repetitive request;
	// this qualification-only bound never changes a selected request's semantics.
	padded := max(64, (o.TopK+63)/64*64)
	uniqueBound := max(1, min(o.N, o.TopK))
	if uint64(uniqueBound)*528+uint64(padded)*4+65536+256 > uint64(compute.FlashMLADecodeQualificationHostLimit) {
		return bad()
	}
	// Ask the existing preparer to validate all original rows and indices before
	// masking, without allocating a selected F32 gather. This local presence hint
	// is never passed to a resident backend and conveys no continuation authority.
	validation := request
	validation.resident = &v41ResidentKVView{}
	if _, err := v41PrepareSharedAttention(validation); err != nil {
		return out, err
	}
	length := o.TopK
	if o.TopKLength != nil {
		length = int(o.TopKLength[0])
	}
	out.Query = make([]uint16, len(request.q))
	for i, value := range request.q {
		bits := math.Float32bits(v41RoundBF16(value))
		if bits&0x7f800000 == 0x7f800000 {
			return compute.FlashMLADecodeQualificationInput{}, fmt.Errorf("model: FlashMLA qualification query overflows BF16 at %d", i)
		}
		out.Query[i] = uint16(bits >> 16)
	}
	out.Sink = append([]float32(nil), request.sink...)
	out.Scale = o.Softmax
	out.Indices = make([]int32, padded)
	for i := range out.Indices {
		out.Indices[i] = -1
	}
	out.Length = int32(length)
	out.PageSize = 1
	layout, _ := v41kv.V41KVCacheLayoutFor(v41kv.V41KVFP8)
	out.KV = make([]byte, 0, uniqueBound*layout.BytesPerToken)
	compact := make(map[int32]int32)
	for slot, id := range request.idx[:length] {
		if id < 0 {
			continue
		}
		rowID, exists := compact[id]
		if !exists {
			var row []float32
			if request.rowKV != nil {
				row = request.rowKV.row(int(id))
			} else {
				row = request.kv[int(id)*512 : (int(id)+1)*512]
			}
			packed, err := layout.V41KVQuantizeRow(row[:448], row[448:])
			if err != nil {
				return compute.FlashMLADecodeQualificationInput{}, err
			}
			rowID = int32(len(compact))
			compact[id] = rowID
			out.KV = append(out.KV, packed...)
		}
		out.Indices[slot] = rowID // Preserve duplicate weights and slot order.
	}
	out.Blocks = len(compact)
	if out.Blocks == 0 {
		// The donor requires storage even when no index references a row. Still
		// invoke it with the real query/sink to witness its zero/+Inf convention.
		out.Blocks = 1
		out.KV = make([]byte, layout.BytesPerToken)
	}
	if err := compute.ValidateFlashMLADecodeQualificationInput(out); err != nil {
		return compute.FlashMLADecodeQualificationInput{}, err
	}
	return out, nil
}

// v41FlashMLAQualificationBinding is an explicit qualification selection only.
// A qualification harness installs this ordinary binding on the session's
// current forward state. No production constructor installs or advertises it.
// Session state/selection mutation remains serial, like other forward bindings.
func (s *Session) v41FlashMLAQualificationBinding() *v41SharedAttentionBinding {
	if s == nil || s.M == nil || !backendHasComparableIdentity(s.Backend) || reflect.ValueOf(s.Backend).IsNil() {
		return newV41SharedAttentionBinding(func(int, v41SharedAttentionRequest) ([]float32, error) {
			return nil, compute.ErrFlashMLADecodeQualificationUnavailable
		})
	}
	model, backend, state, scope, epoch := s.M, s.Backend, s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch
	var busy atomic.Bool
	var poisoned atomic.Bool
	var binding *v41SharedAttentionBinding
	owned := func() bool {
		return s.M == model && backendHasComparableIdentity(s.Backend) && s.Backend == backend && s.v41Forward == state && state != nil &&
			state.sharedAttentionBinding == binding && state.callbackOwner == s && s.backendExecutionScope == scope && scope != nil && scope.live &&
			len(scope.sessions) == 1 && scope.sessions[0] == s && s.cacheGeometryEpoch == epoch
	}
	callback := func(layer int, request v41SharedAttentionRequest) (values []float32, cause error) {
		if !busy.CompareAndSwap(false, true) {
			poisoned.Store(true)
			return nil, fmt.Errorf("model: FlashMLA qualification is already in use")
		}
		defer busy.Store(false)
		fail := func(err error) ([]float32, error) {
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-flashmla-qualification", Layer: layer, Stage: "qualification", Cause: err}
			// A stale callback has no authority to close a newly rebound owner.
			if owned() {
				s.halFailure = closed
				s.closeBackendOwned()
			}
			return nil, closed
		}
		defer func() {
			if recovered := recover(); recovered != nil {
				err, ok := recovered.(error)
				if !ok {
					err = fmt.Errorf("model: FlashMLA qualification panic: %v", recovered)
				}
				values, cause = fail(err)
			}
		}()
		if !owned() || s.halClosed || s.halFailure != nil || poisoned.Load() || (scope != nil && !scope.live) {
			return fail(fmt.Errorf("model: FlashMLA qualification owner changed or closed"))
		}
		input, err := v41PackFlashMLAQualification(request)
		if err != nil {
			return fail(err)
		}
		result, err := compute.QualifyFlashMLADecode(backend, input)
		if err != nil {
			return fail(err)
		}
		if poisoned.Load() || !owned() {
			// A concurrent/reentrant selection may not close an active owner's
			// resources. Retire only after its checked native call has returned.
			return fail(fmt.Errorf("model: FlashMLA qualification concurrent selection"))
		}
		values = make([]float32, len(result.ValuesBF16))
		for i, bits := range result.ValuesBF16 {
			values[i] = math.Float32frombits(uint32(bits) << 16)
		}
		// Compute already validated donor LSE; ordinary model output has no LSE
		// field, so it is omitted, never reinterpreted as the F32 oracle's LSE.
		return values, nil
	}
	binding = newV41SharedAttentionBinding(callback)
	return binding
}
