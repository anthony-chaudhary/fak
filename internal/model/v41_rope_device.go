package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A nil callback keeps the existing host rotation. Selection happens only when
// the session binds an available operation; a selected call has no decline arm.
type v41TailRoPEFunc func(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int) (qOut, kvOut []float32, err error)

// V41TailRoPEOperationError identifies a selected tail-RoPE failure. Continuation
// callers must not treat a nested stage sentinel as permission for host replay.
type V41TailRoPEOperationError struct {
	Layer int
	Cause error
}

func (e *V41TailRoPEOperationError) Error() string {
	return fmt.Sprintf("model: selected V4.1 tail-RoPE failed at layer %d: %v", e.Layer, e.Cause)
}

func (e *V41TailRoPEOperationError) Unwrap() error { return e.Cause }

var errV41TailRoPEResult = errors.New("model: invalid selected V4.1 tail-RoPE result")

func v41TailRoPEInPlace(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int, rotate v41TailRoPEFunc) error {
	if rotate == nil {
		for h := 0; h < heads; h++ {
			applyRopeTailInterleaved(q[h*headDim:(h+1)*headDim], cos, sin, rotaryDim)
		}
		applyRopeTailInterleaved(kv, cos, sin, rotaryDim)
		return nil
	}
	qOut, kvOut, err := rotate(layer, q, kv, cos, sin, heads, headDim, rotaryDim)
	if err == nil && (len(qOut) != len(q) || len(kvOut) != len(kv)) {
		err = errV41TailRoPEResult
	}
	if err != nil {
		return &V41TailRoPEOperationError{Layer: layer, Cause: v41StageErr(v41StageAttention, layer, err)}
	}
	// Publish both owned rows together, only after both readbacks succeeded.
	copy(q, qOut)
	copy(kv, kvOut)
	return nil
}

// v41TailRoPEHostTable snapshots already validated sin/cos operands before any
// backend callback. Prior tables are immutable callback-owned host copies, not
// caller slices. Reusing their exact bits does not admit their device handles;
// the ordinary live-handle/model/epoch checks still run at the upload site.
func v41TailRoPEHostTable(cos, sin []float32, prior [2][]float32) []float32 {
	for _, table := range prior {
		if len(table) != 2*len(sin) {
			continue
		}
		equal := true
		for j, value := range sin {
			if math.Float32bits(value) != math.Float32bits(table[2*j]) || math.Float32bits(cos[j]) != math.Float32bits(table[2*j+1]) {
				equal = false
				break
			}
		}
		if equal {
			return table
		}
	}
	table := make([]float32, 2*len(sin))
	for j := range sin {
		table[2*j], table[2*j+1] = sin[j], cos[j]
	}
	return table
}

// Refs #13668. This leaf uses the exact host-generated table, including its
// current per-layer theta and amplitude. Q and KV remain transient. At most two
// exact tables are session-owned; a miss adds 4*R uploaded bytes to the unchanged
// 4*(heads+1)*D Q/KV transfers. Refs #13846; no latency gain is implied.
func (s *Session) v41TailRoPEFunc() v41TailRoPEFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41TailRoPEBackend)
	if !ok || !operation.SupportsV41TailRoPE() {
		return nil
	}
	// These keys are session-local, never model-owned immutable weights. A newly
	// bound callback may replace them, so a hit must also match the live handle.
	keys := [2]string{"v41-tail-rope-table:0", "v41-tail-rope-table:1"}
	type tableEntry struct {
		values []float32
		tensor compute.Tensor
		model  *Model
		epoch  uint64
	}
	var tables [2]tableEntry
	next := 0
	return func(layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int) (qResult, kvResult []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-tail-rope", Layer: layer, Stage: stage, Cause: err}
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
						qResult, kvResult, cause = nil, nil, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					qResult, kvResult, cause = nil, nil, closeFailure(err)
					return
				}
				// Preserve programming panic identity, while closing the selected
				// session before the existing transactional unwind restores state.
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		qSize, valid := checkedMulInt(heads, headDim)
		if !valid || heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 ||
			len(q) != qSize || len(kv) != headDim || len(cos) != rotaryDim/2 || len(sin) != rotaryDim/2 {
			return nil, nil, closeFailure(errV41TailRoPEResult)
		}
		for _, row := range [][]float32{q, kv, cos, sin} {
			for _, value := range row {
				if !finite32(value) {
					return nil, nil, closeFailure(errV41TailRoPEResult)
				}
			}
		}
		table := v41TailRoPEHostTable(cos, sin, [2][]float32{tables[0].values, tables[1].values})
		run := func() ([]float32, []float32, error) {
			stage = "query upload"
			qt := s.uploadHostF32([]int{heads, headDim}, q, compute.MemoryActivation, "V4.1 tail-RoPE query")
			defer s.Backend.Free(qt)
			stage = "kv upload"
			kt := s.uploadHostF32([]int{headDim}, kv, compute.MemoryActivation, "V4.1 tail-RoPE KV")
			defer s.Backend.Free(kt)
			stage = "table upload"
			var tt compute.Tensor
			for i, entry := range tables {
				live, present := s.halW[keys[i]]
				if !present || live.Buf() == nil || live.Buf() != entry.tensor.Buf() ||
					live.Backend() != s.Backend || entry.model != s.M || entry.epoch != s.cacheGeometryEpoch || len(entry.values) != len(table) {
					continue
				}
				equal := true
				for j, value := range table {
					if math.Float32bits(value) != math.Float32bits(entry.values[j]) {
						equal = false
						break
					}
				}
				if equal {
					tt = live
					break
				}
			}
			if tt.Buf() == nil {
				if s.halW == nil {
					s.halW = make(map[string]compute.Tensor)
				}
				key := keys[next]
				if old, present := s.halW[key]; present {
					// The prior call completed its readbacks before this replacement.
					delete(s.halW, key)
					s.Backend.Free(old)
				}
				tt = s.uploadHostF32([]int{rotaryDim / 2, 2}, table, compute.MemoryActivation, "V4.1 tail-RoPE table")
				s.halW[key] = tt
				tables[next] = tableEntry{values: table, tensor: tt, model: s.M, epoch: s.cacheGeometryEpoch}
				next = (next + 1) % len(tables)
			}
			stage = "tail-rope"
			qo, ko, err := operation.V41TailRoPEQK(qt, kt, tt, heads, headDim, rotaryDim)
			// Even an error may return partially allocated outputs. Free each
			// owned, fresh output once, without double-freeing malformed aliases.
			fresh := func(x compute.Tensor) bool {
				return x.Buf() != nil && x.Backend() == s.Backend && x.Buf() != qt.Buf() && x.Buf() != kt.Buf() && x.Buf() != tt.Buf()
			}
			if fresh(qo) {
				defer s.Backend.Free(qo)
			}
			if fresh(ko) && ko.Buf() != qo.Buf() {
				defer s.Backend.Free(ko)
			}
			if err != nil {
				return nil, nil, err
			}
			if !fresh(qo) || !fresh(ko) || qo.Buf() == ko.Buf() ||
				qo.Dtype != compute.F32 || ko.Dtype != compute.F32 || qo.Layout != compute.RowMajor || ko.Layout != compute.RowMajor || qo.Quant != nil || ko.Quant != nil ||
				len(qo.Shape) != 2 || qo.Shape[0] != heads || qo.Shape[1] != headDim || len(ko.Shape) != 1 || ko.Shape[0] != headDim {
				return nil, nil, errV41TailRoPEResult
			}
			own := func(values, input []float32, independent bool) ([]float32, error) {
				if len(values) != len(input) {
					return nil, errV41TailRoPEResult
				}
				for i, value := range values {
					if !finite32(value) || (i%headDim < headDim-rotaryDim && math.Float32bits(value) != math.Float32bits(input[i])) {
						return nil, errV41TailRoPEResult
					}
				}
				// Read may expose reusable backend memory. Own Q before reading
				// KV, and own both rows before their device allocations are freed.
				if independent {
					return values, nil
				}
				return append([]float32(nil), values...), nil
			}
			if grouped, ok := s.Backend.(compute.V41DeferredStatusBackend); ok {
				// A zero-status read needs no SwiGLU/BF16 module, so the
				// unrelated SupportsV41DeferredStatus probe does not gate it.
				// Both outputs share one checked host fence. A transport error
				// can precede Q's numeric validation; never retry with Read.
				stage = "paired readback"
				verdicts, result, err := compute.ReadV41DeferredResult(grouped, nil, []compute.Tensor{qo, ko})
				if err != nil {
					return nil, nil, err
				}
				if len(verdicts) != 0 || result.RowCount() != 2 {
					return nil, nil, errV41TailRoPEResult
				}
				// Taking ownership here remains private to this callback. Both
				// rows must validate before either is published to its caller.
				rows, independent := result.TakeRows()
				stage = "query readback"
				qValues, err := own(rows[0], q, independent)
				if err != nil {
					return nil, nil, err
				}
				stage = "kv readback"
				kvValues, err := own(rows[1], kv, independent)
				if err != nil {
					return nil, nil, err
				}
				return qValues, kvValues, nil
			}
			stage = "query readback"
			qValues, err := own(s.Backend.Read(qo), q, false)
			if err != nil {
				return nil, nil, err
			}
			stage = "kv readback"
			kvValues, err := own(s.Backend.Read(ko), kv, false)
			return qValues, kvValues, err
		}
		qResult, kvResult, cause = run()
		if cause != nil {
			return nil, nil, closeFailure(cause)
		}
		return qResult, kvResult, nil
	}
}

// v41AttentionQKRoPE owns full-graph BF16 publications around the unchanged
// F32 rotary operation. The source Linear publishes Q in BF16 before rotation;
// apply_rotary_emb widens for arithmetic then copies back into its original
// dtype (model.py772 and392-405 at dba1be0a40aa45a94ad051997016db3960a90277).
// Reduced callers retain their original in-place arithmetic and storage.
func v41AttentionQKRoPE(cfg Config, layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int, rotate v41TailRoPEFunc) ([]float32, []float32, error) {
	return v41AttentionQKRoPEWithPublish(cfg, layer, q, kv, cos, sin, heads, headDim, rotaryDim, rotate, nil)
}

// v41RoPEBF16Pair publishes Q and KV as one owned row, so a bound device
// publication costs one dispatch. Element-wise rounding makes the split exact.
func v41RoPEBF16Pair(layer int, site string, q, kv []float32, publish v41BF16BoundaryFunc) ([]float32, []float32, error) {
	joined := make([]float32, 0, len(q)+len(kv))
	joined = append(append(joined, q...), kv...)
	published, err := v41RoPEBF16Publish(layer, site, joined, publish)
	if err != nil {
		return nil, nil, err
	}
	return published[:len(q):len(q)], published[len(q):], nil
}

func v41AttentionQKRoPEWithPublish(cfg Config, layer int, q, kv, cos, sin []float32, heads, headDim, rotaryDim int, rotate v41TailRoPEFunc, publish v41BF16BoundaryFunc) ([]float32, []float32, error) {
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, nil, err
	}
	if !full {
		err = v41TailRoPEInPlace(layer, q, kv, cos, sin, heads, headDim, rotaryDim, rotate)
		return q, kv, err
	}
	width, ok := checkedMulInt(heads, headDim)
	if !ok || heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim%2 != 0 || rotaryDim > headDim || len(q) != width || len(kv) != headDim || len(cos) != rotaryDim/2 || len(sin) != rotaryDim/2 {
		return nil, nil, v41RoPEPublicationErr(layer, errV41TailRoPEResult)
	}
	for i := range cos {
		if !finite32(cos[i]) || !finite32(sin[i]) {
			return nil, nil, v41RoPEPublicationErr(layer, errV41TailRoPEResult)
		}
	}
	query, key, err := v41RoPEBF16Pair(layer, v41TailRoPEInputSite, q, kv, publish)
	if err != nil {
		return nil, nil, err
	}
	var qOut, kOut []float32
	if rotate == nil {
		if err = v41TailRoPEInPlace(layer, query, key, cos, sin, heads, headDim, rotaryDim, nil); err != nil {
			return nil, nil, err
		}
		qOut, kOut = query, key
	} else {
		qOut, kOut, err = rotate(layer, query, key, cos, sin, heads, headDim, rotaryDim)
		if err != nil {
			return nil, nil, v41RoPEPublicationErr(layer, err)
		}
	}
	if len(qOut) != width || len(kOut) != headDim {
		return nil, nil, v41RoPEPublicationErr(layer, errV41TailRoPEResult)
	}
	// Both callbacks' outputs may alias their inputs or one another. Neither
	// original row is written; finish owning both outputs before returning either.
	return v41RoPEBF16Pair(layer, v41TailRoPEOutputSite, qOut, kOut, publish)
}

func v41RoPEPublicationErr(layer int, cause error) error {
	return &V41TailRoPEOperationError{Layer: layer, Cause: v41StageErr(v41StageAttention, layer, cause)}
}

func v41RoPEBF16Copy(layer int, values []float32) ([]float32, error) {
	out := make([]float32, len(values))
	for i, v := range values {
		if !finite32(v) {
			return nil, v41RoPEPublicationErr(layer, errV41TailRoPEResult)
		}
		out[i] = v41RoundBF16(v)
		if !finite32(out[i]) {
			return nil, v41RoPEPublicationErr(layer, errV41TailRoPEResult)
		}
	}
	return out, nil
}
