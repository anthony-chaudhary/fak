package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This private cold composer changes only complete ratio-two producer groups.
// Ordinary/replaced callbacks, incomplete tails and incremental Step keep their
// scalar calls. Projection results stay F32 until the existing pool/norm tail.
func (m *Model) v41ColdCompressedRows(l, ratio int, kvRows, inputs [][]float32, full bool, st *v41ForwardState, scratch *v41ProjScratch) ([][]float32, error) {
	legacy := func() ([][]float32, error) {
		return m.v41CompressedRowsWithDeviceOps(l, ratio, kvRows, inputs, scratch.denseProjection(), scratch.compressorNorm, scratch.gatherOps.compressorPoolFunc())
	}
	if !full || ratio != 2 || len(inputs) < 2 {
		return legacy()
	}
	if rows, handled, err := m.v41ColdCompressorHostPairs(l, inputs, scratch); handled {
		return rows, err
	}
	pair, err := m.v41PrepareColdCompressor(l, inputs, st, scratch)
	if err != nil {
		return nil, err
	}
	if pair == nil {
		return legacy()
	}
	width, hidden := v41CompressorWidth(m.Cfg), m.Cfg.HiddenSize
	pool, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		return nil, v41StageErr(v41StageCompress, l, err)
	}
	if normalize := scratch.compressorNorm; normalize != nil {
		pool.normalize = func(x, gain []float32, eps float32) ([]float32, error) {
			return normalize(l, x, gain, eps)
		}
	}
	pool.bindGatedPool(l, scratch.gatherOps.compressorPoolFunc())
	gain, eps := m.tensor(layerName(l, "attn.compressor.norm.weight")), float32(m.Cfg.RMSNormEps)
	// PushNormalized checks these operands even on a group's first row. An
	// invalid tail must not move behind the second row's speculative projection.
	validPoolOperands := func() bool {
		if len(gain) != width || !finite32(eps) || eps <= 0 {
			return false
		}
		for _, value := range gain {
			if !finite32(value) {
				return false
			}
		}
		return true
	}
	if !validPoolOperands() {
		return legacy()
	}
	var result [][]float32
	for pos := 0; pos < len(inputs); {
		if !pair.valid() {
			return nil, pair.failure(0, "scope", errV41ProjectionResult)
		}
		var projected [2][]float32
		rows := 1
		if pos+1 < len(inputs) && validPoolOperands() {
			rows = 2
			projected, err = pair.project(inputs[pos : pos+2])
		} else {
			for leaf := range projected {
				projected[leaf], err = m.v41ProjMatRowsWithProjection(l, v41ColdCompressorLeaves[leaf], inputs[pos], width, hidden, scratch.denseProjection())
				if err != nil {
					break
				}
			}
		}
		if err != nil {
			return nil, err
		}
		for row := 0; row < rows; row++ {
			pooled, emitted, err := pool.PushNormalized(pos+row, projected[0][row*width:(row+1)*width], projected[1][row*width:(row+1)*width], gain, eps)
			if err != nil {
				return nil, v41StageErr(v41StageCompress, l, err)
			}
			if emitted {
				result = append(result, pooled)
			}
		}
		pos += rows
	}
	return result, nil
}

var v41ColdCompressorLeaves = [2]string{"attn.compressor.wkv.weight", "attn.compressor.wgate.weight"}

type v41ColdCompressorPair struct {
	s       *Session
	model   *Model
	layer   int
	backend compute.Backend
	read    compute.V41DeferredStatusBackend
	valid   func() bool
}

func (m *Model) v41PrepareColdCompressor(l int, inputs [][]float32, st *v41ForwardState, scratch *v41ProjScratch) (*v41ColdCompressorPair, error) {
	if st == nil || scratch == nil || scratch.projectionRoute == nil || m.lora != nil || m.prism != nil {
		return nil, nil
	}
	binding := scratch.projectionRoute
	native := binding.nativeProjection
	s := native.owner
	if native.core == nil || native.ordinary == nil || binding.projection == nil || s == nil || native.model != m || s.M != m ||
		!backendHasComparableIdentity(native.backend) || !backendHasComparableIdentity(s.Backend) || s.Backend != native.backend ||
		s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.halW == nil ||
		st.callbackOwner != s || st.projectionRoute != binding || s.v41Forward == nil || s.v41Forward.callbackOwner != s || s.v41Forward.projectionRoute != binding {
		return nil, nil
	}
	scope, forward, epoch, backend := s.backendExecutionScope, s.v41Forward, s.cacheGeometryEpoch, s.Backend
	if scope == nil || !scope.live || scope.owner == nil {
		return nil, nil
	}
	owner := scope.owner
	valid := func() bool {
		if s.M != m || !backendHasComparableIdentity(s.Backend) || s.Backend != backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld ||
			s.v41Forward != forward || forward.callbackOwner != s || forward.projectionRoute != binding || st.callbackOwner != s || st.projectionRoute != binding ||
			scratch.projectionRoute != binding || s.cacheGeometryEpoch != epoch || s.backendExecutionScope != scope || !scope.live || scope.owner != owner || m.lora != nil || m.prism != nil {
			return false
		}
		for _, member := range scope.sessions {
			if member == s {
				return true
			}
		}
		return false
	}
	if !valid() {
		return nil, nil
	}
	read, haveRead := backend.(compute.V41DeferredStatusBackend)
	provider, haveOwner := backend.(compute.ForwardExecutionOwnerBackend)
	if !haveRead || !haveOwner {
		return nil, nil
	}
	H, width := m.Cfg.HiddenSize, v41CompressorWidth(m.Cfg)
	// A later malformed payload must retain its original scalar failure site.
	// Metadata admission performs no staging, publication or projection.
	for _, input := range inputs {
		if len(input) != H {
			return nil, nil
		}
		for _, value := range input {
			if !finite32(value) {
				return nil, nil
			}
		}
	}
	pair := &v41ColdCompressorPair{s: s, model: m, layer: l, backend: backend, read: read, valid: valid}
	for i, leaf := range v41ColdCompressorLeaves {
		layout, ok := m.v41HeadLayoutSnapshot(layerName(l, leaf), width, H, false)
		// Quantized P>1 contractions can select different cooperative kernels.
		// Dtype support alone does not certify their scalar arithmetic contract.
		if !ok || layout.dtype != compute.F32 {
			return nil, nil
		}
		_, supported := v41HeadLayoutSupported(backend, layout)
		if !valid() {
			return nil, pair.failure(i, "admission", errV41ProjectionResult)
		}
		if !supported {
			return nil, nil
		}
	}
	// A zero-status grouped read needs no checked-arithmetic capability probe.
	actualOwner := provider.ForwardExecutionOwner()
	if !valid() || actualOwner != owner {
		return nil, pair.failure(0, "admission", errV41ProjectionResult)
	}
	return pair, nil
}

func (p *v41ColdCompressorPair) failure(leaf int, stage string, cause error) error {
	s := p.s
	if s.halFailure == nil {
		s.halFailure = &BackendForwardOperationError{Backend: p.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-compressor-projection", Layer: p.layer, Stage: stage, Cause: cause}
	}
	if !s.halClosed {
		func() {
			defer func() { s.halClosed = true; _ = recover() }()
			s.closeBackendOwned()
		}()
	}
	return v41ProjectionOperationErr(p.layer, v41ColdCompressorLeaves[leaf], s.halFailure)
}

// Physical dispatch is leaf-major. Numeric verdicts remain row-major; a later
// recording/cleanup fault cannot hide an earlier known arithmetic failure.
// Transport failure makes every numeric verdict unknown. No issued work replays.
func (p *v41ColdCompressorPair) project(inputs [][]float32) (result [2][]float32, err error) {
	s, backend := p.s, p.backend
	H, width := p.model.Cfg.HiddenSize, v41CompressorWidth(p.model.Cfg)
	opened := p.model.v41NowNanos()
	calls, successful := 0, 0
	var uploaded, readback int64
	defer func() { p.model.v41NoteCompressorProjection(calls, 0, successful, 0, uploaded, readback, opened) }()
	leaf, stage, fault := 0, "scope", 4
	var primary any
	var cause error
	completed := false
	defer func() {
		if !completed {
			retirement := cause
			if retirement == nil {
				retirement, _ = primary.(error)
			}
			if retirement == nil {
				retirement = errV41ProjectionResult
			}
			p.failure(leaf%2, stage, retirement)
		}
	}()
	setFailure := func(order int, at string, value error, panicValue any) {
		if order <= fault {
			fault, leaf, stage, cause, primary = order, order%2, at, value, panicValue
		}
	}
	type ownedTensor struct {
		tensor compute.Tensor
		order  int
	}
	var owned []ownedTensor
	var storageBackend compute.Backend
	seen := map[compute.Buffer]bool{}
	for _, weight := range s.halW {
		seen[weight.Buf()] = true
	}
	own := func(t compute.Tensor, order int) bool {
		if t.Buf() == nil || !backendHasComparableIdentity(t.Backend()) || !backendHasComparableIdentity(storageBackend) || t.Backend() != storageBackend || seen[t.Buf()] {
			return false
		}
		seen[t.Buf()] = true
		owned = append(owned, ownedTensor{t, order})
		return true
	}
	var outputs []compute.Tensor
	var input compute.Tensor
	inputReady := false
	readAttempted, transportFailed, released := false, false, false
	drainRows := func() []compute.Tensor {
		if len(outputs) == 0 && inputReady {
			return []compute.Tensor{input}
		}
		return outputs
	}
	release := func() {
		released = true
		// Schedule every release before invoking any backend method. Goexit in
		// the drain or one Free still runs all remaining defers and retirement.
		for i := range owned {
			item := owned[i]
			defer func() {
				freed := false
				defer func() {
					value := recover()
					if !freed && !transportFailed && (cause == nil && primary == nil || item.order < fault) {
						setFailure(item.order, "cleanup", nil, value)
						if value == nil {
							setFailure(item.order, "cleanup", errV41ProjectionResult, nil)
						}
					}
				}()
				backend.Free(item.tensor)
				freed = true
			}()
		}
		if rows := drainRows(); !readAttempted && len(rows) != 0 {
			readAttempted = true
			_, _, drainErr := p.read.V41ReadDeferred(nil, rows)
			if drainErr != nil {
				setFailure(0, "paired readback", drainErr, nil)
			}
		}
	}
	defer func() {
		if !released {
			release()
		}
	}()
	value, panicked := v41ColdCompressorCapture(func() {
		if !p.valid() {
			setFailure(0, stage, errV41ProjectionResult, nil)
			return
		}
		panel := make([]float32, 0, 2*H)
		panel = append(append(panel, inputs[0]...), inputs[1]...)
		for leaf = 0; leaf < 2; leaf++ {
			calls++
			stage = "weight upload"
			w := s.v41HeadWeightHAL(layerName(p.layer, v41ColdCompressorLeaves[leaf]), compute.F32, width, H)
			seen[w.Buf()] = true
			if leaf == 0 {
				storageBackend = w.Backend()
			}
			if !p.valid() || w.Buf() == nil || !backendHasComparableIdentity(w.Backend()) || w.Backend() != storageBackend || w.Dtype != compute.F32 || w.Layout != compute.RowMajor ||
				len(w.Shape) != 2 || w.Shape[0] != width || w.Shape[1] != H {
				setFailure(leaf, stage, errV41ProjectionResult, nil)
				return
			}
			if leaf == 0 {
				stage = "activation upload"
				input = s.uploadHostF32([]int{2, H}, panel, compute.MemoryActivation, "V4.1 cold compressor pair")
				fresh := own(input, 0)
				uploaded = int64(len(panel)) * 4
				inputReady = fresh && input.Dtype == compute.F32 && input.Layout == compute.RowMajor && input.Quant == nil &&
					len(input.Shape) == 2 && input.Shape[0] == 2 && input.Shape[1] == H
				if !inputReady || !p.valid() {
					setFailure(0, stage, errV41ProjectionResult, nil)
					return
				}
			}
			stage = "matmul"
			y := backend.BatchedMatMul(w, input, 2)
			fresh := own(y, leaf)
			if !fresh || y.Backend() != w.Backend() || y.Dtype != compute.F32 || y.Layout != compute.RowMajor || y.Quant != nil ||
				len(y.Shape) != 2 || y.Shape[0] != 2 || y.Shape[1] != width {
				setFailure(leaf, stage, errV41ProjectionResult, nil)
				return
			}
			outputs = append(outputs, y)
			if !p.valid() {
				setFailure(leaf, stage, errV41ProjectionResult, nil)
				return
			}
		}
	})
	if panicked {
		setFailure(leaf, stage, nil, value)
		if value == nil {
			setFailure(leaf, stage, errV41ProjectionResult, nil)
		}
	}
	// A failed/malformed first contraction may still have recorded work. Fence
	// through the valid activation when no valid output can be read.
	drain := drainRows()
	if len(drain) > 0 {
		func() {
			completed := false
			defer func() {
				value := recover()
				if !completed {
					transportFailed = true
					fault, leaf, stage, cause, primary = 0, 0, "paired readback", nil, value
					if value == nil {
						cause = errV41ProjectionResult
					}
				}
			}()
			readAttempted = true
			verdicts, values, readErr := p.read.V41ReadDeferred(nil, drain)
			completed = true
			if readErr != nil {
				transportFailed = true
				fault, leaf, stage, cause, primary = 0, 0, "paired readback", readErr, nil
				return
			}
			if len(verdicts) != 0 || len(values) != len(drain) || !p.valid() {
				setFailure(0, "paired readback", errV41ProjectionResult, nil)
				return
			}
			for _, values := range values {
				readback += int64(len(values)) * 4
			}
			for row := 0; row < 2; row++ {
				for which := range outputs {
					order := row*2 + which
					if order >= fault {
						continue
					}
					if len(values[which]) != 2*width {
						setFailure(order, "readback", errV41ProjectionResult, nil)
						continue
					}
					for _, value := range values[which][row*width : (row+1)*width] {
						if !finite32(value) {
							setFailure(order, "readback", errV41ProjectionResult, nil)
							break
						}
					}
				}
			}
			if fault == 4 {
				for which := range outputs {
					result[which] = append([]float32(nil), values[which]...)
				}
			}
		}()
	}
	release()
	if fault == 4 && !p.valid() {
		setFailure(0, "cleanup", errV41ProjectionResult, nil)
	}
	if fault != 4 {
		if primary != nil {
			if payload, ok := primary.(v41HeadPayloadError); ok {
				cause, primary = payload.cause, nil
			} else if value, ok := primary.(error); ok {
				var typed *compute.BackendError
				if errors.As(value, &typed) {
					cause, primary = value, nil
				}
			}
		}
		if primary != nil {
			if converted, ok := compute.ConvertCUDAPanic(primary, "", ""); ok {
				cause, primary = converted, nil
			}
		}
		if primary != nil {
			value, ok := primary.(error)
			if !ok {
				value = fmt.Errorf("unclassified backend panic: %v", primary)
			}
			p.failure(leaf, stage, value)
			completed = true
			panic(primary)
		}
		err = p.failure(leaf, stage, cause)
		completed = true
		return [2][]float32{}, err
	}
	successful = 4
	completed = true
	return result, nil
}

// A completion bit also detects legacy GODEBUG=panicnil=1 panics.
func v41ColdCompressorCapture(run func()) (value any, panicked bool) {
	completed := false
	defer func() {
		value = recover()
		panicked = !completed
	}()
	run()
	completed = true
	return nil, false
}
