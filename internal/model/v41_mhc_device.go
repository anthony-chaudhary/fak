package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// V41MHCOperationError identifies a selected mHC stream operation that failed
// on the device, or a device-only refusal because no device operation exists.
// It never wraps ErrV41ForwardStage, so no caller may treat it as permission to
// replay the forward on the host.
type V41MHCOperationError struct {
	Layer int
	Op    string
	Cause error
}

func (e *V41MHCOperationError) Error() string {
	return fmt.Sprintf("model: selected V4.1 mHC %s failed at layer %d: %v", e.Op, e.Layer, e.Cause)
}

func (e *V41MHCOperationError) Unwrap() error { return e.Cause }

var (
	errV41MHCResult = errors.New("model: invalid selected V4.1 mHC result")
	// errV41MHCDeviceUnavailable is the device-only refusal when the backend
	// has no mHC stream operation, or the forward is the reduced geometry that
	// has no device contract. There is no host-compute escape hatch.
	errV41MHCDeviceUnavailable = errors.New("model: V4.1 mHC stream operations have no device dispatch on this backend")
)

// v41MHCDevice is the session-bound device seam for the mHC stream operations:
// initial streams, the Sinkhorn split, the pre-collapse and the post-mix. A nil
// *v41MHCDevice keeps the host reference. A non-nil seam never runs host math:
// either every operation dispatches on the backend, or (device-only policy with
// no capability) every operation refuses.
type v41MHCDevice struct {
	s       *Session
	op      compute.V41MHCBackend
	refusal error
}

// v41MHCDeviceSeam binds the mHC seam for a session. A backend with the
// capability selects it under every policy; device-only without it binds a
// refusing seam; otherwise the host reference stays selected.
func (s *Session) v41MHCDeviceSeam() *v41MHCDevice {
	if s == nil || s.M == nil {
		return nil
	}
	if s.Backend != nil && s.Backend.Caps().DeviceMemory && compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		if op, ok := s.Backend.(compute.V41MHCBackend); ok && op.SupportsV41MHC() {
			return &v41MHCDevice{s: s, op: op}
		}
	}
	if s.executionPolicy == ExecutionPolicyDeviceOnly {
		return &v41MHCDevice{s: s, refusal: errV41MHCDeviceUnavailable}
	}
	return nil
}

// refuseReduced returns the device-only refusal for the reduced (fixture)
// geometry, whose unrounded pre/post arithmetic has no device contract.
func (d *v41MHCDevice) refuseReduced() error {
	if d == nil || d.s == nil || d.s.executionPolicy != ExecutionPolicyDeviceOnly {
		return nil
	}
	return d.fail(-1, "reduced-geometry", "admission", errV41MHCDeviceUnavailable)
}

// fail closes the selected session and returns the typed operation error. The
// session is retired exactly like any other selected device failure.
func (d *v41MHCDevice) fail(layer int, op, stage string, cause error) error {
	backend := ""
	if d.s.Backend != nil {
		backend = d.s.Backend.Name()
	}
	closed := &BackendForwardOperationError{Backend: backend, Forward: ForwardPathKind("deepseek41"), Path: "v41-mhc-" + op, Layer: layer, Stage: stage, Cause: cause}
	if errors.Is(cause, errV41MHCDeviceUnavailable) {
		d.s.hostFallbackObserved = true
	}
	d.s.halFailure = closed
	d.s.halClosed = true
	d.s.Close()
	return &V41MHCOperationError{Layer: layer, Op: op, Cause: closed}
}

// run executes one device operation: upload the host operands, dispatch, read
// the output back and own it before every device allocation is released.
// Backend panics close the session; programming panics keep their identity.
func (d *v41MHCDevice) run(layer int, op string, want int, inputs []v41MHCHostOperand, dispatch func([]compute.Tensor) (compute.Tensor, error)) (result []float32, cause error) {
	if d.refusal != nil {
		return nil, d.fail(layer, op, "admission", d.refusal)
	}
	s := d.s
	s.ensureOpenBackendSession()
	stage := "payload"
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok {
				var closed *BackendForwardOperationError
				if errors.As(err, &closed) {
					panic(r)
				}
				var backend *compute.BackendError
				if errors.As(err, &backend) {
					result, cause = nil, d.fail(layer, op, stage, err)
					return
				}
			}
			if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
				if original, ok := r.(error); ok {
					err = original
				}
				result, cause = nil, d.fail(layer, op, stage, err)
				return
			}
			err, ok := r.(error)
			if !ok {
				err = fmt.Errorf("unclassified backend panic: %v", r)
			}
			d.fail(layer, op, stage, err)
			panic(r)
		}
	}()
	uploaded := make([]compute.Tensor, 0, len(inputs))
	defer func() {
		for _, t := range uploaded {
			s.Backend.Free(t)
		}
	}()
	for _, in := range inputs {
		stage = in.name + " upload"
		uploaded = append(uploaded, s.uploadHostF32(in.shape, in.values, compute.MemoryActivation, "V4.1 mHC "+op+" "+in.name))
	}
	stage = op
	out, err := dispatch(uploaded)
	// Even an error may return a partially allocated output. Free a fresh output
	// once, never an input it might alias.
	fresh := out.Buf() != nil && out.Backend() == s.Backend
	for _, t := range uploaded {
		fresh = fresh && out.Buf() != t.Buf()
	}
	if fresh {
		defer s.Backend.Free(out)
	}
	if err != nil {
		return nil, d.fail(layer, op, stage, err)
	}
	if !fresh || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil {
		return nil, d.fail(layer, op, stage, errV41MHCResult)
	}
	stage = "readback"
	values := s.Backend.Read(out)
	if len(values) != want {
		return nil, d.fail(layer, op, stage, errV41MHCResult)
	}
	for _, v := range values {
		if !finite32(v) {
			return nil, d.fail(layer, op, stage, errV41MHCResult)
		}
	}
	// Read may expose reusable backend memory; own the row before release.
	return append([]float32(nil), values...), nil
}

type v41MHCHostOperand struct {
	name   string
	shape  []int
	values []float32
}

func v41MHCStreamsRow(streams [][]float32) ([]float32, int, error) {
	if len(streams) != compute.V41MHCStreams || len(streams[0]) == 0 {
		return nil, 0, errV41MHCResult
	}
	H := len(streams[0])
	flat := make([]float32, 0, compute.V41MHCStreams*H)
	for _, row := range streams {
		if len(row) != H {
			return nil, 0, errV41MHCResult
		}
		flat = append(flat, row...)
	}
	return flat, H, nil
}

func v41MHCCoeffRow(pre, post, comb []float32) []float32 {
	row := make([]float32, compute.V41MHCCoeffWidth)
	copy(row[0:4], pre)
	copy(row[4:8], post)
	copy(row[8:], comb)
	return row
}

func v41MHCSplitStreams(flat []float32, H int) [][]float32 {
	out := make([][]float32, compute.V41MHCStreams)
	for h := range out {
		out[h] = flat[h*H : (h+1)*H : (h+1)*H]
	}
	return out
}

// initialStreams is the device twin of v41FullInitialStreams.
func (d *v41MHCDevice) initialStreams(x []float32) ([][]float32, error) {
	for _, v := range x {
		if !finite32(v) {
			// The host refuses the same input before any selection.
			return v41FullInitialStreams(x)
		}
	}
	H := len(x)
	if H == 0 {
		return nil, d.fail(-1, "initial-streams", "admission", errV41MHCResult)
	}
	flat, err := d.run(-1, "initial-streams", compute.V41MHCStreams*H,
		[]v41MHCHostOperand{{"x", []int{1, H}, x}},
		func(t []compute.Tensor) (compute.Tensor, error) { return d.op.V41MHCInitialStreams(t[0], 1, H) })
	if err != nil {
		return nil, err
	}
	return v41MHCSplitStreams(flat, H), nil
}

// split is the device twin of v41MHCSplit (hc fixed at four).
func (d *v41MHCDevice) split(layer int, mixes, scale, base []float32, hc, iters int, eps float32) (v41MHCMix, error) {
	n := (2 + hc) * hc
	if hc != compute.V41MHCStreams || iters < 1 || !finite32(eps) || eps <= 0 || len(mixes) != n || len(base) != n || len(scale) != 3 {
		// Geometry refusals stay the host's typed stage errors.
		mix, err := v41MHCSplit(mixes, scale, base, hc, iters, eps)
		if err == nil {
			err = errV41MHCResult
		}
		return mix, v41StageErr(v41StageMHC, layer, err)
	}
	for _, row := range [][]float32{mixes, scale, base} {
		for _, v := range row {
			if !finite32(v) {
				_, err := v41MHCSplit(mixes, scale, base, hc, iters, eps)
				return v41MHCMix{}, v41StageErr(v41StageMHC, layer, err)
			}
		}
	}
	coeffs, err := d.run(layer, "split", compute.V41MHCCoeffWidth,
		[]v41MHCHostOperand{{"mixes", []int{1, compute.V41MHCCoeffWidth}, mixes}, {"scale", []int{3}, scale}, {"base", []int{compute.V41MHCCoeffWidth}, base}},
		func(t []compute.Tensor) (compute.Tensor, error) {
			return d.op.V41MHCSplit(t[0], t[1], t[2], 1, iters, eps)
		})
	if err != nil {
		return v41MHCMix{}, err
	}
	return v41MHCMix{pre: coeffs[0:4:4], post: coeffs[4:8:8], comb: coeffs[8:24:24]}, nil
}

// preBF16 is the device twin of v41MHCPreBF16.
func (d *v41MHCDevice) preBF16(layer int, streams [][]float32, pre []float32) ([]float32, error) {
	flat, H, err := v41MHCStreamsRow(streams)
	if err != nil || len(pre) != compute.V41MHCStreams {
		return nil, v41StageErr(v41StageMHC, layer, fmt.Errorf("model: invalid V41 mHC pre shape"))
	}
	return d.run(layer, "pre-collapse", H,
		[]v41MHCHostOperand{{"streams", []int{1, compute.V41MHCStreams, H}, flat}, {"coeffs", []int{1, compute.V41MHCCoeffWidth}, v41MHCCoeffRow(pre, nil, nil)}},
		func(t []compute.Tensor) (compute.Tensor, error) { return d.op.V41MHCPre(t[0], t[1], 1, H) })
}

// postBF16 is the device twin of v41MHCPostBF16.
func (d *v41MHCDevice) postBF16(layer int, x []float32, streams [][]float32, mix v41MHCMix) ([][]float32, error) {
	for _, v := range x {
		if !finite32(v) {
			// The host's BF16 publication refuses a non-finite sublayer result.
			return v41MHCPostBF16(layer, x, streams, mix)
		}
	}
	flat, H, err := v41MHCStreamsRow(streams)
	if err != nil || len(x) != H || len(mix.pre) != 4 || len(mix.post) != 4 || len(mix.comb) != 16 {
		return nil, v41StageErr(v41StageMHC, layer, fmt.Errorf("model: invalid V41 mHC post shape"))
	}
	next, err := d.run(layer, "post-mix", compute.V41MHCStreams*H,
		[]v41MHCHostOperand{{"x", []int{1, H}, x}, {"streams", []int{1, compute.V41MHCStreams, H}, flat}, {"coeffs", []int{1, compute.V41MHCCoeffWidth}, v41MHCCoeffRow(mix.pre, mix.post, mix.comb)}},
		func(t []compute.Tensor) (compute.Tensor, error) { return d.op.V41MHCPost(t[0], t[1], t[2], 1, H) })
	if err != nil {
		return nil, err
	}
	return v41MHCSplitStreams(next, H), nil
}

// Host-or-device dispatch for the full-geometry mHC call sites. A nil seam is
// the host reference byte-for-byte.

func v41FullInitialStreamsOn(d *v41MHCDevice, x []float32) ([][]float32, error) {
	if d == nil {
		return v41FullInitialStreams(x)
	}
	return d.initialStreams(x)
}

func v41MHCSplitOn(d *v41MHCDevice, layer int, mixes, scale, base []float32, hc, iters int, eps float32) (v41MHCMix, error) {
	if d == nil {
		mix, err := v41MHCSplit(mixes, scale, base, hc, iters, eps)
		if err != nil {
			return v41MHCMix{}, v41StageErr(v41StageMHC, layer, err)
		}
		return mix, nil
	}
	return d.split(layer, mixes, scale, base, hc, iters, eps)
}

func v41MHCPreBF16On(d *v41MHCDevice, layer int, streams [][]float32, pre []float32) ([]float32, error) {
	if d == nil {
		return v41MHCPreBF16(layer, streams, pre)
	}
	return d.preBF16(layer, streams, pre)
}

func v41MHCPostBF16On(d *v41MHCDevice, layer int, x []float32, streams [][]float32, mix v41MHCMix) ([][]float32, error) {
	if d == nil {
		return v41MHCPostBF16(layer, x, streams, mix)
	}
	return d.postBF16(layer, x, streams, mix)
}
