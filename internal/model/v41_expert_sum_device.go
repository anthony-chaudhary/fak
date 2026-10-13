package model

import (
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Device seams for the full-geometry routed-expert sum and the shared-expert
// add (Refs #13668). Both compose the backend's F32 AddInPlace, so no new
// kernel is needed. A nil callback keeps the host sites; a selected call has no
// decline arm and never replays on the host.
type v41RoutedSumFunc func(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error)
type v41SharedAddFunc func(layer int, routed, shared []float32) ([]float32, error)

var errV41DeviceSumResult = errors.New("model: invalid selected V4.1 device expert sum result")

func v41FullRoutedSumWith(layer int, picks []routePick, outputs [][]float32, width int, device v41RoutedSumFunc) ([]float32, error) {
	if device == nil {
		return v41FullRoutedSum(layer, picks, outputs, width)
	}
	sum, err := device(layer, picks, outputs, width)
	if err != nil {
		return nil, err
	}
	if len(sum) != width {
		return nil, v41ExpertOperationErr(layer, "sum", errV41DeviceSumResult)
	}
	return sum, nil
}

func v41SharedExpertAddWith(layer int, routed, shared []float32, cfg v41RouterConfig, device v41SharedAddFunc) ([]float32, error) {
	if device == nil {
		return v41SharedExpertAdd(routed, shared, cfg)
	}
	if cfg.SharedCount < 1 {
		return nil, &v4RouteError{Field: "v41_shared", Reason: fmt.Sprintf("want >=1, got %d", cfg.SharedCount)}
	}
	out, err := device(layer, routed, shared)
	if err != nil {
		return nil, err
	}
	if len(out) != len(routed) {
		return nil, &v4RouteError{Field: "shared_add", Reason: errV41DeviceSumResult.Error()}
	}
	return out, nil
}

// v41RoutedSumOrder is the host v41FullRoutedSum contribution order: ascending
// expert, stable slot order on equal experts.
func v41RoutedSumOrder(picks []routePick) []int {
	order := make([]int, len(picks))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return picks[order[i]].expert < picks[order[j]].expert })
	return order
}

// v41DeviceAccumulate adds device-resident rows into acc in the given order,
// one F32 AddInPlace per row. With acc zero-initialized this is the host
// sum[i] = float32(sum[i] + v) sequence, including +0 + -0 = +0. Bit parity
// with the host holds on a backend whose add is IEEE round-to-nearest; a
// backend that flushes F32 subnormals differs only on subnormal operands.
func v41DeviceAccumulate(be compute.Backend, acc compute.Tensor, rows []compute.Tensor, order []int) {
	for _, slot := range order {
		be.AddInPlace(acc, rows[slot])
	}
}

func (s *Session) v41DeviceSumAvailable() bool {
	return s != nil && s.M != nil && s.Backend != nil && s.Backend.Caps().DeviceMemory
}

// v41SelectedDeviceCall runs one selected V4.1 device operation. Every
// failure after selection closes the backend session, like the other seams.
func v41SelectedDeviceCall[T any](s *Session, layer int, path string, run func(stage *string) (T, error)) (result T, cause error) {
	var zero T
	s.ensureOpenBackendSession()
	stage := "payload"
	closeSelected := func(primary error) {
		if s.halFailure == nil {
			s.halFailure = primary
		}
		if s.halClosed {
			return
		}
		// An incoming typed fault can belong to another operation or owner.
		// Retire this owner without replacing its primary if Close also fails.
		defer func() { s.halClosed = true; _ = recover() }()
		s.closeBackendOwned()
	}
	closeFailure := func(err error) error {
		if s.halFailure != nil {
			closeSelected(s.halFailure)
			return s.halFailure
		}
		closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: path, Layer: layer, Stage: stage, Cause: err}
		closeSelected(closed)
		return closed
	}
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok {
				var closed *BackendForwardOperationError
				if errors.As(err, &closed) {
					closeSelected(err)
					panic(r)
				}
				var backend *compute.BackendError
				if errors.As(err, &backend) {
					result, cause = zero, closeFailure(err)
					return
				}
			}
			if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
				if original, ok := r.(error); ok {
					err = original
				}
				result, cause = zero, closeFailure(err)
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
	result, cause = run(&stage)
	if cause != nil {
		return zero, closeFailure(cause)
	}
	return result, nil
}

// v41DeviceSumReadback owns a checked copy of a device row. Read may expose
// reusable backend memory, so the copy is taken before any Free.
func (s *Session) v41DeviceSumReadback(t compute.Tensor, width int, invalid error) ([]float32, error) {
	values := s.Backend.Read(t)
	if len(values) != width {
		return nil, invalid
	}
	for _, v := range values {
		if !finite32(v) {
			return nil, invalid
		}
	}
	return append([]float32(nil), values...), nil
}

// v41RoutedSumFunc binds the full-geometry routed-expert sum to the session
// backend. It uploads a zero accumulator and each expert row, adds them on the
// device in host order, and reads back one H-wide row: 4*H*(k+1) upload bytes
// and 4*H readback bytes per call. A resident caller uses v41DeviceAccumulate
// directly and skips both transfers.
func (s *Session) v41RoutedSumFunc() v41RoutedSumFunc {
	if !s.v41DeviceSumAvailable() {
		return nil
	}
	return func(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error) {
		invalid := v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
		return v41SelectedDeviceCall(s, layer, "v41-routed-sum", func(stage *string) ([]float32, error) {
			if len(picks) != len(outputs) || width <= 0 {
				return nil, invalid
			}
			for _, row := range outputs {
				if len(row) != width {
					return nil, invalid
				}
				for _, v := range row {
					if !finite32(v) {
						return nil, invalid
					}
				}
			}
			*stage = "accumulator upload"
			acc := s.uploadHostF32([]int{width}, make([]float32, width), compute.MemoryActivation, "V4.1 routed-sum accumulator")
			defer s.Backend.Free(acc)
			rows := make([]compute.Tensor, len(outputs))
			*stage = "expert upload"
			for slot, row := range outputs {
				rows[slot] = s.uploadHostF32([]int{width}, row, compute.MemoryActivation, "V4.1 routed-sum expert output")
				defer s.Backend.Free(rows[slot])
			}
			*stage = "routed-sum"
			v41DeviceAccumulate(s.Backend, acc, rows, v41RoutedSumOrder(picks))
			*stage = "readback"
			return s.v41DeviceSumReadback(acc, width, invalid)
		})
	}
}

// v41SharedAddFunc binds the routed + shared add to the session backend: it
// uploads both rows, adds shared into an owned copy of routed on the device,
// and reads back one row (8*H upload, 4*H readback bytes per call).
func (s *Session) v41SharedAddFunc() v41SharedAddFunc {
	if !s.v41DeviceSumAvailable() {
		return nil
	}
	return func(layer int, routed, shared []float32) ([]float32, error) {
		return v41SelectedDeviceCall(s, layer, "v41-shared-add", func(stage *string) ([]float32, error) {
			if len(routed) == 0 || len(shared) == 0 {
				return nil, &v4RouteError{Field: "shared_add", Reason: "empty operand"}
			}
			if len(routed) != len(shared) {
				return nil, &v4RouteError{Field: "shared_add", Reason: fmt.Sprintf("width %d does not match %d", len(routed), len(shared))}
			}
			for i := range routed {
				if !finite32(routed[i]) {
					return nil, &v4RouteError{Field: "routed", Reason: fmt.Sprintf("non-finite value at %d", i)}
				}
				if !finite32(shared[i]) {
					return nil, &v4RouteError{Field: "shared", Reason: fmt.Sprintf("non-finite value at %d", i)}
				}
			}
			width := len(routed)
			*stage = "routed upload"
			acc := s.uploadHostF32([]int{width}, append([]float32(nil), routed...), compute.MemoryActivation, "V4.1 shared-add accumulator")
			defer s.Backend.Free(acc)
			*stage = "shared upload"
			sh := s.uploadHostF32([]int{width}, shared, compute.MemoryActivation, "V4.1 shared-add shared output")
			defer s.Backend.Free(sh)
			*stage = "shared-add"
			s.Backend.AddInPlace(acc, sh)
			*stage = "readback"
			return s.v41DeviceSumReadback(acc, width, &v4RouteError{Field: "shared_add", Reason: "non-finite or malformed device result"})
		})
	}
}
