package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RoutedExpertOutcome is the closed result vocabulary of one
// v41RoutedExpertFunc call: declined leaves the pick to the gate/up and down
// seams (or the host triple), handled returns the H-wide down output, and
// error is a selected failure that must surface, never a decline.
type v41RoutedExpertOutcome uint8

const (
	v41RoutedExpertDeclined v41RoutedExpertOutcome = iota
	v41RoutedExpertHandled
	v41RoutedExpertError
)

// v41RoutedExpertFunc runs one full-profile routed expert on the device: the
// gate/up GEMMs, the fused BF16 clamp-SwiGLU route-weight publication
// (compute.V41RoutedSwiGLUBackend) and the down GEMM, with the I-wide
// activation resident between them (Ref #13815). x is the BF16-published
// expert input. On handled it returns the H-wide down output, already published
// as BF16 on a backend with the ordinary publication; the caller's own BF16
// publication is then the identity.
type v41RoutedExpertFunc func(layer int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error)

// v41RoutedExpertFunc binds either the fused resident routed expert or the packed
// expert GEMV path. Per-pick native-weight admission remains a decline; selecting
// any packed projection makes missing siblings and kernel failures fail closed.
func (s *Session) v41RoutedExpertFunc() v41RoutedExpertFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	op, ok := s.Backend.(compute.V41RoutedSwiGLUBackend)
	cfg := s.M.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil
	}
	fusedReady := full && ok && op.V41RoutedSwiGLUAvailable()
	_, packedReady := s.Backend.(compute.RoutedExpertDeviceKernel)
	if !fusedReady && !packedReady {
		return nil
	}
	limit := float32(cfg.SwigluLimit)
	if !finite32(limit) {
		return nil
	}
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	return func(layer int, stem string, x []float32, weight float32) (result []float32, outcome v41RoutedExpertOutcome, cause error) {
		s.ensureOpenBackendSession()
		stage := "gate/up"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-expert", Layer: layer, Stage: stage, Cause: err}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		// Admission stages native expert weights, so its failures belong to this
		// selected callback, like the GEMM, activation and readback failures.
		convertPanic := func(r any) {
			err, isErr := r.(error)
			if isErr {
				var closed *BackendForwardOperationError
				if errors.As(err, &closed) {
					panic(r)
				}
				var backend *compute.BackendError
				if errors.As(err, &backend) {
					result, outcome, cause = nil, v41RoutedExpertError, closeFailure(err)
					return
				}
			}
			if converted, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
				if isErr {
					converted = err
				}
				result, outcome, cause = nil, v41RoutedExpertError, closeFailure(converted)
				return
			}
			if !isErr {
				err = fmt.Errorf("unclassified backend panic: %v", r)
			}
			closeFailure(err)
			panic(r)
		}
		var recovered any
		completed := false
		func() {
			defer func() { recovered = recover() }()
			result, outcome, cause = func() ([]float32, v41RoutedExpertOutcome, error) {
				if H <= 0 || I <= 0 || len(x) != H || !finite32(weight) {
					return nil, v41RoutedExpertError, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
				}
				gateName := layerName(layer, stem+".w1.weight")
				upName := layerName(layer, stem+".w3.weight")
				downName := layerName(layer, stem+".w2.weight")
				weightSuffix := len("weight")
				if s.M.has(gateName[:len(gateName)-weightSuffix]+"bias") ||
					s.M.has(upName[:len(upName)-weightSuffix]+"bias") ||
					s.M.has(downName[:len(downName)-weightSuffix]+"bias") {
					return nil, v41RoutedExpertDeclined, nil
				}
				packedStem := "layers." + itoa(layer) + "." + stem
				packedW1Name := packedStem + ".w1.weight"
				packedUpName := packedStem + ".w3.weight"
				packedDownName := packedStem + ".w2.weight"
				packedCount := 0
				for _, name := range [...]string{packedW1Name, packedUpName, packedDownName} {
					if s.M.V41ExpertDeviceInstalled(name) {
						packedCount++
					}
				}
				if packedCount != 0 {
					if packedCount != 3 {
						metadata := v4QuantMetadataf("V4.1 packed routed expert %s has %d/3 projections", packedStem, packedCount)
						return nil, v41RoutedExpertError, closeFailure(v41ExpertOperationErr(layer, stage, metadata))
					}
					gate, err := s.M.V41ExpertDeviceMatmul(packedW1Name, s.Backend, x)
					if err != nil {
						return nil, v41RoutedExpertError, closeFailure(err)
					}
					up, err := s.M.V41ExpertDeviceMatmul(packedUpName, s.Backend, x)
					if err != nil {
						return nil, v41RoutedExpertError, closeFailure(err)
					}
					gate, err = v41RoutedBF16(layer, "gate/up", gate, I)
					if err != nil {
						return nil, v41RoutedExpertError, err
					}
					up, err = v41RoutedBF16(layer, "gate/up", up, I)
					if err != nil {
						return nil, v41RoutedExpertError, err
					}
					clampSwiGLUProjections(gate, up, limit)
					for i := range gate {
						gate[i] = silu(gate[i]) * up[i] * weight
					}
					activation, err := v41RoutedBF16(layer, "down", gate, I)
					if err != nil {
						return nil, v41RoutedExpertError, err
					}
					stage = "down"
					out, err := s.M.V41ExpertDeviceMatmul(packedDownName, s.Backend, activation)
					if err != nil {
						return nil, v41RoutedExpertError, closeFailure(err)
					}
					out, err = v41RoutedBF16(layer, stage, out, H)
					if err != nil {
						return nil, v41RoutedExpertError, err
					}
					return out, v41RoutedExpertHandled, nil
				}
				if !fusedReady {
					return nil, v41RoutedExpertDeclined, nil
				}
				gateW, releaseGate, upW, releaseUp, input, ok := expertInputDeviceAdmitted(s, gateName, upName, x, H)
				if !ok {
					return nil, v41RoutedExpertDeclined, nil
				}
				defer releaseGate()
				defer releaseUp()
				downW, releaseDown, ok := s.q4kExpertDownDeviceWeight(downName)
				if !ok {
					return nil, v41RoutedExpertDeclined, nil
				}
				defer releaseDown()
				opened := s.M.v41NowNanos()
				xd := s.uploadHostF32([]int{H}, input, compute.MemoryActivation, "moe expert gate/up activation")
				defer s.Backend.Free(xd)
				g := s.Backend.MatMul(gateW, xd)
				defer s.Backend.Free(g)
				u := s.Backend.MatMul(upW, xd)
				defer s.Backend.Free(u)
				stage = "activation"
				activation, err := op.V41RoutedSwiGLU(g, u, limit, weight)
				var fault *compute.V41RoutedSwiGLUFault
				if errors.As(err, &fault) {
					// A numeric verdict on the projections, typed exactly as the host
					// publication would report it. Nothing was published.
					faultStage := "gate/up"
					if fault.Stage == compute.V41RoutedSwiGLUStageDownOperand {
						faultStage = "down"
					}
					return nil, v41RoutedExpertError, v41ExpertOperationErr(layer, faultStage, errV41ExpertResult)
				}
				if err != nil {
					return nil, v41RoutedExpertError, closeFailure(err)
				}
				defer s.Backend.Free(activation)
				s.M.v41NoteExpertActivation(1, 0, 16, opened)
				stage = "down"
				out := s.Backend.MatMul(downW, activation)
				defer s.Backend.Free(out)
				// The down output is published as BF16 on the device before its one
				// readback; a backend without the ordinary publication keeps the caller's
				// host rounding. An overflow is the same numeric verdict the host reports.
				read := out
				if published, ok, err := s.v41PublishResidentBF16(out, compute.V41BF16PublishRoutedDown); ok {
					var numeric *compute.V41BF16CheckedArithmeticError
					if errors.As(err, &numeric) {
						return nil, v41RoutedExpertError, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
					}
					if err != nil {
						return nil, v41RoutedExpertError, closeFailure(err)
					}
					defer s.Backend.Free(published)
					read = published
				}
				// Read may return borrowed scratch; own it before any deferred Free.
				values := append([]float32(nil), s.Backend.Read(read)...)
				if len(values) != H {
					return nil, v41RoutedExpertError, v41ExpertOperationErr(layer, stage, errV41ExpertResult)
				}
				return values, v41RoutedExpertHandled, nil
			}()
			completed = true
		}()
		if !completed {
			convertPanic(recovered)
		}
		return
	}
}

// v41DeviceOnlyRoutedExpert is the device-only routed-expert wrapper (see
// v41BindDeviceOnlyDeclineRefusals): an unbound or declining fused seam refuses
// typed, so the host clamp and SwiGLU arm is unreachable under device-only.
func v41DeviceOnlyRoutedExpert(s *Session, f v41RoutedExpertFunc) v41RoutedExpertFunc {
	return func(layer int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
		if s != nil && s.M != nil {
			packedStem := "layers." + itoa(layer) + "." + stem
			for _, projection := range [...]string{"w1", "w3", "w2"} {
				if s.M.V41ExpertDeviceInstalled(packedStem + "." + projection + ".weight") {
					return nil, v41RoutedExpertError, s.v41DeviceOnlyDeclined(layer, "routed-swiglu-clamp")
				}
			}
		}
		if f != nil {
			y, outcome, err := f(layer, stem, x, weight)
			if outcome != v41RoutedExpertDeclined {
				return y, outcome, err
			}
		}
		return nil, v41RoutedExpertError, s.v41DeviceOnlyDeclined(layer, "routed-swiglu-clamp")
	}
}

// v41DeviceOnlyRoutedSwiGLUBound reports whether this session runs the routed
// clamp and SwiGLU on the device. The full profile needs the fused routed seam;
// the reduced profile clamps through the generic device activation, which needs
// SwiGLUWithLimit for a positive finite limit and none otherwise.
func v41DeviceOnlyRoutedSwiGLUBound(s *Session) bool {
	if s == nil || s.M == nil || s.Backend == nil {
		return false
	}
	full, err := v41ForwardGeometry(s.M.Cfg)
	if err != nil {
		return false
	}
	if full {
		if s.M.V41ExpertDevicePackedPreserved() || !s.Backend.Caps().DeviceMemory {
			return false
		}
		op, ok := s.Backend.(compute.V41RoutedSwiGLUBackend)
		return ok && op.V41RoutedSwiGLUAvailable()
	}
	limit := float32(s.M.Cfg.SwigluLimit)
	if !finite32(limit) {
		return false
	}
	if limit <= 0 {
		return true
	}
	_, limited := s.Backend.(interface {
		SwiGLUWithLimit(compute.Tensor, compute.Tensor, float32) compute.Tensor
	})
	return limited
}
