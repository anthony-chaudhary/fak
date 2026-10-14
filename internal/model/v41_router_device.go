package model

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RouterTopKFunc routes one token on the session backend (Refs #13668). A
// nil callback keeps the host route; a selected call has no decline arm and
// never replays on the host.
type v41RouterTopKFunc func(layer int, logits, correctionBias []float32, cfg v41RouterConfig, full bool) ([]routePick, error)

var errV41DeviceRouteResult = errors.New("model: invalid selected V4.1 device route result")

func v41RouteForGeometryWith(layer int, logits, correctionBias []float32, cfg v41RouterConfig, full bool, device v41RouterTopKFunc) ([]routePick, error) {
	if device == nil {
		return v41RouteForGeometry(logits, correctionBias, cfg, full)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	picks, err := device(layer, logits, correctionBias, cfg, full)
	if err != nil {
		return nil, err
	}
	if len(picks) != cfg.TopK {
		return nil, &v4RouteError{Field: "device_route", Reason: errV41DeviceRouteResult.Error()}
	}
	return picks, nil
}

// v41DeviceRoute routes rows of device-resident router logits in one dispatch.
// Only the k picks per row (indices, weights, status words) are read back: the
// host needs them to schedule routed-expert paging into the activated-expert
// ring, and the logits stay on the device. A numeric fault maps to the host
// router's refusal field.
func v41DeviceRoute(op compute.V41RouterTopKBackend, logits, bias compute.Tensor, rows int, cfg v41RouterConfig, full, hasBias bool) ([][]routePick, error) {
	rowsOut, _, err := v41DeviceRouteCore(op, logits, bias, rows, cfg, full, hasBias)
	return rowsOut, err
}

func v41DeviceRouteCore(op compute.V41RouterTopKBackend, logits, bias compute.Tensor, rows int, cfg v41RouterConfig, full, hasBias bool) ([][]routePick, bool, error) {
	got, err := op.V41RouterTopK(logits, bias, rows, cfg.Experts, cfg.TopK, cfg.RouteScale, full, hasBias)
	var fault *compute.V41RouterTopKFault
	numericFault := errors.As(err, &fault)
	completed := err == nil || numericFault
	if numericFault {
		return nil, completed, v41RouterFaultError(fault)
	}
	if err != nil {
		return nil, completed, err
	}
	if len(got.Experts) != rows*cfg.TopK || len(got.Weights) != rows*cfg.TopK {
		return nil, completed, &v4RouteError{Field: "device_route", Reason: errV41DeviceRouteResult.Error()}
	}
	out := make([][]routePick, rows)
	for row := range out {
		out[row] = make([]routePick, cfg.TopK)
		for slot := range out[row] {
			i := row*cfg.TopK + slot
			e, w := int(got.Experts[i]), got.Weights[i]
			if e < 0 || e >= cfg.Experts || !finite32(w) {
				return nil, completed, &v4RouteError{Field: "device_route", Reason: errV41DeviceRouteResult.Error()}
			}
			out[row][slot] = routePick{expert: e, weight: w}
		}
	}
	return out, completed, nil
}

func v41RouterFaultError(f *compute.V41RouterTopKFault) error {
	switch f.Stage {
	case compute.V41RouterTopKStageLogit:
		return &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite value at expert %d", f.Expert)}
	case compute.V41RouterTopKStageScore:
		return &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite score at expert %d", f.Expert)}
	case compute.V41RouterTopKStageBias:
		return &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("non-finite value at expert %d", f.Expert)}
	case compute.V41RouterTopKStageSelection:
		return &v4RouteError{Field: "selection_score", Reason: fmt.Sprintf("non-finite value at expert %d", f.Expert)}
	case compute.V41RouterTopKStageNormalization:
		return &v4RouteError{Field: "normalization", Reason: "selected score sum must be finite and positive"}
	case compute.V41RouterTopKStageWeight:
		return &v4RouteError{Field: "weight", Reason: fmt.Sprintf("non-finite result for expert %d", f.Expert)}
	}
	return &v4RouteError{Field: "device_route", Reason: f.Error()}
}

// v41RouterTopKFunc binds the router to the session backend when it offers the
// optional V4.1 router primitive. This host-row adapter uploads the 4*E logit
// row per call, borrowing canonical checkpoint bias from the immutable owner.
// Arbitrary caller bias remains a transient 4*E upload. It reads back 8*k pick
// bytes plus 16 status bytes; v41DeviceRoute also accepts resident logits.
func (s *Session) v41RouterTopKFunc() v41RouterTopKFunc { return s.v41BindRouterTopK().ordinary }

func (s *Session) v41BindRouterTopK() v41BoundRouterTopK {
	if !s.v41DeviceSumAvailable() {
		return v41BoundRouterTopK{}
	}
	boundBackend := s.Backend
	op, ok := boundBackend.(compute.V41RouterTopKBackend)
	if !ok || !op.V41RouterTopKAvailable() {
		return v41BoundRouterTopK{}
	}
	core := func(layer int, logits, correctionBias []float32, cfg v41RouterConfig, full bool, residentInput *v41RouterResidentInput) ([]routePick, error) {
		return v41SelectedDeviceCall(s, layer, "v41-router-topk", func(stage *string) (result []routePick, cause error) {
			if residentInput != nil {
				defer func() {
					primary := recover()
					cleanup := residentInput.loan.release()
					if primary != nil {
						panic(primary)
					}
					if cause == nil && cleanup != nil {
						panic(cleanup)
					}
				}()
			}
			// Release only call-owned uploads. Preserve a primary selected
			// failure if a later Free also fails, while attempting both frees.
			var owned []compute.Tensor
			defer func() {
				primaryPanic := recover()
				var cleanupPanic any
				for i := len(owned) - 1; i >= 0; i-- {
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
			if err := cfg.validate(); err != nil {
				return nil, err
			}
			width := len(logits)
			if residentInput != nil {
				if residentInput.loan.valid == nil || !residentInput.loan.valid() {
					return nil, errV41RouterProjectionScope
				}
				lt := residentInput.tensor
				if lt.Buf() == nil || lt.Backend() != s.Backend || lt.Dtype != compute.F32 || lt.Layout != compute.RowMajor || lt.Quant != nil || len(lt.Shape) != 2 || lt.Shape[0] != 1 {
					return nil, errV41DeviceRouteResult
				}
				width = lt.Shape[1]
			}
			if width != cfg.Experts {
				return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("width %d, want %d", width, cfg.Experts)}
			}
			hasBias := len(correctionBias) != 0
			if hasBias && len(correctionBias) != cfg.Experts {
				return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("width %d does not match experts width %d", len(correctionBias), cfg.Experts)}
			}
			var lt compute.Tensor
			if residentInput == nil {
				*stage = "logit upload"
				lt = s.uploadHostF32([]int{1, cfg.Experts}, logits, compute.MemoryActivation, "V4.1 router logits")
				owned = append(owned, lt)
			} else {
				lt = residentInput.tensor
			}
			*stage = "bias upload"
			bias := []float32{0}
			if hasBias {
				bias = correctionBias
			}
			var bt compute.Tensor
			resident := false
			// Production passes the immutable checkpoint view. The callback
			// also accepts arbitrary caller slices: equal values, a layer id,
			// or a backend name do not establish checkpoint identity.
			// A bare callback session without a held model lifetime stays transient.
			name := layerName(layer, "ffn.gate.e_score_correction_bias")
			if hasBias && s.modelWeightsHeld && s.halW != nil {
				meta, present := s.M.manifest[name]
				bytes, valid := checkedMulInt(len(bias), 4)
				if present && valid && strings.EqualFold(meta.Dtype, "F32") &&
					len(meta.Shape) == 1 && meta.Shape[0] == len(bias) && meta.Nbytes == bytes &&
					meta.Offset >= 0 && meta.Offset <= len(s.M.raw) && bytes <= len(s.M.raw)-meta.Offset {
					canonical := s.M.tensor(name)
					if &canonical[0] == &bias[0] {
						// Existing cache ownership separates loaded-model and
						// exact-backend identities and owns final release.
						bt, resident = s.weightHAL(name), true
					}
				}
			}
			if !resident {
				bt = s.uploadHostF32([]int{len(bias)}, bias, compute.MemoryActivation, "V4.1 router correction bias")
				owned = append(owned, bt)
			}
			*stage = "router-topk"
			if residentInput != nil {
				*stage = "projection/router command window"
			}
			rows, completed, err := v41DeviceRouteCore(op, lt, bt, 1, cfg, full, hasBias)
			if residentInput != nil {
				*residentInput.completed = completed
			}
			if err != nil {
				return nil, err
			}
			if residentInput != nil && !residentInput.loan.valid() {
				return nil, errV41RouterProjectionScope
			}
			return rows[0], nil
		})
	}
	ordinary := func(layer int, logits, bias []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
		return core(layer, logits, bias, cfg, full, nil)
	}
	return v41BoundRouterTopK{ordinary: ordinary, core: core, owner: s, model: s.M, backend: boundBackend}
}
