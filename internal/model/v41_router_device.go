package model

import (
	"errors"
	"fmt"

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
	got, err := op.V41RouterTopK(logits, bias, rows, cfg.Experts, cfg.TopK, cfg.RouteScale, full, hasBias)
	var fault *compute.V41RouterTopKFault
	if errors.As(err, &fault) {
		return nil, v41RouterFaultError(fault)
	}
	if err != nil {
		return nil, err
	}
	if len(got.Experts) != rows*cfg.TopK || len(got.Weights) != rows*cfg.TopK {
		return nil, &v4RouteError{Field: "device_route", Reason: errV41DeviceRouteResult.Error()}
	}
	out := make([][]routePick, rows)
	for row := range out {
		out[row] = make([]routePick, cfg.TopK)
		for slot := range out[row] {
			i := row*cfg.TopK + slot
			e, w := int(got.Experts[i]), got.Weights[i]
			if e < 0 || e >= cfg.Experts || !finite32(w) {
				return nil, &v4RouteError{Field: "device_route", Reason: errV41DeviceRouteResult.Error()}
			}
			out[row][slot] = routePick{expert: e, weight: w}
		}
	}
	return out, nil
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
// row and the 4*E correction bias per call and reads back 8*k pick bytes plus
// 16 status bytes. A resident caller hands its logits tensor to v41DeviceRoute.
func (s *Session) v41RouterTopKFunc() v41RouterTopKFunc {
	if !s.v41DeviceSumAvailable() {
		return nil
	}
	op, ok := s.Backend.(compute.V41RouterTopKBackend)
	if !ok || !op.V41RouterTopKAvailable() {
		return nil
	}
	return func(layer int, logits, correctionBias []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
		return v41SelectedDeviceCall(s, layer, "v41-router-topk", func(stage *string) ([]routePick, error) {
			if err := cfg.validate(); err != nil {
				return nil, err
			}
			if len(logits) != cfg.Experts {
				return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("width %d, want %d", len(logits), cfg.Experts)}
			}
			hasBias := len(correctionBias) != 0
			if hasBias && len(correctionBias) != cfg.Experts {
				return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("width %d does not match experts width %d", len(correctionBias), cfg.Experts)}
			}
			*stage = "logit upload"
			lt := s.uploadHostF32([]int{1, cfg.Experts}, logits, compute.MemoryActivation, "V4.1 router logits")
			defer s.Backend.Free(lt)
			*stage = "bias upload"
			bias := []float32{0}
			if hasBias {
				bias = correctionBias
			}
			bt := s.uploadHostF32([]int{len(bias)}, bias, compute.MemoryActivation, "V4.1 router correction bias")
			defer s.Backend.Free(bt)
			*stage = "router-topk"
			rows, err := v41DeviceRoute(op, lt, bt, 1, cfg, full, hasBias)
			if err != nil {
				return nil, err
			}
			return rows[0], nil
		})
	}
}
