package model

import (
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RouterGateFusedEnv set to "0" keeps the router gate projection and the
// router top-k as two device operations.
const v41RouterGateFusedEnv = "FAK_V41_ROUTER_GATE_FUSED"

// v41RouterPicksFused routes one token with the gate projection fused ahead of
// the top-k in one dispatch, replacing the projection MatMul and the router
// top-k on the native step. The backend admits the fused module only after it
// witnessed MatMul + top-k bits on its device, so the picks are the separate
// path's. It takes only the F32 manifest gate, the dtype the projection core
// stages first; every other gate returns handled=false before any device work.
// A selected call has no decline arm and never replays on the host.
func (m *Model) v41RouterPicksFused(s *Session, layer int, input []float32, hidden int, cfg v41RouterConfig, full bool, bias func() []float32, valid func() bool) (picks []routePick, handled bool, err error) {
	if os.Getenv(v41RouterGateFusedEnv) == "0" || m.lora != nil || m.prism != nil {
		return nil, false, nil
	}
	op, ok := s.Backend.(compute.V41RouterGateTopKBackend)
	if !ok {
		return nil, false, nil
	}
	name := layerName(layer, "ffn.gate.weight")
	meta, present := s.M.manifest[name]
	if !present || len(meta.Shape) != 2 || meta.Shape[0] != cfg.Experts || meta.Shape[1] != hidden || len(input) != hidden ||
		!op.V41RouterGateTopKAvailable() {
		return nil, false, nil
	}
	correction := bias() // A panic here retains its original unwrapped identity.
	opened := m.v41NowNanos()
	var uploaded int64
	deviceRows := 0
	defer func() { m.v41NoteDenseProjection(1, 0, deviceRows, 0, uploaded, 0, opened) }()
	rows, err := v41SelectedDeviceCall(s, layer, "v41-router-gate-topk", func(stage *string) (out [][]routePick, cause error) {
		var owned []compute.Tensor
		defer func() {
			for i := len(owned) - 1; i >= 0; i-- {
				s.Backend.Free(owned[i])
			}
		}()
		if err := cfg.validate(); err != nil {
			return nil, err
		}
		hasBias := len(correction) != 0
		if hasBias && len(correction) != cfg.Experts {
			return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("width %d does not match experts width %d", len(correction), cfg.Experts)}
		}
		*stage = "weight upload"
		weight := s.weightHAL(name)
		*stage = "activation upload"
		x := s.uploadHostF32([]int{hidden}, input, compute.MemoryActivation, "V4.1 router gate activation")
		owned = append(owned, x)
		uploaded = int64(hidden) * 4
		*stage = "bias upload"
		bt, transient := s.v41RouterBiasTensor(layer, correction)
		if transient {
			owned = append(owned, bt)
		}
		if !valid() {
			return nil, errV41RouterProjectionScope
		}
		*stage = "router-gate-topk"
		got, err := op.V41RouterGateTopK(weight, x, bt, cfg.Experts, hidden, cfg.TopK, cfg.RouteScale, full, hasBias)
		var fault *compute.V41RouterTopKFault
		if errors.As(err, &fault) {
			return nil, v41RouterFaultError(fault)
		}
		if err != nil {
			return nil, err
		}
		out, err = v41RoutePicksFromDevice(got, 1, cfg)
		if err != nil {
			return nil, err
		}
		if !valid() {
			return nil, errV41RouterProjectionScope
		}
		return out, nil
	})
	if err != nil {
		return nil, true, v41StageErr(v41StageMoE, layer, err)
	}
	deviceRows = 1
	return rows[0], true, nil
}
