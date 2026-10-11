package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// v41DeviceOnlyBinding is one V4.1 forward op whose device seam must be bound by
// the session backend before device-only admission. A nil seam keeps a host arm
// in the forward, so an unbound op the model runs is a host op for this backend
// even though the static inventory no longer lists it (Ref #13668).
type v41DeviceOnlyBinding struct {
	op    v41HostOp
	uses  func(Config) bool
	bound func(*Session) bool
}

var v41DeviceOnlyBindings = []v41DeviceOnlyBinding{
	{op: v41HostOp{name: "embedding-panel", site: "V41EmbeddingGatherAvailable"}, bound: func(s *Session) bool { return s.v41EmbeddingGatherFunc() != nil }},
	{op: v41HostOp{name: "mhc-streams", site: "SupportsV41MHC"}, bound: v41DeviceOnlyMHCBound},
	{op: v41HostOp{name: "rms-norm", site: "v41RMSNormFunc"}, bound: func(s *Session) bool { return s.v41RMSNormFunc() != nil }},
	{op: v41HostOp{name: "tail-rope-qk", site: "SupportsV41TailRoPE"}, bound: func(s *Session) bool { return s.v41TailRoPEFunc() != nil }},
	{op: v41HostOp{name: "compressor-norm", site: "SupportsV41CompressorNorm"}, uses: v41UsesCompressedKV, bound: func(s *Session) bool { return s.v41CompressorNormFunc() != nil }},
	{op: v41HostOp{name: "index-key-norm", site: "SupportsV41IndexKeyNorm"}, uses: v41UsesIndexer, bound: func(s *Session) bool { return s.v41IndexKeyNormFunc() != nil }},
	{op: v41HostOp{name: "attention-indexer-score", site: "V41IndexerScoreAdmission"}, uses: v41UsesIndexer, bound: func(s *Session) bool { return s.v41IndexerScoreFunc() != nil }},
	{op: v41HostOp{name: "attention-index-list", site: "V41IndexSelectAvailable"}, uses: v41UsesIndexer, bound: func(s *Session) bool { return s.v41IndexSelectFunc() != nil }},
	{op: v41HostOp{name: "shared-attention", site: "SupportsV41SharedAttention"}, bound: func(s *Session) bool { return s.v41SharedAttentionFunc() != nil }},
	{op: v41HostOp{name: "attention-output-rotation", site: "V41TailRoPERowsAvailable"}, bound: func(s *Session) bool { return s.v41InverseOutputRoPEFunc() != nil }},
	{op: v41HostOp{name: "moe-router-topk", site: "V41RouterTopKAvailable"}, bound: func(s *Session) bool { return s.v41RouterTopKFunc() != nil }},
	{op: v41HostOp{name: "shared-expert-swiglu", site: "v41SharedActivationFunc"}, bound: func(s *Session) bool { return s.v41SharedActivationFunc() != nil }},
}

func v41DeviceOnlyMHCBound(s *Session) bool {
	if s == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory || !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return false
	}
	op, ok := s.Backend.(compute.V41MHCBackend)
	return ok && op.SupportsV41MHC()
}

func v41UsesCompressedKV(cfg Config) bool {
	if cfg.DeepSeekV41 == nil {
		return false
	}
	for _, ratio := range cfg.DeepSeekV41.CompressRatios {
		if ratio > 1 {
			return true
		}
	}
	return false
}

func v41UsesIndexer(cfg Config) bool {
	if cfg.DeepSeekV41 == nil || cfg.IndexHeadDim <= 0 {
		return false
	}
	for layer := range cfg.DeepSeekV41.CompressRatios {
		if indexSourceAt(cfg.DeepSeekV41, layer) {
			return true
		}
	}
	return false
}

// v41DeviceOnlyUnboundOps names, in forward order, each op the model runs whose
// device seam this session's backend does not bind.
func (s *Session) v41DeviceOnlyUnboundOps() []v41HostOp {
	if s == nil || s.M == nil {
		return nil
	}
	var unbound []v41HostOp
	for _, b := range v41DeviceOnlyBindings {
		if b.uses != nil && !b.uses(s.M.Cfg) {
			continue
		}
		if !b.bound(s) {
			unbound = append(unbound, b.op)
		}
	}
	return unbound
}

// v41DeviceOnlyAdmissionRefusal joins the static host inventory with the ops
// this backend leaves unbound. Nil is the only V4.1 device-only admission.
func (s *Session) v41DeviceOnlyAdmissionRefusal() error {
	ops := append([]v41HostOp(nil), v41DeviceOnlyHostOps...)
	listed := make(map[string]bool, len(ops))
	for _, op := range ops {
		listed[op.name] = true
	}
	for _, op := range s.v41DeviceOnlyUnboundOps() {
		if !listed[op.name] {
			ops = append(ops, op)
		}
	}
	if len(ops) == 0 {
		return nil
	}
	return &v41DeviceOnlyHostOpError{ops: ops}
}
