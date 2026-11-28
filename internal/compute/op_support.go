package compute

// Op-level offload veto for Backend placement.
//
// both supports_op and offload_op, moving an op onto a candidate backend only
// when both hold.
//
// Clean-room Go design: OpSupport / OpOffloader is an OPTIONAL capability
// interface discovered by type assertion alongside Backend.Caps() — matching
// KQuantDeviceKernel, RankUploader, and AsyncUploader — so existing Backend
// implementers (cpu-ref, etc.) compile unchanged without breaking changes.
// Backends that do not implement OpSupport default to supported-but-not-
// offloaded (BackendSupportsOp == true, BackendCanOffload == false).

// OpSupport is an OPTIONAL Backend capability advertising per-op execution
// support and offload preference. Both SupportsOp(op) and OffloadOp(op) must
// return true for placement to offload `op` onto the backend.
type OpSupport interface {
	SupportsOp(op string) bool
	OffloadOp(op string) bool
}

// OpOffloader aliases OpSupport so callers targeting either capability name
// share the exact same interface contract.
type OpOffloader = OpSupport

// OpOffloadDecision summarizes a backend's placement verdict for an op together
// with the backend's DeviceMemory capability flag.
type OpOffloadDecision struct {
	Supported    bool
	Offloadable  bool
	DeviceMemory bool
	VetoReason   string
}

// BackendSupportsOp reports whether backend b can execute op.
//
// Fail-closed on nil backend or empty op name. Backends that do not implement
// OpSupport default to true (core portable floor supports ops without offloading).
func BackendSupportsOp(b Backend, op string) bool {
	if b == nil || op == "" {
		return false
	}
	if s, ok := b.(OpSupport); ok {
		return s.SupportsOp(op)
	}
	return true
}

// BackendCanOffload reports whether op may be offloaded onto backend b.
//
// Returns true iff b is non-nil, op is non-empty, b implements OpSupport (or
// OpOffloader), and both b.SupportsOp(op) and b.OffloadOp(op) return true.
// Backends that do not implement OpSupport default to supported-but-not-offloaded
// (returns false).
func BackendCanOffload(b Backend, op string) bool {
	if b == nil || op == "" {
		return false
	}
	s, ok := b.(OpSupport)
	if !ok {
		return false
	}
	return s.SupportsOp(op) && s.OffloadOp(op)
}

// CanOffload is an alias for BackendCanOffload matching the ggml offload check.
func CanOffload(b Backend, op string) bool {
	return BackendCanOffload(b, op)
}

// InspectOpOffload returns the structured support/offload verdict for (b, op)
// alongside b.Caps().DeviceMemory without altering BackendCanOffload semantics.
func InspectOpOffload(b Backend, op string) OpOffloadDecision {
	if b == nil {
		return OpOffloadDecision{VetoReason: "nil-backend"}
	}
	if op == "" {
		return OpOffloadDecision{DeviceMemory: b.Caps().DeviceMemory, VetoReason: "empty-op"}
	}
	devMem := b.Caps().DeviceMemory
	s, ok := b.(OpSupport)
	if !ok {
		return OpOffloadDecision{
			Supported:    true,
			Offloadable:  false,
			DeviceMemory: devMem,
			VetoReason:   "not-offloader",
		}
	}
	sup := s.SupportsOp(op)
	off := s.OffloadOp(op)
	switch {
	case !sup:
		return OpOffloadDecision{Supported: false, Offloadable: false, DeviceMemory: devMem, VetoReason: "unsupported-op"}
	case !off:
		return OpOffloadDecision{Supported: true, Offloadable: false, DeviceMemory: devMem, VetoReason: "offload-vetoed"}
	default:
		return OpOffloadDecision{Supported: true, Offloadable: true, DeviceMemory: devMem}
	}
}
