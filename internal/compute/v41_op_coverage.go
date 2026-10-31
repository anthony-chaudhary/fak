package compute

import "strings"

// V4.1 op-coverage reader: which optional V41*Backend op contracts a backend
// serves on its device in THIS build, and which fall back to the host path.
// Build tags decide whether a CUDA adapter compiles at all, so the same source
// tree yields different answers per artifact; the reader is the single place
// that says so. Status values are a closed set (V41OpDevice, V41OpHostFallback,
// V41OpAbsent); a new `type V41*Backend interface` must be added to
// v41OpProbes or TestV41OpCoverageListIsClosed fails.

const (
	// V41OpDevice: the backend implements the contract and its probe admits it.
	V41OpDevice = "device"
	// V41OpHostFallback: the contract is implemented but refuses (runtime kill
	// switch, missing native admission) or the backend is a host backend.
	V41OpHostFallback = "host-fallback"
	// V41OpAbsent: this build does not implement the contract; callers take
	// the generic host path.
	V41OpAbsent = "absent"

	// Probe kinds.
	V41ProbeAvailable   = "available"   // a no-argument (or scalar) availability probe was called
	V41ProbeShapeGated  = "shape-gated" // admission depends on operand tensors; implementation presence only
	V41ProbeImplemented = "implemented" // the contract has no availability probe
)

// V41OpStatus is one row of the coverage table.
type V41OpStatus struct {
	Op     string `json:"op"`
	Status string `json:"status"`
	Probe  string `json:"probe,omitempty"`
}

// V41DtypeStatus reports whether a weight dtype is resident on the device.
type V41DtypeStatus struct {
	Dtype  string `json:"dtype"`
	Device bool   `json:"device"`
}

// V41CoverageReport is the per-artifact op coverage receipt.
type V41CoverageReport struct {
	Schema      string           `json:"schema"`
	Backend     string           `json:"backend"`
	Tier        string           `json:"tier,omitempty"`
	DeviceOps   int              `json:"device_ops"`
	TotalOps    int              `json:"total_ops"`
	Ops         []V41OpStatus    `json:"ops"`
	WeightTypes []V41DtypeStatus `json:"weight_dtypes"`
}

// V41CoverageReportSchema names the JSON receipt shape.
const V41CoverageReportSchema = "fak.v41-op-coverage/1"

type v41OpProbe struct {
	name  string
	probe func(Backend) (implemented, available bool, kind string)
}

func v41Avail[T any](f func(T) bool) func(Backend) (bool, bool, string) {
	return func(be Backend) (bool, bool, string) {
		x, ok := any(be).(T)
		if !ok {
			return false, false, ""
		}
		return true, f(x), V41ProbeAvailable
	}
}

func v41Present[T any](kind string) func(Backend) (bool, bool, string) {
	return func(be Backend) (bool, bool, string) {
		_, ok := any(be).(T)
		return ok, ok, kind
	}
}

// v41OpProbes lists every optional V4.1 backend contract in internal/compute.
var v41OpProbes = []v41OpProbe{
	{"V41AttentionBF16Backend", v41Avail(func(b V41AttentionBF16Backend) bool { return b.SupportsV41AttentionBF16(1) })},
	{"V41BF16BoundaryBackend", v41Avail(func(b V41BF16BoundaryBackend) bool { return b.V41BF16BoundaryAvailable() })},
	{"V41BF16CheckedBackend", v41Avail(func(b V41BF16CheckedBackend) bool { return b.V41BF16CheckedAvailable() })},
	{"V41BF16PublishBackend", v41Avail(func(b V41BF16PublishBackend) bool { return b.SupportsV41BF16Publish(1) })},
	{"V41CompressorPoolBackend", v41Avail(func(b V41CompressorPoolBackend) bool { return b.SupportsV41CompressorPool() })},
	{"V41CompressorRingBackend", v41Avail(func(b V41CompressorRingBackend) bool { return b.SupportsV41CompressorRing() })},
	{"V41DeferredStatusBackend", v41Avail(func(b V41DeferredStatusBackend) bool { return b.SupportsV41DeferredStatus(1) })},
	{"V41EmbeddingGatherBackend", v41Avail(func(b V41EmbeddingGatherBackend) bool { return b.V41EmbeddingGatherAvailable() })},
	{"V41EngramGateBackend", v41Avail(func(b V41EngramGateBackend) bool { return b.SupportsV41EngramGate() })},
	{"V41EngramRowDecodeBackend", v41Avail(func(b V41EngramRowDecodeBackend) bool { return b.SupportsV41EngramRowDecode() })},
	// Encoding refinement alone is not device decode support. Probe only known
	// dialects and preserve the broad decode kill switch; this is not a witness.
	{"V41EngramRowEncodingBackend", v41Avail(func(b V41EngramRowEncodingBackend) bool {
		decode, ok := b.(V41EngramRowDecodeBackend)
		return ok && (V41EngramRowDecodeSupportsEncoding(decode, V41EngramRowPacked) ||
			V41EngramRowDecodeSupportsEncoding(decode, V41EngramRowQ2K) ||
			V41EngramRowDecodeSupportsEncoding(decode, V41EngramRowF32))
	})},
	{"V41ExactRMSNormBackend", v41Avail(func(b V41ExactRMSNormBackend) bool { return b.SupportsV41ExactRMSNorm(4096, 1e-6) })},
	{"V41F32FiniteBackend", v41Avail(func(b V41F32FiniteBackend) bool { return b.SupportsV41F32Finite(1) })},
	{"V41FP8WKVBackend", v41Avail(func(b V41FP8WKVBackend) bool { return b.V41FP8WKVSupported() })},
	{"V41IndexScaleBF16Backend", v41Present[V41IndexScaleBF16Backend](V41ProbeShapeGated)},
	{"V41IndexSelectBackend", v41Avail(func(b V41IndexSelectBackend) bool { return b.V41IndexSelectAvailable() })},
	{"V41IndexerScoreBackend", v41Avail(func(b V41IndexerScoreBackend) bool { return b.SupportsV41IndexerScore() })},
	{"V41LogitScaleBackend", v41Avail(func(b V41LogitScaleBackend) bool { return b.V41LogitScaleAvailable() })},
	{"V41MHCBackend", v41Avail(func(b V41MHCBackend) bool { return b.SupportsV41MHC() })},
	{"V41MHCPrenormBackend", v41Avail(func(b V41MHCPrenormBackend) bool { return b.V41MHCPrenormAvailable() })},
	{"V41Q2KEmbeddingGatherBackend", v41Avail(func(b V41Q2KEmbeddingGatherBackend) bool { return b.V41Q2KEmbeddingGatherAvailable() })},
	{"V41Q2KTokenPairBackend", v41Present[V41Q2KTokenPairBackend](V41ProbeShapeGated)},
	{"V41Q2KTokenPanelBackend", v41Present[V41Q2KTokenPanelBackend](V41ProbeShapeGated)},
	{"V41Q3KTokenPanelBackend", v41Present[V41Q3KTokenPanelBackend](V41ProbeShapeGated)},
	{"V41QueryPrefixBackend", v41Avail(func(b V41QueryPrefixBackend) bool { return b.V41QueryPrefixAvailable() })},
	{"V41RoutedDownPanelChunkBackend", v41Present[V41RoutedDownPanelChunkBackend](V41ProbeShapeGated)},
	{"V41RoutedDownPanelRowBackend", v41Present[V41RoutedDownPanelRowBackend](V41ProbeShapeGated)},
	{"V41RoutedFusedBackend", v41Avail(func(b V41RoutedFusedBackend) bool { return b.V41RoutedFusedAvailable() })},
	{"V41RoutedIDBackend", v41Avail(func(b V41RoutedIDBackend) bool { return b.V41RoutedIDAvailable() })},
	{"V41RoutedSwiGLUBackend", v41Avail(func(b V41RoutedSwiGLUBackend) bool { return b.V41RoutedSwiGLUAvailable() })},
	{"V41RoutedSwiGLUPanelChunkBackend", v41Present[V41RoutedSwiGLUPanelChunkBackend](V41ProbeShapeGated)},
	{"V41RoutedSwiGLUPanelRowBackend", v41Present[V41RoutedSwiGLUPanelRowBackend](V41ProbeShapeGated)},
	{"V41RouterGateTopKBackend", v41Avail(func(b V41RouterGateTopKBackend) bool { return b.V41RouterGateTopKAvailable() })},
	{"V41RouterHashBackend", v41Avail(func(b V41RouterHashBackend) bool { return b.V41RouterHashAvailable() })},
	{"V41RouterTopKBackend", v41Avail(func(b V41RouterTopKBackend) bool { return b.V41RouterTopKAvailable() })},
	{"V41RowLogBackend", v41Avail(func(b V41RowLogBackend) bool { return b.V41RowLogAvailable() })},
	{"V41ScaleExactBackend", v41Avail(func(b V41ScaleExactBackend) bool { return b.V41ScaleExactAvailable() })},
	{"V41SharedAttentionBackend", v41Avail(func(b V41SharedAttentionBackend) bool { return b.SupportsV41SharedAttention() })},
	{"V41SparsePrefillBackend", v41Avail(func(b V41SparsePrefillBackend) bool { return b.SupportsV41SparsePrefill() })},
	{"V41StepBatchBackend", v41Present[V41StepBatchBackend](V41ProbeImplemented)},
	{"V41StepBatchRecordingBackend", v41Present[V41StepBatchRecordingBackend](V41ProbeImplemented)},
	{"V41TailRoPEBackend", v41Avail(func(b V41TailRoPEBackend) bool { return b.SupportsV41TailRoPE() })},
	{"V41TailRoPEExactBackend", v41Avail(func(b V41TailRoPEExactBackend) bool { return b.V41TailRoPEExactAvailable() })},
	{"V41TailRoPERowsBackend", v41Avail(func(b V41TailRoPERowsBackend) bool { return b.V41TailRoPERowsAvailable() })},
}

// V41CoverageWeightDtypes are the weight dtypes the DS4.1 artifacts carry.
var V41CoverageWeightDtypes = []Dtype{Q2_K, Q3_K, Q4_K, Q6_K, Q8_0, IQ1_M, IQ2_XXS, IQ2_XS, IQ4_NL}

// v41HostBackend reports whether be computes on the host, where an
// implemented contract is still host arithmetic.
func v41HostBackend(be Backend) bool {
	return strings.HasPrefix(be.Name(), "cpu")
}

// V41OpCoverage classifies every V4.1 op contract for be. A probe that panics
// (uninitialized native state) is reported as host-fallback.
func V41OpCoverage(be Backend) []V41OpStatus {
	out := make([]V41OpStatus, 0, len(v41OpProbes))
	if be == nil {
		for _, p := range v41OpProbes {
			out = append(out, V41OpStatus{Op: p.name, Status: V41OpAbsent})
		}
		return out
	}
	host := v41HostBackend(be)
	for _, p := range v41OpProbes {
		implemented, available, kind := v41SafeProbe(p.probe, be)
		row := V41OpStatus{Op: p.name, Status: V41OpAbsent}
		if implemented {
			row.Probe = kind
		}
		switch {
		case !implemented:
		case available && !host:
			row.Status = V41OpDevice
		default:
			row.Status = V41OpHostFallback
		}
		out = append(out, row)
	}
	return out
}

func v41SafeProbe(probe func(Backend) (bool, bool, string), be Backend) (implemented, available bool, kind string) {
	defer func() {
		if recover() != nil {
			implemented, available, kind = true, false, V41ProbeAvailable
		}
	}()
	return probe(be)
}

// V41Coverage builds the full report: op rows plus resident weight dtypes.
func V41Coverage(be Backend) V41CoverageReport {
	r := V41CoverageReport{Schema: V41CoverageReportSchema, Ops: V41OpCoverage(be)}
	r.TotalOps = len(r.Ops)
	for _, row := range r.Ops {
		if row.Status == V41OpDevice {
			r.DeviceOps++
		}
	}
	if be != nil {
		r.Backend, r.Tier = be.Name(), be.Tier()
	}
	host := be == nil || v41HostBackend(be)
	for _, dt := range V41CoverageWeightDtypes {
		r.WeightTypes = append(r.WeightTypes, V41DtypeStatus{Dtype: dt.String(), Device: !host && BackendSupportsDeviceWeightDtype(be, dt)})
	}
	return r
}
