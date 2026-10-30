package compute

import (
	"math"
	"sort"
	"strings"
	"sync/atomic"
)

// KernelRow is one (kernel, groups) shape's cumulative GPU timestamp fold.
// Kernel is a stable name, never the shim's integer id: ids move between
// commits, names do not.
type KernelRow struct {
	Kernel string
	GX, GY uint32
	Calls  uint64
	NS     float64
}

// KernelProfile is a backend's cumulative per-kernel GPU time. Every recorded
// op is timed, so the rows telescope to GPUNS. Truncated counts recorded ops
// with no timestamp: past the query pool, or in a batch whose timestamps were
// never read (a failed query or submit). UntimedCompute counts compute
// submitted outside a timed batch (its time is in no row), UntimedTransfer the
// same for copies, and BarriersElided the dispatches recorded without a
// preceding barrier, whose timestamp delta is not that dispatch's own time.
type KernelProfile struct {
	Backend         string
	Batches, Ops    uint64
	Truncated       uint64
	UntimedCompute  uint64
	UntimedTransfer uint64
	BarriersElided  uint64
	GPUNS, WallNS   float64
	Rows            []KernelRow
}

type kernelProfileReader struct {
	backend string
	read    func() (KernelProfile, bool)
}

var kernelProfileReaderSlot atomic.Pointer[kernelProfileReader]

// RegisterKernelProfileReader installs the per-kernel GPU-time reader. A
// backend registers only when its timestamp profile is on, so the default
// process has no reader and KernelProfileSnapshot costs one nil load. Unlike a
// DeviceWork reader, read may take the backend's device lock.
func RegisterKernelProfileReader(backend string, read func() (KernelProfile, bool)) (unregister func()) {
	if backend == "" || read == nil {
		return func() {}
	}
	r := &kernelProfileReader{backend: backend, read: read}
	kernelProfileReaderSlot.Store(r)
	return func() { kernelProfileReaderSlot.CompareAndSwap(r, nil) }
}

// KernelProfileSnapshot reads the registered backend's cumulative profile; nil
// when no reader is registered or the backend reports the profile unusable.
// With a reader it may block behind in-flight device work, so it must never be
// called while that backend's device lock is held.
func KernelProfileSnapshot() *KernelProfile {
	r := kernelProfileReaderSlot.Load()
	if r == nil {
		return nil
	}
	p, ok := r.read()
	if !ok {
		return nil
	}
	p.Backend = r.backend
	return &p
}

// KernelProfileReaderRegistered reports whether a kernel profile is being kept.
func KernelProfileReaderRegistered() bool { return kernelProfileReaderSlot.Load() != nil }

type kernelRowKey struct {
	kernel string
	gx, gy uint32
}

func (p *KernelProfile) totalsFinite() bool {
	return finiteNonnegative(p.GPUNS) && finiteNonnegative(p.WallNS)
}

func finiteNonnegative(v float64) bool {
	return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func indexKernelRows(rows []KernelRow) (map[kernelRowKey]KernelRow, bool) {
	out := make(map[kernelRowKey]KernelRow, len(rows))
	for _, r := range rows {
		if !finiteNonnegative(r.NS) {
			return nil, false
		}
		k := kernelRowKey{r.Kernel, r.GX, r.GY}
		prev := out[k]
		prev.Kernel, prev.GX, prev.GY = r.Kernel, r.GX, r.GY
		prev.Calls += r.Calls
		prev.NS += r.NS
		out[k] = prev
	}
	return out, true
}

func sortKernelRows(rows []KernelRow) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.NS != b.NS {
			return a.NS > b.NS
		}
		if a.Kernel != b.Kernel {
			return a.Kernel < b.Kernel
		}
		if a.GX != b.GX {
			return a.GX < b.GX
		}
		return a.GY < b.GY
	})
}

// KernelProfileDelta is the work between two cumulative snapshots, joined by
// kernel name and groups. It is nil when either side is missing, the backends
// differ, or any counter or row went backwards (a reset or a different
// process). Rows absent before are kept whole; rows with no new calls drop.
func KernelProfileDelta(before, after *KernelProfile) *KernelProfile {
	if before == nil || after == nil || before.Backend != after.Backend {
		return nil
	}
	if !before.totalsFinite() || !after.totalsFinite() {
		return nil
	}
	if after.Batches < before.Batches || after.Ops < before.Ops || after.Truncated < before.Truncated ||
		after.UntimedCompute < before.UntimedCompute || after.UntimedTransfer < before.UntimedTransfer ||
		after.BarriersElided < before.BarriersElided || after.GPUNS < before.GPUNS || after.WallNS < before.WallNS {
		return nil
	}
	prev, ok := indexKernelRows(before.Rows)
	if !ok {
		return nil
	}
	cur, ok := indexKernelRows(after.Rows)
	if !ok {
		return nil
	}
	out := &KernelProfile{
		Backend:         after.Backend,
		Batches:         after.Batches - before.Batches,
		Ops:             after.Ops - before.Ops,
		Truncated:       after.Truncated - before.Truncated,
		UntimedCompute:  after.UntimedCompute - before.UntimedCompute,
		UntimedTransfer: after.UntimedTransfer - before.UntimedTransfer,
		BarriersElided:  after.BarriersElided - before.BarriersElided,
		GPUNS:           after.GPUNS - before.GPUNS,
		WallNS:          after.WallNS - before.WallNS,
	}
	for k, b := range prev {
		a, ok := cur[k]
		if !ok || a.Calls < b.Calls || a.NS < b.NS {
			return nil
		}
	}
	for k, a := range cur {
		b := prev[k]
		if a.Calls == b.Calls {
			continue
		}
		out.Rows = append(out.Rows, KernelRow{Kernel: a.Kernel, GX: a.GX, GY: a.GY, Calls: a.Calls - b.Calls, NS: a.NS - b.NS})
	}
	sortKernelRows(out.Rows)
	return out
}

// KernelProfileAdd sums two window deltas of one backend; nil when either is
// missing or the backends differ, so a caller folding windows sees the gap.
func KernelProfileAdd(a, b *KernelProfile) *KernelProfile {
	if a == nil || b == nil || a.Backend != b.Backend || !a.totalsFinite() || !b.totalsFinite() {
		return nil
	}
	rows, ok := indexKernelRows(append(append(make([]KernelRow, 0, len(a.Rows)+len(b.Rows)), a.Rows...), b.Rows...))
	if !ok {
		return nil
	}
	out := &KernelProfile{
		Backend:         a.Backend,
		Batches:         a.Batches + b.Batches,
		Ops:             a.Ops + b.Ops,
		Truncated:       a.Truncated + b.Truncated,
		UntimedCompute:  a.UntimedCompute + b.UntimedCompute,
		UntimedTransfer: a.UntimedTransfer + b.UntimedTransfer,
		BarriersElided:  a.BarriersElided + b.BarriersElided,
		GPUNS:           a.GPUNS + b.GPUNS,
		WallNS:          a.WallNS + b.WallNS,
		Rows:            make([]KernelRow, 0, len(rows)),
	}
	for _, r := range rows {
		out.Rows = append(out.Rows, r)
	}
	sortKernelRows(out.Rows)
	return out
}

// Kernel op classes: the closed set KernelOpClass maps every kernel name into.
const (
	KernelOpMatvec    = "matvec"
	KernelOpMatmul    = "matmul"
	KernelOpAttention = "attention"
	KernelOpGDN       = "gdn"
	KernelOpNorm      = "norm"
	KernelOpRope      = "rope"
	KernelOpSwiGLU    = "swiglu"
	KernelOpArgmax    = "argmax"
	KernelOpCopy      = "copy"
	KernelOpOther     = "other"
)

// KernelOpClasses is the closed op-class vocabulary, in precedence order.
var KernelOpClasses = []string{
	KernelOpCopy, KernelOpAttention, KernelOpGDN, KernelOpMatvec, KernelOpMatmul,
	KernelOpSwiGLU, KernelOpRope, KernelOpNorm, KernelOpArgmax, KernelOpOther,
}

// Timestamp pseudo-kernels: transfer and fill commands recorded into a timed
// batch. Their names come from the shim's fvk_ts_kernel_name.
var kernelTransferNames = map[string]bool{
	"d2d_copy": true, "h2d_upload": true, "d2d_gather": true, "fill": true, "readback": true,
}

// KernelOpClass maps a kernel name to its op class. A fused kernel takes the
// class of the op that dominates its cost, which is why attention, GDN and the
// matrix products are tested before the elementwise ops they fuse, and the
// fused projections whose names carry no matvec token (a GDN in_proj, a routed
// gate_up or down_sum) before GDN and SwiGLU. A routed *_panel_* kernel is a
// publication or activation over an already projected row, not a projection.
// CUDA names spell the same ops gemv/gemm (cuBLAS rows cublas_*gemm*), mul_mat
// (the llama.cpp MMQ port), q3k_panel (the Q3_K panel product), and
// fattn/attend/spec_verify/dsa_index for the attention family.
func KernelOpClass(name string) string {
	switch {
	case kernelTransferNames[name]:
		return KernelOpCopy
	case strings.Contains(name, "attention"), strings.Contains(name, "fattn"), strings.Contains(name, "attend"),
		strings.Contains(name, "spec_verify"), strings.Contains(name, "dsa_index"):
		return KernelOpAttention
	case strings.Contains(name, "in_proj"), strings.Contains(name, "gate_up"), strings.Contains(name, "down_sum"):
		return KernelOpMatvec
	case strings.Contains(name, "gdn"), strings.Contains(name, "kda"):
		return KernelOpGDN
	case strings.Contains(name, "matvec"), strings.Contains(name, "gemv"):
		return KernelOpMatvec
	case strings.Contains(name, "matmul"), strings.Contains(name, "token_panel"), strings.Contains(name, "token_pair"),
		strings.Contains(name, "gemm"), strings.Contains(name, "mul_mat"), strings.Contains(name, "q3k_panel"),
		strings.Contains(name, "w4a16"):
		return KernelOpMatmul
	case strings.Contains(name, "swiglu"):
		return KernelOpSwiGLU
	case strings.Contains(name, "rope"):
		return KernelOpRope
	case strings.Contains(name, "norm"):
		return KernelOpNorm
	case strings.Contains(name, "argmax"):
		return KernelOpArgmax
	default:
		return KernelOpOther
	}
}
