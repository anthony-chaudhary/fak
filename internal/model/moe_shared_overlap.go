package model

import (
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The launch/separate-output/join/combine ordering adapts Niko1221/Strata
// c62d5e3a15b40ea51c96aae9e8d36928965f53ab, include/strata/core/hit_hook.hpp
// and src/core/session.cpp. See STRATA_LICENSE.txt for the MIT notice.
// Fak retains its current router, placement, kernels and ordered reduction.
type glmOverlapBatch struct {
	cfg      Config
	picks    []routePick
	gate, up []*q4kTensor
	down     []*kQuantTensor
	x, delta []float32
}

// overlapGLMSharedExpert declines without starting either branch. Once started,
// it never falls back or returns until its CPU branch has joined.
func (m *Model) overlapGLMSharedExpert(layer int, x any, picks []routePick, mat matKernel) ([]float32, bool) {
	split, ok := mat.(splitKernel)
	if !ok || split.onHost == nil {
		return nil, false
	}
	host, ok := split.host.(sessionResidentKernel)
	if !ok || host.s == nil {
		return nil, false
	}
	device, ok := split.device.(backendKernel)
	s := host.s
	if !ok || device.s != s || s.M != m || s.GLMSharedExpertOverlapWorkspaceBytes <= 0 ||
		!s.modelWeightsHeld || s.halClosed || s.halFailure != nil || s.backendExecutionScope == nil || !s.backendExecutionScope.live ||
		s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.Q4_K) {
		return nil, false
	}
	xf, ok := x.([]float32)
	if !ok || len(picks) == 0 || m.lora != nil || !splitHostResidentExperts(mat, layer, picks) {
		return nil, false
	}
	h, mi := m.Cfg.HiddenSize, m.Cfg.expertIntermediate()
	if h <= 0 || mi <= 0 || h%256 != 0 || mi%256 != 0 || len(xf) != h || m.Cfg.NSharedExperts <= 0 {
		return nil, false
	}
	sharedN, ok := glmOverlapProduct(m.Cfg.MoEIntermediateSize, m.Cfg.NSharedExperts)
	if !ok {
		return nil, false
	}
	if sharedN == 0 {
		sharedN = m.Cfg.IntermediateSize
	}
	if sharedN <= 0 || sharedN%256 != 0 {
		return nil, false
	}
	need, ok := glmOverlapPayloadBound(len(picks), h, mi, sharedN)
	if !ok || need > s.GLMSharedExpertOverlapWorkspaceBytes {
		return nil, false
	}
	prefix := layerName(layer, "mlp.shared_experts.")
	names := [3]string{prefix + "gate_proj.weight", prefix + "up_proj.weight", prefix + "down_proj.weight"}
	shared := [3]*q4kTensor{}
	for i, name := range names {
		out, in := sharedN, h
		if i == 2 {
			out, in = h, sharedN
		}
		q := m.q4kw[name]
		if split.onHost(name) || m.q8w[name] != nil || m.kqw[name] != nil || !glmOverlapQ4(q, out, in) {
			return nil, false
		}
		shared[i] = q
	}
	b := glmOverlapBatch{cfg: m.Cfg, gate: make([]*q4kTensor, len(picks)), up: make([]*q4kTensor, len(picks)), down: make([]*kQuantTensor, len(picks))}
	for i, pick := range picks {
		if pick.expert < 0 {
			return nil, false
		}
		gn, un, dn := expertName(layer, pick.expert, "gate_proj.weight"), expertName(layer, pick.expert, "up_proj.weight"), expertName(layer, pick.expert, "down_proj.weight")
		g, u, d := m.q4kw[gn], m.q4kw[un], m.kqw[dn]
		if !glmOverlapQ4(g, mi, h) || !glmOverlapQ4(u, mi, h) || !glmOverlapK(d, h, mi) ||
			m.has(expertName(layer, pick.expert, "gate_proj.bias")) || m.has(expertName(layer, pick.expert, "up_proj.bias")) || m.has(expertName(layer, pick.expert, "down_proj.bias")) {
			return nil, false
		}
		for _, q := range shared {
			if glmOverlapRawAliases(g.raw, q.raw) || glmOverlapRawAliases(u.raw, q.raw) || glmOverlapRawAliases(d.raw, q.raw) {
				return nil, false
			}
		}
		// Private descriptors retain only immutable payload/geometry, not lazy owners,
		// retention ledgers, NUMA pools or mutable published raw slice headers.
		b.gate[i] = &q4kTensor{out: g.out, in: g.in, nblk: g.nblk, raw: g.raw}
		b.up[i] = &q4kTensor{out: u.out, in: u.in, nblk: u.nblk, raw: u.raw}
		b.down[i] = &kQuantTensor{out: d.out, in: d.in, nblk: d.nblk, kind: d.kind, raw: d.raw}
	}
	s.admitHostStage(HostStageHostMatmul) // Owner only, before any worker exists.
	b.picks = append([]routePick(nil), picks...)
	b.x = append([]float32(nil), xf...)
	b.delta = make([]float32, h)
	sharedOut := glmOverlapJoin(func() {
		recordHostBatchedExpertDispatch(len(b.picks))
		batchedExpertDelta(b.cfg, b.picks, b.gate, b.up, b.down, b.x, b.delta)
	}, func() []float32 { return glmSharedExperts(m, layer, x, mat) })
	if len(sharedOut) != h {
		panic("model: GLM shared overlap result length mismatch")
	}
	for i := range b.delta {
		b.delta[i] += sharedOut[i]
	}
	s.glmSharedExpertOverlapCount++
	return b.delta, true
}

// glmOverlapJoin leaves the owner panic/Goexit authoritative while always
// joining the worker. It does not recover a panic on an existing pool worker.
func glmOverlapJoin(cpu func(), owner func() []float32) []float32 {
	done := make(chan struct{})
	var failure any
	completed := false
	go func() {
		defer close(done)
		defer func() { failure = recover() }()
		cpu()
		completed = true
	}()
	defer func() { <-done }()
	result := owner()
	<-done
	if !completed {
		if failure != nil {
			panic(failure)
		}
		panic("model: GLM overlap CPU branch did not complete")
	}
	return result
}

func glmOverlapQ4(q *q4kTensor, out, in int) bool {
	if q == nil || q.out != out || q.in != in || q.nblk != in/256 || in%256 != 0 ||
		q.lazy != nil || q.replicas != nil || q.numaPool != nil {
		return false
	}
	n, ok := glmOverlapProduct(out, q.nblk, q4kBlockBytes)
	return ok && n > 0 && len(q.raw) == n
}
func glmOverlapK(q *kQuantTensor, out, in int) bool {
	if q == nil || q.out != out || q.in != in || q.nblk != in/256 || in%256 != 0 ||
		(q.kind != kindQ5K && q.kind != kindQ6K) || q.lazy != nil || q.denseBound != nil || q.replicas != nil || q.numaPool != nil || q.w3MLP {
		return false
	}
	n, ok := glmOverlapProduct(out, q.nblk, q.kind.blockBytes())
	return ok && n > 0 && len(q.raw) == n
}
func glmOverlapRawAliases(a, b []byte) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	ap, bp := uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&b[0]))
	ae, be := ap+uintptr(len(a)), bp+uintptr(len(b))
	if ae < ap || be < bp {
		return true
	}
	return ap < be && bp < ae
}
func glmOverlapProduct(values ...int) (int, bool) {
	n := 1
	for _, v := range values {
		if v < 0 || (v != 0 && n > int(^uint(0)>>1)/v) {
			return 0, false
		}
		n *= v
	}
	return n, true
}

// glmOverlapPayloadBound bounds explicit host slice payloads, including total
// per-range scratch allocation, independently of worker count. It is not an RSS
// ceiling: runtime stacks/allocator metadata and backend staging remain outside it.
func glmOverlapPayloadBound(k, h, m, i int) (int64, bool) {
	if k <= 0 || h <= 0 || m <= 0 || i <= 0 || h%256 != 0 || m%256 != 0 {
		return 0, false
	}
	var sum int64
	add := func(v ...int) bool {
		n, ok := glmOverlapProduct(v...)
		if !ok || int64(n) > int64(^uint64(0)>>1)-sum {
			return false
		}
		sum += int64(n)
		return true
	}
	q := func(n int) (int, bool) {
		a, ok := glmOverlapProduct(3, n)
		if !ok || a > int(^uint(0)>>1)-n/8 {
			return 0, false
		}
		return a + n/8, true
	}
	sh, ok := glmOverlapProduct(128, h/256)
	if !ok {
		return 0, false
	}
	if sh < 1024 {
		sh = 1024
	}
	sm, ok := glmOverlapProduct(128, m/256)
	if !ok {
		return 0, false
	}
	if sm < 1024 {
		sm = 1024
	}
	qh, ok := q(h)
	if !ok {
		return 0, false
	}
	qm, ok := q(m)
	if !ok {
		return 0, false
	}
	terms := [][]int{{8, k, m}, {4, k, h}, {2, qh}, {k, qm}, {2, k, m, sh}, {k, h, sm}, {12, h}, {8, i}, {4, h}}
	for _, term := range terms {
		if !add(term...) {
			return 0, false
		}
	}
	// Exact descriptor/header/value allocations (not allocator rounding).
	metadata := [][]int{
		{k, int(unsafe.Sizeof(routePick{}))}, {2, k, int(unsafe.Sizeof(q4kTensor{}))}, {k, int(unsafe.Sizeof(kQuantTensor{}))},
		{3, k, int(unsafe.Sizeof((*q4kTensor)(nil)))}, {4, k, int(unsafe.Sizeof([]float32(nil)))},
		{k, int(unsafe.Sizeof(q8Vec{}))}, {k, int(unsafe.Sizeof(false))},
		{1, int(unsafe.Sizeof(glmOverlapBatch{}))}, {1, int(unsafe.Sizeof(Config{}))},
	}
	for _, term := range metadata {
		if !add(term...) {
			return 0, false
		}
	}
	// Branch control values and the two gate/up quantization descriptors.
	if !add(2, int(unsafe.Sizeof(q8Vec{}))) || !add(1, int(unsafe.Sizeof((chan struct{})(nil)))) || !add(1, int(unsafe.Sizeof(struct{ value any }{}))) || !add(1, int(unsafe.Sizeof(false))) {
		return 0, false
	}
	return sum, true
}
