package model

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ColdSharedMath struct {
	kind         string
	dtype        compute.Dtype
	shape        []int
	x, second, y []float32
	limit        float32
}

// The arithmetic is cpu-ref MatMul plus the independent shared activation
// oracle. Only public host transfers are counted; inspecting bytes inside this
// test backend is not represented as a device readback.
type v41ColdSharedRecorder struct {
	compute.Backend
	*v41DeferredLedger
	math                                        []v41ColdSharedMath
	created, freed                              map[compute.Buffer]int
	activationUploads, weightUploads, dataReads int
	activationBytes, dataBytes                  int
	sharedWeights                               map[compute.Buffer]int
	sharedReads                                 []int
	deferredReadSites                           [][]uint32
	deferredReadRows                            []int
	active, lastProjection                      int
	deny, denyQ6                                bool
	failMatMul                                  int
	matmuls                                     int
	fault                                       error
	malformed                                   string
	foreign, unrelated                          compute.Tensor
	cleanupPanic                                bool
}

func newV41ColdSharedRecorder() *v41ColdSharedRecorder {
	b := &v41ColdSharedRecorder{Backend: compute.Default(), created: map[compute.Buffer]int{}, freed: map[compute.Buffer]int{}, sharedWeights: map[compute.Buffer]int{}, active: -1}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return b
}
func (b *v41ColdSharedRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *v41ColdSharedRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !b.deny && (dt == compute.F32 || dt == compute.Q2_K || dt == compute.Q4_K || dt == compute.Q6_K)
}
func (b *v41ColdSharedRecorder) SupportsQ6KMatMul() bool { return !b.denyQ6 }
func (b *v41ColdSharedRecorder) tensor(shape []int, values []float32) compute.Tensor {
	t := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
	b.created[t.Buf()]++
	return t
}
func (b *v41ColdSharedRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, _ string) compute.Tensor {
	if class == compute.MemoryActivation {
		b.activationUploads++
		b.activationBytes += x.Numel() * 4
	} else {
		b.weightUploads++
	}
	return b.upload(x, dt)
}
func (b *v41ColdSharedRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if len(x.Shape) == 1 {
		b.activationUploads++
		b.activationBytes += x.Numel() * 4
	} else {
		b.weightUploads++
	}
	return b.upload(x, dt)
}
func (b *v41ColdSharedRecorder) upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt == compute.F32 {
		return b.tensor(x.Shape, b.Backend.Read(x))
	}
	host := x.Buf().(compute.HostBuffer)
	raw := make([]byte, len(host.I8()))
	for i, v := range host.I8() {
		raw[i] = byte(v)
	}
	var out compute.Tensor
	switch dt {
	case compute.Q2_K:
		out = compute.NewQ2K(b, x.Shape, raw)
	case compute.Q4_K:
		out = compute.NewQ4K(b, x.Shape, raw)
	case compute.Q6_K:
		out = compute.NewQ6K(b, x.Shape, raw)
	default:
		panic("unexpected test upload dtype")
	}
	b.created[out.Buf()]++
	return out
}
func (b *v41ColdSharedRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	if b.matmuls == b.failMatMul {
		panic(b.fault)
	}
	if index, ok := b.sharedWeights[w.Buf()]; ok {
		if index == 0 {
			b.sharedReads = append(b.sharedReads, 0)
			b.active = len(b.sharedReads) - 1
		}
		b.lastProjection = index
	}
	y := b.Backend.MatMul(w, x)
	values := append([]float32(nil), b.Backend.Read(y)...)
	b.math = append(b.math, v41ColdSharedMath{kind: "matmul", dtype: w.Dtype, shape: append([]int(nil), w.Shape...), x: append([]float32(nil), b.Backend.Read(x)...), y: values})
	return b.tensor(y.Shape, values)
}
func (b *v41ColdSharedRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	y := b.Backend.BatchedMatMul(w, x, rows)
	return b.tensor(y.Shape, b.Backend.Read(y))
}
func (b *v41ColdSharedRecorder) SwiGLU(g, u compute.Tensor) compute.Tensor {
	return b.SwiGLUWithLimit(g, u, 0)
}
func (b *v41ColdSharedRecorder) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	gate, up := b.Backend.Read(g), b.Backend.Read(u)
	values := make([]float32, len(gate))
	for i := range values {
		a, c := gate[i], up[i]
		if limit > 0 {
			a = min(a, limit)
			c = max(-limit, min(c, limit))
		}
		values[i] = a / (1 + float32(math.Exp(float64(-a)))) * c
	}
	b.math = append(b.math, v41ColdSharedMath{kind: "swiglu", x: append([]float32(nil), gate...), second: append([]float32(nil), up...), y: append([]float32(nil), values...), limit: limit})
	return b.tensor(g.Shape, values)
}
func (b *v41ColdSharedRecorder) Read(x compute.Tensor) []float32 {
	b.dataReads++
	b.dataBytes += x.Numel() * 4
	if b.active >= 0 {
		b.sharedReads[b.active]++
	}
	return b.Backend.Read(x)
}
func (b *v41ColdSharedRecorder) Free(x compute.Tensor) {
	_, status := b.deferredPending[x.Buf()]
	b.freed[x.Buf()]++
	delete(b.deferredPending, x.Buf())
	b.Backend.Free(x)
	if b.cleanupPanic && !status {
		panic("injected cleanup failure")
	}
}
func (b *v41ColdSharedRecorder) SupportsV41BF16Publish(n int) bool { return !b.deny && n > 0 }
func (b *v41ColdSharedRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	p, st, err := b.V41BF16PublishDeferred(x, site)
	if err != nil {
		return p, err
	}
	v, _, err := b.V41ReadDeferred([]compute.V41DeferredStatus{st}, nil)
	if err != nil {
		return p, err
	}
	return p, v[0]
}
func (b *v41ColdSharedRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	p, st, err := b.v41DeferredLedger.V41BF16PublishDeferred(x, site)
	b.created[p.Buf()]++
	b.created[st.Record.Buf()]++
	if b.deferredPublishes == 1 {
		switch b.malformed {
		case "output-input":
			b.Free(p)
			p = x
		case "output-unrelated":
			b.Free(p)
			p = b.unrelated
		case "record-unrelated":
			b.Free(st.Record)
			st.Record = b.unrelated
		case "output-foreign":
			b.Free(p)
			p = b.foreign
		case "record-output":
			b.Free(st.Record)
			st.Record = p
		case "status-shape":
			st.Record.Shape = []int{2, 2}
		case "status-site":
			st.Site++
		case "partial":
			err = b.fault
		}
	}
	return p, st, err
}
func (b *v41ColdSharedRecorder) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	sites := make([]uint32, len(statuses))
	for i := range statuses {
		sites[i] = statuses[i].Site
	}
	b.deferredReadSites = append(b.deferredReadSites, sites)
	b.deferredReadRows = append(b.deferredReadRows, len(rows))
	b.dataReads += len(rows)
	for _, row := range rows {
		b.dataBytes += row.Numel() * 4
	}
	if b.active >= 0 {
		b.sharedReads[b.active] += len(rows)
		if b.lastProjection == 2 && len(statuses) > 0 && statuses[len(statuses)-1].Site == compute.V41BF16PublishSharedExpertProjection {
			b.active = -1
		}
	}
	return b.v41DeferredLedger.V41ReadDeferred(statuses, rows)
}

func v41ColdSharedFixture(t *testing.T, dtype compute.Dtype) (*Model, []float32) {
	t.Helper()
	H, I := 4, 3
	if dtype != compute.F32 {
		H, I = 256, 256
	}
	m := &Model{Cfg: Config{NumLayers: 1, HeadDim: 512, HiddenSize: H, MoEIntermediateSize: I, SwigluLimit: .125}, manifest: map[string]tensorMeta{}, kqw: map[string]*kQuantTensor{}, q4kw: map[string]*q4kTensor{}}
	for n, leaf := range []string{"w1", "w3", "w2"} {
		out, in := I, H
		if n == 2 {
			out, in = H, I
		}
		name := layerName(0, "ffn.shared_experts."+leaf+".weight")
		if dtype == compute.Q2_K {
			m.kqw[name] = q2kFixtureTensor(out, in, uint64(71+n))
			continue
		}
		if dtype == compute.Q4_K {
			m.q4kw[name] = quantizeQ4KFromRaw(makeTestQ4KRaw(out, in), out, in)
			continue
		}
		if dtype == compute.Q6_K {
			raw := make([]byte, out*(in/qkK)*q6kBlockBytes)
			lcgBytes(raw, uint64(91+n))
			for at := 0; at < len(raw); at += q6kBlockBytes {
				binary.LittleEndian.PutUint16(raw[at+208:], 0x1000)
			}
			m.kqw[name] = quantizeKQuantFromRaw(raw, out, in, kindQ6K)
			continue
		}
		start := len(m.raw)
		for i := 0; i < out*in; i++ {
			m.raw = binary.LittleEndian.AppendUint32(m.raw, math.Float32bits(float32((i+3*n)%11-4)*.137+.023))
		}
		m.manifest[name] = tensorMeta{Dtype: "F32", Shape: []int{out, in}, Offset: start, Nbytes: len(m.raw) - start}
	}
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(i%7-2) / 8
	}
	return m, x
}
func v41ColdSharedSession(m *Model, b *v41ColdSharedRecorder) *Session {
	return &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
}

func v41ColdSharedEqualWords(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func v41ColdSharedEqualMath(a, b []v41ColdSharedMath) bool {
	if len(a) != len(b) {
		return false
	}
	for i, op := range a {
		other := b[i]
		if op.kind != other.kind || op.dtype != other.dtype || !reflect.DeepEqual(op.shape, other.shape) ||
			math.Float32bits(op.limit) != math.Float32bits(other.limit) || !v41ColdSharedEqualWords(op.x, other.x) ||
			!v41ColdSharedEqualWords(op.second, other.second) || !v41ColdSharedEqualWords(op.y, other.y) {
			return false
		}
	}
	return true
}

func v41ColdSharedAssertReleased(t *testing.T, b *v41ColdSharedRecorder, s *Session) {
	t.Helper()
	cached := map[compute.Buffer]bool{}
	for _, w := range s.halW {
		cached[w.Buf()] = true
	}
	if !s.M.VisitImmutableDeviceWeights(b, func(w compute.Tensor) bool { cached[w.Buf()] = true; return true }) {
		t.Fatal("model weight inventory unavailable")
	}
	for ptr, creates := range b.created {
		want := creates
		if cached[ptr] {
			want--
		}
		if b.freed[ptr] != want {
			t.Fatalf("buffer release count=%d want=%d (cached=%t)", b.freed[ptr], want, cached[ptr])
		}
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestV41ColdSharedResidentScalarChain(t *testing.T) {
	if v41ColdSharedEqualWords([]float32{math.Float32frombits(0x80000000)}, []float32{0}) {
		t.Fatal("word comparator failed to distinguish signed zero")
	}
	for _, dtype := range []compute.Dtype{compute.F32, compute.Q2_K, compute.Q4_K, compute.Q6_K} {
		for _, limit := range []float64{0, .125} {
			name := "limited"
			if limit == 0 {
				name = "unlimited"
			}
			t.Run(dtype.String()+"/"+name, func(t *testing.T) {
				m, input := v41ColdSharedFixture(t, dtype)
				m.Cfg.SwigluLimit = limit
				t.Cleanup(func() { _ = m.CloseWeights() })
				full, old, next := newV41ColdSharedRecorder(), newV41ColdSharedRecorder(), newV41ColdSharedRecorder()
				reference, control, candidate := v41ColdSharedSession(m, full), v41ColdSharedSession(m, old), v41ColdSharedSession(m, next)
				defer reference.Close()
				defer control.Close()
				defer candidate.Close()
				want, err := m.v41FullSharedExpertSwiGLU(0, input, m.Cfg, reference.v41DenseProjectionFunc(), reference.v41SharedActivationFunc(), reference.v41BF16PublishFunc())
				if err != nil {
					t.Fatal(err)
				}
				oldChain := control.v41SharedExpertChain(0, false, nil, nil)
				oldWant, oldSelected, err := oldChain(input)
				if err != nil || !oldSelected {
					t.Fatalf("old four-read control selected=%t error=%v", oldSelected, err)
				}
				chain := candidate.v41ColdSharedExpertChain(0)
				got, selected, err := chain(input)
				if err != nil || !selected {
					t.Fatalf("selected=%t error=%v", selected, err)
				}
				if !v41ColdSharedEqualWords(got, want) || !v41ColdSharedEqualWords(got, oldWant) || !v41ColdSharedEqualMath(next.math, full.math) || !v41ColdSharedEqualMath(next.math, old.math) {
					t.Fatal("result or actual scalar arithmetic differs from independent full helper or old resident control")
				}
				if len(next.math) != 4 || next.math[0].dtype != dtype || next.math[1].dtype != dtype || next.math[3].dtype != dtype {
					t.Fatal("wrong kernel sequence or weight dtype")
				}
				if next.activationUploads != 1 || next.dataReads != 1 || next.deferredFences != 1 || old.dataReads != 1 || old.deferredFences != 4 || full.dataReads <= next.dataReads {
					t.Fatalf("uploads=%d candidate rows/fences=%d/%d old rows/fences=%d/%d full-helper rows=%d", next.activationUploads, next.dataReads, next.deferredFences, old.dataReads, old.deferredFences, full.dataReads)
				}
				if next.activationBytes != 4*len(input) || next.dataBytes != 4*len(input) {
					t.Fatalf("activation/data bytes=%d/%d want=%d/%d", next.activationBytes, next.dataBytes, 4*len(input), 4*len(input))
				}
				orderedSites := []uint32{compute.V41BF16PublishSharedExpertProjection, compute.V41BF16PublishSharedExpertProjection, compute.V41BF16PublishSharedExpertDownInput, compute.V41BF16PublishSharedExpertProjection}
				if !reflect.DeepEqual(next.deferredSites, orderedSites) || !reflect.DeepEqual(next.deferredReadSites, [][]uint32{orderedSites}) || !reflect.DeepEqual(next.deferredReadRows, []int{1}) {
					t.Fatal("BF16 boundary order changed")
				}
				if !reflect.DeepEqual(old.deferredReadSites, [][]uint32{{orderedSites[0]}, {orderedSites[1]}, {orderedSites[2]}, {orderedSites[3]}}) || !reflect.DeepEqual(old.deferredReadRows, []int{0, 0, 0, 1}) {
					t.Fatal("old four-read control no longer witnesses one status per fence and one final row")
				}
				for _, op := range next.math[1:] {
					if !v41BF16Widened(op.x) {
						t.Fatal("dependent kernel received unpublished operand")
					}
				}
				before := next.weightUploads
				again, selected, err := chain(input)
				if err != nil || !selected || !v41ColdSharedEqualWords(again, got) || next.weightUploads != before {
					t.Fatal("warm chain changed output or restaged weights")
				}
				v41ColdSharedAssertReleased(t, next, candidate)
			})
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ColdSharedResidentAdmission(t *testing.T) {
	for _, mode := range []string{"dtype", "q6-native", "lora", "prism", "deferred-disabled", "input-not-bf16"} {
		t.Run(mode, func(t *testing.T) {
			dtype := compute.F32
			if mode == "q6-native" {
				dtype = compute.Q6_K
			}
			m, x := v41ColdSharedFixture(t, dtype)
			b := newV41ColdSharedRecorder()
			s := v41ColdSharedSession(m, b)
			switch mode {
			case "dtype":
				b.deny = true
			case "q6-native":
				b.denyQ6 = true
			case "lora":
				m.lora = NewLoRASet()
			case "prism":
				m.prism = &prismHadamardState{}
			case "deferred-disabled":
				t.Setenv(v41DeferredBF16PublishEnv, "0")
			case "input-not-bf16":
				x[0] = 1.00390625
			}
			chain := s.v41ColdSharedExpertChain(0)
			if out, selected, err := chain(x); out != nil || selected || err != nil {
				t.Fatalf("decline=%t out=%v err=%v", !selected, out, err)
			}
			if len(b.created) != 0 || len(s.halW) != 0 || len(b.math) != 0 || b.dataReads != 0 || s.halClosed {
				t.Fatal("decline had side effects")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ColdSharedResidentFailureOrderAndOwnership(t *testing.T) {
	for _, mode := range []string{"gate-status", "up-status", "activation-status", "down-status", "matmul", "transport", "output-input", "output-unrelated", "record-unrelated", "output-foreign", "record-output", "status-shape", "status-site", "partial", "primary-before-cleanup"} {
		t.Run(mode, func(t *testing.T) {
			m, x := v41ColdSharedFixture(t, compute.F32)
			b := newV41ColdSharedRecorder()
			s := v41ColdSharedSession(m, b)
			t.Cleanup(func() { b.cleanupPanic = false; _ = m.CloseWeights() })
			primary := errors.New("first publication failure")
			later := &compute.BackendError{Backend: "test", Site: "later", Err: errors.New("later failure")}
			b.fault, b.malformed = later, mode
			b.foreign = compute.NewF32(compute.Default(), []int{3}, []float32{1, 2, 3})
			b.unrelated = compute.NewF32(b, []int{3}, []float32{4, 5, 6})
			s.halW["unrelated"] = b.unrelated
			s.borrowedHALW = map[string]struct{}{"unrelated": {}}
			wantMatmuls := 1
			switch mode {
			case "gate-status", "primary-before-cleanup":
				b.deferredFaultAt, b.deferredFault = 1, primary
				b.failMatMul = 2
				wantMatmuls = 2
			case "up-status":
				b.deferredFaultAt, b.deferredFault = 2, primary
				wantMatmuls = 3
			case "activation-status":
				b.deferredFaultAt, b.deferredFault = 3, primary
				wantMatmuls = 3
			case "down-status":
				b.deferredFaultAt, b.deferredFault = 4, primary
				wantMatmuls = 3
			case "matmul":
				b.failMatMul = 1
			case "transport":
				b.deferredReadErr = primary
				wantMatmuls = 3
			}
			if mode == "primary-before-cleanup" {
				b.cleanupPanic = true
			}
			chain := s.v41ColdSharedExpertChain(0)
			out, selected, err := chain(x)
			if !selected || err == nil || out != nil || !s.halClosed || b.matmuls != wantMatmuls {
				t.Fatalf("selected=%t out=%v err=%v closed=%t matmuls=%d want=%d", selected, out, err, s.halClosed, b.matmuls, wantMatmuls)
			}
			if b.deferredFaultAt > 0 && (!errors.Is(err, primary) || errors.Is(err, later)) {
				t.Fatalf("fault precedence lost: %v", err)
			}
			if b.deferredFaultAt > 0 && (b.deferredFences != 1 || len(b.deferredReadSites) != 1 || len(b.deferredPending) != 0) {
				t.Fatalf("issued statuses were not drained once: fences=%d batches=%d pending=%d", b.deferredFences, len(b.deferredReadSites), len(b.deferredPending))
			}
			if b.freed[b.unrelated.Buf()] != 0 || b.freed[b.foreign.Buf()] != 0 {
				t.Fatal("borrowed or foreign tensor freed")
			}
			v41ColdSharedAssertReleased(t, b, s)
			before := b.matmuls
			if retry := recoverError(func() { _, _, _ = chain(x) }); retry != s.halFailure || b.matmuls != before {
				t.Fatal("retired chain retried")
			}
		})
	}
}

// This is the RED control: install this helper and test but retain the original
// v41Layer call site. Actual cold Prefill then fails the one-read-per-shared-row
// assertion. Changing only the call site makes this same test GREEN.
// fak-test:runtime medium est=3s lane=default
func TestV41ColdSharedResidentPrefillBridge(t *testing.T) {
	m := v41FullStepModel(t)
	m.Cfg.SwigluLimit = .125
	b := newV41ColdSharedRecorder()
	s := v41DenseTestSession(t, m, b)
	for i, leaf := range []string{"w1", "w3", "w2"} {
		name := layerName(0, "ffn.shared_experts."+leaf+".weight")
		w := s.weightHAL(name)
		b.sharedWeights[w.Buf()] = i
	}
	got := s.Prefill([]int{1, 2})
	want := lastLogits(m.Forward([]int{1, 2}))
	v41GroupedParity(t, got, want, 1e-4)
	if !reflect.DeepEqual(b.sharedReads, []int{1, 1}) {
		t.Fatalf("cold shared data reads=%v want [1 1]", b.sharedReads)
	}
	before := len(b.sharedReads)
	step := s.Step(3)
	v41GroupedParity(t, step, lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
	if len(b.sharedReads) != before+1 || b.sharedReads[before] <= 1 {
		t.Fatal("incremental scalar bridge was changed")
	}
}
