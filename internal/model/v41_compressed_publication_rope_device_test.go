package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder exercises the selected production owner with cpu-ref storage.
// The structural strict marker is a routing fixture, not a device qualification.
type v41PublicationRoPERecorder struct {
	*v41DenseTestBackend
	exact                      bool
	calls                      []v41InverseRowsCall
	otherCalls                 int
	fault, cleanup             string
	failCall                   int
	cause                      error
	armed                      bool
	cleanupCalls               int
	resources                  map[compute.Buffer]*v41IndexRoPEResource
	outputs                    map[compute.Buffer][]float32
	foreign                    compute.Buffer
	foreignReads, foreignFrees int
}

func v41PublicationRoPERequireCPU(t *testing.T) {
	t.Helper()
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("publication software witness requires cpu-ref")
	}
}
func newV41PublicationRoPERecorder() *v41PublicationRoPERecorder {
	return &v41PublicationRoPERecorder{v41DenseTestBackend: newV41DenseTestBackend(), exact: true, resources: map[compute.Buffer]*v41IndexRoPEResource{}, outputs: map[compute.Buffer][]float32{}}
}
func (b *v41PublicationRoPERecorder) V41TailRoPEExactAvailable() bool { return b.exact }
func (b *v41PublicationRoPERecorder) V41TailRoPERowsAvailable() bool  { return true }
func (b *v41PublicationRoPERecorder) BeginBatch()                     {}
func (b *v41PublicationRoPERecorder) FlushBatch() {
	if b.armed && b.cleanup == "flush" {
		b.cleanupCalls++
		panic(errors.New("secondary publication Close failure"))
	}
}
func (b *v41PublicationRoPERecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	selected := strings.HasPrefix(site, "V4.1 compressed publication RoPE")
	if selected && ((b.fault == "tail-upload" && !strings.HasSuffix(site, " table")) || (b.fault == "table-upload" && strings.HasSuffix(site, " table"))) {
		b.armed = true
		panic(b.cause)
	}
	if class != compute.MemoryActivation || dt != compute.F32 {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	if selected {
		b.resources[out.Buf()] = &v41IndexRoPEResource{}
	}
	return out
}
func (b *v41PublicationRoPERecorder) Read(x compute.Tensor) []float32 {
	if x.Buf() == b.foreign && b.foreign != nil {
		b.foreignReads++
	}
	values := b.Backend.Read(x)
	if _, ok := b.outputs[x.Buf()]; ok && b.fault == "read-length" && len(b.calls) == b.failCall {
		return values[:len(values)-1]
	}
	return values
}
func (b *v41PublicationRoPERecorder) Free(x compute.Tensor) {
	if x.Buf() == b.foreign && b.foreign != nil {
		b.foreignFrees++
	}
	r := b.resources[x.Buf()]
	if r != nil {
		r.attempts++
	}
	if r != nil && b.armed && b.cleanup == "free" && b.cleanupCalls == 0 {
		b.cleanupCalls++
		panic(errors.New("secondary publication Free failure"))
	}
	// Deliberately invalidate a borrowed Read slice before releasing its tensor.
	if values, ok := b.outputs[x.Buf()]; ok {
		for i := range values {
			values[i] = 91
		}
		delete(b.outputs, x.Buf())
	}
	b.v41DenseTestBackend.Free(x)
	if r != nil {
		r.released = true
	}
}
func (b *v41PublicationRoPERecorder) V41TailRoPERows(x, table compute.Tensor, rows, hd, rd int) (compute.Tensor, error) {
	if x.Backend() != b || table.Backend() != b || !reflect.DeepEqual(x.Shape, []int{rows, hd}) || !reflect.DeepEqual(table.Shape, []int{rd / 2, 2}) {
		return compute.Tensor{}, errors.New("publication recorder operand contract")
	}
	c := v41InverseRowsCall{x: append([]float32(nil), b.Backend.Read(x)...), table: append([]float32(nil), b.Backend.Read(table)...), rows: rows, headDim: hd, rotaryDim: rd}
	selected := hd == rd && (rows == 1 || rows == 2)
	if selected {
		b.calls = append(b.calls, c)
	} else {
		b.otherCalls++
	}
	fail := selected && len(b.calls) == b.failCall
	if fail {
		b.armed = true
		switch b.fault {
		case "error":
			return compute.Tensor{}, b.cause
		case "panic":
			panic(b.cause)
		case "alias-input":
			return x, nil
		case "alias-table":
			return table, nil
		case "nil":
			return compute.Tensor{}, nil
		case "foreign":
			out := compute.NewF32(b.Backend, []int{rows, hd}, v41TailRoPEExactOracle(c.x, c.table, hd, rd))
			b.foreign = out.Buf()
			return out, nil
		}
	}
	values := v41TailRoPEExactOracle(c.x, c.table, hd, rd)
	if fail {
		switch b.fault {
		case "index-nonfinite":
			values[0] = float32(math.NaN())
		case "latent-nonfinite":
			values[len(values)-1] = float32(math.Inf(1))
		case "index-overflow":
			values[0] = math.MaxFloat32
		case "latent-overflow":
			values[len(values)-1] = math.MaxFloat32
		}
	}
	out := compute.NewF32(b, []int{rows, hd}, values)
	if selected {
		b.resources[out.Buf()] = &v41IndexRoPEResource{}
		b.outputs[out.Buf()] = b.Backend.Read(out)
	}
	if fail {
		switch b.fault {
		case "partial-error":
			return out, b.cause
		case "shape":
			out.Shape = []int{1, rows * hd}
		case "dtype":
			out.Dtype = compute.F16
		}
	}
	return out, nil
}
func v41PublicationRoPECensus(t *testing.T, b *v41PublicationRoPERecorder, failedFree bool) {
	t.Helper()
	unreleased := 0
	for _, r := range b.resources {
		if r.attempts != 1 {
			t.Fatalf("selected resource Free attempts=%d, want1", r.attempts)
		}
		if !r.released {
			unreleased++
		}
	}
	want := 0
	if failedFree {
		want = 1
	}
	if unreleased != want {
		t.Fatalf("unreleased selected resources=%d, want%d", unreleased, want)
	}
	if b.foreignReads != 0 || b.foreignFrees != 0 {
		t.Fatal("foreign output was read or freed")
	}
}
func v41PublicationRoPEIsolate(s *Session) {
	v41ScoreOnly(s)
	st := s.v41State()
	st.gatherOps.indexRoPE = nil
	st.gatherOps.inverseOutput = nil
}
func v41PublicationRoPEFixture(t *testing.T) *Model {
	t.Helper()
	m := v41CompressorTestFixtureIndex(t, 128)
	m.Cfg.QKRopeHeadDim, m.Cfg.QKNopeHeadDim = 2, 510
	m.Cfg.RopeScaling = ""
	return m
}

// Four separately rounded binary32 products precede the separately rounded
// subtract/add. Binary64 exactly represents each product of binary32 operands;
// explicit narrowing prevents the oracle from inheriting FMA contraction.
// Only after that operation order do integer round-to-nearest-even BF16 bits
// replace the consumed tails. Prefix bits are never interpreted or rounded.
func v41PublicationRoPEOracle(input []float32, rd int, cos, sin []float32) []float32 {
	out := append([]float32(nil), input...)
	for j := 0; j < rd/2; j++ {
		i := len(out) - rd + 2*j
		a, b := float64(input[i]), float64(input[i+1])
		c, s := float64(cos[j]), float64(sin[j])
		ac, bs, bc, as := float32(a*c), float32(b*s), float32(b*c), float32(a*s)
		x, y := float32(float64(ac)-float64(bs)), float32(float64(bc)+float64(as))
		cast := func(v float32) float32 {
			u := math.Float32bits(v)
			return math.Float32frombits((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000)
		}
		out[i], out[i+1] = cast(x), cast(y)
	}
	return out
}

// fak-test:runtime medium est=2s lane=default
// fak-test:justify Existing-API behavioral witness for both production call sites.
func TestV41CompressedPublicationRoPEReachability(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	m := v41PublicationRoPEFixture(t)
	b := newV41PublicationRoPERecorder()
	s := v41DenseTestSession(t, m, b)
	v41PublicationRoPEIsolate(s)
	host := v41DenseTestSession(t, m, newV41DenseTestBackend())
	v41ScoreOnly(host)
	v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}))
	if len(b.calls) != 1 {
		t.Fatalf("cold completed-group publication calls=%d, want1", len(b.calls))
	}
	v41ScoreTestBits(t, s.Step(4), host.Step(4))
	if len(b.calls) != 2 {
		t.Fatalf("incremental completed-group publication calls=%d, want2", len(b.calls))
	}
	v41ScoreTestBits(t, s.Prefill([]int{5, 6}), host.Prefill([]int{5, 6}))
	if len(b.calls) != 3 {
		t.Fatalf("suffix completed-group publication calls=%d, want3", len(b.calls))
	}
	for i, c := range b.calls {
		if c.rows != 2 || c.headDim != 2 || c.rotaryDim != 2 || len(c.x) != 4 {
			t.Fatalf("call%d packed unequal full widths incorrectly: %+v", i, c)
		}
		want := []float32{float32(math.Sin(float64(2 * i))), float32(math.Cos(float64(2 * i)))}
		v41TailRoPECheckBits(t, "first-group position", c.table, want)
	}
	if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(host.v41Forward)) {
		t.Fatal("selected publication changed continuation")
	}
	v41PublicationRoPECensus(t, b, false)
}

// Candidate-only owner API tests below are deliberately excluded from the
// separately frozen, byte-identical existing-API parent/candidate repro.
// fak-test:runtime fast est=30ms lane=default
// fak-test:justify Exact bit publication, tail-only validation and owned readback.
func TestV41CompressedPublicationRoPEBits(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	type vector struct {
		name string
		pos  int
		a, b uint32
		want [2]uint32
	}
	vectors := []vector{
		{"negative-positive-zero", 0, 0x80000000, 0, [2]uint32{0x80000000, 0}},
		{"positive-negative-zero", 0, 0, 0x80000000, [2]uint32{0, 0}},
		{"both-negative-zero", 0, 0x80000000, 0x80000000, [2]uint32{0, 0x80000000}},
		{"minimum-subnormal", 0, 0x00010000, 0, [2]uint32{0x00010000, 0}},
		{"negative-minimum-subnormal", 0, 0x80010000, 0, [2]uint32{0x80010000, 0}},
		{"zero-subnormal-tie", 0, 0x00008000, 0, [2]uint32{0, 0}},
		{"odd-subnormal-tie", 0, 0x00018000, 0, [2]uint32{0x00020000, 0}},
		{"positive-subnormal", 0, 0x00400000, 0, [2]uint32{0x00400000, 0}},
		{"negative-subnormal", 0, 0x80400000, 0, [2]uint32{0x80400000, 0}},
		{"tie-even-positive", 0, 0x3f808000, 0, [2]uint32{0x3f800000, 0}},
		{"tie-odd-positive", 0, 0x3f818000, 0, [2]uint32{0x3f820000, 0}},
		{"tie-even-negative", 0, 0xbf808000, 0, [2]uint32{0xbf800000, 0}},
		{"tie-odd-negative", 0, 0xbf818000, 0, [2]uint32{0xbf820000, 0}},
		{"cancellation-rounding", 2, 0x41820000, 0xc0ee0000, [2]uint32{0x3a060000, 0x418f0000}},
	}
	for _, v := range vectors {
		t.Run(v.name, func(t *testing.T) {
			m := &Model{Cfg: Config{HeadDim: 8, IndexHeadDim: 6, QKRopeHeadDim: 2, RopeTheta: 10000}}
			b := newV41PublicationRoPERecorder()
			s := &Session{M: m, Backend: b}
			t.Cleanup(s.Close)
			t.Cleanup(func() { b.armed = false })
			rotate := s.v41CompressedPublicationRoPEFunc()
			if rotate == nil {
				t.Fatal("strict owner did not bind")
			}
			prefix := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234), 1.001, -2.003, 3, 4}
			latent := append(append([]float32(nil), prefix...), math.Float32frombits(v.a), math.Float32frombits(v.b))
			key := append(append([]float32(nil), prefix[:4]...), math.Float32frombits(v.b), math.Float32frombits(v.a))
			originalL, originalK := append([]float32(nil), latent...), append([]float32(nil), key...)
			cos, sin := v41RopeTableForLayer(m.Cfg, 0, v.pos)
			wantL, wantK := v41PublicationRoPEOracle(latent, 2, cos, sin), v41PublicationRoPEOracle(key, 2, cos, sin)
			for i, u := range v.want {
				if math.Float32bits(wantL[6+i]) != u {
					t.Fatalf("literal oracle disagreement %08x != %08x", math.Float32bits(wantL[6+i]), u)
				}
			}
			if err := m.v41CompressedPublicationRoPEWithDevice(0, v.pos, latent, key, rotate); err != nil {
				t.Fatal(err)
			}
			v41TailRoPECheckBits(t, "latent exact publication", latent, wantL)
			v41TailRoPECheckBits(t, "index exact publication", key, wantK)
			v41TailRoPECheckBits(t, "latent untouched prefix", latent[:6], originalL[:6])
			v41TailRoPECheckBits(t, "index untouched prefix", key[:4], originalK[:4])
			if len(b.calls) != 1 || b.calls[0].rows != 2 {
				t.Fatal("unequal-width tails were not packed in one call")
			}
			v41TailRoPECheckBits(t, "index-first packed input", b.calls[0].x, []float32{originalK[4], originalK[5], originalL[6], originalL[7]})
			v41PublicationRoPECensus(t, b, false)
		})
	}
	t.Run("fused-negative-control", func(t *testing.T) {
		a, b := float64(math.Float32frombits(0x41820000)), float64(math.Float32frombits(0xc0ee0000))
		c, s := float32(math.Cos(2)), float32(math.Sin(2))
		bs := float32(b * float64(s))
		fused := float32(math.FMA(a, float64(c), -float64(bs)))
		u := math.Float32bits(fused)
		rounded := (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
		if rounded != 0x3a070000 {
			t.Fatalf("FMA control=%08x, want3a070000, distinct from ordered3a060000", rounded)
		}
	})
	t.Run("latent-only", func(t *testing.T) {
		m := &Model{Cfg: Config{HeadDim: 4, QKRopeHeadDim: 2, RopeTheta: 10000}}
		b := newV41PublicationRoPERecorder()
		s := &Session{M: m, Backend: b}
		t.Cleanup(s.Close)
		t.Cleanup(func() { b.armed = false })
		x := []float32{1.003, -2.007, .25, -.75}
		cos, sin := v41RopeTableForLayer(m.Cfg, 0, 2)
		want := v41PublicationRoPEOracle(x, 2, cos, sin)
		if err := m.v41CompressedPublicationRoPEWithDevice(0, 2, x, nil, s.v41CompressedPublicationRoPEFunc()); err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "single latent tail", x, want)
		if len(b.calls) != 1 || b.calls[0].rows != 1 {
			t.Fatal("missing index should dispatch one tail")
		}
		v41PublicationRoPECensus(t, b, false)
	})
}

// fak-test:runtime fast est=50ms lane=default
// fak-test:justify Selected failure is transactional, typed, owned and nonreplayable.
func TestV41CompressedPublicationRoPEFailures(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	faults := []string{"tail-upload", "table-upload", "error", "partial-error", "panic", "alias-input", "alias-table", "foreign", "nil", "shape", "dtype", "read-length", "index-nonfinite", "latent-nonfinite", "index-overflow", "latent-overflow"}
	for _, fault := range faults {
		t.Run(fault, func(t *testing.T) {
			m := &Model{Cfg: Config{HeadDim: 8, IndexHeadDim: 6, QKRopeHeadDim: 2, RopeTheta: 10000}}
			b := newV41PublicationRoPERecorder()
			s := &Session{M: m, Backend: b}
			t.Cleanup(s.Close)
			t.Cleanup(func() { b.armed = false })
			b.fault, b.failCall, b.cause = fault, 1, &compute.BackendError{Backend: "publication-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			latent, key := []float32{1, 2, 3, 4, 5, 6, .5, -.25}, []float32{2, 3, 4, 5, 1.5, -.75}
			beforeL, beforeK := append([]float32(nil), latent...), append([]float32(nil), key...)
			rotate := s.v41CompressedPublicationRoPEFunc()
			err := m.v41CompressedPublicationRoPEWithDevice(0, 2, latent, key, rotate)
			var closed *BackendForwardOperationError
			var named *V41TailRoPEOperationError
			if !errors.As(err, &closed) || closed.Path != "v41-compressed-publication-rope" || !errors.As(err, &named) || !s.BackendSessionClosed() {
				t.Fatalf("selected failure lost owner/type: %v", err)
			}
			if (fault == "error" || fault == "partial-error" || fault == "panic" || strings.HasSuffix(fault, "upload")) && !errors.Is(err, b.cause) {
				t.Fatal("selected cause lost")
			}
			v41TailRoPECheckBits(t, "no latent publication", latent, beforeL)
			v41TailRoPECheckBits(t, "no index publication", key, beforeK)
			calls := len(b.calls)
			if retry := recoverError(func() { _, _ = rotate(0, []float32{1, 2}, []float32{1}, []float32{0}, 1, 2) }); retry != s.halFailure || len(b.calls) != calls {
				t.Fatal("retired owner retried")
			}
			v41PublicationRoPECensus(t, b, false)
		})
	}
	for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fffffff, 0xffffffff} {
		t.Run(fmt.Sprintf("invalid-input-%08x", bits), func(t *testing.T) {
			m := &Model{Cfg: Config{HeadDim: 4, QKRopeHeadDim: 2, RopeTheta: 10000}}
			b := newV41PublicationRoPERecorder()
			s := &Session{M: m, Backend: b}
			t.Cleanup(s.Close)
			t.Cleanup(func() { b.armed = false })
			x := []float32{1, 2, math.Float32frombits(bits), 0}
			before := append([]float32(nil), x...)
			err := m.v41CompressedPublicationRoPEWithDevice(0, 0, x, nil, s.v41CompressedPublicationRoPEFunc())
			var closed *BackendForwardOperationError
			if !errors.As(err, &closed) || closed.Stage != "payload" || len(b.calls) != 0 || len(b.resources) != 0 || !s.BackendSessionClosed() {
				t.Fatalf("invalid tail did not refuse before upload: %v", err)
			}
			v41TailRoPECheckBits(t, "invalid tail unchanged", x, before)
		})
	}
	for _, cleanup := range []string{"", "free", "flush"} {
		t.Run("wrapped-primary-"+cleanup, func(t *testing.T) {
			m := &Model{Cfg: Config{HeadDim: 4, QKRopeHeadDim: 2, RopeTheta: 10000}}
			b := newV41PublicationRoPERecorder()
			s := &Session{M: m, Backend: b}
			t.Cleanup(s.Close)
			t.Cleanup(func() { b.armed = false })
			inner := &BackendForwardOperationError{Backend: "publication-recorder", Path: "nested-selected-operation", Cause: ErrV41ForwardStage}
			primary := fmt.Errorf("outer attribution: %w", inner)
			b.fault, b.failCall, b.cause, b.cleanup = "panic", 1, primary, cleanup
			rotate := s.v41CompressedPublicationRoPEFunc()
			x := []float32{1, 2, .5, -.25}
			before := append([]float32(nil), x...)
			first := recoverError(func() { _ = m.v41CompressedPublicationRoPEWithDevice(0, 2, x, nil, rotate) })
			if first != primary || s.halFailure != primary || !s.BackendSessionClosed() {
				t.Fatalf("wrapped primary identity was replaced: %v", first)
			}
			if retry := recoverError(func() { _, _ = rotate(0, []float32{1, 2}, []float32{1}, []float32{0}, 1, 2) }); retry != primary || len(b.calls) != 1 {
				t.Fatal("retry did not retain exact wrapper")
			}
			if cleanup != "" && b.cleanupCalls != 1 {
				t.Fatal("secondary cleanup fault was not reached")
			}
			v41TailRoPECheckBits(t, "panic atomicity", x, before)
			v41PublicationRoPECensus(t, b, cleanup == "free")
		})
	}
}

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Real current-owner restore, both gather callbacks and group rollback.
func TestV41CompressedPublicationRoPERestore(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	m := v41PublicationRoPEFixture(t)
	b := newV41PublicationRoPERecorder()
	source := v41DenseTestSession(t, m, b)
	v41ScoreOnly(source)
	host := v41DenseTestSession(t, m, newV41DenseTestBackend())
	v41ScoreOnly(host)
	v41ScoreTestBits(t, source.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}))
	if len(b.calls) != 1 || b.otherCalls == 0 {
		t.Fatal("source did not execute publication plus other gather rotations")
	}
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	target := v41DenseTestSession(t, m, b)
	t.Cleanup(func() { b.armed = false })
	if err := snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	v41ScoreOnly(target)
	source.Close()
	if target.v41State().gatherOps.owner != target || target.v41Forward.callbackOwner != target {
		t.Fatal("restored carrier owner is stale")
	}
	prior := b.otherCalls
	v41ScoreTestBits(t, target.Prefill([]int{4}), host.Prefill([]int{4}))
	if len(b.calls) != 2 || b.otherCalls <= prior {
		t.Fatal("restored suffix omitted publication or another gather rotation")
	}
	v41ScoreTestBits(t, target.Step(5), host.Step(5))
	if len(b.calls) != 2 {
		t.Fatal("incomplete group re-rotated a completed publication")
	}
	before := captureV41ForwardSnapshot(target.v41Forward)
	primary := fmt.Errorf("restored wrapper: %w", &BackendForwardOperationError{Backend: "publication-recorder", Path: "restored-selected-operation", Cause: ErrV41ForwardStage})
	b.fault, b.failCall, b.cause, b.cleanup = "panic", 3, primary, "flush"
	first := recoverError(func() { target.Step(6) })
	if first != primary || target.halFailure != primary || !target.BackendSessionClosed() || b.cleanupCalls != 1 {
		t.Fatalf("restored current owner lost wrapped primary: %v", first)
	}
	if !reflect.DeepEqual(before, captureV41ForwardSnapshot(target.v41Forward)) {
		t.Fatal("failed completing group changed cursor, pending rows or registry")
	}
	if retry := recoverError(func() { target.Step(6) }); retry != primary || len(b.calls) != 3 {
		t.Fatal("retired restored owner replayed")
	}
	v41PublicationRoPECensus(t, b, false)
}

// fak-test:runtime medium est=2s lane=default
// fak-test:justify A cold refold reuses completed keys but still rotates fresh latent tails.
func TestV41CompressedPublicationRoPEColdReusedKeys(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	m := v41PublicationRoPEFixture(t)
	b := newV41PublicationRoPERecorder()
	s := v41DenseTestSession(t, m, b)
	v41PublicationRoPEIsolate(s)
	s.Prefill([]int{1, 2, 3})
	if len(b.calls) != 1 || b.calls[0].rows != 2 {
		t.Fatal("first cold fold did not publish fresh pair")
	}
	before := captureV41ForwardSnapshot(s.v41Forward)
	if _, err := m.forwardV41(nil, s.v41State()); err != nil {
		t.Fatal(err)
	}
	if len(b.calls) != 2 || b.calls[1].rows != 1 {
		t.Fatal("memoized key was rotated again instead of dispatching only fresh latent")
	}
	if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
		t.Fatal("cold refold changed already published continuation")
	}
	v41PublicationRoPECensus(t, b, false)
}

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Selected readback faults reach the actual group rollback and no-replay boundary.
func TestV41CompressedPublicationRoPEStepFailures(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	for _, fault := range []string{"index-nonfinite", "latent-overflow", "partial-error"} {
		t.Run(fault, func(t *testing.T) {
			m := v41PublicationRoPEFixture(t)
			b := newV41PublicationRoPERecorder()
			s := v41DenseTestSession(t, m, b)
			v41PublicationRoPEIsolate(s)
			s.Prefill([]int{1, 2, 3})
			if len(b.calls) != 1 {
				t.Fatal("warm group did not use selected publication")
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.fault, b.failCall, b.cause = fault, 2, &compute.BackendError{Backend: "publication-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			err := recoverError(func() { s.Step(4) })
			var closed *BackendForwardOperationError
			var named *V41TailRoPEOperationError
			if !errors.As(err, &closed) || !errors.As(err, &named) || closed.Path != "v41-compressed-publication-rope" || !s.BackendSessionClosed() {
				t.Fatalf("actual consumer did not fail closed: %v", err)
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("failed group changed pending rows/cursor/registry")
			}
			if retry := recoverError(func() { s.Step(4) }); retry != s.halFailure || len(b.calls) != 2 {
				t.Fatal("selected failure replayed")
			}
			v41PublicationRoPECensus(t, b, false)
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
// fak-test:justify An ordinary Rows backend cannot silently acquire the strict arithmetic promise.
func TestV41CompressedPublicationRoPEUnavailable(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	m := &Model{Cfg: Config{HeadDim: 4, IndexHeadDim: 4, QKRopeHeadDim: 2, RopeTheta: 10000}}
	b := newV41PublicationRoPERecorder()
	b.exact = false
	s := &Session{M: m, Backend: b}
	t.Cleanup(s.Close)
	if s.v41CompressedPublicationRoPEFunc() != nil {
		t.Fatal("strict availability=false bound a callback")
	}
	got, want := []float32{1.003, -2.007, .25, -.75}, []float32{1.003, -2.007, .25, -.75}
	if err := m.v41CompressedPublicationRoPE(0, 2, want, nil); err != nil {
		t.Fatal(err)
	}
	if err := m.v41CompressedPublicationRoPEWithDevice(0, 2, got, nil, s.v41CompressedPublicationRoPEFunc()); err != nil {
		t.Fatal(err)
	}
	v41TailRoPECheckBits(t, "unchanged portable arm", got, want)
	if len(b.calls) != 0 || s.BackendSessionClosed() {
		t.Fatal("portable arm dispatched or retired")
	}
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	ops := s.v41DeviceGatherOpsFunc()
	before := append([]float32(nil), got...)
	err := m.v41CompressedPublicationRoPEWithDevice(0, 2, got, nil, ops.compressedPublicationRoPEFunc())
	if !errors.Is(err, errV41DeviceOnlyHostOp) || !s.BackendSessionClosed() || len(b.calls) != 0 {
		t.Fatalf("missing strict capability failed open: %v", err)
	}
	v41TailRoPECheckBits(t, "device-only no publication", got, before)
}

// fak-test:runtime medium est=2s lane=default
// fak-test:justify The cold assembly also carries strict publication in admitted reduced geometry.
func TestV41CompressedPublicationRoPEReducedCold(t *testing.T) {
	v41PublicationRoPERequireCPU(t)
	m := v41IncrementalPlainModel(t, 2)
	c := &m.Cfg
	c.DeepSeekV41.Attention = DeepSeekV41AttentionGeometry{}
	c.DeepSeekV41.CompressRatios = []int{2, 2}
	c.DeepSeekV41.KVSourceLayerIDs = []int{0}
	c.DeepSeekV41.IndexSourceLayerIDs = []int{0}
	c.IndexNHeads, c.IndexHeadDim, c.IndexTopK = 1, 16, 2
	c.QKRopeHeadDim, c.QKNopeHeadDim = 2, c.HeadDim-2
	c.RopeScaling = ""
	var extra []synthTensor
	for layer := 0; layer < c.NumLayers; layer++ {
		for _, spec := range []struct {
			leaf  string
			shape []int
		}{
			{"attn.compressor.wkv.weight", []int{c.HeadDim, c.HiddenSize}},
			{"attn.compressor.wgate.weight", []int{c.HeadDim, c.HiddenSize}},
			{"attn.compressor.norm.weight", []int{c.HeadDim}},
			{"indexer.wq_b.weight", []int{c.IndexHeadDim, c.QLoraRank}},
			{"indexer.wk.weight", []int{c.IndexHeadDim, c.HeadDim}},
			{"indexer.k_norm.weight", []int{c.IndexHeadDim}},
			{"indexer.weights_proj.weight", []int{1, c.HiddenSize}},
		} {
			extra = append(extra, synthTensor{layerName(layer, spec.leaf), spec.shape})
		}
	}
	man, raw := synthBuildRaw(extra, func(name string, next func() float32) float32 {
		if strings.HasSuffix(name, "norm.weight") {
			return 1
		}
		return synthMatmulFill(name, next)
	})
	for name, meta := range man {
		meta.Offset += len(m.raw)
		m.manifest[name] = meta
	}
	m.raw = append(m.raw, raw...)
	if full, err := v41ForwardGeometry(m.Cfg); err != nil || full {
		t.Fatalf("fixture is not reduced: %t %v", full, err)
	}
	b := newV41PublicationRoPERecorder()
	s := v41DenseTestSession(t, m, b)
	v41PublicationRoPEIsolate(s)
	host := v41DenseTestSession(t, m, newV41DenseTestBackend())
	v41ScoreOnly(host)
	v41ScoreTestBits(t, s.Prefill([]int{1, 2, 3, 4}), host.Prefill([]int{1, 2, 3, 4}))
	if len(b.calls) != 2 {
		t.Fatalf("reduced cold completed-group calls=%d,want2", len(b.calls))
	}
	for group, call := range b.calls {
		if call.rows != 2 || call.headDim != 2 {
			t.Fatal("reduced unequal full widths leaked into packed geometry")
		}
		v41TailRoPECheckBits(t, "reduced group-first table", call.table, []float32{float32(math.Sin(float64(2 * group))), float32(math.Cos(float64(2 * group)))})
	}
	v41PublicationRoPECensus(t, b, false)
}
