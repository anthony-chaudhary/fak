package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A cpu-ref recorder for the strict scalar multiply on top of the ordinary
// publication recorder. It witnesses the Session wiring, operation order,
// ownership and fail-closed policy, not a physical Vulkan command.
type v41IndexWeightScaleRecorder struct {
	*v41BF16PublishRecorder
	available bool
	scales    []float32
	inputs    [][]float32
	failure   error
	malformed bool
}

func newV41IndexWeightScaleRecorder() *v41IndexWeightScaleRecorder {
	b := &v41IndexWeightScaleRecorder{v41BF16PublishRecorder: newV41BF16PublishRecorder(), available: true}
	b.self = b
	return b
}

func (b *v41IndexWeightScaleRecorder) V41ScaleExactAvailable() bool { return b.available }
func (b *v41IndexWeightScaleRecorder) V41ScaleExact(x compute.Tensor, scale float32) (compute.Tensor, error) {
	if len(x.Shape) != 1 || x.Dtype != compute.F32 || x.Backend() != compute.Backend(b) {
		panic("exact scale operand")
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	b.scales = append(b.scales, scale)
	b.inputs = append(b.inputs, append([]float32(nil), values...))
	if b.failure != nil {
		return compute.Tensor{}, b.failure
	}
	for i, v := range values {
		values[i] = v * scale
	}
	if b.malformed {
		values = values[:len(values)-1]
	}
	return compute.NewF32(b, []int{len(values)}, values), nil
}
func (b *v41IndexWeightScaleRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 index weight scale input" {
		return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	}
	return b.v41BF16PublishRecorder.UploadClass(x, dt, class, site)
}

func v41IndexWeightScaleRequireCPU(t *testing.T) {
	t.Helper()
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("index weight scale software witness requires cpu-ref")
	}
}

// v41IndexWeightScaleRows are finite projections: BF16 ties both ways, signed
// zero, subnormals, the largest finite BF16 and values whose F32 product is
// not BF16, so both boundaries and the multiply's own rounding are exercised.
func v41IndexWeightScaleRows(heads int) [][]float32 {
	words := []uint32{
		0x3f800000, 0x3f848000, 0x3f808000, 0x3f818000, 0xbf808000, 0x80000000, 0x00000000,
		0x00010000, 0x80010000, 0x007f0000, 0x00800000, 0x7f7f0000, 0xff7f0000, 0x3f807fff,
		0x41820000, 0xc0ee0000, 0x3eaaaaab, 0x00018000, 0x80018000, 0x3fc00000,
	}
	var rows [][]float32
	for shift := 0; shift < len(words); shift += heads {
		row := make([]float32, heads)
		for h := range row {
			row[h] = math.Float32frombits(words[(shift+h)%len(words)])
		}
		rows = append(rows, row)
	}
	return rows
}

// The bound seam is bit-exact against the host arm for every scale geometry:
// it uploads the published projection, multiplies once on the strict module by
// the host's own F32 scalar and publishes the resident product at site 25.
// fak-test:runtime fast est=200ms lane=default
func TestV41IndexWeightScaleDeviceMatchesHost(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	for _, tc := range []struct{ dim, heads int }{{128, 64}, {128, 1}, {3, 4}, {64, 2}} {
		m := v41IndexerTestFixture(t)
		m.Cfg.IndexNHeads = tc.heads
		b := newV41IndexWeightScaleRecorder()
		s := v41DenseTestSession(t, m, b)
		scaled := s.v41IndexWeightScaleFunc()
		publish := s.v41BF16PublishFunc()
		if scaled == nil || publish == nil || s.v41State().gatherOps.indexWeightScale == nil {
			t.Fatalf("dim=%d heads=%d: seam not bound for a backend that admits it", tc.dim, tc.heads)
		}
		for r, row := range v41IndexWeightScaleRows(tc.heads) {
			want, err := v41IndexWeightsBF16(0, row, tc.dim, tc.heads)
			if err != nil {
				t.Fatal(err)
			}
			before := append([]float32(nil), row...)
			multiplies, publications := len(b.scales), b.calls
			var got []float32
			if rounds := v41BF16PublishHostRounds(t, func() {
				got, err = v41IndexWeightsBF16Device(0, row, tc.dim, tc.heads, publish, scaled)
			}); rounds != 0 || err != nil {
				t.Fatalf("dim=%d heads=%d row=%d: host rounds=%d err=%v", tc.dim, tc.heads, r, rounds, err)
			}
			if !slices.Equal(row, before) {
				t.Fatal("seam changed its caller's projection")
			}
			if len(b.scales) != multiplies+1 || b.scales[len(b.scales)-1] != v41IndexWeightScale(tc.dim, tc.heads) {
				t.Fatalf("multiplies=%d scale=%v, want one at the host scalar", len(b.scales)-multiplies, b.scales[multiplies:])
			}
			if b.sites[len(b.sites)-1] != compute.V41BF16PublishIndexerScaledWeight || b.calls-publications > 2 {
				t.Fatalf("publications %v, want the scaled weight last", b.sites[publications:])
			}
			if !v41BF16Widened(b.inputs[len(b.inputs)-1]) {
				t.Fatal("multiply received an unpublished projection")
			}
			if len(got) != len(want) {
				t.Fatalf("width %d want %d", len(got), len(want))
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("dim=%d heads=%d row=%d lane=%d device %08x host %08x", tc.dim, tc.heads, r, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
		}
		if s.BackendSessionClosed() {
			t.Fatal("successful seams retired the session")
		}
	}
}

// Admission binds the row only with the strict multiply and the publication;
// under device-only an unbound seam refuses typed before any host multiply.
// fak-test:runtime fast est=200ms lane=default
func TestV41IndexWeightScaleDeviceOnlyBinding(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	for _, op := range v41DeviceOnlyHostOps {
		if op.name == "indexer-weight-scale" {
			t.Fatal("indexer-weight-scale runs on its device seam but is still statically inventoried as host")
		}
	}
	names := func(s *Session) []string { return v41HostOpNames(s.v41DeviceOnlyUnboundOps()) }
	m := v41IndexerTestFixture(t)
	if !v41UsesIndexer(m.Cfg) {
		t.Fatal("fixture must run the indexer")
	}
	if got := names(v41DenseTestSession(t, m, newV41IndexWeightScaleRecorder())); slices.Contains(got, "indexer-weight-scale") {
		t.Fatalf("scale backend leaves the seam unbound: %v", got)
	}
	unavailable := newV41IndexWeightScaleRecorder()
	unavailable.available = false
	for name, b := range map[string]compute.Backend{"no-multiply": newV41BF16PublishRecorder(), "multiply-unavailable": unavailable} {
		if got := names(v41DenseTestSession(t, m, b)); !slices.Contains(got, "indexer-weight-scale") {
			t.Fatalf("%s backend is admitted for the index weight scale: %v", name, got)
		}
	}
	plain := &Session{M: &Model{Cfg: Config{DeepSeekV41: &DeepSeekV41Config{CompressRatios: []int{1, 1}}}}, Backend: newV41BF16PublishRecorder()}
	if got := names(plain); slices.Contains(got, "indexer-weight-scale") {
		t.Fatalf("model without an indexer names the unused scale op: %v", got)
	}

	s := v41DenseTestSession(t, m, newV41BF16PublishRecorder())
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	scaled := s.v41DeviceGatherOpsFunc().indexWeightScaleFunc()
	if scaled == nil {
		t.Fatal("device-only accessor returned no refusal for an unbound scale")
	}
	row := []float32{1.5}
	var err error
	if rounds := v41BF16PublishHostRounds(t, func() {
		_, err = v41IndexWeightsBF16Device(0, row, 128, 1, s.v41DeviceGatherOpsFunc().bf16PublishFunc(), scaled)
	}); rounds != 0 {
		t.Fatalf("device-only refusal rounded %d values on the host", rounds)
	}
	var closed *BackendForwardOperationError
	if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Path != "device-only" || !s.BackendSessionClosed() {
		t.Fatalf("unbound device-only scale = %v (closed=%t), want typed refusal and retirement", err, s.BackendSessionClosed())
	}

	refused := v41DenseTestSession(t, m, newV41BF16PublishRecorder())
	refused.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	err = recoverError(func() { refused.validateDeviceOnlyExecution("decode") })
	var hostOps *v41DeviceOnlyHostOpError
	if !errors.As(err, &hostOps) || !slices.Contains(v41HostOpNames(hostOps.ops), "indexer-weight-scale") || !refused.BackendSessionClosed() {
		t.Fatalf("device-only admission = %v, want a refusal naming indexer-weight-scale", err)
	}
}

// A selected failure retires the session typed and never replays on the host;
// an input outside the published-BF16 domain refuses before any dispatch.
// fak-test:runtime fast est=200ms lane=default
func TestV41IndexWeightScaleDeviceFailsClosed(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	m := v41IndexerTestFixture(t)
	m.Cfg.IndexNHeads = 2
	scale := v41IndexWeightScale(128, 2)
	for _, tc := range []struct {
		name      string
		arm       func(*v41IndexWeightScaleRecorder)
		weights   []float32
		scale     float32
		multiples int
	}{
		{"multiply-error", func(b *v41IndexWeightScaleRecorder) {
			b.failure = &compute.BackendError{Backend: "vulkan", Site: "V41ScaleExact", Err: compute.ErrVulkanExecutionFailed}
		}, []float32{1, -2}, scale, 1},
		{"malformed-product", func(b *v41IndexWeightScaleRecorder) { b.malformed = true }, []float32{1, -2}, scale, 1},
		{"publication-fault", func(b *v41IndexWeightScaleRecorder) {
			b.v41BF16PublishRecorder.faultAt = 1
			b.v41BF16PublishRecorder.fault = &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishIndexerScaledWeight, Index: 1}
		}, []float32{1, -2}, scale, 1},
		{"unpublished-input", nil, []float32{1, math.Float32frombits(0x3f808000)}, scale, 0},
		{"nonfinite-scale", nil, []float32{1, -2}, float32(math.Inf(1)), 0},
		{"wrong-width", nil, []float32{1}, scale, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newV41IndexWeightScaleRecorder()
			if tc.arm != nil {
				tc.arm(b)
			}
			s := v41DenseTestSession(t, m, b)
			scaled := s.v41IndexWeightScaleFunc()
			if scaled == nil {
				t.Fatal("seam not bound")
			}
			in := append([]float32(nil), tc.weights...)
			out, err := scaled(3, in, tc.scale)
			var closed *BackendForwardOperationError
			if out != nil || !errors.Is(err, ErrV41IndexWeightScaleOperation) || !errors.As(err, &closed) ||
				closed.Path != "v41-index-weight-scale" || closed.Layer != 3 || !s.BackendSessionClosed() || s.halFailure != err {
				t.Fatalf("failure = %v out=%v closed=%t, want typed retirement", err, out, s.BackendSessionClosed())
			}
			if errors.Is(err, ErrV41ForwardStage) {
				t.Fatal("selected failure wraps the forward stage sentinel and could authorize host replay")
			}
			if len(b.scales) != tc.multiples || !slices.Equal(in, tc.weights) {
				t.Fatalf("multiplies=%d (want %d) or input changed", len(b.scales), tc.multiples)
			}
			if wantPublish := tc.name == "publication-fault"; (b.calls == 1) != wantPublish {
				t.Fatalf("publications=%d, want one only when the product reaches publication", b.calls)
			}
			if again := recoverError(func() { _, _ = scaled(3, []float32{1, 1}, scale) }); !errors.Is(again, ErrV41IndexWeightScaleOperation) {
				t.Fatalf("retired session accepted another seam: %v", again)
			}
		})
	}
}

// A real indexer session routes every weight scale through the seam and keeps
// the host-multiply session's logits bit for bit.
// fak-test:runtime medium est=3s lane=default
func TestV41IndexWeightScaleSessionParity(t *testing.T) {
	v41IndexWeightScaleRequireCPU(t)
	prompt := []int{1, 2, 3, 4}
	run := func(b compute.Backend) [][]float32 {
		m := v41IndexerTestFixture(t)
		s := v41DenseTestSession(t, m, b)
		var logits [][]float32
		last := s.Prefill(prompt[:2])
		logits = append(logits, append([]float32(nil), last...))
		for _, id := range prompt[2:] {
			logits = append(logits, append([]float32(nil), s.Step(id)...))
		}
		if s.BackendSessionClosed() {
			t.Fatal("session retired")
		}
		return logits
	}
	host := run(newV41BF16PublishRecorder())
	b := newV41IndexWeightScaleRecorder()
	stop := enableV41BF16Witness()
	device := run(b)
	calls := v41IndexWeightScaleCalls.Load()
	stop()
	if calls == 0 || int64(len(b.scales)) != calls {
		t.Fatalf("seam calls=%d multiplies=%d, want every index weight scale on the seam", calls, len(b.scales))
	}
	for step := range host {
		if len(device[step]) != len(host[step]) {
			t.Fatalf("step %d width %d want %d", step, len(device[step]), len(host[step]))
		}
		for i := range host[step] {
			if math.Float32bits(device[step][i]) != math.Float32bits(host[step][i]) {
				t.Fatalf("step %d logit %d device %08x host %08x", step, i, math.Float32bits(device[step][i]), math.Float32bits(host[step][i]))
			}
		}
	}
	t.Logf("index weight scale seam calls=%d over %d steps, logits bit-identical", calls, len(host))
}
