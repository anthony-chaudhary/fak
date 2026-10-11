package model

import (
	"errors"
	"math/rand"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RouterTestBackend routes with the host oracle over its own copied
// storage, so these tests witness plumbing, ordering, readback scope and fault
// typing; numeric parity of a real kernel is the vulkan-tagged witness.
type v41RouterTestBackend struct {
	*v41SumTestBackend
	supported bool
	calls     int
	reads     int
	fault     error
}

func newV41RouterTestBackend() *v41RouterTestBackend {
	return &v41RouterTestBackend{v41SumTestBackend: newV41SumTestBackend(), supported: true}
}

func (b *v41RouterTestBackend) V41RouterTopKAvailable() bool { return b.supported }

func (b *v41RouterTestBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.v41SumTestBackend.Read(x)
}

func (b *v41RouterTestBackend) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, routeScale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	b.calls++
	if !b.live[logits.Buf()] || !b.live[bias.Buf()] || !reflect.DeepEqual(logits.Shape, []int{rows, experts}) {
		return compute.V41RouterPicks{}, errors.New("router operands are not live device tensors of the declared shape")
	}
	if b.fault != nil {
		return compute.V41RouterPicks{}, b.fault
	}
	all := b.Backend.Read(logits)
	var corr []float32
	if hasBias {
		corr = b.Backend.Read(bias)
	}
	cfg := v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: routeScale}
	var out compute.V41RouterPicks
	for row := 0; row < rows; row++ {
		picks, err := v41RouteForGeometry(all[row*experts:(row+1)*experts], corr, cfg, full)
		if err != nil {
			return compute.V41RouterPicks{}, err
		}
		for _, p := range picks {
			out.Experts = append(out.Experts, int32(p.expert))
			out.Weights = append(out.Weights, p.weight)
		}
	}
	return out, nil
}

func v41RouterFixture(rng *rand.Rand) (logits, bias []float32) {
	logits = make([]float32, V41RouterExperts)
	bias = make([]float32, V41RouterExperts)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 2)
		bias[i] = float32(rng.NormFloat64() * 0.1)
	}
	return logits, bias
}

// fak-test:runtime fast est=20ms lane=default
func TestV41RouterTopKDeviceSeamMatchesHostAndReadsBackOnlyPicks(t *testing.T) {
	rng := rand.New(rand.NewSource(13668))
	cfg := v41DefaultRouterConfig()
	for _, full := range []bool{false, true} {
		for _, withBias := range []bool{false, true} {
			logits, bias := v41RouterFixture(rng)
			if !withBias {
				bias = nil
			}
			// Exact ties: equal logits and bias keep the lower expert first.
			logits[17], logits[260] = 9, 9
			if withBias {
				bias[17], bias[260] = 0.25, 0.25
			}
			want, err := v41RouteForGeometry(logits, bias, cfg, full)
			if err != nil {
				t.Fatal(err)
			}
			b := newV41RouterTestBackend()
			s := &Session{M: &Model{}, Backend: b}
			got, err := v41RouteForGeometryWith(5, logits, bias, cfg, full, s.v41RouterTopKFunc())
			if err != nil {
				t.Fatalf("full=%t bias=%t: %v", full, withBias, err)
			}
			if !reflect.DeepEqual(got, want) || got[0].expert != 17 || got[1].expert != 260 {
				t.Fatalf("full=%t bias=%t: device picks %v, host %v", full, withBias, got, want)
			}
			if b.calls != 1 || b.reads != 0 || len(b.live) != 0 {
				t.Fatalf("calls=%d logit/bias readbacks=%d live=%d, want one call, no readback, nothing live", b.calls, b.reads, len(b.live))
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeviceRouteBatchesRows(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	cfg := v41DefaultRouterConfig()
	a, bias := v41RouterFixture(rng)
	c, _ := v41RouterFixture(rng)
	b := newV41RouterTestBackend()
	lt := b.Upload(compute.NewF32(compute.Default(), []int{2, cfg.Experts}, append(append([]float32(nil), a...), c...)), compute.F32)
	bt := b.Upload(compute.NewF32(compute.Default(), []int{cfg.Experts}, bias), compute.F32)
	rows, err := v41DeviceRoute(b, lt, bt, 2, cfg, true, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, logits := range [][]float32{a, c} {
		want, _ := v41RouteForGeometry(logits, bias, cfg, true)
		if !reflect.DeepEqual(rows[i], want) {
			t.Fatalf("row %d: %v, want %v", i, rows[i], want)
		}
	}
	if b.calls != 1 || b.reads != 0 {
		t.Fatalf("a two-row route took %d calls and %d logit readbacks", b.calls, b.reads)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41RouterTopKDeviceFailuresAreTypedAndClose(t *testing.T) {
	logits, bias := v41RouterFixture(rand.New(rand.NewSource(3)))
	cfg := v41DefaultRouterConfig()
	for _, tc := range []struct {
		name  string
		fault error
		field string
	}{
		{"logit", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageLogit, Expert: 4}, "logits"},
		{"score", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageScore, Expert: 4}, "logits"},
		{"bias", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageBias, Expert: 3}, "correction_bias"},
		{"selection", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageSelection, Expert: 3}, "selection_score"},
		{"normalization", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageNormalization, Expert: -1}, "normalization"},
		{"weight", &compute.V41RouterTopKFault{Stage: compute.V41RouterTopKStageWeight, Expert: 9}, "weight"},
		{"backend", &compute.BackendError{Backend: "router-test", Site: "V41RouterTopK", Err: errors.New("injected")}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newV41RouterTestBackend()
			b.fault = tc.fault
			s := &Session{M: &Model{}, Backend: b}
			got, err := v41RouteForGeometryWith(2, logits, bias, cfg, true, s.v41RouterTopKFunc())
			var closed *BackendForwardOperationError
			if got != nil || !errors.As(err, &closed) || closed.Path != "v41-router-topk" || !s.BackendSessionClosed() || len(b.live) != 0 {
				t.Fatalf("selected route failure: got=%v err=%v closed=%t live=%d", got, err, s.BackendSessionClosed(), len(b.live))
			}
			if tc.field != "" {
				var route *v4RouteError
				if !errors.As(err, &route) || route.Field != tc.field {
					t.Fatalf("fault mapped to %v, want field %q", err, tc.field)
				}
			} else if backend := new(compute.BackendError); !errors.As(err, &backend) {
				t.Fatalf("backend failure lost its identity: %v", err)
			}
			if b.calls != 1 {
				t.Fatalf("selected failure was retried: %d calls", b.calls)
			}
		})
	}
	b := newV41RouterTestBackend()
	s := &Session{M: &Model{}, Backend: b}
	if _, err := v41RouteForGeometryWith(0, logits[:10], bias, cfg, true, s.v41RouterTopKFunc()); err == nil || b.calls != 0 {
		t.Fatalf("short logit row reached the device: err=%v calls=%d", err, b.calls)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41RouterTopKUnavailableKeepsHostRoute(t *testing.T) {
	logits, bias := v41RouterFixture(rand.New(rand.NewSource(11)))
	cfg := v41DefaultRouterConfig()
	b := newV41RouterTestBackend()
	b.supported = false
	s := &Session{M: &Model{}, Backend: b}
	if s.v41RouterTopKFunc() != nil {
		t.Fatal("an unsupported router primitive bound the device seam")
	}
	if (&Session{M: &Model{}, Backend: newV41SumTestBackend()}).v41RouterTopKFunc() != nil {
		t.Fatal("a backend without the router primitive bound the device seam")
	}
	host, _ := v41RouteForGeometry(logits, bias, cfg, true)
	got, err := v41RouteForGeometryWith(0, logits, bias, cfg, true, nil)
	if err != nil || !reflect.DeepEqual(got, host) {
		t.Fatalf("nil router callback changed the host route: %v %v", got, err)
	}
}
