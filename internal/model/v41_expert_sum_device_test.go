package model

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41SumTestBackend owns copied storage per upload, so a seam that mutates an
// input or reads a freed row is visible. DeviceMemory tests routing only.
type v41SumTestBackend struct {
	compute.Backend
	live      map[compute.Buffer]bool
	adds      int
	uploads   int
	faultAdd  int
	faultRead bool
	shortRead bool
	nanRead   bool
}

func newV41SumTestBackend() *v41SumTestBackend {
	return &v41SumTestBackend{Backend: compute.Default(), live: map[compute.Buffer]bool{}}
}

func (b *v41SumTestBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

func (b *v41SumTestBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	values := append([]float32(nil), b.Backend.Read(x)...)
	y := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.live[y.Buf()] = true
	b.uploads++
	return y
}

func (b *v41SumTestBackend) AddInPlace(dst, src compute.Tensor) {
	if !b.live[dst.Buf()] || !b.live[src.Buf()] {
		panic("add on a freed or foreign buffer")
	}
	b.adds++
	if b.faultAdd != 0 && b.adds == b.faultAdd {
		panic(&compute.BackendError{Backend: "sum-test", Site: "AddInPlace", Err: errors.New("injected add fault")})
	}
	b.Backend.AddInPlace(dst, src)
}

func (b *v41SumTestBackend) Read(x compute.Tensor) []float32 {
	if !b.live[x.Buf()] {
		panic("read of a freed buffer")
	}
	if b.faultRead {
		panic(&compute.BackendError{Backend: "sum-test", Site: "Read", Err: errors.New("injected read fault")})
	}
	values := b.Backend.Read(x)
	if b.shortRead {
		return values[:len(values)-1]
	}
	if b.nanRead {
		out := append([]float32(nil), values...)
		out[0] = float32(math.NaN())
		return out
	}
	return values
}

func (b *v41SumTestBackend) Free(x compute.Tensor) {
	if !b.live[x.Buf()] {
		panic("double free")
	}
	delete(b.live, x.Buf())
}

func v41SumFixture(rng *rand.Rand, picks, width int) [][]float32 {
	outputs := make([][]float32, picks)
	for slot := range outputs {
		outputs[slot] = make([]float32, width)
		for i := range outputs[slot] {
			outputs[slot][i] = v41RoundBF16(float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(16)-8))))
		}
	}
	return outputs
}

func v41SameBits(a, b []float32) bool {
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

// fak-test:runtime fast est=20ms lane=default
func TestV41RoutedSumDeviceMatchesHostBits(t *testing.T) {
	rng := rand.New(rand.NewSource(13668))
	const width = 96
	cases := []struct {
		name  string
		picks []routePick
		edit  func([][]float32)
	}{
		{"shuffled-top6", []routePick{{expert: 301, weight: 0.4}, {expert: 7, weight: 0.3}, {expert: 199}, {expert: 42}, {expert: 383}, {expert: 0}}, nil},
		{"duplicate-experts-stable-slots", []routePick{{expert: 9}, {expert: 3}, {expert: 9}, {expert: 3}, {expert: 1}, {expert: 9}}, nil},
		{"signed-zero-and-subnormal", []routePick{{expert: 5}, {expert: 2}, {expert: 4}}, func(rows [][]float32) {
			rows[0][0], rows[1][0], rows[2][0] = float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))
			rows[0][1], rows[1][1], rows[2][1] = math.Float32frombits(1), math.Float32frombits(3), -math.Float32frombits(2)
		}},
		{"cancellation-order", []routePick{{expert: 3}, {expert: 1}, {expert: 2}}, func(rows [][]float32) {
			// (1e8 + 1) - 1e8 differs from 1e8 - 1e8 + 1 only by order.
			rows[0][2], rows[1][2], rows[2][2] = -1e8, 1e8, 1
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			outputs := v41SumFixture(rng, len(tc.picks), width)
			if tc.edit != nil {
				tc.edit(outputs)
			}
			snapshot := make([][]float32, len(outputs))
			for i := range outputs {
				snapshot[i] = append([]float32(nil), outputs[i]...)
			}
			want, err := v41FullRoutedSum(4, tc.picks, outputs, width)
			if err != nil {
				t.Fatal(err)
			}
			b := newV41SumTestBackend()
			s := &Session{M: &Model{}, Backend: b}
			got, err := v41FullRoutedSumWith(4, tc.picks, outputs, width, s.v41RoutedSumFunc())
			if err != nil {
				t.Fatal(err)
			}
			if !v41SameBits(got, want) {
				t.Fatalf("device routed sum differs from host bits:\n got=%v\nwant=%v", got, want)
			}
			if b.adds != len(tc.picks) || b.uploads != len(tc.picks)+1 || len(b.live) != 0 {
				t.Fatalf("adds=%d uploads=%d live=%d, want %d adds, %d uploads, none live", b.adds, b.uploads, len(b.live), len(tc.picks), len(tc.picks)+1)
			}
			for i := range outputs {
				if !v41SameBits(outputs[i], snapshot[i]) {
					t.Fatalf("seam mutated expert output %d", i)
				}
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41SharedAddDeviceMatchesHostBits(t *testing.T) {
	rng := rand.New(rand.NewSource(1294))
	rows := v41SumFixture(rng, 2, 128)
	routed, shared := rows[0], rows[1]
	routed[0], shared[0] = float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))
	routed[1], shared[1] = 1e30, 3e30
	want, err := v41SharedExpertAdd(routed, shared, v41DefaultRouterConfig())
	if err != nil {
		t.Fatal(err)
	}
	routedCopy := append([]float32(nil), routed...)
	b := newV41SumTestBackend()
	s := &Session{M: &Model{}, Backend: b}
	got, err := v41SharedExpertAddWith(2, routed, shared, v41DefaultRouterConfig(), s.v41SharedAddFunc())
	if err != nil {
		t.Fatal(err)
	}
	if !v41SameBits(got, want) || !v41SameBits(routed, routedCopy) || b.adds != 1 || len(b.live) != 0 {
		t.Fatalf("shared add: bits=%v routedIntact=%v adds=%d live=%d", v41SameBits(got, want), v41SameBits(routed, routedCopy), b.adds, len(b.live))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeviceSumNilCallbackKeepsHostAndUnavailableBackendDeclines(t *testing.T) {
	outputs := [][]float32{{1, 2}, {3, 4}}
	picks := []routePick{{expert: 2}, {expert: 1}}
	host, _ := v41FullRoutedSum(0, picks, outputs, 2)
	got, err := v41FullRoutedSumWith(0, picks, outputs, 2, nil)
	if err != nil || !v41SameBits(got, host) {
		t.Fatalf("nil routed-sum callback changed the host site: %v %v", got, err)
	}
	cpu := &Session{M: &Model{}, Backend: compute.Default()}
	if cpu.v41RoutedSumFunc() != nil || cpu.v41SharedAddFunc() != nil {
		t.Fatal("a backend without device memory must not bind the device sum seams")
	}
	var nilSession *Session
	if nilSession.v41RoutedSumFunc() != nil || nilSession.v41SharedAddFunc() != nil {
		t.Fatal("nil session bound a device sum seam")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41DeviceSumFailuresAreTypedAndCloseWithoutHostRetry(t *testing.T) {
	outputs := [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	picks := []routePick{{expert: 8}, {expert: 4}, {expert: 6}}
	for _, tc := range []struct {
		name   string
		arm    func(*v41SumTestBackend)
		rows   [][]float32
		result bool
	}{
		{name: "add-fault", arm: func(b *v41SumTestBackend) { b.faultAdd = 2 }},
		{name: "read-fault", arm: func(b *v41SumTestBackend) { b.faultRead = true }},
		{name: "short-readback", arm: func(b *v41SumTestBackend) { b.shortRead = true }, result: true},
		{name: "non-finite-readback", arm: func(b *v41SumTestBackend) { b.nanRead = true }, result: true},
		{name: "non-finite-input", arm: func(*v41SumTestBackend) {}, rows: [][]float32{{1, 2, 3}, {float32(math.Inf(1)), 5, 6}, {7, 8, 9}}, result: true},
		{name: "ragged-input", arm: func(*v41SumTestBackend) {}, rows: [][]float32{{1, 2, 3}, {4, 5}, {7, 8, 9}}, result: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newV41SumTestBackend()
			tc.arm(b)
			s := &Session{M: &Model{}, Backend: b}
			rows := outputs
			if tc.rows != nil {
				rows = tc.rows
			}
			got, err := v41FullRoutedSumWith(3, picks, rows, 3, s.v41RoutedSumFunc())
			var closed *BackendForwardOperationError
			if got != nil || !errors.As(err, &closed) || closed.Path != "v41-routed-sum" || closed.Layer != 3 || !s.BackendSessionClosed() {
				t.Fatalf("selected failure was not a typed closing error: got=%v err=%v closed=%v", got, err, s.BackendSessionClosed())
			}
			var backend *compute.BackendError
			var expert *V41ExpertOperationError
			if tc.result && (!errors.As(err, &expert) || expert.Stage != "sum") {
				t.Fatalf("result failure lost the expert sum stage: %v", err)
			}
			if !tc.result && !errors.As(err, &backend) {
				t.Fatalf("backend failure lost its BackendError identity: %v", err)
			}
			if len(b.live) != 0 {
				t.Fatalf("%d device rows leaked on failure", len(b.live))
			}
			if retry := recoverError(func() { _, _ = s.v41RoutedSumFunc()(3, picks, outputs, 3) }); retry == nil {
				t.Fatal("closed session admitted a second selected call")
			}
		})
	}
	b := newV41SumTestBackend()
	b.faultAdd = 1
	s := &Session{M: &Model{}, Backend: b}
	got, err := v41SharedExpertAddWith(1, []float32{1, 2}, []float32{3, 4}, v41DefaultRouterConfig(), s.v41SharedAddFunc())
	var closed *BackendForwardOperationError
	if got != nil || !errors.As(err, &closed) || closed.Path != "v41-shared-add" || !s.BackendSessionClosed() || len(b.live) != 0 {
		t.Fatalf("shared add fault: got=%v err=%v", got, err)
	}
	b = newV41SumTestBackend()
	s = &Session{M: &Model{}, Backend: b}
	if _, err := v41SharedExpertAddWith(1, []float32{1, float32(math.NaN())}, []float32{3, 4}, v41DefaultRouterConfig(), s.v41SharedAddFunc()); err == nil {
		t.Fatal("shared add admitted a non-finite routed row")
	} else if route := new(v4RouteError); !errors.As(err, &route) || route.Field != "routed" {
		t.Fatalf("shared add input refusal lost its route field: %v", err)
	}
}
