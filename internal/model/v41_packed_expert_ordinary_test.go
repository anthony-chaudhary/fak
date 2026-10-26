package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41PackedOrdinaryBackend struct {
	*v41RoutedSwiGLUTestBackend
	calls   int
	failAt  int
	failure error
}

func newV41PackedOrdinaryBackend() *v41PackedOrdinaryBackend {
	return &v41PackedOrdinaryBackend{v41RoutedSwiGLUTestBackend: newV41RoutedSwiGLUTestBackend()}
}

func (b *v41PackedOrdinaryBackend) GemvExpertMXFP4E8M0(o, i int, weight, scales []byte, x []float32) ([]float32, error) {
	b.calls++
	if b.failAt == b.calls {
		return nil, b.failure
	}
	return compute.GemvMXFP4E8M0("ordinary-packed-test", o, i, weight, scales, x)
}

func (b *v41PackedOrdinaryBackend) GemvExpertFP8Block32E8M0(int, int, []byte, []byte, []float32) ([]float32, error) {
	panic("packed MXFP4 fixture selected the FP8 expert kernel")
}

func (b *v41PackedOrdinaryBackend) SupportsV41BF16Publish(n int) bool { return n > 0 }

func (b *v41PackedOrdinaryBackend) V41BF16Publish(input compute.Tensor, site uint32) (compute.Tensor, error) {
	if site != compute.V41BF16PublishRoutedDown {
		return compute.Tensor{}, errors.New("unexpected BF16 publication site")
	}
	values := b.Backend.Read(input)
	for i := range values {
		values[i] = v41RoundBF16(values[i])
	}
	return b.Backend.Upload(compute.NewF32(compute.Default(), []int{len(values)}, values), compute.F32), nil
}

type v41PackedOrdinaryProjection struct {
	name    string
	weights []byte
	scales  []byte
}

func v41PackedOrdinaryInstall(t *testing.T, m *Model, projection string, seed byte) v41PackedOrdinaryProjection {
	t.Helper()
	const rows, packedCols, scaleCols = 32, 16, 1
	name := "layers.0.ffn.experts.0." + projection + ".weight"
	weights := make([]byte, rows*packedCols)
	for i := range weights {
		weights[i] = byte(int(seed) + i*5 + i/7)
	}
	scales := make([]byte, rows*scaleCols)
	for i := range scales {
		scales[i] = byte(125 + (int(seed)+i)%5)
	}
	if err := installV41ExpertQuantDevice(
		name, "layers.0.ffn.experts.0."+projection+".scale",
		stEntry{Dtype: "I8", Shape: []int{rows, packedCols}},
		stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}},
		weights, scales, m,
	); err != nil {
		t.Fatalf("install %s: %v", projection, err)
	}
	return v41PackedOrdinaryProjection{name: name, weights: weights, scales: scales}
}

func v41PackedOrdinaryOracle(t *testing.T, w1, w3, w2 v41PackedOrdinaryProjection, x []float32, limit, routeWeight float32) []float32 {
	t.Helper()
	const width = 32
	input := make([]float32, len(x))
	for i := range x {
		input[i] = v41RoundBF16(x[i])
	}
	gate, err := compute.GemvMXFP4E8M0(w1.name, width, width, w1.weights, w1.scales, input)
	if err != nil {
		t.Fatal(err)
	}
	up, err := compute.GemvMXFP4E8M0(w3.name, width, width, w3.weights, w3.scales, input)
	if err != nil {
		t.Fatal(err)
	}
	activation := make([]float32, width)
	for i := range activation {
		g, u := v41RoundBF16(gate[i]), v41RoundBF16(up[i])
		if limit > 0 {
			g = min(g, limit)
			u = max(-limit, min(u, limit))
		}
		activation[i] = v41RoundBF16(silu(g) * u * routeWeight)
	}
	out, err := compute.GemvMXFP4E8M0(w2.name, width, width, w2.weights, w2.scales, activation)
	if err != nil {
		t.Fatal(err)
	}
	for i := range out {
		out[i] = v41RoundBF16(out[i])
	}
	return out
}

// TestV41PackedExpertOrdinaryConsumer pins the ordinary routed-expert callback's
// packed V4.1 consumer. The reduced process-global quant profile makes this a
// wiring test, not a full-checkpoint allocation; keep it nonparallel.
// fak-test:runtime fast est=250ms lane=default
func TestV41PackedExpertOrdinaryConsumer(t *testing.T) {
	saved := v41ExpertQuantSpecs
	v41ExpertQuantSpecs = map[string]v41ExpertQuantSpec{
		"w1": {weightRows: 32, weightCols: 16, scaleRows: 32, scaleCols: 1},
		"w2": {weightRows: 32, weightCols: 16, scaleRows: 32, scaleCols: 1},
		"w3": {weightRows: 32, weightCols: 16, scaleRows: 32, scaleCols: 1},
	}
	defer func() { v41ExpertQuantSpecs = saved }()

	newModel := func() *Model {
		return &Model{Cfg: Config{
			NumLayers: 1, HiddenSize: 32, MoEIntermediateSize: 32, SwigluLimit: .75,
			DeepSeekV41: &DeepSeekV41Config{},
		}}
	}
	newSession := func(m *Model, b *v41PackedOrdinaryBackend) *Session {
		return &Session{M: m, Backend: b}
	}
	requireCallback := func(t *testing.T, s *Session) v41RoutedExpertFunc {
		t.Helper()
		callback := s.v41RoutedExpertFunc()
		if callback == nil {
			t.Fatal("ordinary routed-expert callback did not select the installed packed MXFP4 expert")
		}
		return callback
	}

	t.Run("packed w1 w3 w2 exact semantics", func(t *testing.T) {
		m := newModel()
		defer m.CloseWeights()
		w1 := v41PackedOrdinaryInstall(t, m, "w1", 1)
		w3 := v41PackedOrdinaryInstall(t, m, "w3", 7)
		w2 := v41PackedOrdinaryInstall(t, m, "w2", 13)
		b := newV41PackedOrdinaryBackend()
		s := newSession(m, b)
		defer s.Close()
		callback := requireCallback(t, s)
		x := make([]float32, 32)
		for i := range x {
			x[i] = v41RoundBF16(float32(math.Sin(float64(i)*.31)) * .5)
		}
		const routeWeight = float32(.501953125)
		want := v41PackedOrdinaryOracle(t, w1, w3, w2, x, float32(m.Cfg.SwigluLimit), routeWeight)
		host := 0
		got, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", x, routeWeight, m.Cfg, callback,
			func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				host++
				return nil, v41GateUpDeclined, nil
			},
			func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
				host++
				return nil, v41DownDeclined, nil
			},
			func() ([]float32, []float32, []float32, error) {
				host++
				return nil, nil, nil, errors.New("host triple replay")
			},
			func() ([]float32, error) {
				host++
				return nil, errors.New("host down replay")
			}, false)
		if err != nil || !reflect.DeepEqual(got, want) || b.calls != 3 || host != 0 || s.HostFallbackObserved() {
			t.Fatalf("packed ordinary result mismatch: err=%v kernels=%d host=%d fallback=%t", err, b.calls, host, s.HostFallbackObserved())
		}
	})

	t.Run("partial metadata refuses before kernels", func(t *testing.T) {
		m := newModel()
		defer m.CloseWeights()
		v41PackedOrdinaryInstall(t, m, "w1", 1)
		v41PackedOrdinaryInstall(t, m, "w3", 7)
		b := newV41PackedOrdinaryBackend()
		s := newSession(m, b)
		defer s.Close()
		host := 0
		got, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", make([]float32, 32), .5, m.Cfg, requireCallback(t, s), nil, nil,
			func() ([]float32, []float32, []float32, error) {
				host++
				return nil, nil, nil, errors.New("host replay")
			},
			func() ([]float32, error) {
				host++
				return nil, errors.New("host replay")
			}, false)
		var selected *V41ExpertOperationError
		var closed *BackendForwardOperationError
		if got != nil || !errors.Is(err, ErrV4ExpertQuantMetadata) || !errors.As(err, &selected) || !errors.As(err, &closed) ||
			!s.BackendSessionClosed() || b.calls != 0 || host != 0 {
			t.Fatalf("partial packed metadata: result=%v err=%v closed=%t kernels=%d host=%d", got, err, s.BackendSessionClosed(), b.calls, host)
		}
	})

	t.Run("kernel fault closes without host replay", func(t *testing.T) {
		m := newModel()
		defer m.CloseWeights()
		v41PackedOrdinaryInstall(t, m, "w1", 1)
		v41PackedOrdinaryInstall(t, m, "w3", 7)
		v41PackedOrdinaryInstall(t, m, "w2", 13)
		sentinel := &compute.BackendError{Backend: "ordinary-packed-test", Site: "GemvExpertMXFP4E8M0", Err: errors.New("injected packed kernel fault")}
		b := newV41PackedOrdinaryBackend()
		b.failAt, b.failure = 1, sentinel
		s := newSession(m, b)
		defer s.Close()
		host := 0
		got, err := m.v41FullRoutedExpertWithDevice(0, "ffn.experts.0", make([]float32, 32), .5, m.Cfg, requireCallback(t, s), nil, nil,
			func() ([]float32, []float32, []float32, error) {
				host++
				return nil, nil, nil, errors.New("host replay")
			},
			func() ([]float32, error) {
				host++
				return nil, errors.New("host replay")
			}, false)
		var closed *BackendForwardOperationError
		if got != nil || !errors.Is(err, sentinel) || !errors.As(err, &closed) || !s.BackendSessionClosed() || b.calls != 1 || host != 0 {
			t.Fatalf("packed kernel fault: result=%v err=%v closed=%t kernels=%d host=%d", got, err, s.BackendSessionClosed(), b.calls, host)
		}
	})
}
