package model

import (
	"context"
	"encoding/binary"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// glmOverlapTestBackend runs actual cpu-ref primitives while modeling the device
// ownership/capability contract. It supplies no device/performance evidence.
type glmOverlapTestBackend struct {
	compute.Backend
	mu            sync.Mutex
	allow         bool
	calls, failAt int
	pending       bool
}

func (b *glmOverlapTestBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}
func (b *glmOverlapTestBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return b.allow && dt == compute.Q4_K
}
func (b *glmOverlapTestBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *glmOverlapTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.pending {
		panic("overlapping backend owner calls")
	}
	b.calls++
	if b.failAt > 0 && b.calls == b.failAt {
		panic("shared projection failure")
	}
	b.pending = true
	return b.Backend.MatMul(w, x)
}
func (b *glmOverlapTestBackend) Read(x compute.Tensor) []float32 {
	y := b.Backend.Read(x)
	b.pending = false
	return y
}
func (b *glmOverlapTestBackend) Free(x compute.Tensor) {
	if b.pending {
		panic("Free before Read fence")
	}
	b.Backend.Free(x)
}

// fak-test:runtime fast est=1s
func TestGLMSharedOverlapActualApply(t *testing.T) {
	be, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref unavailable")
	}
	cfg := Config{ModelType: "glm_moe_dsa", HiddenSize: 256, HeadDim: 256, IntermediateSize: 256, MoEIntermediateSize: 256, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, NumExperts: 2, NumExpertsPerTok: 2, NSharedExperts: 1, NormTopKProb: true}
	m := &Model{Cfg: cfg, q4kw: make(map[string]*q4kTensor), kqw: make(map[string]*kQuantTensor)}
	raw := func(rows, block int, code byte) []byte {
		b := make([]byte, rows*block)
		for r := 0; r < rows; r++ {
			p := b[r*block : (r+1)*block]
			binary.LittleEndian.PutUint16(p, 0x1400)
			for j := 4; j < 16; j++ {
				p[j] = 1
			}
			start := 16
			if block == q5kBlockBytes {
				start = 48
			}
			for j := start; j < block; j++ {
				p[j] = code
			}
		}
		return b
	}
	routerRaw := raw(2, q4kBlockBytes, 0x12)
	for i := q4kBlockBytes + 16; i < len(routerRaw); i++ {
		routerRaw[i] = 0x23
	}
	m.q4kw[routerName(0)] = quantizeQ4KFromRaw(routerRaw, 2, 256)
	for e := 0; e < 2; e++ {
		for _, p := range []string{"gate_proj.weight", "up_proj.weight"} {
			m.q4kw[expertName(0, e, p)] = quantizeQ4KFromRaw(raw(256, q4kBlockBytes, byte(0x11+e)), 256, 256)
		}
		m.kqw[expertName(0, e, "down_proj.weight")] = quantizeKQuantFromRaw(raw(256, q5kBlockBytes, byte(0x12+e)), 256, 256, kindQ5K)
	}
	for _, p := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		m.q4kw[layerName(0, "mlp.shared_experts."+p)] = quantizeQ4KFromRaw(raw(256, q4kBlockBytes, 0x13), 256, 256)
	}
	b := &glmOverlapTestBackend{Backend: be, allow: true}
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatalf("live backend constructor refused: %v", err)
	}
	s.CPUOffloadExperts = true
	defer s.Close()
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%7+1) / 32
	}
	apply := func() []float32 {
		scope := s.beginBackendExecution()
		defer scope.finish()
		kind := m.ffnForLayer(0)
		if _, ok := kind.(glmMoeFFN); !ok {
			t.Fatal("real architecture dispatch did not select GLM MoE")
		}
		return kind.apply(m, 0, x, s.glmDsaMatKernel())
	}
	baseline := apply()
	if s.glmSharedExpertOverlapCount != 0 {
		t.Fatal("default activated overlap")
	}
	budget, ok := glmOverlapPayloadBound(2, 256, 256, 256)
	if !ok {
		t.Fatal("valid bound declined")
	}
	s.GLMSharedExpertOverlapWorkspaceBytes = budget - 1
	if got := apply(); !reflect.DeepEqual(got, baseline) || s.glmSharedExpertOverlapCount != 0 {
		t.Fatal("over-budget changed serial path")
	}
	s.GLMSharedExpertOverlapWorkspaceBytes = budget
	got := apply()
	if !reflect.DeepEqual(got, baseline) || s.glmSharedExpertOverlapCount != 1 {
		t.Fatal("actual constructor/apply overlap absent or differs")
	}
	nonzero := false
	for _, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("nonfinite result")
		}
		nonzero = nonzero || v != 0
	}
	if !nonzero {
		t.Fatal("vacuous zero fixture")
	}
	b.allow = false
	if got := apply(); !reflect.DeepEqual(got, baseline) || s.glmSharedExpertOverlapCount != 1 {
		t.Fatal("unsupported device dtype activated overlap")
	}
	b.allow = true
	reject := func(name string, change func() func()) {
		t.Run(name, func(t *testing.T) {
			restore := change()
			defer restore()
			scope := s.beginBackendExecution()
			defer scope.finish()
			if _, ok := m.overlapGLMSharedExpert(0, x, []routePick{{0, .25}, {1, .75}}, s.glmDsaMatKernel()); ok {
				t.Fatal("negative admission started overlap")
			}
			if s.glmSharedExpertOverlapCount != 1 {
				t.Fatal("negative published success")
			}
		})
	}
	reject("zero-budget", func() func() {
		old := s.GLMSharedExpertOverlapWorkspaceBytes
		s.GLMSharedExpertOverlapWorkspaceBytes = 0
		return func() { s.GLMSharedExpertOverlapWorkspaceBytes = old }
	})
	reject("missing-shared", func() func() {
		n := layerName(0, "mlp.shared_experts.gate_proj.weight")
		old := m.q4kw[n]
		delete(m.q4kw, n)
		return func() { m.q4kw[n] = old }
	})
	reject("raw-alias", func() func() {
		n := layerName(0, "mlp.shared_experts.gate_proj.weight")
		old := m.q4kw[n]
		q := *old
		q.raw = m.q4kw[expertName(0, 0, "gate_proj.weight")].raw
		m.q4kw[n] = &q
		return func() { m.q4kw[n] = old }
	})
	reject("malformed-host-shape", func() func() {
		n := expertName(0, 0, "gate_proj.weight")
		old := m.q4kw[n]
		q := *old
		q.in = 128
		m.q4kw[n] = &q
		return func() { m.q4kw[n] = old }
	})
	reject("all-device", func() func() {
		old := s.CPUOffloadExperts
		s.CPUOffloadExperts = false
		return func() { s.CPUOffloadExperts = old }
	})

	for _, picks := range [][]routePick{{{0, .25}, {1, .75}}, {{1, .75}, {0, .25}}} {
		func() {
			scope := s.beginBackendExecution()
			defer scope.finish()
			mat := s.glmDsaMatKernel()
			want := make([]float32, 256)
			if !m.hostBatchedGLMExperts(0, x, want, picks) {
				t.Fatal("serial batch refused")
			}
			shared := glmSharedExperts(m, 0, x, mat)
			for i := range want {
				want[i] += shared[i]
			}
			got, ok := m.overlapGLMSharedExpert(0, x, picks, mat)
			if !ok || !reflect.DeepEqual(got, want) {
				t.Fatal("current nonuniform route order not preserved")
			}
		}()
	}

	// Destructive failure is last: later admissions never reuse a possibly failed Session.
	before := b.calls
	successfulBeforeFailure := s.glmSharedExpertOverlapCount
	b.failAt = before + 2 // Router succeeds; first shared projection fails after launch.
	func() {
		defer func() {
			if recover() != "shared projection failure" {
				t.Error("owner failure changed")
			}
		}()
		apply()
	}()
	if b.calls != before+2 || s.glmSharedExpertOverlapCount != successfulBeforeFailure {
		t.Fatal("failed overlap retried or published success")
	}
	b.failAt = 0

}

// fak-test:runtime fast est=100ms
func TestGLMSharedOverlapJoinAndBounds(t *testing.T) {
	t.Run("owner-panic-joins", func(t *testing.T) {
		started, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
		var once sync.Once
		releaseCPU := func() { once.Do(func() { close(release) }) }
		defer releaseCPU()
		primary := &struct{}{}
		go func() {
			defer close(finished)
			defer func() {
				if recover() != primary {
					t.Error("primary panic replaced")
				}
			}()
			glmOverlapJoin(func() { close(started); <-release }, func() []float32 { <-started; panic(primary) })
		}()
		glmOverlapAwait(t, started)
		select {
		case <-finished:
			t.Fatal("escaped before CPU joined")
		default:
		}
		releaseCPU()
		glmOverlapAwait(t, finished)
	})
	for _, cpuFirst := range []bool{false, true} {
		cpuDone, ownerReady := make(chan struct{}), make(chan struct{})
		got := glmOverlapJoin(func() {
			if !cpuFirst {
				<-ownerReady
			}
			close(cpuDone)
		}, func() []float32 {
			close(ownerReady)
			if cpuFirst {
				<-cpuDone
			}
			return []float32{3, 1}
		})
		if !reflect.DeepEqual(got, []float32{3, 1}) {
			t.Fatal("completion order changed result")
		}
	}
	if _, ok := glmOverlapPayloadBound(int(^uint(0)>>1), 256, 256, 256); ok {
		t.Fatal("overflow admitted")
	}
	a := make([]byte, 32)
	if !glmOverlapRawAliases(a[:16], a[8:]) || glmOverlapRawAliases(a[:8], a[8:]) {
		t.Fatal("backing alias check")
	}
}

func glmOverlapAwait(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("owned branch did not settle")
	}
}

type glmOverlapTestCloser struct {
	closed  *atomic.Bool
	cpuDone *atomic.Bool
}

func (c glmOverlapTestCloser) Close() error {
	if !c.cpuDone.Load() {
		panic("weight backing retired before CPU joined")
	}
	c.closed.Store(true)
	return nil
}

// fak-test:runtime fast est=100ms
func TestGLMSharedOverlapCancellationRetirement(t *testing.T) {
	be, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref unavailable")
	}
	m := &Model{Cfg: Config{HiddenSize: 256, NumLayers: 1, NumHeads: 1, NumKVHeads: 1}}
	var closed, cpuDone atomic.Bool
	m.SetWeightCloser(glmOverlapTestCloser{closed: &closed, cpuDone: &cpuDone})
	s := m.NewSession()
	s.Backend = &glmOverlapTestBackend{Backend: be, allow: true}
	scope := s.beginBackendExecution()
	defer func() {
		if scope.live {
			scope.finish()
		}
	}()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseCPU := func() { releaseOnce.Do(func() { close(release) }) }
	result := glmOverlapJoin(func() { close(started); <-release; cpuDone.Store(true) }, func() []float32 {
		defer releaseCPU()
		glmOverlapAwait(t, started)
		cancel()
		if ctx.Err() != context.Canceled {
			t.Fatal("cancellation not delivered")
		}
		if m.CloseWeights() == nil {
			t.Fatal("active model pin not retained")
		}
		s.closeBackendOwned()
		if closed.Load() {
			t.Fatal("inner teardown released CPU backing")
		}
		releaseCPU()
		return nil
	})
	if result != nil || !cpuDone.Load() || closed.Load() {
		t.Fatal("cancel return escaped join or scope retired early")
	}
	scope.finish()
	if !closed.Load() {
		t.Fatal("model backing not retired after owned join/scope")
	}
}
