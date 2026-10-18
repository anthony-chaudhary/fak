package model

import (
	"errors"
	"math"
	"reflect"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder exercises the existing ordinary control. A future composer can
// reuse its source-buffer and event observations without enabling a new path here.
// CPU arithmetic and API byte counts are not device-performance evidence.
type v41HeadScaleLoanRecorder struct {
	*v41DenseTestBackend
	executionOwner                     sync.Mutex
	retainedScaleCalls                 int
	headWeight, headOutput, scaleInput compute.Buffer
	headLive                           bool
	scaleLive                          map[compute.Buffer]bool
	events                             []string
	uploadBytes, scaleCalls            int
	rawNonfinite                       bool
	scaleFault                         error
	scalePanic                         string
	scaleAlias                         bool
	cleanupPanic                       bool
	cleanupExit                        string
	headCleanupPanic                   bool
}

// This float-only fixture uses synchronous CPU arithmetic underneath the dense
// recorder. Rehome fresh outputs before that recorder records buffer identities;
// every projection input/output then belongs to the same outer execution owner.
// Copy before retiring the temporary CPU tensor, never create an owning alias.
type v41HeadScaleLoanCPU struct {
	compute.Backend
	identity compute.Backend
}

// Only the F32 resident-weight path is implemented by this fixture adapter.
func (b *v41HeadScaleLoanCPU) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32
}

func (b *v41HeadScaleLoanCPU) owned(t compute.Tensor) compute.Tensor {
	defer b.Backend.Free(t)
	if t.Dtype != compute.F32 || t.Layout != compute.RowMajor || t.Quant != nil {
		panic("head-scale fixture requires plain F32")
	}
	return compute.NewF32(b.identity, append([]int(nil), t.Shape...), append([]float32(nil), b.Backend.Read(t)...))
}
func (b *v41HeadScaleLoanCPU) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.owned(b.Backend.Upload(x, dt))
}
func (b *v41HeadScaleLoanCPU) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.owned(b.Backend.MatMul(w, x))
}
func (b *v41HeadScaleLoanCPU) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.owned(b.Backend.BatchedMatMul(w, x, rows))
}

func newV41HeadScaleLoanRecorder() *v41HeadScaleLoanRecorder {
	b := &v41HeadScaleLoanRecorder{
		v41DenseTestBackend: newV41DenseTestBackend(),
		scaleLive:           map[compute.Buffer]bool{},
	}
	b.v41DenseTestBackend.Backend = &v41HeadScaleLoanCPU{Backend: compute.Default(), identity: b}
	return b
}

// A retained loan requires the same explicit execution domain as native backends.
func (b *v41HeadScaleLoanRecorder) ForwardExecutionOwner() *sync.Mutex {
	return &b.executionOwner
}

func (b *v41HeadScaleLoanRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	y := b.v41DenseTestBackend.MatMul(w, x)
	if w.Buf() == b.headWeight {
		b.headOutput, b.headLive = y.Buf(), true
		b.events = append(b.events, "head-project")
	}
	return y
}

func (b *v41HeadScaleLoanRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 logit scale input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("scale input lost F32 activation class")
	}
	values := append([]float32(nil), compute.Default().Read(x)...)
	b.uploadBytes += 4 * len(values)
	b.events = append(b.events, "scale-upload")
	y := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.scaleLive[y.Buf()] = true
	return y
}

func (b *v41HeadScaleLoanRecorder) V41LogitScaleAvailable() bool { return true }

func (b *v41HeadScaleLoanRecorder) V41LogitScale(x compute.Tensor, factor float32) (compute.Tensor, error) {
	if x.Buf() == b.headOutput && !b.headLive {
		panic("borrowed head was retired before scale")
	}
	if x.Buf() == b.headOutput && b.headLive {
		b.retainedScaleCalls++
	}
	b.scaleCalls++
	b.scaleInput = x.Buf()
	b.events = append(b.events, "scale-multiply")
	switch b.scalePanic {
	case "nil":
		panic(nil)
	case "goexit":
		runtime.Goexit()
	}
	if b.scaleAlias {
		return x, nil
	}
	values := append([]float32(nil), compute.Default().Read(x)...)
	for i := range values {
		values[i] *= factor
	}
	y := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.scaleLive[y.Buf()] = true
	return y, b.scaleFault
}

func (b *v41HeadScaleLoanRecorder) Read(x compute.Tensor) []float32 {
	if b.scaleLive[x.Buf()] {
		b.events = append(b.events, "scale-read")
		return compute.Default().Read(x)
	}
	values := b.v41DenseTestBackend.Read(x)
	if x.Buf() == b.headOutput {
		b.events = append(b.events, "head-read")
		if b.rawNonfinite && len(values) != 0 {
			values = append([]float32(nil), values...)
			values[0] = float32(math.NaN())
		}
	}
	return values
}

func (b *v41HeadScaleLoanRecorder) Free(x compute.Tensor) {
	if x.Buf() == b.headOutput {
		b.headLive = false
		b.events = append(b.events, "head-free")
	}
	if b.scaleLive[x.Buf()] {
		delete(b.scaleLive, x.Buf())
		compute.Default().Free(x)
		switch b.cleanupExit {
		case "nil":
			b.cleanupExit = ""
			panic(nil)
		case "goexit":
			b.cleanupExit = ""
			runtime.Goexit()
		}
		if b.cleanupPanic {
			b.cleanupPanic = false
			panic("scale cleanup marker")
		}
		return
	}
	b.v41DenseTestBackend.Free(x)
	if x.Buf() == b.headOutput && b.headCleanupPanic {
		b.headCleanupPanic = false
		panic("secondary head cleanup marker")
	}
}

// This fixes the ordinary control and raw-finite fence before any production
// head-to-scale loan is added. Replacing the callback intentionally removes
// canonical composition authority, even if its ordinary implementation is equal.
// fak-test:runtime medium est=1s lane=default
func TestV41HeadScaleLoanOrdinaryControlAndRawFence(t *testing.T) {
	t.Setenv("FAK_V41_HEAD_SCALE_LOAN", "1")
	for _, nonfinite := range []bool{false, true} {
		name := "ordinary-control"
		if nonfinite {
			name = "raw-failure-before-scale"
		}
		t.Run(name, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 1)
			m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = .5, 0, nil
			b := newV41HeadScaleLoanRecorder()
			s := v41DenseTestSession(t, m, b)
			b.headWeight = s.weightHAL(m.headName()).Buf()
			st := s.v41State()
			st.setDenseProjection(st.denseProjection())
			if len(s.Prefill([]int{1})) == 0 || len(s.Step(2)) == 0 {
				t.Fatal("ordinary warm decode control unavailable")
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.events, b.uploadBytes, b.scaleCalls = nil, 0, 0
			b.rawNonfinite = nonfinite
			marker := errors.New("scale must not override raw projection failure")
			if nonfinite {
				b.scaleFault = marker
			}
			var result []float32
			var recovered any
			returned := false
			func() {
				defer func() { recovered = recover() }()
				result = s.Step(3)
				returned = true
			}()
			if nonfinite {
				cause, ok := recovered.(error)
				if returned || result != nil || !ok || !errors.Is(cause, errV41ProjectionResult) || errors.Is(cause, marker) ||
					b.scaleCalls != 0 || b.uploadBytes != 0 || !s.halClosed || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
					t.Fatalf("raw failure lost precedence/rollback: returned=%v panic=%v scale=%d upload=%d closed=%v", returned, recovered, b.scaleCalls, b.uploadBytes, s.halClosed)
				}
			} else {
				if !returned || recovered != nil || len(result) != m.Cfg.VocabSize || b.scaleCalls != 1 ||
					b.uploadBytes != 4*m.Cfg.VocabSize || b.scaleInput == b.headOutput {
					t.Fatalf("ordinary transfer control changed: returned=%v panic=%v width=%d scale=%d upload=%d", returned, recovered, len(result), b.scaleCalls, b.uploadBytes)
				}
				want := []string{"head-project", "head-read", "head-free", "scale-upload", "scale-multiply", "scale-read"}
				if !reflect.DeepEqual(b.events, want) {
					t.Fatalf("ordinary operation order=%v want=%v", b.events, want)
				}
			}
			if b.headLive || len(b.scaleLive) != 0 {
				t.Fatal("head or scale transient survived the completed step")
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41HeadScaleLoanWarmStepParity(t *testing.T) {
	probe := newV41HeadScaleLoanRecorder()
	if !compute.BackendSupportsDeviceWeightDtype(probe, compute.F32) ||
		compute.BackendSupportsDeviceWeightDtype(probe, compute.BF16) ||
		compute.BackendSupportsDeviceWeightDtype(probe, compute.Q4_K) {
		t.Fatal("fixture must advertise exactly its F32 resident-weight capability")
	}
	t.Setenv("FAK_V41_HEAD_SCALE_LOAN", "1")
	var control []float32
	for _, ordinary := range []bool{true, false} {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = .5, 0, nil
		b := newV41HeadScaleLoanRecorder()
		s := v41DenseTestSession(t, m, b)
		b.headWeight = s.weightHAL(m.headName()).Buf()
		if ordinary {
			s.v41State().setLogitScale(s.v41State().logitScale())
		}
		s.Prefill([]int{1})
		s.Step(2)
		if !ordinary && b.retainedScaleCalls == 0 {
			t.Fatal("fixture did not enter retained head-scale operation")
		}
		b.events, b.uploadBytes, b.scaleCalls = nil, 0, 0
		result := s.Step(3)
		if ordinary {
			control = append([]float32(nil), result...)
			continue
		}
		if !reflect.DeepEqual(result, control) || b.scaleCalls != 1 || b.uploadBytes != 0 || b.scaleInput != b.headOutput {
			t.Fatalf("loan parity/transfer: calls=%d upload=%d sameInput=%v", b.scaleCalls, b.uploadBytes, b.scaleInput == b.headOutput)
		}
		want := []string{"head-project", "head-read", "scale-multiply", "scale-read", "head-free"}
		if !reflect.DeepEqual(b.events, want) || b.headLive || len(b.scaleLive) != 0 {
			t.Fatalf("loan order/lifetime: %v", b.events)
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41HeadScaleLoanRawFailurePrecedesSelectedScale(t *testing.T) {
	t.Setenv("FAK_V41_HEAD_SCALE_LOAN", "1")
	m := v41IncrementalPlainModel(t, 1)
	m.Cfg.LogitScale = .5
	b := newV41HeadScaleLoanRecorder()
	s := v41DenseTestSession(t, m, b)
	b.headWeight = s.weightHAL(m.headName()).Buf()
	s.Prefill([]int{1})
	s.Step(2)
	if b.retainedScaleCalls == 0 {
		t.Fatal("fixture did not enter retained head-scale operation")
	}
	before := captureV41ForwardSnapshot(s.v41Forward)
	b.events, b.uploadBytes, b.scaleCalls = nil, 0, 0
	b.rawNonfinite = true
	marker := errors.New("scale must remain unselected")
	b.scaleFault = marker
	var recovered any
	func() { defer func() { recovered = recover() }(); s.Step(3) }()
	err, ok := recovered.(error)
	if !ok || !errors.Is(err, errV41ProjectionResult) || errors.Is(err, marker) || b.scaleCalls != 0 || b.uploadBytes != 0 || !s.halClosed || b.headLive || len(b.scaleLive) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
		t.Fatalf("raw fence/rollback lost: %v calls=%d uploads=%d", recovered, b.scaleCalls, b.uploadBytes)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41HeadScaleLoanSelectedFailureRetiresAndRollsBack(t *testing.T) {
	t.Setenv("FAK_V41_HEAD_SCALE_LOAN", "1")
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"error", "alias", "panic-nil", "goexit", "error-and-cleanup", "cleanup-nil", "cleanup-goexit", "dual-cleanup"} {
		t.Run(mode, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 1)
			m.Cfg.LogitScale = .5
			b := newV41HeadScaleLoanRecorder()
			s := v41DenseTestSession(t, m, b)
			b.headWeight = s.weightHAL(m.headName()).Buf()
			s.Prefill([]int{1})
			s.Step(2)
			if b.retainedScaleCalls == 0 {
				t.Fatal("fixture did not enter retained head-scale operation")
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.events, b.uploadBytes, b.scaleCalls = nil, 0, 0
			marker := errors.New("selected scale fault")
			switch mode {
			case "error":
				b.scaleFault = marker
			case "alias":
				b.scaleAlias = true
			case "panic-nil":
				b.scalePanic = "nil"
			case "goexit":
				b.scalePanic = "goexit"
			case "dual-cleanup":
				b.cleanupPanic = true
				b.headCleanupPanic = true
			case "cleanup-nil":
				b.cleanupExit = "nil"
			case "cleanup-goexit":
				b.cleanupExit = "goexit"
			case "error-and-cleanup":
				b.scaleFault = marker
				b.cleanupPanic = true
			}
			var recovered any
			returned := false
			done := make(chan struct{})
			resumed := false
			go func() {
				defer close(done)
				func() { defer func() { recovered = recover() }(); s.Step(3); returned = true }()
				resumed = true
			}()
			<-done
			if resumed != (mode != "goexit" && mode != "cleanup-goexit") {
				t.Fatalf("control flow changed: mode=%s resumed=%v", mode, resumed)
			}
			if returned || b.scaleCalls != 1 || b.uploadBytes != 0 || !s.halClosed || b.headLive || len(b.scaleLive) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatalf("selected failure escaped retirement/rollback: mode=%s returned=%v panic=%v", mode, returned, recovered)
			}
			if mode == "dual-cleanup" && recovered != "scale cleanup marker" {
				t.Fatalf("earlier cleanup fault replaced: %v", recovered)
			}
			if mode == "error" || mode == "error-and-cleanup" {
				err, ok := recovered.(error)
				if !ok || !errors.Is(err, marker) {
					t.Fatalf("primary scale fault replaced: %v", recovered)
				}
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41HeadScaleLoanCustomClockKeepsOrdinaryBoundary(t *testing.T) {
	t.Setenv("FAK_V41_HEAD_SCALE_LOAN", "1")
	m := v41IncrementalPlainModel(t, 1)
	m.Cfg.LogitScale = .5
	b := newV41HeadScaleLoanRecorder()
	s := v41DenseTestSession(t, m, b)
	b.headWeight = s.weightHAL(m.headName()).Buf()
	s.Prefill([]int{1})
	s.Step(2)
	if b.retainedScaleCalls == 0 {
		t.Fatal("fixture did not enter retained head-scale operation")
	}
	ticks := int64(0)
	m.v41SetExpertTimeClock(func() int64 { ticks++; return ticks })
	defer m.v41SetExpertTimeClock(nil)
	b.events, b.uploadBytes, b.scaleCalls = nil, 0, 0
	s.Step(3)
	if ticks == 0 || b.scaleCalls != 1 || b.uploadBytes != 4*m.Cfg.VocabSize || b.scaleInput == b.headOutput {
		t.Fatal("custom clock did not decline composition")
	}
	want := []string{"head-project", "head-read", "head-free", "scale-upload", "scale-multiply", "scale-read"}
	if !reflect.DeepEqual(b.events, want) {
		t.Fatalf("custom clock order=%v", b.events)
	}
}
