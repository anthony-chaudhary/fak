package model

import (
	"context"
	"errors"
	"math"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// Software recorder only. Every operation uses cpu-ref arithmetic; its device
// capability exercises the actual Session adapter and ownership contract.
type v41FinalNormRecorder struct {
	*v41BF16PublishRecorder
	mu                                                    sync.Mutex
	gain                                                  compute.Buffer
	grouped                                               int
	finalDispatches                                       int
	fault                                                 string
	frees                                                 map[compute.Buffer]int
	lastRaw                                               compute.Buffer
	activationBytes, payloadBytes, statusBytes, hostReads int
	probePanic                                            bool
}

func newV41FinalNormRecorder() *v41FinalNormRecorder {
	b := &v41FinalNormRecorder{v41BF16PublishRecorder: newV41DeferredPublishRecorder(), frees: map[compute.Buffer]int{}}
	b.self = b
	return b
}
func (b *v41FinalNormRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41FinalNormRecorder) SupportsV41DeferredStatus(n int) bool {
	if b.probePanic {
		panic(nil)
	}
	return b.v41DeferredLedger.SupportsV41DeferredStatus(n)
}
func (b *v41FinalNormRecorder) Read(x compute.Tensor) []float32 {
	out := b.v41BF16PublishRecorder.Read(x)
	if x.Buf() == b.lastRaw {
		b.hostReads++
		b.payloadBytes += 4 * len(out)
	}
	return out
}

func (b *v41FinalNormRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		return b.v41BF16PublishRecorder.UploadClass(x, dt, class, site)
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), compute.Default().Read(x)...))
	if class == compute.MemoryActivation {
		b.activationBytes += 4 * len(compute.Default().Read(x))
	}
	if site == "hal-weight model.norm.weight" {
		b.gain = out.Buf()
	}
	if site == "V4.1 BF16 publication input" {
		b.frees[out.Buf()] = 0
	}
	return out
}
func (b *v41FinalNormRecorder) RMSNorm(x, w compute.Tensor, eps float32) compute.Tensor {
	if w.Buf() == b.gain {
		switch b.fault {
		case "rms":
			panic(&compute.BackendError{Backend: "cpu-recorder", Class: compute.VulkanClassExecutionFailed, Err: errors.New("final norm injected RMSNorm")})
		case "nil":
			panic(nil)
		case "goexit":
			runtime.Goexit()
		}
	}
	raw := compute.Default().RMSNorm(x, w, eps)
	defer compute.Default().Free(raw)
	values := append([]float32(nil), compute.Default().Read(raw)...)
	if w.Buf() == b.gain && (b.fault == "raw" || b.fault == "ordered") {
		values[0] = float32(math.NaN())
	}
	out := compute.NewF32(b, append([]int(nil), raw.Shape...), values)
	if w.Buf() == b.gain {
		b.frees[out.Buf()] = 0
		b.lastRaw = out.Buf()
	}
	return out
}
func (b *v41FinalNormRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	out, status, err := b.v41BF16PublishRecorder.V41BF16PublishDeferred(x, site)
	b.frees[out.Buf()], b.frees[status.Record.Buf()] = 0, 0
	if site == compute.V41BF16PublishFinalNormInput || site == compute.V41BF16PublishFinalNormOutput {
		b.finalDispatches++
		if (site == compute.V41BF16PublishFinalNormInput && (b.fault == "input" || b.fault == "ordered")) || (site == compute.V41BF16PublishFinalNormOutput && (b.fault == "output" || b.fault == "ordered")) {
			b.deferredPending[status.Record.Buf()] = &compute.V41BF16CheckedArithmeticError{Site: site, Index: 0}
		}
	}
	return out, status, err
}
func (b *v41FinalNormRecorder) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.hostReads++
	for _, st := range statuses {
		b.statusBytes += 16 * compute.V41DeferredRecords(st)
	}
	for _, row := range rows {
		n := 1
		for _, size := range row.Shape {
			n *= size
		}
		b.payloadBytes += 4 * n
	}
	grouped := len(statuses) == 2 && statuses[0].Site == compute.V41BF16PublishFinalNormInput && statuses[1].Site == compute.V41BF16PublishFinalNormOutput
	if grouped {
		b.grouped++
	}
	if grouped && b.fault == "read" {
		b.deferredReadErr = errors.New("final norm injected grouped read")
	}
	verdicts, values, err := b.v41BF16PublishRecorder.V41ReadDeferred(statuses, rows)
	b.deferredReadErr = nil
	return verdicts, values, err
}
func (b *v41FinalNormRecorder) Free(x compute.Tensor) {
	if _, ok := b.frees[x.Buf()]; ok {
		b.frees[x.Buf()]++
	}
	delete(b.deferredPending, x.Buf())
	b.v41BF16PublishRecorder.Free(x)
	if x.Buf() == b.lastRaw {
		if b.fault == "cleanup-nil" {
			panic(nil)
		}
		if b.fault == "cleanup-goexit" {
			runtime.Goexit()
		}
	}
}
func v41FinalNormFixture(t *testing.T) (*Model, *Session, *v41FinalNormRecorder) {
	t.Helper()
	m := v41FullStepPatchedNorms(t)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	b := newV41FinalNormRecorder()
	s := v41DenseTestSession(t, m, b)
	return m, s, b
}
func v41FinalNormPrepared(t *testing.T, m *Model, s *Session, b *v41FinalNormRecorder) []float32 {
	t.Helper()
	scope := s.beginBackendExecution()
	complete := false
	defer scope.finishCompleted(&complete)
	st := s.v41State()
	if st.headNorm == nil || st.headNorm.owner != s {
		t.Fatal("canonical owned binding absent")
	}
	_ = s.weightHAL("model.norm.weight")
	if b.gain == nil {
		t.Fatal("actual learned gain upload absent")
	}
	row := make([]float32, m.Cfg.HiddenSize)
	for i := range row {
		row[i] = float32(i%11-5) / 19
	}
	complete = true
	return row
}
func v41FinalNormAttempt(m *Model, s *Session, row []float32) ([]float32, bool, error) {
	scope := s.beginBackendExecution()
	complete := false
	defer scope.finishCompleted(&complete)
	out, selected, err := s.v41Forward.headNorm.try(m, s.v41Forward, row)
	complete = true
	return out, selected, err
}

// fak-test:runtime medium est=1s lane=default
func TestV41FinalNormCompositionSession(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "")
	withV41ActivationTraceEnv(t, "")
	m, s, b := v41FinalNormFixture(t)
	controlB := newV41FinalNormRecorder()
	control := v41DenseTestSession(t, m, controlB)
	st := control.v41State()
	st.setFinalNorm(st.finalNorm()) // same callback, explicitly no composition authority
	v41DeferredBitsEqual(t, "cold prefill", s.Prefill([]int{1}), control.Prefill([]int{1}))
	if b.grouped != 0 || controlB.grouped != 0 {
		t.Fatal("uncached gain selected composition")
	}
	v41DeferredBitsEqual(t, "step", s.Step(2), control.Step(2))
	if b.grouped != 1 || controlB.grouped != 0 {
		t.Fatalf("actual Step grouped reads=%d control=%d", b.grouped, controlB.grouped)
	}
	if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(control.v41Forward)) {
		t.Fatal("continuation mismatch")
	}
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(snap.Close)
	restored := v41DenseTestSession(t, m, b)
	if err := snap.Restore(restored); err != nil {
		t.Fatal(err)
	}
	old := s.v41Forward.headNorm
	fresh := restored.v41State().headNorm
	if fresh == old || fresh.owner != restored {
		t.Fatal("snapshot copied callback ownership")
	}
	scope := restored.beginBackendExecution()
	_, selected, err := old.try(m, restored.v41Forward, make([]float32, m.Cfg.HiddenSize))
	scope.finish()
	if selected || err != nil {
		t.Fatal("old session product borrowed restored owner")
	}
	v41DeferredBitsEqual(t, "restore first step", restored.Step(3), control.Step(3))
	before := b.grouped
	v41DeferredBitsEqual(t, "restore warm step", restored.Step(4), control.Step(4))
	if b.grouped != before+1 {
		t.Fatal("restored actual Step did not rebind composition")
	}
}

// Logical recorder payloads count API transfers, not physical device timing.
// fak-test:runtime medium est=1s lane=default
func TestV41FinalNormCompositionPayloadAndRollback(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "")
	withV41ActivationTraceEnv(t, "")
	t.Run("payload", func(t *testing.T) {
		m, s, b := v41FinalNormFixture(t)
		row := v41FinalNormPrepared(t, m, s, b)
		cb := newV41FinalNormRecorder()
		control := v41DenseTestSession(t, m, cb)
		_ = v41FinalNormPrepared(t, m, control, cb)
		var want []float32
		func() {
			scope := control.beginBackendExecution()
			complete := false
			defer scope.finishCompleted(&complete)
			st := control.v41State()
			var err error
			want, err = m.v41PrepareHeadInput(row, st.finalNorm(), st.bf16Boundary())
			if err != nil {
				t.Fatal(err)
			}
			complete = true
		}()
		got, selected, err := v41FinalNormAttempt(m, s, row)
		if !selected || err != nil {
			t.Fatalf("selected=%v err=%v", selected, err)
		}
		v41DeferredBitsEqual(t, "ordinary callback sequence", got, want)
		H := len(row)
		if cb.activationBytes != 12*H || cb.payloadBytes != 12*H || cb.statusBytes != 32 || cb.hostReads != 3 {
			t.Fatalf("ordinary API payload upload/read/status=%d/%d/%d boundaries=%d", cb.activationBytes, cb.payloadBytes, cb.statusBytes, cb.hostReads)
		}
		if b.activationBytes != 4*H || b.payloadBytes != 8*H || b.statusBytes != 32 || b.hostReads != 1 {
			t.Fatalf("composed API payload upload/read/status=%d/%d/%d boundaries=%d", b.activationBytes, b.payloadBytes, b.statusBytes, b.hostReads)
		}
	})
	t.Run("actual Step rollback", func(t *testing.T) {
		_, s, b := v41FinalNormFixture(t)
		if len(s.Prefill([]int{1})) == 0 || len(s.Step(2)) == 0 || b.grouped != 1 {
			t.Fatal("actual warm Step route absent")
		}
		before := captureV41ForwardSnapshot(s.v41Forward)
		b.fault = "read"
		var output []float32
		var failure any
		returned := false
		func() { defer func() { failure = recover() }(); output = s.Step(3); returned = true }()
		if returned || output != nil || failure == nil || !s.halClosed || s.halFailure == nil || b.grouped != 2 {
			t.Fatalf("Step failure returned=%v output=%v panic=%v closed=%v grouped=%d", returned, output, failure, s.halClosed, b.grouped)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatal("failed head changed committed continuation")
		}
		count, groups := b.finalDispatches, b.grouped
		b.fault = ""
		returned = false
		failure = nil
		func() { defer func() { failure = recover() }(); _ = s.Step(3); returned = true }()
		if returned || failure == nil || b.finalDispatches != count || b.grouped != groups {
			t.Fatal("retired Session replayed work")
		}
	})
}

// Each partial failure checks the actual producer counter, independently of the
// returned error. Failed selected operations never publish a row or stay reusable.
// fak-test:runtime medium est=1s lane=default
func TestV41FinalNormCompositionFailures(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "")
	withV41ActivationTraceEnv(t, "")
	for _, fault := range []string{"healthy", "rms", "input", "raw", "output", "read", "ordered"} {
		t.Run(fault, func(t *testing.T) {
			m, s, b := v41FinalNormFixture(t)
			row := v41FinalNormPrepared(t, m, s, b)
			b.fault = fault
			stop := enableV41BF16Witness()
			defer stop()
			out, selected, err := v41FinalNormAttempt(m, s, row)
			want := int64(2)
			if fault == "rms" {
				want = 1
			}
			if !selected || int64(b.finalDispatches) != want || v41BF16PublishCalls.Load() != want || v41BF16DeferredCalls.Load() != want {
				t.Fatalf("selection=%v dispatch=%d counters=%d/%d want=%d", selected, b.finalDispatches, v41BF16PublishCalls.Load(), v41BF16DeferredCalls.Load(), want)
			}
			if fault == "healthy" {
				if err != nil || len(out) != len(row) || s.halClosed {
					t.Fatalf("healthy result %d err=%v closed=%v", len(out), err, s.halClosed)
				}
				staged := make([]float32, len(row))
				for i, v := range row {
					staged[i] = v41RoundBF16(v)
				}
				wantRow := rmsnormCfg(staged, m.tensor("model.norm.weight"), float32(m.Cfg.RMSNormEps), m.Cfg)
				for i, v := range wantRow {
					wantRow[i] = v41RoundBF16(v)
				}
				v41DeferredBitsEqual(t, "same BF16 math", out, wantRow)
			} else if out != nil || err == nil || !s.halClosed || s.halFailure == nil {
				t.Fatalf("failed result=%v err=%v closed=%v failure=%v", out, err, s.halClosed, s.halFailure)
			}
			var arithmetic *compute.V41BF16CheckedArithmeticError
			switch fault {
			case "input", "ordered":
				if !errors.As(err, &arithmetic) || arithmetic.Site != compute.V41BF16PublishFinalNormInput {
					t.Fatalf("input verdict lost precedence: %v", err)
				}
			case "output":
				if !errors.As(err, &arithmetic) || arithmetic.Site != compute.V41BF16PublishFinalNormOutput {
					t.Fatalf("output verdict missing: %v", err)
				}
			case "raw":
				if errors.As(err, &arithmetic) || !errors.Is(err, errV41ProjectionResult) {
					t.Fatalf("raw validation did not precede output verdict: %v", err)
				}
			}
			for _, count := range b.frees {
				if count != 1 {
					t.Fatalf("transient free count=%d", count)
				}
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41FinalNormCompositionReplacementAndTrace(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "")
	withV41ActivationTraceEnv(t, "")
	for _, kind := range []string{"norm", "round", "nil-norm", "nil-round", "disabled", "missing-gain"} {
		t.Run(kind, func(t *testing.T) {
			m, s, b := v41FinalNormFixture(t)
			row := v41FinalNormPrepared(t, m, s, b)
			st := s.v41Forward
			switch kind {
			case "norm":
				st.setFinalNorm(st.finalNorm())
			case "round":
				st.setBF16Boundary(st.bf16Boundary())
			case "nil-norm":
				st.setFinalNorm(nil)
			case "nil-round":
				st.setBF16Boundary(nil)
			case "disabled":
				t.Setenv(v41DeferredBF16PublishEnv, "0")
			case "missing-gain":
				gain := s.halW["model.norm.weight"]
				delete(s.halW, "model.norm.weight")
				defer func() { s.halW["model.norm.weight"] = gain }()
			}
			_, selected, err := v41FinalNormAttempt(m, s, row)
			if selected || err != nil || b.grouped != 0 {
				t.Fatalf("replacement selected=%v err=%v grouped=%d", selected, err, b.grouped)
			}
		})
	}
	// A global recorder without an env path creates a real trace object. This
	// exercises full-forward preparation rather than passing a synthetic nil state.
	m, s, b := v41FinalNormFixture(t)
	_ = v41FinalNormPrepared(t, m, s, b)
	rec, disable := computetrace.Enable(4096, "finalnorm", "full")
	defer disable()
	if len(s.Prefill([]int{1})) == 0 || len(rec.Artifact().Events) == 0 {
		t.Fatal("actual full-forward trace absent")
	}
	if b.grouped != 0 {
		t.Fatal("diagnostic full-forward composed final normalization")
	}
	beforeEvents := len(rec.Artifact().Events)
	trace := m.v41ActivationTracer([]int{1, 2})
	if trace == nil {
		t.Fatal("explicit transaction tracer absent")
	}
	func() {
		scope := s.beginBackendExecution()
		complete := false
		defer scope.finishCompleted(&complete)
		got, stats, err := m.forwardV41Step(2, s.v41State(), &v41ProjScratch{activationTrace: trace})
		if err != nil || len(got) == 0 || !stats.Committed {
			t.Fatalf("traced transaction: stats=%+v err=%v", stats, err)
		}
		complete = true
	}()
	if b.grouped != 0 || len(rec.Artifact().Events) <= beforeEvents {
		t.Fatal("explicit Step trace lost ordinary boundaries or emitted no events")
	}
}

// Isolate the legacy runtime setting. The child runs actual selected callbacks
// in a joined goroutine so Goexit, a nil panic and normal return are distinct.
// fak-test:runtime medium est=2s lane=default
func TestV41FinalNormCompositionLegacyPanic(t *testing.T) {
	const marker = "FAK_TEST_V41_FINAL_NORM_PANIC_CHILD"
	if os.Getenv(marker) != "1" {
		exe, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "-test.run=^TestV41FinalNormCompositionLegacyPanic$", "-test.timeout=6s")
		env := make([]string, 0, len(os.Environ())+2)
		for _, v := range os.Environ() {
			if !strings.HasPrefix(v, "GODEBUG=") && !strings.HasPrefix(v, marker+"=") {
				env = append(env, v)
			}
		}
		cmd.Env = append(env, "GODEBUG="+os.Getenv("GODEBUG")+",panicnil=1", marker+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("legacy child: %v\n%s", err, out)
		}
		return
	}
	t.Setenv(v41DeferredBF16PublishEnv, "")
	withV41ActivationTraceEnv(t, "")
	for _, fault := range []string{"nil", "cleanup-nil", "goexit", "cleanup-goexit", "probe-nil"} {
		t.Run(fault, func(t *testing.T) {
			m, s, b := v41FinalNormFixture(t)
			row := v41FinalNormPrepared(t, m, s, b)
			b.fault = fault
			b.probePanic = fault == "probe-nil"
			returned, boundaryReturned := false, false
			var payload any
			done := make(chan struct{})
			go func() {
				defer close(done)
				func() {
					defer func() { payload = recover() }()
					_, _, _ = v41FinalNormAttempt(m, s, row)
					returned = true
				}()
				boundaryReturned = true
			}()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Fatal("selected operation did not drain")
			}
			b.fault = ""
			b.probePanic = false
			if returned || payload != nil || boundaryReturned != (fault != "goexit" && fault != "cleanup-goexit") || !s.halClosed || s.halFailure == nil {
				t.Fatalf("fault=%s returned=%v outer=%v panic=%v closed=%v", fault, returned, boundaryReturned, payload, s.halClosed)
			}
			if s.backendExecutionScope != nil || !b.mu.TryLock() {
				t.Fatal("owner scope not released")
			}
			b.mu.Unlock()
			for _, count := range b.frees {
				if count != 1 {
					t.Fatalf("unwind free count=%d", count)
				}
			}
		})
	}
}
