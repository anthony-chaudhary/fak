package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder uses only parent APIs. Its structural uint32 method does not
// require the new compute interface, so parent overlays fail on missing calls.
// CPU-backed storage demonstrates ownership and routing, never device status.
type v41BF16CheckedRecorder struct {
	*v41DenseTestBackend
	checkedLive, rawLive                   map[compute.Buffer]bool
	sites                                  []uint32
	calls, rawCalls, uploads, reads, frees int
	borrowed                               []float32
	unavailable                            bool
	fault                                  error
	panicFault, cleanup, closePanic        any
	faultAt, closeCalls, opsAtFailure      int
	partial, alias, malformed              bool
}

func newV41BF16CheckedRecorder() *v41BF16CheckedRecorder {
	return &v41BF16CheckedRecorder{v41DenseTestBackend: newV41DenseTestBackend(), checkedLive: map[compute.Buffer]bool{}, rawLive: map[compute.Buffer]bool{}}
}
func (b *v41BF16CheckedRecorder) V41BF16CheckedAvailable() bool  { return !b.unavailable }
func (b *v41BF16CheckedRecorder) V41BF16BoundaryAvailable() bool { return true }
func (b *v41BF16CheckedRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	checked := site == "V4.1 checked BF16 input"
	if !checked && site != "V4.1 BF16 boundary input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation {
		panic("BF16 upload lost activation class")
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	if checked {
		b.uploads++
		b.checkedLive[out.Buf()] = true
	} else {
		b.rawLive[out.Buf()] = true
	}
	return out
}
func (b *v41BF16CheckedRecorder) V41BF16Boundary(input compute.Tensor) (compute.Tensor, error) {
	b.rawCalls++
	values := append([]float32(nil), b.Backend.Read(input)...)
	for i, value := range values {
		values[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(value)))
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	b.rawLive[out.Buf()] = true
	return out, nil
}
func (b *v41BF16CheckedRecorder) V41BF16Checked(input compute.Tensor, site uint32) (compute.Tensor, error) {
	b.calls++
	b.sites = append(b.sites, site)
	if input.Backend() != b || len(input.Shape) != 1 || (site != 1 && site != 2) {
		panic("checked BF16 owner, rank or site changed")
	}
	if b.calls == b.faultAt {
		b.opsAtFailure = len(b.ops)
		if b.panicFault != nil {
			panic(b.panicFault)
		}
		if b.fault != nil && !b.partial {
			return compute.Tensor{}, b.fault
		}
	}
	if b.alias {
		return input, nil
	}
	values := append([]float32(nil), b.Backend.Read(input)...)
	for i, value := range values {
		bits := math.Float32bits(value)
		rounded := v41BF16BoundaryOracle(bits)
		if bits&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000 {
			return compute.Tensor{}, &v41BF16CheckedTestFault{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(rounded)
	}
	if b.malformed {
		values[0] = math.Float32frombits(0x3f800001)
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	b.checkedLive[out.Buf()] = true
	if b.calls == b.faultAt && b.fault != nil {
		return out, b.fault
	}
	return out, nil
}
func (b *v41BF16CheckedRecorder) Read(x compute.Tensor) []float32 {
	if b.checkedLive[x.Buf()] {
		b.reads++
		b.borrowed = append(b.borrowed[:0], b.Backend.Read(x)...)
		return b.borrowed
	}
	return b.v41DenseTestBackend.Read(x)
}
func (b *v41BF16CheckedRecorder) Free(x compute.Tensor) {
	if b.checkedLive[x.Buf()] {
		b.frees++
		delete(b.checkedLive, x.Buf())
		b.Backend.Free(x)
		for i := range b.borrowed {
			b.borrowed[i] = 99
		}
		if b.cleanup != nil {
			panic(b.cleanup)
		}
		return
	}
	if b.rawLive[x.Buf()] {
		delete(b.rawLive, x.Buf())
		b.Backend.Free(x)
		return
	}
	b.v41DenseTestBackend.Free(x)
}
func (b *v41BF16CheckedRecorder) Recycle() {
	b.closeCalls++
	if b.closePanic != nil {
		panic(b.closePanic)
	}
}

type v41BF16CheckedTestFault struct {
	Site  uint32
	Index int
}

func (e *v41BF16CheckedTestFault) Error() string {
	return fmt.Sprintf("checked BF16 site %d invalid element %d", e.Site, e.Index)
}

// fak-test:runtime medium est=2s lane=default
func TestV41BF16CheckedFinalNormDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("checked BF16 software witness requires cpu-ref")
	}
	t.Run("prefill-step-restored-owner", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41BF16CheckedRecorder()
		s := v41DenseTestSession(t, m, b)
		v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
		if b.calls != 4 || b.rawCalls != 0 {
			t.Fatalf("missing checked final-norm dispatch: checked=%d raw=%d", b.calls, b.rawCalls)
		}
		v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
		if b.calls != 6 || b.uploads != 6 || b.reads != 6 || b.frees != 12 || len(b.checkedLive) != 0 || !s.v41IncrementalEligible() {
			t.Fatal("checked copies leaked or changed incremental execution")
		}
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer snap.Close()
		target := v41DenseTestSession(t, m, b)
		if err := snap.Restore(target); err != nil {
			t.Fatal(err)
		}
		s.Close()
		v41GroupedParity(t, target.Prefill([]int{4}), lastLogits(m.Forward([]int{1, 2, 3, 4})), 1e-4)
		if b.calls != 8 || target.v41State().callbackOwner != target || len(b.checkedLive) != 0 {
			t.Fatal("restored suffix retained closed source owner")
		}
		for i, site := range b.sites {
			if site != uint32(1+i%2) {
				t.Fatalf("publication %d has site %d", i, site)
			}
		}
	})
	t.Run("absent-keeps-raw-path", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41BF16CheckedRecorder()
		b.unavailable = true
		s := v41DenseTestSession(t, m, b)
		v41GroupedParity(t, s.Prefill([]int{1}), lastLogits(m.Forward([]int{1})), 1e-4)
		if b.calls != 0 || b.rawCalls != 2 || s.BackendSessionClosed() || len(b.rawLive) != 0 {
			t.Fatal("absent checked capability changed pre-selection raw path")
		}
	})
	t.Run("finite-bit-publication", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41BF16CheckedRecorder()
		s := v41DenseTestSession(t, m, b)
		input := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x00018000), math.Float32frombits(0x3f808000), math.Float32frombits(0x3f818000)}
		original := append([]float32(nil), input...)
		round := s.v41State().bf16Boundary
		got, err := round(-1, "model.norm.weight input", input)
		if err != nil || b.calls != 1 || b.rawCalls != 0 || len(got) != len(input) {
			t.Fatalf("checked finite publication not selected: calls=%d err=%v", b.calls, err)
		}
		for i := range got {
			if math.Float32bits(input[i]) != math.Float32bits(original[i]) || math.Float32bits(got[i]) != v41BF16BoundaryOracle(math.Float32bits(original[i])) {
				t.Fatal("input changed or output borrowed freed scratch")
			}
		}
	})
	for _, mode := range []string{"numeric", "partial-cleanup", "typed-panic-step", "typed-panic-suffix"} {
		t.Run(mode, func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41BF16CheckedRecorder()
			s := v41DenseTestSession(t, m, b)
			// LIFO cleanup disarms only after assertions, before fixture Close. The
			// unchanged parent must report missing dispatch, not abort in teardown.
			t.Cleanup(func() { b.closePanic = nil })
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			readsBefore := b.reads
			b.faultAt = 5
			numeric := &v41BF16CheckedTestFault{Site: 1, Index: 3}
			var primary error = &compute.BackendError{Backend: "checked-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage, Recovered: numeric}
			if mode == "typed-panic-step" || mode == "typed-panic-suffix" {
				primary = &BackendForwardOperationError{Backend: "checked-recorder", Forward: ForwardPathKind("deepseek41"), Path: "wrapped-checked-publication", Layer: -1, Stage: "status", Cause: ErrV41ForwardStage}
				b.panicFault = primary
			} else {
				b.fault = primary
			}
			b.partial = mode == "partial-cleanup"
			b.cleanup, b.closePanic = errors.New("secondary checked free failure"), errors.New("secondary checked close failure")
			err := recoverError(func() {
				if mode == "typed-panic-suffix" {
					s.Prefill([]int{3})
				} else {
					s.Step(3)
				}
			})
			var closed *BackendForwardOperationError
			if !errors.Is(err, primary) || !errors.As(err, &closed) || b.calls != 5 || b.reads != readsBefore || b.rawCalls != 0 {
				t.Fatalf("checked failure was replayed/read or lost cause: calls=%d reads=%d err=%v", b.calls, b.reads, err)
			}
			if b.panicFault != nil {
				if err != primary || s.halFailure != primary {
					t.Fatal("typed panic identity replaced")
				}
			} else {
				var selected *V41BF16BoundaryOperationError
				var attributed *v41BF16CheckedTestFault
				if !errors.As(err, &selected) || selected.Layer != -1 || selected.Site != "model.norm.weight input" || closed.Path != "v41-bf16-checked" || !errors.As(err, &attributed) || attributed != numeric {
					t.Fatal("status/site identity lost")
				}
			}
			if !s.BackendSessionClosed() || b.closeCalls != 1 || len(b.checkedLive) != 0 || len(b.ops) != b.opsAtFailure || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("checked failure leaked, advanced state or left owner open")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || b.calls != 5 {
				t.Fatal("retired checked owner retried")
			}
		})
	}
	for _, mode := range []string{"nan-wrap", "bf16-overflow", "alias", "malformed", "unknown-site"} {
		t.Run(mode, func(t *testing.T) {
			m := v41FullStepPatchedNorms(t)
			b := newV41BF16CheckedRecorder()
			s := v41DenseTestSession(t, m, b)
			values := []float32{1}
			site := "model.norm.weight output"
			switch mode {
			case "nan-wrap":
				values[0] = math.Float32frombits(0x7fffffff)
			case "bf16-overflow":
				values[0] = math.Float32frombits(0x7f7f8000)
			case "alias":
				b.alias = true
			case "malformed":
				b.malformed = true
			case "unknown-site":
				site = "unknown"
			}
			got, err := s.v41State().bf16Boundary(-1, site, values)
			var closed *BackendForwardOperationError
			if got != nil || !errors.As(err, &closed) || closed.Path != "v41-bf16-checked" || !s.BackendSessionClosed() || len(b.checkedLive) != 0 {
				t.Fatalf("invalid checked publication not closed: mode=%s err=%v", mode, err)
			}
			if mode == "nan-wrap" || mode == "unknown-site" {
				if b.calls != 0 || b.uploads != 0 {
					t.Fatal("invalid model payload dispatched")
				}
			} else if b.calls != 1 {
				t.Fatal("checked operation was not reached")
			}
			if mode != "malformed" && b.reads != 0 {
				t.Fatal("failed publication was read")
			}
		})
	}
	t.Run("device-only-gate-retained", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		b := newV41BF16CheckedRecorder()
		s := v41DenseTestSession(t, m, b)
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		err := recoverError(func() { s.Prefill([]int{1}) })
		if !errors.Is(err, errV41DeviceOnlyHostOp) || b.calls != 0 || !s.BackendSessionClosed() {
			t.Fatal("checked prerequisite promoted device-only admission")
		}
	})
}
