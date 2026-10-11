package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Independent quotient/remainder RNE oracle. uint16 increment deliberately
// wraps, including NaN payloads whose raw host helper wraps to finite zero.
func v41BF16BoundaryOracle(bits uint32) uint32 {
	upper, lower := uint16(bits>>16), uint16(bits)
	if lower > 0x8000 || (lower == 0x8000 && upper&1 != 0) {
		upper++
	}
	return uint32(upper) << 16
}

// fak-test:runtime fast est=20ms lane=default
func TestV41BF16BoundaryBitContract(t *testing.T) {
	t.Parallel()
	// Every upper word, with both sides of the midpoint and carry extrema,
	// covers signs, all exponents, both tie parities, subnormals and NaN wrap.
	for high := uint32(0); high <= 0xffff; high++ {
		for _, low := range []uint32{0, 1, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff} {
			word := high<<16 | low
			got := math.Float32bits(v41RoundBF16(math.Float32frombits(word)))
			if want := v41BF16BoundaryOracle(word); got != want {
				t.Fatalf("raw boundary %08x: got %08x want %08x", word, got, want)
			}
		}
	}
	for _, tc := range [][2]uint32{{0, 0}, {0x80000000, 0x80000000}, {0x3f808000, 0x3f800000}, {0x3f818000, 0x3f820000}, {0x00018000, 0x00020000}, {0x7f7f7fff, 0x7f7f0000}, {0x7f7f8000, 0x7f800000}, {0x7fffffff, 0x80000000}, {0xffffffff, 0}} {
		if got := v41BF16BoundaryOracle(tc[0]); got != tc[1] {
			t.Fatalf("oracle %08x=%08x want literal %08x", tc[0], got, tc[1])
		}
	}
}

type v41BF16BoundaryRecorder struct {
	*v41DenseTestBackend
	boundaryLive                                  map[compute.Buffer]bool
	boundaryReads, boundaryUploads, boundaryCalls int
	borrowed                                      []float32
	fault                                         error
	faultAt                                       int
	partial, alias, malformed                     bool
	cleanup                                       any
	freeAttempts                                  int
	opsAtFailure                                  int
}

func newV41BF16BoundaryRecorder() *v41BF16BoundaryRecorder {
	return &v41BF16BoundaryRecorder{v41DenseTestBackend: newV41DenseTestBackend(), boundaryLive: map[compute.Buffer]bool{}}
}

func (b *v41BF16BoundaryRecorder) V41BF16BoundaryAvailable() bool { return true }

func (b *v41BF16BoundaryRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 BF16 boundary input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	b.boundaryUploads++
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	b.boundaryLive[out.Buf()] = true
	return out
}

func (b *v41BF16BoundaryRecorder) V41BF16Boundary(input compute.Tensor) (compute.Tensor, error) {
	b.boundaryCalls++
	if input.Backend() != b {
		panic("BF16 input is not owned by recorder")
	}
	if b.alias {
		return input, nil
	}
	if b.fault != nil && b.boundaryCalls == b.faultAt && !b.partial {
		b.opsAtFailure = len(b.ops)
		return compute.Tensor{}, b.fault
	}
	values := append([]float32(nil), b.Backend.Read(input)...)
	for i, value := range values {
		values[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(value)))
	}
	if b.malformed {
		values[0] = math.Float32frombits(0x3f800001)
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	b.boundaryLive[out.Buf()] = true
	if b.fault != nil && b.boundaryCalls == b.faultAt {
		b.opsAtFailure = len(b.ops)
		return out, b.fault
	}
	return out, nil
}

func (b *v41BF16BoundaryRecorder) Read(x compute.Tensor) []float32 {
	if b.boundaryLive[x.Buf()] {
		b.boundaryReads++
		b.borrowed = append(b.borrowed[:0], b.Backend.Read(x)...)
		return b.borrowed
	}
	return b.v41DenseTestBackend.Read(x)
}

func (b *v41BF16BoundaryRecorder) Free(x compute.Tensor) {
	if b.boundaryLive[x.Buf()] {
		b.freeAttempts++
		delete(b.boundaryLive, x.Buf())
		b.Backend.Free(x)
		for i := range b.borrowed {
			b.borrowed[i] = 99
		}
		if b.cleanup != nil {
			panic(b.cleanup)
		}
		return
	}
	b.v41DenseTestBackend.Free(x)
}

// CPU storage proves the production final-norm call chain and failure ownership,
// never physical execution. The two BF16 publications stay outside F32 RMSNorm.
// fak-test:runtime medium est=2s lane=default
func TestV41BF16BoundaryFinalNormDispatch(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" || compute.Default().Caps().DeviceMemory {
		t.Fatal("software boundary witness requires cpu-ref")
	}
	m := v41FullStepPatchedNorms(t)
	b := newV41BF16BoundaryRecorder()
	s := v41DenseTestSession(t, m, b)
	if s.v41State().bf16Boundary == nil {
		t.Fatal("session did not bind BF16 callback")
	}
	v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
	if b.boundaryCalls != 4 || !s.v41IncrementalEligible() {
		t.Fatalf("prefill calls=%d incremental=%t", b.boundaryCalls, s.v41IncrementalEligible())
	}
	v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
	if b.boundaryCalls != 6 || b.boundaryUploads != 6 || b.boundaryReads != 6 || b.freeAttempts != 12 || len(b.boundaryLive) != 0 {
		t.Fatal("decode failed to preserve two owned BF16 publications per token")
	}
	// Prefix restore requires the same backend identity. Close the source after
	// restoring to prove the callback binds to the destination session.
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
	if b.boundaryCalls != 8 || target.v41State().callbackOwner != target {
		t.Fatal("restored suffix used stale BF16 owner")
	}
	before := captureV41ForwardSnapshot(target.v41Forward)
	b.fault = &compute.BackendError{Backend: "bf16-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
	b.faultAt = 9
	err = recoverError(func() { target.Step(5) })
	var closed *BackendForwardOperationError
	var projection *V41ProjectionOperationError
	if !errors.Is(err, ErrV41BF16BoundaryOperation) || !errors.Is(err, b.fault) || !errors.As(err, &closed) || !errors.As(err, &projection) || projection.Stage != string(v41StageFinalNorm) || closed.Path != "v41-bf16-boundary" {
		t.Fatalf("selected boundary lost typed identity: %v", err)
	}
	if !target.BackendSessionClosed() || b.boundaryCalls != 9 || len(b.ops) != b.opsAtFailure || len(b.boundaryLive) != 0 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(target.v41Forward)) {
		t.Fatal("selected BF16 failure replayed, leaked, or advanced history")
	}
	if retry := recoverError(func() { target.Step(5) }); retry != target.halFailure || b.boundaryCalls != 9 {
		t.Fatal("closed BF16 session retried")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41BF16BoundaryOwnershipAndRefusal(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" || compute.Default().Caps().DeviceMemory {
		t.Fatal("software boundary witness requires cpu-ref")
	}
	for _, mode := range []string{"success", "nonfinite", "overflow", "alias", "malformed", "partial-cleanup"} {
		t.Run(mode, func(t *testing.T) {
			b := newV41BF16BoundaryRecorder()
			s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
			t.Cleanup(s.Close)
			input := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x00018000), 1.00390625, 1.01171875}
			switch mode {
			case "nonfinite":
				input[0] = math.Float32frombits(0xffffffff)
			case "overflow":
				input[0] = math.Float32frombits(0x7f7f8000)
			case "alias":
				b.alias = true
			case "malformed":
				b.malformed = true
			case "partial-cleanup":
				b.fault, b.faultAt, b.partial, b.cleanup = ErrV41ForwardStage, 1, true, "secondary release fault"
			}
			original := append([]float32(nil), input...)
			got, err := v41BF16BoundaryCopy(2, "test", input, s.v41BF16BoundaryFunc())
			for i := range input {
				if math.Float32bits(input[i]) != math.Float32bits(original[i]) {
					t.Fatal("input changed")
				}
			}
			if mode == "success" {
				if err != nil || len(got) != len(input) || s.BackendSessionClosed() {
					t.Fatal(err)
				}
				for i, v := range got {
					if math.Float32bits(v) != v41BF16BoundaryOracle(math.Float32bits(input[i])) {
						t.Fatal("owned result changed during Free")
					}
				}
			} else if got != nil || !errors.Is(err, ErrV41BF16BoundaryOperation) || !s.BackendSessionClosed() {
				t.Fatalf("invalid selected output was not refused: %v", err)
			}
			if mode == "partial-cleanup" && (!errors.Is(err, b.fault) || b.freeAttempts != 2) {
				t.Fatal("cleanup replaced primary failure or skipped owner")
			}
			if mode == "nonfinite" && (b.boundaryUploads != 0 || b.boundaryCalls != 0) {
				t.Fatal("NaN reached raw wrapping primitive")
			}
			if len(b.boundaryLive) != 0 {
				t.Fatal("boundary allocation leaked")
			}
		})
	}
	var absent *Session
	if absent.v41BF16BoundaryFunc() != nil || (&Session{M: &Model{}, Backend: compute.Default()}).v41BF16BoundaryFunc() != nil {
		t.Fatal("portable backend bound device boundary")
	}
}

// The witness spine does not remove any global host-operation admission gate.
// fak-test:runtime fast est=100ms lane=default
func TestV41BF16BoundaryRetainsDeviceOnlyGate(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" || compute.Default().Caps().DeviceMemory {
		t.Fatal("software boundary witness requires cpu-ref")
	}
	m := v41FullStepPatchedNorms(t)
	b := newV41BF16BoundaryRecorder()
	s := v41DenseTestSession(t, m, b)
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	err := recoverError(func() { s.Prefill([]int{1}) })
	if !errors.Is(err, errV41DeviceOnlyHostOp) || b.boundaryCalls != 0 || !s.BackendSessionClosed() {
		t.Fatalf("partial BF16 spine changed whole-forward admission: calls=%d err=%v", b.boundaryCalls, err)
	}
}
