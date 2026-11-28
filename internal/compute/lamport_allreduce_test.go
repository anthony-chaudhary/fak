package compute

import (
	"errors"
	"math"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s lane=default
func TestLamportAllreducePosZeroRoundTripAndSentinel(t *testing.T) {
	t.Parallel()

	if KNegZeroF16 != 0x8000 || KNegZeroF32 != 0x80000000 {
		t.Fatalf("unexpected sentinel constants: F16=0x%x F32=0x%x", KNegZeroF16, KNegZeroF32)
	}

	// +0.0 encodes to -0.0 (0x80000000), never colliding with empty 0x00000000.
	if got := EncodeLamportF32(0.0); got != KNegZeroF32 {
		t.Fatalf("EncodeLamportF32(+0.0) = 0x%08x, want 0x%08x", got, KNegZeroF32)
	}
	if got := EncodeLamportF16(0x0000); got != KNegZeroF16 {
		t.Fatalf("EncodeLamportF16(0x0000) = 0x%04x, want 0x%04x", got, KNegZeroF16)
	}

	// Untouched slot buffer must report empty / incomplete.
	buf, err := NewLamportSlotBuffer(4)
	if err != nil {
		t.Fatalf("NewLamportSlotBuffer: %v", err)
	}
	if buf.Ready() {
		t.Fatal("untouched slot buffer reported Ready() = true, want false")
	}
	if _, err := buf.ReadF32(); !errors.Is(err, ErrLamportSlotIncomplete) {
		t.Fatalf("untouched ReadF32 err=%v, want ErrLamportSlotIncomplete", err)
	}

	// Written all-zero payload must be ready and round-trip back to canonical +0.0.
	negZero := float32(math.Copysign(0, -1))
	zeros := []float32{0.0, -0.0, negZero, 0.0}
	if err := buf.WriteF32(zeros); err != nil {
		t.Fatalf("WriteF32: %v", err)
	}
	if !buf.Ready() {
		t.Fatal("written zero payload reported Ready() = false, want true")
	}
	decoded, err := buf.ReadF32()
	if err != nil {
		t.Fatalf("ReadF32: %v", err)
	}
	for i, v := range decoded {
		if math.Float32bits(v) != 0 {
			t.Fatalf("decoded[%d] bits = 0x%08x, want canonical +0.0 (0x00000000)", i, math.Float32bits(v))
		}
	}

	// End-to-end 2-rank all-zero round-trip (F32 and F16).
	resF32, err := LamportAllreduceF32([][]float32{{0.0, negZero, 1.5}, {0.0, negZero, -1.5}})
	if err != nil {
		t.Fatalf("LamportAllreduceF32: %v", err)
	}
	for i, v := range resF32 {
		if math.Float32bits(v) != 0 {
			t.Fatalf("LamportAllreduceF32[%d] = %g (0x%08x), want canonical +0.0", i, v, math.Float32bits(v))
		}
	}

	resF16, err := LamportAllreduceF16Bits([][]uint16{{0x0000, KNegZeroF16, 0x3c00}, {0x0000, 0x0000, 0xbc00}})
	if err != nil {
		t.Fatalf("LamportAllreduceF16Bits: %v", err)
	}
	for i, bits := range resF16 {
		if bits != 0x0000 {
			t.Fatalf("LamportAllreduceF16Bits[%d] = 0x%04x, want 0x0000 (+0.0)", i, bits)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestLamportAllreduceIncompletePeerBlockingAndSentinelDetection(t *testing.T) {
	t.Parallel()

	comm, err := NewLamportCommunicator(2, 4)
	if err != nil {
		t.Fatalf("NewLamportCommunicator: %v", err)
	}

	// Only rank 0 pushes epoch 0; rank 1 has not pushed yet.
	if err := comm.Push(0, 0, []float32{1.0, 0.0, -2.0, 3.5}); err != nil {
		t.Fatalf("Push(rank 0): %v", err)
	}
	if comm.PollReady(0, 0) {
		t.Fatal("PollReady=true while rank 1 slot is empty")
	}
	if _, err := comm.ConsumeAllReduce(0, 0, 5*time.Millisecond); !errors.Is(err, ErrLamportSlotIncomplete) {
		t.Fatalf("ConsumeAllReduce with missing peer: err=%v, want ErrLamportSlotIncomplete", err)
	}

	// Partial write on rank 1 leaving last element as empty sentinel 0x00000000 must still fail closed.
	slot1, err := comm.InboundSlot(0, 1, 0)
	if err != nil {
		t.Fatalf("InboundSlot: %v", err)
	}
	for i, v := range []float32{2.0, 0.0, 5.0} {
		if err := slot1.WriteEncodedWord(i, EncodeLamportF32(v)); err != nil {
			t.Fatalf("WriteEncodedWord(%d): %v", i, err)
		}
	}
	if comm.PollReady(0, 0) {
		t.Fatal("PollReady=true when word 3 of rank 1 still holds empty sentinel")
	}
	if _, err := comm.ConsumeAllReduce(0, 0, 5*time.Millisecond); !errors.Is(err, ErrLamportSlotIncomplete) {
		t.Fatalf("ConsumeAllReduce with partial peer write: err=%v, want ErrLamportSlotIncomplete", err)
	}

	// Completing the final element unblocks reduction cleanly.
	if err := slot1.WriteEncodedWord(3, EncodeLamportF32(-3.5)); err != nil {
		t.Fatalf("WriteEncodedWord(3): %v", err)
	}
	if !comm.PollReady(0, 0) {
		t.Fatal("PollReady=false after all peer words populated")
	}
	got, err := comm.ConsumeAllReduce(0, 0, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("ConsumeAllReduce after completion: %v", err)
	}
	want := []float32{3.0, 0.0, 3.0, 0.0}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("got[%d]=%g (0x%08x), want %g (0x%08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
	// After ConsumeAllReduce, rank 0's epoch-0 slots must be reset to empty sentinel.
	if comm.PollReady(0, 0) {
		t.Fatal("consumed slot was not reset to empty sentinel")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestLamportAllreduceMultiRankExactSum(t *testing.T) {
	t.Parallel()

	const (
		ranks  = 4
		count  = 8
		epochs = 5 // exceeds LamportRingSlots=3 to test ring wrap-around & slot reset
	)

	be := Default()
	comm, err := NewLamportCommunicator(ranks, count)
	if err != nil {
		t.Fatalf("NewLamportCommunicator: %v", err)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		parts := make([][]float32, ranks)
		tensors := make([]Tensor, ranks)
		for r := 0; r < ranks; r++ {
			p := make([]float32, count)
			for i := 0; i < count; i++ {
				if i%3 == 0 {
					p[i] = 0.0
				} else {
					p[i] = float32((r+1)*(epoch+1)*((i%5)-2)) * 0.25
				}
			}
			parts[r] = p
			tensors[r] = NewF32(be, []int{count}, p)
		}

		refT, err := be.(CollectiveBackend).AllReduceSum(tensors)
		if err != nil {
			t.Fatalf("cpuBackend.AllReduceSum: %v", err)
		}
		want := be.Read(refT)

		// Concurrent producers push while concurrent consumers reduce on every rank.
		var wg sync.WaitGroup
		results := make([][]float32, ranks)
		errs := make([]error, ranks)
		for r := 0; r < ranks; r++ {
			r := r
			wg.Add(2)
			go func() {
				defer wg.Done()
				if err := comm.Push(r, epoch, parts[r]); err != nil {
					t.Errorf("epoch %d rank %d push: %v", epoch, r, err)
				}
			}()
			go func() {
				defer wg.Done()
				results[r], errs[r] = comm.ConsumeAllReduce(r, epoch, 2*time.Second)
			}()
		}
		wg.Wait()

		for r := 0; r < ranks; r++ {
			if errs[r] != nil {
				t.Fatalf("epoch %d rank %d consume: %v", epoch, r, errs[r])
			}
			for i := range want {
				w := CanonicalizeLamportF32(want[i])
				if math.Float32bits(results[r][i]) != math.Float32bits(w) {
					t.Fatalf("epoch %d rank %d [%d] = %g (0x%08x), want %g (0x%08x)",
						epoch, r, i, results[r][i], math.Float32bits(results[r][i]), w, math.Float32bits(w))
				}
			}
		}
	}

	// Refusal checks on invalid geometry.
	if _, err := LamportAllreduceF32([][]float32{{1, 2}, {1}}); !errors.Is(err, ErrLamportInvalidArgs) {
		t.Errorf("ragged parts: err=%v, want ErrLamportInvalidArgs", err)
	}
	if _, err := NewLamportCommunicator(0, 4); !errors.Is(err, ErrLamportInvalidArgs) {
		t.Errorf("zero ranks: err=%v, want ErrLamportInvalidArgs", err)
	}
}
