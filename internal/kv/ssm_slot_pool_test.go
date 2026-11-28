package kv

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestSSMStateSlotPool(t *testing.T) {
	shape := SSMStateShape{
		Layers:    2,
		ConvDim:   16,
		DConv:     4,
		NumHeads:  4,
		HeadDim:   8,
		DState:    16,
		ElemBytes: 2,
	}
	pool, err := NewSSMStateSlotPool(shape, 2)
	if err != nil {
		t.Fatalf("NewSSMStateSlotPool: %v", err)
	}
	wantPerSlot := int64(2) * (16*4*2 + 4*8*16*2) // 2 * (128 + 1024) = 2304
	if st := pool.Stats(); st.BytesPerSlot != wantPerSlot || st.TotalPoolBytes != 2*wantPerSlot {
		t.Fatalf("Stats=%+v, want BytesPerSlot=%d TotalPoolBytes=%d", st, wantPerSlot, 2*wantPerSlot)
	}

	s0, err := pool.Allocate("seq-a")
	if err != nil || s0 != 0 {
		t.Fatalf("Allocate seq-a = (%d, %v), want (0, nil)", s0, err)
	}
	// Idempotent re-allocation returns the same slot.
	if s0Again, err := pool.Allocate("seq-a"); err != nil || s0Again != s0 {
		t.Fatalf("idempotent Allocate seq-a = (%d, %v), want (%d, nil)", s0Again, err, s0)
	}
	s1, err := pool.Allocate("seq-b")
	if err != nil || s1 != 1 {
		t.Fatalf("Allocate seq-b = (%d, %v), want (1, nil)", s1, err)
	}
	if _, err := pool.Allocate("seq-c"); !errors.Is(err, ErrSSMStatePoolFull) {
		t.Fatalf("Allocate seq-c err=%v, want ErrSSMStatePoolFull", err)
	}

	// Decode-side state_batch_indices: negative (-1) for pad/unallocated entries.
	indices := pool.BatchIndices([]string{"seq-b", "", "missing", "seq-a"})
	wantIndices := []int32{1, -1, -1, 0}
	if !reflect.DeepEqual(indices, wantIndices) {
		t.Fatalf("BatchIndices=%v, want %v", indices, wantIndices)
	}

	if !pool.Release("seq-a") {
		t.Fatal("Release seq-a returned false")
	}
	s2, err := pool.Allocate("seq-c")
	if err != nil || s2 != 0 {
		t.Fatalf("Allocate seq-c after release = (%d, %v), want slot 0 reused", s2, err)
	}
	if st := pool.Stats(); st.UsedSlots != 2 || st.FreeSlots != 0 {
		t.Fatalf("Stats after reuse=%+v, want Used=2 Free=0", st)
	}
}
