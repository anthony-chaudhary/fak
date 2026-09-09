package ctxmmu

import (
	"errors"
	"testing"
)

func TestPhysicalBlockAllocationIncarnation(t *testing.T) {
	alloc := NewPhysicalBlockAllocator(1, 8)
	old, err := alloc.Allocate()
	if err != nil {
		t.Fatalf("Allocate old: %v", err)
	}
	oldID := old.ID
	if ok, err := alloc.Free(oldID); err != nil || !ok {
		t.Fatalf("Free old: ok=%v err=%v", ok, err)
	}

	replacement, err := alloc.Allocate()
	if err != nil {
		t.Fatalf("Allocate replacement: %v", err)
	}
	if replacement == old {
		t.Fatal("replacement reused the retired PhysicalBlock object")
	}
	if replacement.ID == oldID {
		t.Fatalf("replacement reused retired allocation ID %d", oldID)
	}
	if old.PhysicalSlot() != replacement.PhysicalSlot() {
		t.Fatalf("single-slot allocator did not reuse physical slot: old=%d replacement=%d", old.PhysicalSlot(), replacement.PhysicalSlot())
	}
	replacement.Data[0] = 0x7a
	wantAccess := replacement.LastAccess()

	if got, err := alloc.GetBlock(oldID); !errors.Is(err, ErrBlockNotFound) || got != nil {
		t.Errorf("GetBlock(stale ID) = (%p, %v), want (nil, ErrBlockNotFound)", got, err)
	}
	if err := alloc.Retain(oldID); !errors.Is(err, ErrBlockNotFound) {
		t.Errorf("Retain(stale ID) = %v, want ErrBlockNotFound", err)
	}
	if got := old.Retain(); got != 0 {
		t.Errorf("retired block Retain() = %d, want 0", got)
	}
	if got := old.Release(); got != 0 {
		t.Errorf("retired block Release() = %d, want 0", got)
	}
	old.Data[0] = 0xff

	if got := replacement.RefCount(); got != 1 {
		t.Errorf("replacement refcount = %d, want 1", got)
	}
	if got := replacement.LastAccess(); got != wantAccess {
		t.Errorf("replacement last access changed through stale handle: got %d want %d", got, wantAccess)
	}
	if got := replacement.Data[0]; got != 0x7a {
		t.Errorf("replacement data changed through stale object: got %#x want 0x7a", got)
	}
	if ok, err := alloc.Free(oldID); err != nil || ok {
		t.Errorf("Free(stale ID) = (%v, %v), want (false, nil)", ok, err)
	}
	if got, err := alloc.GetBlock(replacement.ID); err != nil || got != replacement {
		t.Fatalf("replacement after stale operations = (%p, %v), want (%p, nil)", got, err, replacement)
	}

	// Retirement is keyed by the allocator's private incarnation identity, not
	// the exported ID mirror on the block object.
	replacementID := replacement.ID
	replacement.ID = oldID
	if got, err := alloc.GetBlock(replacementID); err != nil || got != replacement {
		t.Fatalf("authoritative lookup after public ID mutation = (%p, %v), want (%p, nil)", got, err, replacement)
	}
	if ok, err := alloc.Free(replacementID); err != nil || !ok {
		t.Fatalf("Free replacement by authoritative ID: ok=%v err=%v", ok, err)
	}
	if ok, err := alloc.Free(replacementID); err != nil || ok {
		t.Fatalf("second Free replacement = (%v, %v), want (false, nil)", ok, err)
	}
	if got := alloc.AllocatedCount(); got != 0 {
		t.Fatalf("AllocatedCount after exactly-once retirement = %d, want 0", got)
	}

	t.Run("ID exhaustion is side-effect free", func(t *testing.T) {
		exhausted := NewPhysicalBlockAllocator(1, 8)
		exhausted.nextID = maxPhysicalBlockID() + 1
		if _, err := exhausted.Allocate(); !errors.Is(err, ErrBlockIDExhausted) {
			t.Fatalf("Allocate at exhausted ID space = %v, want ErrBlockIDExhausted", err)
		}
		if exhausted.AllocatedCount() != 0 || exhausted.FreeCount() != 1 || exhausted.blocks[0] != nil {
			t.Fatalf("ID exhaustion mutated allocator: allocated=%d free=%d block=%p", exhausted.AllocatedCount(), exhausted.FreeCount(), exhausted.blocks[0])
		}
	})
}
