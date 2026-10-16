package strix

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1s
func TestUMAPointerManagerRefusesForeignRelease(t *testing.T) {
	owner, other := NewUMAPointerManager(), NewUMAPointerManager()
	buf, err := owner.AllocateUMABuffer(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	buf.Slice()[0] = 37
	ptr := buf.HostPtr()
	beforeOwner, beforeOther := owner.Counters(), other.Counters()
	for i := 0; i < 2; i++ {
		if err := other.FreeUMABuffer(buf); !errors.Is(err, ErrBufferNotOwned) {
			t.Fatalf("foreign release=%v", err)
		}
		if owner.Counters() != beforeOwner || other.Counters() != beforeOther {
			t.Fatal("foreign release changed allocation counters")
		}
		if buf.HostPtr() != ptr || buf.Slice()[0] != 37 {
			t.Fatal("foreign release invalidated owner's bytes")
		}
	}
	if err := owner.FreeUMABuffer(buf); err != nil {
		t.Fatal(err)
	}
	if owner.Counters().ActiveAllocations != 0 || buf.HostPtr() != 0 {
		t.Fatal("owner release did not retire buffer")
	}
	if err := owner.FreeUMABuffer(buf); !errors.Is(err, ErrBufferClosed) {
		t.Fatalf("repeat release=%v", err)
	}
}

// fak-test:runtime fast est=1s
func TestUMAPointerManagerUnmanagedReleaseBoundary(t *testing.T) {
	mgr := NewUMAPointerManager()
	buf, err := NewUMABuffer(64)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	if err := mgr.FreeUMABuffer(buf); !errors.Is(err, ErrBufferNotOwned) {
		t.Fatalf("unregistered release=%v", err)
	}
	if buf.HostPtr() == 0 {
		t.Fatal("unregistered buffer was closed")
	}
	var unmanaged *UMAPointerManager
	if err := unmanaged.FreeUMABuffer(buf); err != nil {
		t.Fatalf("legacy nil-manager release=%v", err)
	}
	if buf.HostPtr() != 0 {
		t.Fatal("nil-manager release left buffer live")
	}
	if err := mgr.FreeUMABuffer(nil); !errors.Is(err, ErrBufferClosed) {
		t.Fatalf("nil buffer=%v", err)
	}
	if err := mgr.FreeUMABuffer(buf); !errors.Is(err, ErrBufferClosed) {
		t.Fatalf("closed buffer=%v", err)
	}
}
