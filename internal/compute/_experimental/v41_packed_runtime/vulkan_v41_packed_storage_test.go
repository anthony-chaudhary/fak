//go:build ignore

// SOURCE-ONLY: integration sketch for package compute, not a standalone package.
// Original platform constraint: vulkan && (windows || linux) && cgo

package compute

import "testing"

// fak-test:runtime fast est=5ms lane=default
// No Vulkan initialization: exercise only empty-handle registry and accounting.
func TestVulkanV41PackedPrivateOwnership(t *testing.T) {
	e, err := NewV41PackedRowExtent(V41MainKVPer16E4M3NoGlobal, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	p := &vulkanV41PackedRows{live: make(map[*vulkanV41PackedRow]bool)}
	h := &vulkanV41PackedRow{owner: p, extent: e}
	p.live[h] = true
	var typedNil *vulkanV41PackedRow
	foreign := &vulkanV41PackedRow{owner: &vulkanV41PackedRows{}, extent: e}
	for _, bad := range []V41PackedRowStorage{nil, typedNil, foreign} {
		if _, err := p.handle(bad, false); err == nil {
			t.Fatal("accepted invalid owner handle")
		}
	}
	if _, err := p.handle(h, false); err != nil {
		t.Fatal(err)
	}
	h.poisoned = true
	if _, err := p.handle(h, false); err == nil {
		t.Fatal("accepted poisoned handle")
	}
	if _, err := p.handle(h, true); err != nil {
		t.Fatal(err)
	}
	delete(p.live, h)
	if _, err := p.handle(h, true); err == nil {
		t.Fatal("accepted released handle")
	}
	// Synthetic receipt only: no native pointer or physical reservation claim.
	p.live[&vulkanV41PackedRow{buf: &vulkanBuf{n: 576}, reservation: 4096, poisoned: true}] = true
	class, capacity, reservation, err := p.Accounting()
	if err != nil || class != MemoryKVCache || capacity != 576 || reservation != 4096 {
		t.Fatalf("accounting = %v/%d/%d/%v", class, capacity, reservation, err)
	}
}
