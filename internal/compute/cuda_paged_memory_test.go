package compute

import (
	"testing"
)

// TestCUDABlockTableZeroCopyClone verifies #10723:
// block-table indirection allows session cloning with zero-copy prefix sharing,
// refcounted physical pages, and copy-on-write allocation for divergent tokens.
func TestCUDABlockTableZeroCopyClone(t *testing.T) {
	const blockSize = 16
	const stride = 64 // 64 floats per token

	parent := NewCUDABlockTable(blockSize, stride)

	// Append 32 tokens -> exactly 2 blocks of 16 tokens each
	allocated := parent.AppendLogicalTokens(32)
	if len(allocated) != 2 {
		t.Fatalf("expected 2 blocks allocated, got %d", len(allocated))
	}
	if parent.TokenCount() != 32 {
		t.Fatalf("parent token count = %d, want 32", parent.TokenCount())
	}
	if parent.SharedBytes() != 0 {
		t.Fatalf("parent shared bytes = %d, want 0", parent.SharedBytes())
	}
	totalBytes := parent.TotalBytes()
	if parent.UniqueBytes() != totalBytes {
		t.Fatalf("parent unique bytes = %d, want %d", parent.UniqueBytes(), totalBytes)
	}

	// Clone session: shallow copies block table, increments refcount
	child := parent.Clone()
	if child.TokenCount() != 32 {
		t.Fatalf("child token count = %d, want 32", child.TokenCount())
	}
	if child.SharedBytes() != totalBytes {
		t.Fatalf("child shared bytes = %d, want %d (zero-copy sharing)", child.SharedBytes(), totalBytes)
	}
	if child.UniqueBytes() != 0 {
		t.Fatalf("child unique bytes = %d, want 0", child.UniqueBytes())
	}

	// Verify block IDs are identical and refcount is 2
	pBlocks := parent.Blocks()
	cBlocks := child.Blocks()
	for i := range pBlocks {
		if pBlocks[i].ID != cBlocks[i].ID {
			t.Fatalf("block %d ID mismatch: parent %d vs child %d", i, pBlocks[i].ID, cBlocks[i].ID)
		}
		if pBlocks[i].RefCount() != 2 {
			t.Fatalf("block %d refcount = %d, want 2", i, pBlocks[i].RefCount())
		}
	}

	// Append 16 divergent tokens to child: should allocate a new 3rd block
	childAllocated := child.AppendLogicalTokens(16)
	if len(childAllocated) != 1 {
		t.Fatalf("expected 1 new block allocated for child, got %d", len(childAllocated))
	}
	if child.TokenCount() != 48 {
		t.Fatalf("child token count = %d, want 48", child.TokenCount())
	}
	if parent.TokenCount() != 32 {
		t.Fatalf("parent token count modified! got %d, want 32", parent.TokenCount())
	}

	// Child has shared bytes for the 32 prefix tokens, plus unique bytes for the 16 divergent tokens
	if child.SharedBytes() != totalBytes {
		t.Fatalf("child shared bytes after append = %d, want %d", child.SharedBytes(), totalBytes)
	}
	if child.UniqueBytes() <= 0 {
		t.Fatalf("child unique bytes after append = %d, want > 0", child.UniqueBytes())
	}

	// Release child: parent blocks return to refcount 1
	child.Release()
	if parent.SharedBytes() != 0 {
		t.Fatalf("parent shared bytes after child release = %d, want 0", parent.SharedBytes())
	}
}

// TestPageOffloadManagerTransactional verifies #10722:
// page-granular CPU offloading commits host bundles before device page release.
func TestPageOffloadManagerTransactional(t *testing.T) {
	mgr := NewPageOffloadManager()

	data := []byte("serialized-kv-page-bundle-layer0-31")
	bundle, err := mgr.OffloadBundle(101, 32, 16, data)
	if err != nil {
		t.Fatalf("OffloadBundle failed: %v", err)
	}
	if !bundle.Committed {
		t.Fatalf("expected bundle to be committed")
	}
	if mgr.OffloadedBlocks() != 1 {
		t.Fatalf("offloaded blocks = %d, want 1", mgr.OffloadedBlocks())
	}

	// Restore bundle
	restored, ok := mgr.RestoreBundle(101)
	if !ok || string(restored.HostData) != string(data) {
		t.Fatalf("failed to restore committed bundle: ok=%v, data=%q", ok, string(restored.HostData))
	}

	// Remove host bundle
	mgr.RemoveHostBundle(101)
	if mgr.OffloadedBlocks() != 0 {
		t.Fatalf("offloaded blocks after remove = %d, want 0", mgr.OffloadedBlocks())
	}
}

// TestVMMReservationVirtualToPhysical verifies #10720:
// virtual memory reservation decouples context length from physical page mapping.
func TestVMMReservationVirtualToPhysical(t *testing.T) {
	const maxTokens = 131072 // 128k context
	const stride = 128
	const pageSize = 2 * 1024 * 1024 // 2MB

	res := NewVMMReservation(maxTokens, stride, pageSize)
	if res.VirtualBytes <= 0 {
		t.Fatalf("virtual bytes = %d, want > 0", res.VirtualBytes)
	}
	if res.PageCount() != 0 {
		t.Fatalf("initial page count = %d, want 0", res.PageCount())
	}

	// Map physical page for token 0
	page0, err := res.MapPhysicalPage(0)
	if err != nil || page0 != 0 {
		t.Fatalf("MapPhysicalPage(0) = (%d, %v), want (0, nil)", page0, err)
	}
	if res.PageCount() != 1 {
		t.Fatalf("page count = %d, want 1", res.PageCount())
	}

	// Map physical page for token 5000 (crosses into page 1 or higher)
	page5000, err := res.MapPhysicalPage(5000)
	if err != nil {
		t.Fatalf("MapPhysicalPage(5000) failed: %v", err)
	}
	if page5000 <= page0 {
		t.Fatalf("expected page5000 > page0, got %d <= %d", page5000, page0)
	}
	if res.PageCount() != 2 {
		t.Fatalf("page count = %d, want 2", res.PageCount())
	}

	// Out-of-bounds mapping fails
	if _, err := res.MapPhysicalPage(maxTokens + 1); err == nil {
		t.Fatalf("expected error for token beyond maxTokens")
	}

	// Unmap physical page 0
	if !res.UnmapPhysicalPage(page0) {
		t.Fatalf("UnmapPhysicalPage failed for page0")
	}
	if res.PageCount() != 1 {
		t.Fatalf("page count after unmap = %d, want 1", res.PageCount())
	}
}
