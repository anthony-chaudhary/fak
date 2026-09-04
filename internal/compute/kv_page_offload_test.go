package compute

import (
	"bytes"
	"errors"
	"fmt"
	"sync"
	"testing"
)

// TestKVPageOffloadLifecycle verifies page allocation, staging, eviction, and restoration.
func TestKVPageOffloadLifecycle(t *testing.T) {
	cfg := KVOffloadConfig{
		MaxDevicePages: 16,
		MaxHostPages:   64,
		PageSizeBytes:  1024,
	}
	mgr := NewKVPageOffloader(cfg)

	raw := []byte("kv-cache-layer0-31-tokens-0-15-data-payload")
	page, err := mgr.AllocatePage(1001, 32, 16, raw)
	if err != nil {
		t.Fatalf("AllocatePage failed: %v", err)
	}

	if page.State != PageStateDevice {
		t.Fatalf("expected page state %s, got %s", PageStateDevice, page.State)
	}
	if page.PinCount() != 0 {
		t.Fatalf("expected pin count 0, got %d", page.PinCount())
	}

	// Read initial device data
	data, state, err := mgr.ReadPage(1001)
	if err != nil || state != PageStateDevice || !bytes.Equal(data, raw) {
		t.Fatalf("ReadPage before eviction = (%q, %s, %v), want (%q, %s, nil)", data, state, err, raw, PageStateDevice)
	}

	// Transactional eviction to host DRAM
	if err := mgr.EvictPage(1001); err != nil {
		t.Fatalf("EvictPage failed: %v", err)
	}

	stats := mgr.Stats()
	if stats.DevicePages != 0 || stats.HostPages != 1 || stats.OffloadedCount != 1 {
		t.Fatalf("stats after eviction = %+v, want device=0, host=1, offloaded=1", stats)
	}

	// Read host data and verify integrity
	dataHost, stateHost, err := mgr.ReadPage(1001)
	if err != nil || stateHost != PageStateHost || !bytes.Equal(dataHost, raw) {
		t.Fatalf("ReadPage while offloaded = (%q, %s, %v), want (%q, %s, nil)", dataHost, stateHost, err, raw, PageStateHost)
	}

	ok, err := mgr.VerifyDataIntegrity(1001)
	if err != nil || !ok {
		t.Fatalf("VerifyDataIntegrity = (%v, %v), want (true, nil)", ok, err)
	}

	// Restore back to device
	if err := mgr.RestorePage(1001); err != nil {
		t.Fatalf("RestorePage failed: %v", err)
	}

	statsRestored := mgr.Stats()
	if statsRestored.DevicePages != 1 || statsRestored.HostPages != 0 || statsRestored.RestoredCount != 1 {
		t.Fatalf("stats after restore = %+v, want device=1, host=0, restored=1", statsRestored)
	}

	dataRestored, stateRestored, err := mgr.ReadPage(1001)
	if err != nil || stateRestored != PageStateDevice || !bytes.Equal(dataRestored, raw) {
		t.Fatalf("ReadPage after restore = (%q, %s, %v), want (%q, %s, nil)", dataRestored, stateRestored, err, raw, PageStateDevice)
	}
}

// TestKVPageOffloadPinTracking verifies that pinned pages cannot be evicted (#10722).
func TestKVPageOffloadPinTracking(t *testing.T) {
	mgr := NewKVPageOffloader(KVOffloadConfig{})

	raw := []byte("pinned-kv-page-content")
	_, err := mgr.AllocatePage(2001, 16, 8, raw)
	if err != nil {
		t.Fatalf("AllocatePage failed: %v", err)
	}

	// Pin page
	if err := mgr.Pin(2001); err != nil {
		t.Fatalf("Pin failed: %v", err)
	}

	pinned, err := mgr.IsPinned(2001)
	if err != nil || !pinned {
		t.Fatalf("IsPinned = (%v, %v), want (true, nil)", pinned, err)
	}

	// Multiple pins
	if err := mgr.Pin(2001); err != nil {
		t.Fatalf("second Pin failed: %v", err)
	}

	stats := mgr.Stats()
	if stats.PinnedPages != 1 {
		t.Fatalf("pinned pages = %d, want 1", stats.PinnedPages)
	}

	// Attempting to evict pinned page must fail with ErrPagePinned
	err = mgr.EvictPage(2001)
	if !errors.Is(err, ErrPagePinned) {
		t.Fatalf("EvictPage on pinned page = %v, want %v", err, ErrPagePinned)
	}

	// First unpin: still pinned (pinCount = 1)
	if err := mgr.Unpin(2001); err != nil {
		t.Fatalf("Unpin failed: %v", err)
	}
	if err := mgr.EvictPage(2001); !errors.Is(err, ErrPagePinned) {
		t.Fatalf("EvictPage when still pinned = %v, want %v", err, ErrPagePinned)
	}

	// Second unpin: unpinned (pinCount = 0)
	if err := mgr.Unpin(2001); err != nil {
		t.Fatalf("second Unpin failed: %v", err)
	}
	pinned, _ = mgr.IsPinned(2001)
	if pinned {
		t.Fatalf("page should be unpinned")
	}

	// Now eviction should succeed
	if err := mgr.EvictPage(2001); err != nil {
		t.Fatalf("EvictPage failed after unpin: %v", err)
	}

	data, state, err := mgr.ReadPage(2001)
	if err != nil || state != PageStateHost || !bytes.Equal(data, raw) {
		t.Fatalf("ReadPage after unpin and eviction = (%q, %s, %v)", data, state, err)
	}
}

// TestKVPageOffloadTransactionalRollback verifies that staging failures rollback cleanly without data loss.
func TestKVPageOffloadTransactionalRollback(t *testing.T) {
	mgr := NewKVPageOffloader(KVOffloadConfig{})

	raw := []byte("critical-kv-data-never-lose")
	_, err := mgr.AllocatePage(3001, 32, 16, raw)
	if err != nil {
		t.Fatalf("AllocatePage failed: %v", err)
	}

	// Inject a simulated host staging error (e.g. host DRAM allocation failure or DMA error)
	injectedErr := errors.New("simulated host DRAM bus fault")
	mgr.onStageHook = func(page *KVOffloadPage) error {
		return injectedErr
	}

	err = mgr.EvictPage(3001)
	if err == nil || !errors.Is(err, ErrStagingFailed) {
		t.Fatalf("EvictPage under fault = %v, want ErrStagingFailed", err)
	}

	// Verify rollback: state must remain PageStateDevice and device data intact
	data, state, err := mgr.ReadPage(3001)
	if err != nil || state != PageStateDevice || !bytes.Equal(data, raw) {
		t.Fatalf("ReadPage after rollback = (%q, %s, %v), want intact device data", data, state, err)
	}

	stats := mgr.Stats()
	if stats.DevicePages != 1 || stats.HostPages != 0 || stats.FailedEvictions != 1 {
		t.Fatalf("stats after rollback = %+v, want device=1, host=0, failedEvictions=1", stats)
	}

	// Clear fault hook and evict successfully
	mgr.onStageHook = nil
	if err := mgr.EvictPage(3001); err != nil {
		t.Fatalf("subsequent EvictPage failed: %v", err)
	}

	// Now test restore rollback
	injectedRestoreErr := errors.New("simulated VRAM allocation failure during restore")
	mgr.onRestoreHook = func(page *KVOffloadPage) error {
		return injectedRestoreErr
	}

	err = mgr.RestorePage(3001)
	if err == nil || !errors.Is(err, ErrRestoreFailed) {
		t.Fatalf("RestorePage under fault = %v, want ErrRestoreFailed", err)
	}

	// Verify restore rollback: state must remain PageStateHost and host data intact
	dataHost, stateHost, err := mgr.ReadPage(3001)
	if err != nil || stateHost != PageStateHost || !bytes.Equal(dataHost, raw) {
		t.Fatalf("ReadPage after restore rollback = (%q, %s, %v), want intact host data", dataHost, stateHost, err)
	}
}

// TestKVPageOffloadLRU verifies LRU candidate selection and eviction under memory pressure.
func TestKVPageOffloadLRU(t *testing.T) {
	mgr := NewKVPageOffloader(KVOffloadConfig{MaxDevicePages: 4})

	for i := 1; i <= 5; i++ {
		_, err := mgr.AllocatePage(CUDABlockID(i), 16, 8, []byte(fmt.Sprintf("page-%d", i)))
		if err != nil {
			t.Fatalf("AllocatePage %d failed: %v", i, err)
		}
	}

	// Pin page 1
	if err := mgr.Pin(1); err != nil {
		t.Fatalf("Pin(1) failed: %v", err)
	}

	// Touch page 2 so page 3 becomes older
	if err := mgr.Touch(2); err != nil {
		t.Fatalf("Touch(2) failed: %v", err)
	}

	// Evict 2 LRU pages: page 1 is pinned so skipped; candidates in access order: 3, 4, 5, 2
	evicted, err := mgr.EvictLRU(2)
	if err != nil {
		t.Fatalf("EvictLRU failed: %v", err)
	}

	if len(evicted) != 2 || evicted[0] != 3 || evicted[1] != 4 {
		t.Fatalf("evicted pages = %v, want [3, 4]", evicted)
	}

	// Verify page 3 and 4 are now in host DRAM, others remain in device
	_, s3, _ := mgr.ReadPage(3)
	_, s4, _ := mgr.ReadPage(4)
	_, s1, _ := mgr.ReadPage(1)
	_, s2, _ := mgr.ReadPage(2)
	_, s5, _ := mgr.ReadPage(5)

	if s3 != PageStateHost || s4 != PageStateHost {
		t.Fatalf("pages 3 and 4 should be offloaded, got s3=%s, s4=%s", s3, s4)
	}
	if s1 != PageStateDevice || s2 != PageStateDevice || s5 != PageStateDevice {
		t.Fatalf("pages 1, 2, 5 should remain in device, got s1=%s, s2=%s, s5=%s", s1, s2, s5)
	}
}

// TestKVPageOffloadConcurrent verifies concurrent page allocation, pinning, eviction, and restoration.
func TestKVPageOffloadConcurrent(t *testing.T) {
	mgr := NewKVPageOffloader(KVOffloadConfig{})
	const numPages = 20
	const numWorkers = 8

	for i := 1; i <= numPages; i++ {
		_, err := mgr.AllocatePage(CUDABlockID(i), 8, 4, []byte(fmt.Sprintf("concurrent-payload-%04d", i)))
		if err != nil {
			t.Fatalf("AllocatePage %d failed: %v", i, err)
		}
	}

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for iter := 0; iter < 25; iter++ {
				pageID := CUDABlockID(1 + (workerID+iter)%numPages)

				// Pin and read
				if err := mgr.Pin(pageID); err == nil {
					data, _, readErr := mgr.ReadPage(pageID)
					if readErr == nil && len(data) == 0 {
						t.Errorf("read empty data on pinned page %d", pageID)
					}
					_ = mgr.Unpin(pageID)
				}

				// Try evicting
				_ = mgr.EvictPage(pageID)

				// Try restoring
				_ = mgr.RestorePage(pageID)

				// Touch
				_ = mgr.Touch(pageID)
			}
		}(w)
	}

	wg.Wait()

	// Verify all pages survived uncorrupted
	for i := 1; i <= numPages; i++ {
		ok, err := mgr.VerifyDataIntegrity(CUDABlockID(i))
		if err != nil || !ok {
			t.Fatalf("page %d integrity check failed: ok=%v, err=%v", i, ok, err)
		}
	}
}

// TestKVPageOffloadReclaimPinnedPage verifies that attempting to reclaim a pinned page
// fails closed with ErrPagePinned and does NOT delete the page from tracking (#10722).
func TestKVPageOffloadReclaimPinnedPage(t *testing.T) {
	mgr := NewKVPageOffloader(KVOffloadConfig{})

	raw := []byte("pinned-cannot-reclaim-until-unpinned")
	_, err := mgr.AllocatePage(5001, 16, 8, raw)
	if err != nil {
		t.Fatalf("AllocatePage failed: %v", err)
	}

	if err := mgr.Pin(5001); err != nil {
		t.Fatalf("Pin failed: %v", err)
	}

	// Attempting to reclaim while pinned must fail with ErrPagePinned
	err = mgr.ReclaimPage(5001)
	if !errors.Is(err, ErrPagePinned) {
		t.Fatalf("ReclaimPage pinned = %v, want ErrPagePinned", err)
	}

	// Invariant: The page must STILL be tracked and accessible!
	data, state, err := mgr.ReadPage(5001)
	if err != nil || state != PageStateDevice || !bytes.Equal(data, raw) {
		t.Fatalf("ReadPage after failed reclaim = (%q, %s, %v), want original data intact", data, state, err)
	}

	// Unpin page
	if err := mgr.Unpin(5001); err != nil {
		t.Fatalf("Unpin failed: %v", err)
	}

	// Reclaim now succeeds
	if err := mgr.ReclaimPage(5001); err != nil {
		t.Fatalf("ReclaimPage after unpin failed: %v", err)
	}

	// Page is now gone
	_, _, err = mgr.ReadPage(5001)
	if !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("ReadPage after reclaim = %v, want ErrPageNotFound", err)
	}
}

// TestKVPageOffloadDuplicateAndNotFound verifies duplicate allocation and non-existent page guards.
func TestKVPageOffloadDuplicateAndNotFound(t *testing.T) {
	mgr := NewKVPageOffloader(KVOffloadConfig{})

	_, err := mgr.AllocatePage(6001, 8, 4, []byte("page1"))
	if err != nil {
		t.Fatalf("AllocatePage failed: %v", err)
	}

	// Duplicate allocation fails
	_, err = mgr.AllocatePage(6001, 8, 4, []byte("page1-duplicate"))
	if err == nil {
		t.Fatal("AllocatePage duplicate ID should fail")
	}

	// Non-existent page operations
	missing := CUDABlockID(999999)
	if err := mgr.Pin(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("Pin missing = %v, want ErrPageNotFound", err)
	}
	if err := mgr.Unpin(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("Unpin missing = %v, want ErrPageNotFound", err)
	}
	if _, err := mgr.IsPinned(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("IsPinned missing = %v, want ErrPageNotFound", err)
	}
	if err := mgr.Touch(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("Touch missing = %v, want ErrPageNotFound", err)
	}
	if err := mgr.EvictPage(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("EvictPage missing = %v, want ErrPageNotFound", err)
	}
	if err := mgr.RestorePage(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("RestorePage missing = %v, want ErrPageNotFound", err)
	}
	if err := mgr.ReclaimPage(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("ReclaimPage missing = %v, want ErrPageNotFound", err)
	}
	if _, _, err := mgr.ReadPage(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("ReadPage missing = %v, want ErrPageNotFound", err)
	}
	if _, err := mgr.VerifyDataIntegrity(missing); !errors.Is(err, ErrPageNotFound) {
		t.Fatalf("VerifyDataIntegrity missing = %v, want ErrPageNotFound", err)
	}

	// Unpin on unpinned page
	if err := mgr.Unpin(6001); err == nil {
		t.Fatal("Unpin on unpinned page should return error")
	}
}
