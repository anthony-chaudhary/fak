package compute

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// KVPageOffloadState represents the lifecycle state of a page in the KV cache offload hierarchy (#10722).
type KVPageOffloadState string

const (
	// PageStateDevice indicates the page resides in device memory (VRAM) and is ready for GPU kernels.
	PageStateDevice KVPageOffloadState = "DEVICE"
	// PageStateStaging indicates the page is being transactionally copied to host DRAM before release.
	PageStateStaging KVPageOffloadState = "STAGING"
	// PageStateHost indicates the page has been successfully offloaded to host DRAM and freed from device VRAM.
	PageStateHost KVPageOffloadState = "HOST"
	// PageStateRestoring indicates the page is being transferred back from host DRAM to device VRAM.
	PageStateRestoring KVPageOffloadState = "RESTORING"
	// PageStateReclaimed indicates the page has been permanently freed and reclaimed.
	PageStateReclaimed KVPageOffloadState = "RECLAIMED"
)

var (
	// ErrPagePinned is returned when attempting to evict a page whose pin count is positive.
	ErrPagePinned = errors.New("cannot evict pinned KV page")
	// ErrPageNotFound is returned when operating on a non-existent page ID.
	ErrPageNotFound = errors.New("KV page not found")
	// ErrPageNotResident is returned when evicting a page that is not currently device-resident.
	ErrPageNotResident = errors.New("KV page is not resident on device")
	// ErrPageNotOffloaded is returned when restoring a page that is not currently staged on host.
	ErrPageNotOffloaded = errors.New("KV page is not offloaded on host")
	// ErrStagingFailed is returned when transactional staging to host RAM encounters an error.
	ErrStagingFailed = errors.New("transactional staging to host RAM failed")
	// ErrRestoreFailed is returned when restoring a page to device memory fails.
	ErrRestoreFailed = errors.New("restoration from host RAM to device memory failed")
	// ErrChecksumMismatch is returned if data corruption is detected during transfer.
	ErrChecksumMismatch = errors.New("checksum mismatch during KV page transfer")
)

// KVOffloadPage represents a single page of KV cache data subject to page-granular offloading (#10722).
type KVOffloadPage struct {
	mu          sync.RWMutex
	ID          CUDABlockID
	LayerCount  int
	Tokens      int
	SizeBytes   int64
	DeviceData  []byte
	HostData    []byte
	State       KVPageOffloadState
	pinCount    int32
	lastAccess  int64 // monotonic logical timestamp for LRU
	checksum    [32]byte
	allocatedAt time.Time
}

// PinCount returns the current number of active pins on this page.
func (p *KVOffloadPage) PinCount() int {
	return int(atomic.LoadInt32(&p.pinCount))
}

// IsPinned reports whether the page is currently pinned.
func (p *KVOffloadPage) IsPinned() bool {
	return atomic.LoadInt32(&p.pinCount) > 0
}

// KVOffloadConfig configures the transactional page-granular KV offload manager.
type KVOffloadConfig struct {
	MaxDevicePages int   // Max pages resident in device memory before eviction triggers
	MaxHostPages   int   // Max pages staged in host memory
	PageSizeBytes  int64 // Nominal byte size per page
}

// KVPageOffloadStats provides snapshot observability into the offload manager.
type KVPageOffloadStats struct {
	TotalPages       int   `json:"total_pages"`
	DevicePages      int   `json:"device_pages"`
	HostPages        int   `json:"host_pages"`
	PinnedPages      int   `json:"pinned_pages"`
	OffloadedCount   int64 `json:"offloaded_count"`
	RestoredCount    int64 `json:"restored_count"`
	FailedEvictions  int64 `json:"failed_evictions"`
	FailedRestores   int64 `json:"failed_restores"`
	DeviceBytesAlloc int64 `json:"device_bytes_alloc"`
	HostBytesAlloc   int64 `json:"host_bytes_alloc"`
}

// KVPageOffloader coordinates transactional page-granular CPU offloading for in-kernel KV cache (#10722, epic #2236).
// It ensures that:
// 1. Pages are staged and verified in host RAM BEFORE device physical VRAM is released.
// 2. Pinned pages are strictly protected from eviction.
// 3. Any failure during host staging or restoration aborts transactionally, keeping state consistent.
// 4. Memory reclaim and LRU candidate selection operate deterministically.
type KVPageOffloader struct {
	mu           sync.RWMutex
	cfg          KVOffloadConfig
	pages        map[CUDABlockID]*KVOffloadPage
	clockSeq     int64
	offloadedCnt int64
	restoredCnt  int64
	failedEvict  int64
	failedRest   int64

	// Fault injection hooks for testing transactional rollback
	onStageHook   func(page *KVOffloadPage) error
	onRestoreHook func(page *KVOffloadPage) error
}

// NewKVPageOffloader creates an initialized transactional KV page offload manager.
func NewKVPageOffloader(cfg KVOffloadConfig) *KVPageOffloader {
	if cfg.MaxDevicePages <= 0 {
		cfg.MaxDevicePages = 1024
	}
	if cfg.MaxHostPages <= 0 {
		cfg.MaxHostPages = 4096
	}
	return &KVPageOffloader{
		cfg:   cfg,
		pages: make(map[CUDABlockID]*KVOffloadPage),
	}
}

// AllocatePage creates a new device-resident KV page and tracks it in the offloader.
func (o *KVPageOffloader) AllocatePage(id CUDABlockID, layers, tokens int, data []byte) (*KVOffloadPage, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if _, exists := o.pages[id]; exists {
		return nil, fmt.Errorf("KV page %d already allocated", id)
	}

	o.clockSeq++
	devCopy := append([]byte(nil), data...)
	csum := sha256.Sum256(devCopy)

	page := &KVOffloadPage{
		ID:          id,
		LayerCount:  layers,
		Tokens:      tokens,
		SizeBytes:   int64(len(devCopy)),
		DeviceData:  devCopy,
		State:       PageStateDevice,
		lastAccess:  o.clockSeq,
		checksum:    csum,
		allocatedAt: time.Now(),
	}

	o.pages[id] = page
	return page, nil
}

// Pin increments the pin count of a page, preventing eviction while in use by kernels.
func (o *KVPageOffloader) Pin(id CUDABlockID) error {
	o.mu.RLock()
	page, ok := o.pages[id]
	o.mu.RUnlock()

	if !ok {
		return ErrPageNotFound
	}

	page.mu.Lock()
	defer page.mu.Unlock()

	if page.State == PageStateReclaimed {
		return ErrPageNotFound
	}

	atomic.AddInt32(&page.pinCount, 1)
	return nil
}

// Unpin decrements the pin count of a page, allowing eviction when pin count reaches 0.
func (o *KVPageOffloader) Unpin(id CUDABlockID) error {
	o.mu.RLock()
	page, ok := o.pages[id]
	o.mu.RUnlock()

	if !ok {
		return ErrPageNotFound
	}

	page.mu.Lock()
	defer page.mu.Unlock()

	cur := atomic.LoadInt32(&page.pinCount)
	if cur <= 0 {
		return fmt.Errorf("cannot unpin unpinned KV page %d", id)
	}

	atomic.AddInt32(&page.pinCount, -1)
	return nil
}

// IsPinned checks if a page is currently pinned.
func (o *KVPageOffloader) IsPinned(id CUDABlockID) (bool, error) {
	o.mu.RLock()
	page, ok := o.pages[id]
	o.mu.RUnlock()

	if !ok {
		return false, ErrPageNotFound
	}

	return page.IsPinned(), nil
}

// Touch updates the logical access timestamp of a page for LRU tracking.
func (o *KVPageOffloader) Touch(id CUDABlockID) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	page, ok := o.pages[id]
	if !ok {
		return ErrPageNotFound
	}

	o.clockSeq++
	page.mu.Lock()
	page.lastAccess = o.clockSeq
	page.mu.Unlock()
	return nil
}

// EvictPage transactionally offloads a single page from device VRAM to host DRAM (#10722).
// Invariant: Staging to host RAM must be verified and committed before device memory is released.
// If staging fails, device memory is untouched and state is rolled back.
func (o *KVPageOffloader) EvictPage(id CUDABlockID) error {
	o.mu.Lock()
	page, ok := o.pages[id]
	if !ok {
		o.mu.Unlock()
		return ErrPageNotFound
	}
	o.mu.Unlock()

	page.mu.Lock()
	defer page.mu.Unlock()

	if page.State != PageStateDevice {
		atomic.AddInt64(&o.failedEvict, 1)
		return ErrPageNotResident
	}

	if atomic.LoadInt32(&page.pinCount) > 0 {
		atomic.AddInt64(&o.failedEvict, 1)
		return ErrPagePinned
	}

	// Transaction Begin: Transition to STAGING
	origState := page.State
	page.State = PageStateStaging

	// 1. Stage device data to host DRAM
	hostStaging := append([]byte(nil), page.DeviceData...)

	// Verify checksum of staged buffer
	stagedSum := sha256.Sum256(hostStaging)
	if stagedSum != page.checksum {
		// Rollback
		page.State = origState
		atomic.AddInt64(&o.failedEvict, 1)
		return fmt.Errorf("%w: staged host data corrupted", ErrChecksumMismatch)
	}

	// Check optional fault injection hook
	if o.onStageHook != nil {
		if err := o.onStageHook(page); err != nil {
			// Rollback to device resident
			page.State = origState
			atomic.AddInt64(&o.failedEvict, 1)
			return fmt.Errorf("%w: %v", ErrStagingFailed, err)
		}
	}

	// 2. Transaction Commit: Assign host data and release device VRAM
	page.HostData = hostStaging
	page.DeviceData = nil
	page.State = PageStateHost

	atomic.AddInt64(&o.offloadedCnt, 1)
	return nil
}

// RestorePage transactionally restores a page from host DRAM back to device VRAM (#10722).
// Invariant: Host DRAM is retained until device allocation and copy are verified.
func (o *KVPageOffloader) RestorePage(id CUDABlockID) error {
	o.mu.Lock()
	page, ok := o.pages[id]
	if !ok {
		o.mu.Unlock()
		return ErrPageNotFound
	}
	o.mu.Unlock()

	page.mu.Lock()
	defer page.mu.Unlock()

	if page.State != PageStateHost {
		atomic.AddInt64(&o.failedRest, 1)
		return ErrPageNotOffloaded
	}

	// Transaction Begin: Transition to RESTORING
	origState := page.State
	page.State = PageStateRestoring

	// 1. Allocate device VRAM and copy from host
	devRestored := append([]byte(nil), page.HostData...)

	// Verify checksum of restored buffer
	restoredSum := sha256.Sum256(devRestored)
	if restoredSum != page.checksum {
		// Rollback
		page.State = origState
		atomic.AddInt64(&o.failedRest, 1)
		return fmt.Errorf("%w: restored device data corrupted", ErrChecksumMismatch)
	}

	// Check optional fault injection hook
	if o.onRestoreHook != nil {
		if err := o.onRestoreHook(page); err != nil {
			// Rollback to host offloaded
			page.State = origState
			atomic.AddInt64(&o.failedRest, 1)
			return fmt.Errorf("%w: %v", ErrRestoreFailed, err)
		}
	}

	// 2. Transaction Commit: Assign device data, release host DRAM, set DEVICE state
	page.DeviceData = devRestored
	page.HostData = nil
	page.State = PageStateDevice

	atomic.AddInt64(&o.restoredCnt, 1)
	return nil
}

// EvictLRU finds and transactionally evicts unpinned resident pages up to targetCount.
// Returns the slice of successfully evicted page IDs.
func (o *KVPageOffloader) EvictLRU(targetCount int) ([]CUDABlockID, error) {
	if targetCount <= 0 {
		return nil, nil
	}

	o.mu.Lock()
	// Collect candidate unpinned device pages
	type candidate struct {
		id         CUDABlockID
		lastAccess int64
	}
	var candidates []candidate

	for id, p := range o.pages {
		p.mu.RLock()
		if p.State == PageStateDevice && atomic.LoadInt32(&p.pinCount) == 0 {
			candidates = append(candidates, candidate{id: id, lastAccess: p.lastAccess})
		}
		p.mu.RUnlock()
	}
	o.mu.Unlock()

	// Sort candidates by lastAccess ascending (oldest first), breaking ties by ID deterministically
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].lastAccess == candidates[j].lastAccess {
			return candidates[i].id < candidates[j].id
		}
		return candidates[i].lastAccess < candidates[j].lastAccess
	})

	var evicted []CUDABlockID
	for _, c := range candidates {
		if len(evicted) >= targetCount {
			break
		}
		if err := o.EvictPage(c.id); err == nil {
			evicted = append(evicted, c.id)
		}
	}

	return evicted, nil
}

// ReclaimPage permanently frees a page from either device or host memory.
func (o *KVPageOffloader) ReclaimPage(id CUDABlockID) error {
	o.mu.Lock()
	page, ok := o.pages[id]
	if !ok {
		o.mu.Unlock()
		return ErrPageNotFound
	}

	page.mu.Lock()
	defer page.mu.Unlock()

	if atomic.LoadInt32(&page.pinCount) > 0 {
		o.mu.Unlock()
		return ErrPagePinned
	}

	delete(o.pages, id)
	o.mu.Unlock()

	page.DeviceData = nil
	page.HostData = nil
	page.State = PageStateReclaimed
	return nil
}

// ReadPage retrieves page data, returning bytes, residency state, and any error.
func (o *KVPageOffloader) ReadPage(id CUDABlockID) ([]byte, KVPageOffloadState, error) {
	o.mu.RLock()
	page, ok := o.pages[id]
	o.mu.RUnlock()

	if !ok {
		return nil, "", ErrPageNotFound
	}

	page.mu.RLock()
	defer page.mu.RUnlock()

	switch page.State {
	case PageStateDevice:
		return append([]byte(nil), page.DeviceData...), PageStateDevice, nil
	case PageStateHost:
		return append([]byte(nil), page.HostData...), PageStateHost, nil
	default:
		return nil, page.State, fmt.Errorf("page %d in transient state %s", id, page.State)
	}
}

// VerifyDataIntegrity validates that page data matches its original checksum.
func (o *KVPageOffloader) VerifyDataIntegrity(id CUDABlockID) (bool, error) {
	o.mu.RLock()
	page, ok := o.pages[id]
	o.mu.RUnlock()

	if !ok {
		return false, ErrPageNotFound
	}

	page.mu.RLock()
	defer page.mu.RUnlock()

	var data []byte
	switch page.State {
	case PageStateDevice:
		data = page.DeviceData
	case PageStateHost:
		data = page.HostData
	default:
		return false, fmt.Errorf("page %d in transient state %s", id, page.State)
	}

	actual := sha256.Sum256(data)
	return actual == page.checksum, nil
}

// Stats returns a snapshot of memory usage and offload metrics.
func (o *KVPageOffloader) Stats() KVPageOffloadStats {
	o.mu.RLock()
	defer o.mu.RUnlock()

	var stats KVPageOffloadStats
	stats.TotalPages = len(o.pages)
	stats.OffloadedCount = atomic.LoadInt64(&o.offloadedCnt)
	stats.RestoredCount = atomic.LoadInt64(&o.restoredCnt)
	stats.FailedEvictions = atomic.LoadInt64(&o.failedEvict)
	stats.FailedRestores = atomic.LoadInt64(&o.failedRest)

	for _, p := range o.pages {
		p.mu.RLock()
		switch p.State {
		case PageStateDevice:
			stats.DevicePages++
			stats.DeviceBytesAlloc += p.SizeBytes
		case PageStateHost:
			stats.HostPages++
			stats.HostBytesAlloc += p.SizeBytes
		}
		if atomic.LoadInt32(&p.pinCount) > 0 {
			stats.PinnedPages++
		}
		p.mu.RUnlock()
	}

	return stats
}

// Config returns the active offload configuration.
func (o *KVPageOffloader) Config() KVOffloadConfig {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.cfg
}
