//go:build !linux

package alloc

import "fmt"

// regionBackingCap reports the largest backing a single Region may occupy on
// this platform. Non-Linux regions are heap-backed and capped at 64MB (the dev
// ceiling); 0 means uncapped (Linux mmaps the full requested size).
func regionBackingCap() uint64 { return 64 << 20 }

func (r *Region) allocate() error {
	allocSize := r.size
	if cap := regionBackingCap(); cap > 0 && allocSize > cap {
		allocSize = cap
	}
	r.data = make([]byte, allocSize)
	// Keep the logical size in lock-step with the real backing array. The
	// bitmap/offset slot math is derived from Size(), so a logical size larger
	// than the 64MB dev ceiling would let SlotData slice past the array
	// (slice bounds out of range). On a full-size (Linux) region the requested
	// size and the backing length are already equal.
	r.size = allocSize
	r.isMapped = false
	return nil
}

func (r *Region) munmap() error {
	r.data = nil
	return nil
}

// TryMlockall is a no-op on non-Linux platforms.
func TryMlockall() error {
	return nil
}

// PinPages is a no-op on non-Linux platforms.
func (r *Region) PinPages() error {
	return nil
}

// NewDevdaxRegion is not supported on non-Linux platforms.
func NewDevdaxRegion(devPath string, offset, size uint64) (*Region, error) {
	return nil, fmt.Errorf("devdax regions require Linux")
}

// queryDevdaxCapacityImpl is not supported on non-Linux platforms.
func queryDevdaxCapacityImpl(devPath string) (uint64, error) {
	return 0, fmt.Errorf("devdax capacity query requires Linux")
}
