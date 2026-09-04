//go:build !linux

package alloc

import "fmt"

func (r *Region) allocate() error {
	r.data = make([]byte, r.size)
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
