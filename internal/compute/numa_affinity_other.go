//go:build !(linux && amd64)

package compute

import "errors"

// PinCurrentThreadToCPUs has no affinity syscall off linux/amd64. It refuses rather than
// pretending to pin, so a caller can never infer node locality it did not get; decode stays
// correct and simply runs unpinned.
func PinCurrentThreadToCPUs(cpus []int) (unpin func(), err error) {
	return func() {}, errors.New("compute: thread affinity unsupported on this platform")
}
