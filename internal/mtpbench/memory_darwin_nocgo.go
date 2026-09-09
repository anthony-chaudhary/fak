//go:build darwin && !cgo

package mtpbench

import "errors"

func observeMemory() (MemorySnapshot, error) {
	return MemorySnapshot{}, errors.New("mtpbench: current RSS unavailable without cgo")
}
