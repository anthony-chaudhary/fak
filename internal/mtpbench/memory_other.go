//go:build !darwin && !linux

package mtpbench

import "errors"

func observeMemory() (MemorySnapshot, error) {
	return MemorySnapshot{}, errors.New("mtpbench: current RSS and swap unavailable")
}
