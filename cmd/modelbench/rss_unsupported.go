//go:build !linux && !darwin && !windows

package main

import "fmt"

func peakRSSBytes() (uint64, error) {
	return 0, fmt.Errorf("peak RSS is not implemented on this platform")
}
