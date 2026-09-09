//go:build linux

package mtpbench

import (
	"errors"
	"os"
	"strconv"
	"strings"
)

func observeMemory() (MemorySnapshot, error) {
	raw, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return MemorySnapshot{}, err
	}
	fields := strings.Fields(string(raw))
	if len(fields) < 2 {
		return MemorySnapshot{}, errors.New("mtpbench: malformed /proc/self/statm")
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		return MemorySnapshot{}, err
	}
	return MemorySnapshot{CurrentRSSBytes: pages * uint64(os.Getpagesize()), RSSAvailable: true, SwapAvailable: false}, nil
}
