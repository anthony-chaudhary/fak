//go:build darwin

package compute

import (
	"encoding/binary"
	"syscall"
)

func hostSystemMemory() (total, free int64, known bool) {
	raw, err := syscall.Sysctl("hw.memsize")
	if err != nil || raw == "" {
		return 0, FreeUnknown, false
	}
	var buf [8]byte
	copy(buf[:], raw)
	totalBytes := binary.LittleEndian.Uint64(buf[:])
	if totalBytes == 0 {
		return 0, FreeUnknown, false
	}
	return uint64ToCapInt64(totalBytes), FreeUnknown, true
}
