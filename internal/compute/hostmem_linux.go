//go:build linux

package compute

import "syscall"

func hostSystemMemory() (total, free int64, known bool) {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return 0, FreeUnknown, false
	}
	unit := uint64(si.Unit)
	if unit == 0 {
		unit = 1
	}
	total = scaleHostMem(uint64(si.Totalram), unit)
	free = scaleHostMem(uint64(si.Freeram), unit)
	return total, free, total > 0
}

func scaleHostMem(v, unit uint64) int64 {
	if unit == 0 {
		unit = 1
	}
	if v > maxInt64Uint64/unit {
		return int64(maxInt64Uint64)
	}
	return int64(v * unit)
}
