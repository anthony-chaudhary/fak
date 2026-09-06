//go:build !linux && !windows

package l3kv

import "os"

func deallocateFileRangeOS(file *os.File, offset, length int64) error {
	return fallbackDeallocate(file, offset, length)
}
