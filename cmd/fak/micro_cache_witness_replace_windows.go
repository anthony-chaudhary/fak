//go:build windows

package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	microCacheKernel32    = syscall.NewLazyDLL("kernel32.dll")
	microCacheMoveFileExW = microCacheKernel32.NewProc("MoveFileExW")
)

const (
	microCacheMovefileReplaceExisting = 0x1
	microCacheMovefileWriteThrough    = 0x8
)

func replaceMicroCacheWitness(src, dst string) error {
	srcp, err := syscall.UTF16PtrFromString(src)
	if err != nil {
		return err
	}
	dstp, err := syscall.UTF16PtrFromString(dst)
	if err != nil {
		return err
	}
	r, _, callErr := microCacheMoveFileExW.Call(
		uintptr(unsafe.Pointer(srcp)),
		uintptr(unsafe.Pointer(dstp)),
		microCacheMovefileReplaceExisting|microCacheMovefileWriteThrough,
	)
	if r == 0 {
		return fmt.Errorf("replace cache witness: %w", callErr)
	}
	return nil
}
