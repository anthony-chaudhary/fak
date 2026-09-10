//go:build !amd64

package strix

import "unsafe"

func streamCopyAVX512Kernel(dst, src unsafe.Pointer, chunks int64)    {}
func streamCopyAVX512StoreOnly(dst, src unsafe.Pointer, chunks int64) {}
func streamCopyAVX512LoadOnly(dst, src unsafe.Pointer, chunks int64)  {}
func sfenceAsm()                                                      {}

func canExecuteAVX512() bool {
	return false
}

func hardwareHasAVX512() bool {
	return false
}
