//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd

package model

// madvise_advice_other.go - the no-op fallback for platforms without madvise (Windows,
// solaris/illumos, js/wasm, plan9, aix). It mirrors madviseWillneed's stub: the aligned
// read-through path stays byte-for-byte correct everywhere, only the kernel access hint is
// forfeited. It always reports false.
func madviseAdvice(data []byte, off, length int, advice PageAdvice) bool { return false }
