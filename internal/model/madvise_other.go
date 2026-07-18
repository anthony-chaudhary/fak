//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd

package model

// madviseWillneed is the no-op fallback on platforms without a stdlib madvise (Windows,
// solaris/illumos, js/wasm, plan9, aix). The demand-paged read path stays correct everywhere;
// only the readahead/compute overlap is forfeited. It always reports false.
func madviseWillneed(data []byte, off, length int) bool { return false }
