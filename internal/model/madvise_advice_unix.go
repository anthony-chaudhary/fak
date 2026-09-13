//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package model

import "golang.org/x/sys/unix"

// madvise_advice_unix.go - the OS shim that translates the model-local PageAdvice vocabulary
// (#1302) into a real madvise constant and issues it over a mapped expert range. It mirrors
// madviseWillneed (madvise_unix.go) in every discipline: the start is rounded DOWN to a page
// boundary because madvise requires a page-aligned address, every error is swallowed (a failed
// hint only forfeits an IO optimization, never correctness), and the return is only whether a
// hint was issued. It goes through golang.org/x/sys/unix rather than the stdlib syscall because
// syscall.Madvise is undefined on several tagged targets (notably darwin/arm64), whereas
// unix.Madvise is uniform across every platform in the build constraint above.
func madviseAdvice(data []byte, off, length int, advice PageAdvice) bool {
	if length <= 0 || off < 0 || off >= len(data) {
		return false
	}
	end := off + length
	if end > len(data) {
		end = len(data)
	}
	if page := unix.Getpagesize(); page > 0 {
		off -= off % page // round the start down to a page boundary (madvise requires it)
	}
	var kind int
	switch advice {
	case PageAdviceRandom:
		kind = unix.MADV_RANDOM
	case PageAdviceSequential:
		kind = unix.MADV_SEQUENTIAL
	default:
		kind = unix.MADV_NORMAL
	}
	if err := unix.Madvise(data[off:end], kind); err != nil {
		return false
	}
	return true
}
