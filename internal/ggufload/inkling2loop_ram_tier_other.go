//go:build !linux

package ggufload

import "errors"

// Without madvise the prefetch relies on its page-touch loop alone.
func inkling2RAMAdvise([][]byte) {}

func inkling2RAMMlock([]byte) error {
	return errors.New("mlock is only supported on linux")
}

// The cgroup clamp only exists in the Linux host probe.
func inkling2CgroupClamped(int64) bool { return false }
