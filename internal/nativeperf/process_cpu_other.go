//go:build !(linux || darwin || freebsd || netbsd || openbsd || dragonfly || windows)

package nativeperf

import "time"

func readProcessCPUTime() (time.Duration, bool) { return 0, false }
