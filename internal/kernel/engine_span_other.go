//go:build !windows

package kernel

import "time"

type measuredSpanClock struct{ started time.Time }

func startMeasuredSpan() measuredSpanClock { return measuredSpanClock{started: time.Now()} }

func (c measuredSpanClock) elapsedNanos() int64 { return time.Since(c.started).Nanoseconds() }
