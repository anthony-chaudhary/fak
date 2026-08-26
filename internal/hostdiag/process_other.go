//go:build !windows

package hostdiag

import "time"

func processImage(int) (string, bool)        { return "", false }
func processStartedAt(int) (time.Time, bool) { return time.Time{}, false }
