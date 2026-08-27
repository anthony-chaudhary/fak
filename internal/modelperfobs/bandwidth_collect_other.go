//go:build !darwin && !linux && !windows

package modelperfobs

import "time"

func collectHostSnapshot() (hostSnapshot, error) {
	return hostSnapshot{at: time.Now(), collector: "portable-runtime"}, nil
}
