//go:build !linux && !windows

package systembaseline

import "time"

func capturePlatform() Snapshot {
	return Snapshot{At: time.Now().UTC(), ProcessNote: "host and process resource axes unsupported on this platform"}
}
