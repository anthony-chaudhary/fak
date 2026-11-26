//go:build !windows

package main

import "os"

func truncateServeArtifactFixture(path string, size int64) error {
	return os.Truncate(path, size)
}
