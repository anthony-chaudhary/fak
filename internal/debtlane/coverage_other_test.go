//go:build !windows

package debtlane

import "os"

func lockFileExclusively(path string) func() {
	_ = os.Chmod(path, 0000)
	return func() {
		_ = os.Chmod(path, 0644)
	}
}
