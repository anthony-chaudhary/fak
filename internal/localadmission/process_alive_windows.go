//go:build windows

package localadmission

import "os"

func processAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	return err == nil && p != nil
}
