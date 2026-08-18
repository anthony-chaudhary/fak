// Package exclusivefile creates process marker files atomically.
package exclusivefile

import (
	"fmt"
	"os"
	"time"
)

// CreatePIDTime atomically creates path and records the current PID and Unix time.
func CreatePIDTime(path string) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(file, "%d %d\n", os.Getpid(), time.Now().Unix())
	return file.Close()
}
