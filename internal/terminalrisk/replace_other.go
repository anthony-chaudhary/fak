//go:build !windows

package terminalrisk

import "os"

func replace(src, dst string) error { return os.Rename(src, dst) }
