//go:build !windows && !plan9

package codetools

import (
	"fmt"
	"os"
	"syscall"
)

func fileIdentity(_ *os.File, info os.FileInfo) (string, error) {
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return "", fmt.Errorf("filesystem identity is unavailable")
	}
	return fmt.Sprintf("%v:%v", st.Dev, st.Ino), nil
}
