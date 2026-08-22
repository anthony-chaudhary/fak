//go:build plan9

package codetools

import (
	"fmt"
	"os"
	"syscall"
)

func fileIdentity(_ *os.File, info os.FileInfo) (string, error) {
	dir, ok := info.Sys().(*syscall.Dir)
	if !ok {
		return "", fmt.Errorf("filesystem identity is unavailable")
	}
	return fmt.Sprintf("%d:%d:%d", dir.Qid.Path, dir.Qid.Vers, dir.Qid.Type), nil
}
