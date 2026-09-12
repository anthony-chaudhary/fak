//go:build !windows

package codetools

import "os"

func renameReplacing(oldPath, newPath string) error {
	return os.Rename(oldPath, newPath)
}
