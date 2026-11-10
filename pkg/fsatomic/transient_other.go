//go:build !windows

package fsatomic

func isTransientRenameError(error) bool { return false }
