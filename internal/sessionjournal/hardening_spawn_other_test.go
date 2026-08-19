//go:build !windows

package sessionjournal

import "os/exec"

func hideAppendHelperWindow(cmd *exec.Cmd) {}
