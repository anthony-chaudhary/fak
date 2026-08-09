//go:build !windows

package devindex

import "os/exec"

func graphCommand(name string, args ...string) *exec.Cmd { return exec.Command(name, args...) }
