//go:build !windows

package fleetmetrics

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
