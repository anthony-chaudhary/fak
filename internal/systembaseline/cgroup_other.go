//go:build !linux && !windows

package systembaseline

import "syscall"

type unavailableCommandAttributor struct {
	result CgroupV2
}

func newCommandAttributorPlatform() commandAttributorPlatform {
	return &unavailableCommandAttributor{result: unavailableCgroup("cgroup v2 attribution is unavailable on this platform")}
}

func (u *unavailableCommandAttributor) configureSysProcAttr(attr *syscall.SysProcAttr) (*syscall.SysProcAttr, bool) {
	return attr, false
}
func (u *unavailableCommandAttributor) active() bool       { return false }
func (u *unavailableCommandAttributor) started(int) error  { return nil }
func (u *unavailableCommandAttributor) launchFailed(error) {}
func (u *unavailableCommandAttributor) finish() CgroupV2   { return u.result.clone() }
