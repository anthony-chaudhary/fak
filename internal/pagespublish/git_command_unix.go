//go:build !windows

package pagespublish

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
