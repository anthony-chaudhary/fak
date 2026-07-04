//go:build !windows

package procguard

import (
	"errors"
	"fmt"
	"sort"
	"syscall"
)

// killSignal terminates the POSIX process tree rooted at pid. Launchers that
// set pid as a process-group leader take the fast path; older direct PID binds
// fall back to a ps-derived descendant walk.
func killSignal(pid int) error {
	if err := syscall.Kill(-pid, syscall.SIGKILL); err == nil {
		return nil
	} else if !errors.Is(err, syscall.ESRCH) {
		return err
	}

	descendants, collectErr := descendantPIDs(pid)
	for i := len(descendants) - 1; i >= 0; i-- {
		_ = syscall.Kill(descendants[i], syscall.SIGKILL)
	}
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		if collectErr != "" {
			return fmt.Errorf("%w (process-tree scan: %s)", err, collectErr)
		}
		return err
	}
	return nil
}

func descendantPIDs(root int) ([]int, string) {
	procs, errText := collectPOSIXRelations()
	children := map[int][]int{}
	for _, p := range procs {
		if p.PPID == nil {
			continue
		}
		children[*p.PPID] = append(children[*p.PPID], p.PID)
	}
	for ppid := range children {
		sort.Ints(children[ppid])
	}

	var out []int
	var walk func(int)
	walk = func(pid int) {
		for _, child := range children[pid] {
			walk(child)
			out = append(out, child)
		}
	}
	walk(root)
	return out, errText
}
