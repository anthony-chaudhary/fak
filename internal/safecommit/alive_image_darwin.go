//go:build darwin

package safecommit

import (
	"strings"

	"golang.org/x/sys/unix"
)

// processImageName resolves a live PID's executable base name (lowercased, without a
// trailing ".exe"), reporting ok=false when the image cannot be read. On Darwin it reads
// the kern.proc.pid sysctl record via unix.SysctlKinfoProc.
func processImageName(pid int) (string, bool) {
	if pid <= 0 {
		return "", false
	}
	info, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil || info == nil {
		return "", false
	}
	var b []byte
	for _, c := range info.Proc.P_comm {
		if c == 0 {
			break
		}
		b = append(b, byte(c))
	}
	name := normalizeImage(string(b))
	if name != "" {
		return name, true
	}
	return "", false
}

func normalizeImage(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	return strings.TrimSuffix(name, ".exe")
}
