//go:build !windows

package safecommit

import (
	"os"
	"path/filepath"
	"strings"
)

// processImageName resolves a live PID's executable base name (lowercased, without a
// trailing ".exe"), reporting ok=false when the image cannot be read. It powers the
// PID-reuse guard: a lock whose recorded PID is alive but whose image is provably not a
// committer is a reused PID number, safe to break (issue #2339).
//
// On Linux it reads /proc/<pid>/exe (the definitive image path); if that symlink is
// unreadable (a permission boundary, or a platform without procfs such as macOS) it
// falls back to /proc/<pid>/comm. When neither is available it returns ok=false, so an
// unidentifiable live holder is treated as committer-like and never reaped.
func processImageName(pid int) (string, bool) {
	if pid <= 0 {
		return "", false
	}
	proc := "/proc/" + itoa(pid)
	if target, err := os.Readlink(proc + "/exe"); err == nil {
		return normalizeImage(filepath.Base(target)), true
	}
	if data, err := os.ReadFile(proc + "/comm"); err == nil {
		if name := normalizeImage(strings.TrimSpace(string(data))); name != "" {
			return name, true
		}
	}
	return "", false
}

func normalizeImage(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	return strings.TrimSuffix(name, ".exe")
}

// itoa avoids strconv just for a positive pid in the /proc path.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
