//go:build !windows

package main

import (
	"os"
	"path/filepath"
	"runtime"
)

// machineGuardRegistryDir returns the OS-service registry only when it has
// already been installed. Interactive Guard processes merely append lifecycle
// facts; the PID-1 service remains the policy/recovery owner.
func machineGuardRegistryDir() string {
	var p string
	switch runtime.GOOS {
	case "linux":
		p = filepath.Join(string(filepath.Separator), "var", "lib", "fak", "registry")
	case "darwin":
		p = filepath.Join(string(filepath.Separator), "var", "db", "fak", "registry")
	default:
		return ""
	}
	if st, err := os.Stat(p); err == nil && st.IsDir() {
		return p
	}
	return ""
}
