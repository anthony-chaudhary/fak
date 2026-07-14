//go:build windows

package main

import (
	"os"
	"path/filepath"
)

func machineGuardRegistryDir() string {
	p := filepath.Join(os.Getenv("ProgramData"), "fak", "guard-control", "registry")
	if st, e := os.Stat(p); e == nil && st.IsDir() {
		return p
	}
	return ""
}
