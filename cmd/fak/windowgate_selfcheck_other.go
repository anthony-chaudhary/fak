//go:build !windows

package main

import (
	"io"
	"runtime"
)

func runDesktopConsoleSelfcheckChild(io.Writer, io.Writer) (int, bool) { return 0, false }

func runDesktopConsoleSelfcheck(stdout, stderr io.Writer, asJSON bool) int {
	rep := desktopConsoleSelfcheckReport{
		Schema:     desktopConsoleSelfcheckSchema,
		OK:         true,
		Applicable: false,
		Platform:   runtime.GOOS,
		Reason:     "Windows console-window semantics are not applicable on this host",
	}
	if err := writeDesktopConsoleSelfcheck(stdout, rep, asJSON); err != nil {
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	return 0
}
