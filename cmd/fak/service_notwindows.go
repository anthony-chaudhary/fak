//go:build !windows

package main

import "io"

func runWindowsServiceDispatcher(io.Writer, io.Writer, ...string) int { return 2 }
func windowsServiceAction(string, io.Writer, io.Writer, bool, ...string) (serviceResult, int) {
	return serviceResult{}, 2
}
