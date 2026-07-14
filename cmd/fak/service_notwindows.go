//go:build !windows

package main

import "io"

func runWindowsServiceDispatcher(io.Writer, io.Writer) int { return 2 }
