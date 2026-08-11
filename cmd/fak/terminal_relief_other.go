//go:build !windows

package main

import (
	"fmt"
	"os"
)

func cmdTerminalRelief([]string) {
	fmt.Fprintln(os.Stderr, "fak terminal-relief: Windows only")
	os.Exit(1)
}
