package main

import (
	"fmt"
	"unicode/utf16"
)

// CreateProcessW accepts at most 32,767 UTF-16 code units including the
// terminating NUL. Keep the check next to the guarded-child spawn chokepoint so
// every brokered launch inherits the same structural ceiling.
const guardWindowsCommandLineLimit = 32767

func guardWindowsCommandLineUnits(command []string) int {
	total := 1
	for i, arg := range command {
		if i > 0 {
			total++
		}
		total += guardWindowsQuotedArgUnits(arg)
	}
	return total
}
func guardWindowsArgvPreflight(command []string, goos string) error {
	if goos != "windows" {
		return nil
	}
	total := guardWindowsCommandLineUnits(command)
	longestIndex, longestBytes := -1, 0
	for i, arg := range command {
		if len(arg) > longestBytes {
			longestIndex, longestBytes = i, len(arg)
		}
	}
	if total < guardWindowsCommandLineLimit {
		return nil
	}
	return fmt.Errorf("guarded child argv exceeds Windows command-line ceiling: assembled=%d UTF-16 units limit=%d; largest arg[%d]=%d bytes", total, guardWindowsCommandLineLimit, longestIndex, longestBytes)
}

// guardWindowsQuotedArgUnits mirrors the CommandLineToArgvW quoting contract
// used by os/exec: spaces require surrounding quotes, backslashes before a
// quote are doubled, and trailing backslashes are doubled before the close.
func guardWindowsQuotedArgUnits(arg string) int {
	units := utf16.Encode([]rune(arg))
	if len(units) == 0 {
		return 2 // ""
	}
	needsQuotes := false
	for _, u := range units {
		if u == ' ' || u == '\t' || u == '"' {
			needsQuotes = true
			break
		}
	}
	if !needsQuotes {
		return len(units)
	}
	total := 2 // surrounding quotes
	backslashes := 0
	for _, u := range units {
		switch u {
		case '\\':
			backslashes++
		case '"':
			total += 2*backslashes + 2
			backslashes = 0
		default:
			total += backslashes + 1
			backslashes = 0
		}
	}
	return total + 2*backslashes
}
