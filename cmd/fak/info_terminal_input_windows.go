//go:build windows

package main

import "golang.org/x/sys/windows"

// prepareInfoTerminalInput keeps Windows console input on the VT byte stream that
// infoInputScanner consumes. x/term MakeRaw enables VT input but deliberately leaves
// ENABLE_MOUSE_INPUT untouched; when that flag is set, Windows reports mouse clicks as
// INPUT_RECORD values instead of the SGR bytes requested with DECSET 1000/1006. os.Stdin.Read
// cannot decode those records, so the overlay looks alive while every click is inert.
func prepareInfoTerminalInput(fd int) (func(), error) {
	h := windows.Handle(fd)
	var previous uint32
	if err := windows.GetConsoleMode(h, &previous); err != nil {
		return nil, err
	}
	mode := infoTerminalInputMode(previous)
	if err := windows.SetConsoleMode(h, mode); err != nil {
		return nil, err
	}
	return func() { _ = windows.SetConsoleMode(h, previous) }, nil
}

func infoTerminalInputMode(mode uint32) uint32 {
	return (mode | windows.ENABLE_VIRTUAL_TERMINAL_INPUT | windows.ENABLE_EXTENDED_FLAGS) &^
		(windows.ENABLE_MOUSE_INPUT | windows.ENABLE_QUICK_EDIT_MODE)
}
