//go:build !windows

package main

func prepareInfoTerminalInput(_ int) (func(), error) {
	return func() {}, nil
}
