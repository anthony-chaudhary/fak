//go:build !windows

package main

func installGuardHostTerminationObserver(string) error { return nil }
