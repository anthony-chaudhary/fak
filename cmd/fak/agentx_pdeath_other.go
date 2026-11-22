//go:build !linux

package main

// armParentDeathSignal is a no-op off Linux, which has no PR_SET_PDEATHSIG. The
// ppid verdict still runs, but on OSes that never reparent an orphan (Windows)
// it only catches a --parent-pid that was not the direct parent at launch.
func armParentDeathSignal() error { return nil }
