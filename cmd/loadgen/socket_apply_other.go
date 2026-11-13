//go:build !unix

package main

// applySocketOptions is a no-op on non-unix platforms. Windows deliberately
// skips the upstream SO_RCVBUF/SO_SNDBUF sizing (it disables TCP Auto-Tuning)
// and has no SO_KEEPALIVE tuning of the kind upstream sets, so there is nothing
// to do here; the dialer still runs it so the process is uniformly wired.
func applySocketOptions(fd uintptr, plan socketPlan) error { return nil }
