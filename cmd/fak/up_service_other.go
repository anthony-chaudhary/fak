//go:build !darwin

package main

import "errors"

// spawnUpServiceLapseWaker is darwin-only: the service verbs refuse with
// NOT_SUPPORTED before they could reach it elsewhere.
func spawnUpServiceLapseWaker(string, []string, string) (int, error) {
	return 0, errors.New("the `fak up off --for` waker requires darwin (launchd)")
}
