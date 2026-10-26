//go:build !windows

package main

import "errors"

func probePlatform() platformProbe {
	return platformProbe{err: errors.New("Windows WebAuthn probe is unsupported on this platform")}
}
