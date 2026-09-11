//go:build windows

// Windows host of the metal-check gate. When the Makefile's metal-check target runs
// through a GNU make built for Windows (win32/MSYS), a `go run` built from that make
// still reports the host arch through runtime.GOARCH; nothing else is needed. This
// runner exists to keep the gate buildable and runnable on windows/amd64, where it
// always reaches the same not-applicable verdict as any other non-darwin/arm64 host.
package main

import "os"

func main() {
	os.Exit(Run(runtimeArch()))
}
