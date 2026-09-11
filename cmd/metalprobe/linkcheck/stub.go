//go:build !darwin

// The non-darwin build of the linkcheck leaf. The Metal cgo probe never compiles off
// macOS, so this stub keeps the package resolvable for non-darwin `go build ./...`
// and `go vet` sweeps, and refuses to run: cmd/metalprobe invokes the real leaf only
// on darwin/arm64 hosts.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "metalprobe: linkcheck leaf requires darwin (cmd/metalprobe runs it only on darwin/arm64)")
	os.Exit(1)
}
