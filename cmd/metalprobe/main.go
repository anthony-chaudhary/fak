//go:build !windows

// Non-windows host of the metal-check gate. The gate logic itself is pure Go and
// portable (metalcheck.go); this thin main simply executes it against the arch the
// running binary was built for, which matches what `go run` compiled.
package main

import "os"

func main() {
	os.Exit(Run(runtimeArch()))
}
