//go:build !linux

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "v41epguard: Linux pidfd and parent-death containment are required")
	os.Exit(2)
}
