//go:build darwin

// Command linkcheck is the cgo probe leaf invoked by cmd/metalprobe on darwin/arm64.
// It links internal/compute + internal/metalgemm — the two go-metal leaves whose cgo
// pieces (metal_shim.m, metal.m) clang compiles in-process — so building and running
// this leaf on a provisioned host proves the whole cgo Metal build path links AND can
// initialize a live device. It prints one machine line for cmd/metalprobe to parse:
//
//	probe: compiled=<bool> available=<bool> device=<name>
//
// and a diagnostic stderr line whether the compute registry's picked backend is
// metal-backed. The portable gate (cmd/metalprobe) must never import this leaf
// statically; it is reached only via `go run ./cmd/metalprobe/linkcheck` on a
// darwin/arm64 host.
package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func main() {
	compiled := metalgemm.Compiled()
	available := metalgemm.Available()
	name := metalgemm.DeviceName()
	if name == "" {
		name = "none"
	}
	fmt.Printf("probe: compiled=%t available=%t device=%s\n", compiled, available, name)
	metalBacked := false
	for _, b := range compute.Registered() {
		if compute.Pick(b) != nil && b == "metal" {
			metalBacked = true
			break
		}
	}
	fmt.Fprintf(os.Stderr, "linkcheck: compute registry metal-backed=%t\n", metalBacked)
	os.Exit(0)
}
