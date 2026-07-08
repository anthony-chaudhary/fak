// Package buildwitness is a structural CI guard: it fails when the primary binary package
// (cmd/fak) does not compile with the DEFAULT build tags. It exists because the recurring, costly
// failure mode in this repo is an author leaving an UNCOMMITTED or TAGLESS .go file in cmd/fak
// that references a symbol which was never committed — the package builds fine in the author's
// working tree (where the symbol's other half is also uncommitted) but is red for every other
// session and for CI. `go build ./cmd/fak` from a clean checkout is the ground truth, and the
// witness test runs exactly that from a package that compiles INDEPENDENTLY of cmd/fak, so the
// guard still runs (and reports a precise compiler error) even when cmd/fak itself is broken.
//
// The durable convention this enforces (see AGENTS.md): work-in-progress that cannot yet compile
// against committed symbols must be gated behind a `//go:build wip_<feature>` tag, so the default
// build — and this witness — stays green while the WIP lives on disk.
package buildwitness

import "runtime"

// TargetPackage is the import path this witness compiles to prove the default build is green.
const TargetPackage = "./cmd/fak"

// BuildArgs returns the `go build` argv that compiles TargetPackage to a throwaway output under
// default tags. Writing to `out` (the null device) rather than the in-tree binary avoids the
// Windows exclusive-lock false positive where a running peer holds fak.exe (#2373).
func BuildArgs(out string) []string {
	return []string{"build", "-o", out, TargetPackage}
}

// NullDevice is the platform bit-bucket path the throwaway build artifact is written to.
func NullDevice() string {
	if runtime.GOOS == "windows" {
		return "NUL"
	}
	return "/dev/null"
}
