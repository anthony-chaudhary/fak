package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/trunkbuildprobe"
)

// cmdTrunkBuildProbe is the thin shell over internal/trunkbuildprobe — the
// read-only diagnosis of whether the release gate's red trunk is a forgotten
// `git add` (Go port of the retired tools/trunk_build_probe.py). Exit codes
// mirror release_decide: 0 builds, 2 broken, 1 probe failure.
func cmdTrunkBuildProbe(argv []string) {
	os.Exit(trunkbuildprobe.Run(os.Stdout, os.Stderr, argv))
}
