package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	fixture, err := BuildFixture()
	if err != nil {
		panic(err)
	}
	path := filepath.Join(os.TempDir(), "fak-replay-fixture.json")
	if err := WriteFixture(path, fixture); err != nil {
		panic(err)
	}
	loaded, err := ReadFixture(path)
	if err != nil {
		panic(err)
	}
	actual, outcome, err := Replay(loaded)
	if err != nil {
		panic(err)
	}
	if err := Compare(loaded.Expected, actual, Strict, loaded.Nondeterminism); err != nil {
		panic(err)
	}
	fmt.Printf("fixture=%s products=%d cursor=%d scrubbed=%d strict=pass\n", loaded.Provenance.FixtureID, len(actual), outcome.FinalCursor, loaded.ScrubReport.Replacements)
}
