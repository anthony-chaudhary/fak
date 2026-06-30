package milestonereport

// epicsfrom.go wires the `--epics-from` override: the tracked-epic set becomes a
// committed, reviewable DATA file instead of a code edit to the TrackedEpics slice.
// A file may carry just the specs (the resolver then reads each epic's children live
// via `gh`), OR pre-resolved counts alongside them so a fold is fully OFFLINE — the
// hermetic path the milestone tests and a gh-free CI run take. The default stays the
// in-code TrackedEpics slice when the flag is absent (zero-config back-compat).

import (
	"encoding/json"
	"fmt"
	"os"
)

// DefaultTrackedEpicsRel is the committed seed that mirrors the in-code TrackedEpics
// default exactly, so `--epics-from docs/milestones/tracked-epics.json` is behavior-
// identical to the zero-config default. It lives under docs/ as durable trunk
// evidence — adding/removing a tracked epic becomes a reviewable diff of this file.
const DefaultTrackedEpicsRel = "docs/milestones/tracked-epics.json"

// EpicsFile is the on-disk shape of an `--epics-from` data file: the tracked-epic
// SPECS, and an OPTIONAL pre-resolved COUNTS block. When counts is present the fold
// is hermetic (no `gh` call); when it is absent the resolver reads each epic's
// children live. A count whose Number has no matching spec is ignored — the spec set
// is the source of truth for WHICH epics are tracked.
type EpicsFile struct {
	// Schema is an optional, ignored-on-read provenance tag for the committed seed.
	Schema string `json:"schema,omitempty"`
	// Specs is the tracked-epic set: the analog of the in-code TrackedEpics slice.
	Specs []EpicSpec `json:"specs"`
	// Counts, when present, pre-resolves each epic's child tally so the fold needs no
	// `gh` — the offline override the acceptance criterion drives in tests.
	Counts []EpicCounts `json:"counts,omitempty"`
}

// LoadEpicsFile reads and decodes an `--epics-from` data file. A missing file, a
// malformed JSON body, or an empty spec set is an error — the caller falls back to
// the in-code default ONLY when the flag is absent, never when a named file is
// unreadable (a typo'd path must not silently track the wrong set).
func LoadEpicsFile(path string) (EpicsFile, error) {
	var f EpicsFile
	raw, err := os.ReadFile(path)
	if err != nil {
		return f, fmt.Errorf("read epics-from file: %w", err)
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return f, fmt.Errorf("parse epics-from file %s: %w", path, err)
	}
	if len(f.Specs) == 0 {
		return f, fmt.Errorf("epics-from file %s has no specs", path)
	}
	return f, nil
}

// HasCounts reports whether the file carries a pre-resolved counts block — i.e.
// whether the fold can run fully offline. An empty Counts slice means "resolve live".
func (f EpicsFile) HasCounts() bool { return len(f.Counts) > 0 }

// FoldOffline folds the file's specs against its pre-resolved counts with the pure
// InterpretEpics — no `gh`, deterministic. It is the hermetic path: CountsFromSpecs
// with the file's own data. Callers use this only when HasCounts() is true.
func (f EpicsFile) FoldOffline() Epics {
	return CountsFromSpecs(f.Specs, f.Counts)
}

// MarshalEpicsFile renders an EpicsFile as the committed-seed JSON (stable 2-space
// indent, trailing newline). It is the inverse of LoadEpicsFile, used to author or
// regenerate the seed so the on-disk file and the in-code default never drift.
func MarshalEpicsFile(f EpicsFile) ([]byte, error) {
	b, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
