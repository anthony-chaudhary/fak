package modver

// Maturity-scorecard-to-module adapter (#2468).
//
// MaturityScores folds a `fak maturity --json` payload (fak-maturity-scorecard/1)
// into the same flat {module: score} map LoadScores and `fak version modules
// --scores` already consume, so "is this leaf production-grade at its current
// rev?" becomes one query against the module-version series. internal/maturity
// already grades every declared capability, but it grades it by LANE; until the
// grade is keyed the way modver keys a module it cannot ride next to that
// module's rev in the fak-module-versions/1 ledger.
//
// The join is a RE-KEYING, not a new scale. A capability's module score is its
// ladder position as a percentage of the top rung, docked a whole capability's
// worth for a ladder-skip — exactly the two terms internal/maturity's own fleet
// index is built from (100*sum(rung)/(n*max) - 100*skips/n). The arithmetic mean
// of these per-module scores therefore reproduces that published index, which is
// the invariant maturity_adapter_test.go pins: modver cannot quietly drift into
// grading maturity differently from the scorecard that owns the grade.
//
// A ladder-skip goes NEGATIVE and is deliberately not clamped at zero. A skip is
// an overclaim — the capability looks more mature than its evidence supports —
// and the fleet index charges it a full capability; clamping here would let a
// skipped module read as merely immature and would break the mean invariant
// above. (The ledger already accepts negative scores; see score_adapter.go.)
//
// The top rung is read from the payload's own ladder rather than hardcoded, so
// adding a rung to internal/maturity rescales this join automatically instead of
// silently inflating every module.
//
// The SUPPORT-maturity sibling (internal/supportmaturityscore) is deliberately
// absent: it grades covmatrix cells (model family x backend), which carry no
// module key at all, so there is nothing to join per module. Minting one would
// misfile a hardware fact onto a source tree.
//
// Input is bytes, like the two sibling adapters (score_adapter.go, coverage.go),
// which keeps modver free of a dependency on the scorecards it joins.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// MaturityCapability is one capability row of a maturity scorecard — the subset
// of internal/maturity.Capability this join reads. Dir is the capability's tree
// root (internal/<leaf>), which is also its module key.
type MaturityCapability struct {
	Dir  string `json:"dir"`
	Rung int    `json:"rung"`
	Skip bool   `json:"skip"`
}

// MaturityScorecard is the stable adapter input envelope: the capability rows
// plus the closed rung ladder they are scored against. Schema is advisory for
// forward compatibility; capabilities and corpus.ladder are load-bearing.
type MaturityScorecard struct {
	Schema string `json:"schema,omitempty"`
	Corpus struct {
		Ladder []string `json:"ladder"`
	} `json:"corpus"`
	Capabilities []MaturityCapability `json:"capabilities"`
}

// moduleDirProbe is a representative file name used to classify a capability's
// tree ROOT through moduleOf, which classifies FILE paths (a directory path has
// one segment fewer). It is never opened.
const moduleDirProbe = "probe.go"

// MaturityScores decodes a maturity scorecard and returns the flat module ->
// ladder-score map. A row that cannot be keyed to a module, a duplicate row, or
// a rung outside the closed ladder is an error rather than a partial fold: a
// maturity score that silently dropped or misfiled rows would understate a
// module in the ledger while looking like a real grade.
func MaturityScores(data []byte) (map[string]float64, error) {
	var in MaturityScorecard
	if err := json.Unmarshal(data, &in); err != nil {
		return nil, fmt.Errorf("modver: decode maturity scorecard: %w", err)
	}
	if len(in.Capabilities) == 0 {
		return nil, fmt.Errorf("modver: maturity scorecard has no capabilities")
	}
	// The ladder is what makes a rung a percentage. Without it the fold would
	// have to guess a denominator, and a guess that drifted from
	// internal/maturity's ladder would rescale every module at once.
	maxRung := len(in.Corpus.Ladder) - 1
	if maxRung < 1 {
		return nil, fmt.Errorf("modver: maturity scorecard has no rung ladder (want corpus.ladder with at least 2 rungs)")
	}

	out := make(map[string]float64, len(in.Capabilities))
	for i, row := range in.Capabilities {
		dir := strings.TrimSpace(row.Dir)
		if dir == "" {
			return nil, fmt.Errorf("modver: maturity capability %d has an empty dir", i+1)
		}
		module, ok := moduleOfDir(dir)
		if !ok {
			return nil, fmt.Errorf("modver: cannot map maturity capability dir %q to a module", dir)
		}
		if _, dup := out[module]; dup {
			return nil, fmt.Errorf("modver: duplicate maturity capability for module %q", module)
		}
		if row.Rung < 0 || row.Rung > maxRung {
			return nil, fmt.Errorf("modver: maturity capability %q has rung %d outside the ladder 0..%d",
				dir, row.Rung, maxRung)
		}
		out[module] = maturityLadderScore(row.Rung, maxRung, row.Skip)
	}
	return out, nil
}

// maturityLadderScore places one capability on the 0-100 ladder scale and
// charges a ladder-skip a whole capability, mirroring the two terms of
// internal/maturity's fleet index term-for-term.
func maturityLadderScore(rung, maxRung int, skip bool) float64 {
	score := 100 * float64(rung) / float64(maxRung)
	if skip {
		score -= 100
	}
	return score
}

// moduleOfDir maps a capability's tree ROOT to its module key by probing the one
// keyspace authority (moduleOf) with a file directly inside it. The probe result
// must be the directory itself: a nested dir like internal/leaf/sub would
// classify to internal/leaf, and silently folding it there would attribute a
// subpackage's grade to its parent module.
func moduleOfDir(dir string) (string, bool) {
	clean := strings.Trim(strings.TrimSpace(strings.ReplaceAll(dir, "\\", "/")), "/")
	if clean == "" {
		return "", false
	}
	name, _, ok := moduleOf(clean + "/" + moduleDirProbe)
	if !ok || name != clean {
		return "", false
	}
	return name, true
}

// MaturityEntries labels a maturity fold as witnessed and returns the ScoreEntry
// map JoinScores takes. Every rung in internal/maturity is re-derived from
// evidence the capability's author did not write — code on disk, a *_test.go, an
// import from cmd/, a documented verb — so the grade is measured first-hand off
// the tree rather than modeled, and earns ProvenanceWitnessed (#2498).
func MaturityEntries(scores map[string]float64) map[string]ScoreEntry {
	entries := make(map[string]ScoreEntry, len(scores))
	for module, score := range scores {
		entries[module] = ScoreEntry{Score: score, Provenance: ProvenanceWitnessed}
	}
	return entries
}
