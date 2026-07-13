package modver

// Scorecard-to-module adapter (#2466).
//
// ScorecardFileScores folds a scorecard's per-file rows into the exact flat
// {module: number} map LoadScores and `fak version modules --scores` consume.
// Each tracked file has equal weight and a module score is the arithmetic mean
// of its file scores. Equal file weighting keeps the result explainable and
// prevents a scorecard-specific defect taxonomy from leaking into modver.

import (
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
)

// FileScore is one scorecard observation. Path is repository-relative and Score
// is conventionally 0..100; negative scorecard values are accepted because the
// code-slop scorecard can legitimately fall below zero when debt is large.
type FileScore struct {
	Path  string  `json:"path"`
	Score float64 `json:"score"`
}

// FileScorecard is the stable adapter input envelope. The schema field is
// advisory for forward compatibility; files is the load-bearing payload.
type FileScorecard struct {
	Schema string      `json:"schema,omitempty"`
	Files  []FileScore `json:"files"`
}

// ScorecardFileScores decodes per-file scorecard output and returns a flat
// module-score map. Duplicate paths and unclassifiable paths are rejected rather
// than silently weighting a file twice or joining it to the wrong module.
func ScorecardFileScores(data []byte) (map[string]float64, error) {
	var in FileScorecard
	if err := json.Unmarshal(data, &in); err != nil {
		return nil, fmt.Errorf("modver: decode per-file scorecard: %w", err)
	}
	if len(in.Files) == 0 {
		return nil, fmt.Errorf("modver: per-file scorecard has no files")
	}
	type aggregate struct {
		total float64
		count int
	}
	aggs := map[string]aggregate{}
	seen := map[string]bool{}
	for i, row := range in.Files {
		path := normalizeScorePath(row.Path)
		if path == "" {
			return nil, fmt.Errorf("modver: per-file scorecard row %d has an empty path", i+1)
		}
		if seen[path] {
			return nil, fmt.Errorf("modver: duplicate per-file score path %q", path)
		}
		seen[path] = true
		if math.IsNaN(row.Score) || math.IsInf(row.Score, 0) {
			return nil, fmt.Errorf("modver: non-finite score for %q", path)
		}
		module, _, ok := moduleOf(path)
		if !ok {
			return nil, fmt.Errorf("modver: cannot map scorecard path %q to a module", path)
		}
		a := aggs[module]
		a.total += row.Score
		a.count++
		aggs[module] = a
	}

	out := make(map[string]float64, len(aggs))
	for module, a := range aggs {
		out[module] = a.total / float64(a.count)
	}
	return out, nil
}

func normalizeScorePath(path string) string {
	path = filepath.ToSlash(strings.TrimSpace(path))
	path = strings.TrimPrefix(path, "./")
	path = strings.TrimPrefix(path, "/")
	return path
}

// MarshalModuleScores emits deterministic JSON for process substitution or a
// temporary file consumed by `fak version modules --scores`.
func MarshalModuleScores(scores map[string]float64) ([]byte, error) {
	// encoding/json already sorts string map keys. Copy through a sorted key walk
	// to make that determinism explicit at this adapter boundary.
	keys := make([]string, 0, len(scores))
	for key := range scores {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	ordered := make(map[string]float64, len(keys))
	for _, key := range keys {
		ordered[key] = scores[key]
	}
	return json.MarshalIndent(ordered, "", "  ")
}
