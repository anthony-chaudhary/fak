package modver

// Coverage-profile-to-module adapter (#2467).
//
// CoverageScores folds a Go coverage profile (`go test -coverprofile=coverage.out`)
// into the same flat {module: percent} map LoadScores and `fak version modules
// --scores` already consume, so statement coverage can ride the module-version
// series as a score. Coverage is the cheapest real score to trend against module
// revs: the test gate already produces it per package, and it is a WITNESSED
// measurement — read off a real run's artifact, not modeled — so the joined score
// carries ProvenanceWitnessed at the CLI seam.
//
// The fold is statement-WEIGHTED, not file-averaged: a module's percent is its
// covered statements over its total statements across every file that maps to it.
// That is the number `go tool cover -func` reports, and it stops a 5-statement
// helper file from carrying the same weight as a 500-statement core file. This is
// the one place the adapter deliberately differs from the per-file scorecard
// adapter (score_adapter.go), which has no statement counts to weight by and so
// takes the arithmetic mean.
//
// Package-to-module is deliberately many-to-one: a profile is keyed per PACKAGE,
// but internal/<leaf>/<subpkg> folds into the module internal/<leaf> (moduleOf),
// so a leaf with subpackages gets one statement-weighted percent across all of
// them rather than an arbitrary per-package pick.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// coverageBlock keys one profile block by its file and source range. A merged
// profile (several packages' runs concatenated, or the same package measured
// twice) can repeat a block; keying by file+span folds the repeats into one, so
// its statements are counted once and the block reads as covered when ANY
// occurrence executed it.
type coverageBlock struct {
	file string
	span string
}

// CoverageScores decodes a Go coverage profile and returns the flat
// module -> statement-coverage-percent map, rounded to one decimal (the
// precision `go tool cover -func` itself reports).
//
// modulePath is the Go module path from go.mod: a profile names files by IMPORT
// path, and because this repo's Go module is the repository root, trimming that
// prefix yields the repo-relative path moduleOf classifies. Rows outside the
// prefix (a dependency caught by a merged -coverpkg profile) and rows under no
// tracked keyspace (a root-package file) belong to no module and are skipped —
// the same rule ModulesForPaths applies — never misfiled onto a neighbour. An
// empty modulePath treats the profile's file names as already repo-relative.
//
// A malformed profile is an error rather than a partial fold: a coverage score
// that silently dropped half its blocks would understate a module in the ledger
// while looking like a real measurement.
func CoverageScores(data []byte, modulePath string) (map[string]float64, error) {
	prefix := strings.TrimSuffix(strings.TrimSpace(modulePath), "/")
	if prefix != "" {
		prefix += "/"
	}
	type blockStat struct {
		module  string
		stmts   int
		covered bool
	}
	blocks := map[coverageBlock]*blockStat{}
	sawMode, rows := false, 0
	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if !sawMode {
			// The mode header is mandatory in the format `go test -coverprofile`
			// writes; requiring it makes "wrong file entirely" fail loud instead of
			// folding to an empty, innocent-looking score map.
			if !strings.HasPrefix(line, "mode:") {
				return nil, fmt.Errorf("modver: coverage profile line %d: want a leading \"mode:\" header, got %q", i+1, line)
			}
			sawMode = true
			continue
		}
		file, span, stmts, count, err := parseCoverageLine(line)
		if err != nil {
			return nil, fmt.Errorf("modver: coverage profile line %d: %w", i+1, err)
		}
		rows++
		rel := file
		if prefix != "" {
			if !strings.HasPrefix(rel, prefix) {
				continue
			}
			rel = strings.TrimPrefix(rel, prefix)
		}
		module, _, ok := moduleOf(rel)
		if !ok {
			continue
		}
		key := coverageBlock{file: file, span: span}
		b := blocks[key]
		if b == nil {
			b = &blockStat{module: module, stmts: stmts}
			blocks[key] = b
		}
		if count > 0 {
			b.covered = true
		}
	}
	if !sawMode {
		return nil, fmt.Errorf("modver: coverage profile is empty")
	}
	if rows == 0 {
		return nil, fmt.Errorf("modver: coverage profile has a mode header but no block rows")
	}

	type tally struct{ covered, total int }
	tallies := map[string]*tally{}
	for _, b := range blocks {
		t := tallies[b.module]
		if t == nil {
			t = &tally{}
			tallies[b.module] = t
		}
		t.total += b.stmts
		if b.covered {
			t.covered += b.stmts
		}
	}
	out := make(map[string]float64, len(tallies))
	for module, t := range tallies {
		if t.total == 0 {
			// A statement-free module has no percent to report; emitting 0 would
			// read as "measured at zero coverage" rather than "nothing to measure".
			continue
		}
		out[module] = math.Round(float64(t.covered)*1000/float64(t.total)) / 10
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("modver: coverage profile maps to no versioned module (want file names under module path %q)", modulePath)
	}
	return out, nil
}

// parseCoverageLine splits one profile block row:
//
//	<file>:<startLine>.<startCol>,<endLine>.<endCol> <numStmts> <count>
//
// The file name is everything before the LAST colon: an import path is
// colon-free, but a Windows-absolute path (which a -coverpkg run over an
// absolute directory can emit) carries a drive colon that a first-colon split
// would cut on.
func parseCoverageLine(line string) (file, span string, stmts, count int, err error) {
	fields := strings.Fields(line)
	if len(fields) != 3 {
		return "", "", 0, 0, fmt.Errorf("want \"<file>:<span> <stmts> <count>\", got %q", line)
	}
	cut := strings.LastIndex(fields[0], ":")
	if cut <= 0 || cut == len(fields[0])-1 {
		return "", "", 0, 0, fmt.Errorf("no <file>:<span> separator in %q", fields[0])
	}
	n, err := strconv.Atoi(fields[1])
	if err != nil || n < 0 {
		return "", "", 0, 0, fmt.Errorf("bad statement count %q", fields[1])
	}
	c, err := strconv.Atoi(fields[2])
	if err != nil || c < 0 {
		return "", "", 0, 0, fmt.Errorf("bad execution count %q", fields[2])
	}
	return fields[0][:cut], fields[0][cut+1:], n, c, nil
}

// CoverageEntries labels a coverage fold as witnessed and returns the
// ScoreEntry map JoinScores takes. A coverage percent is measured first-hand
// from a real run's artifact, so it earns ProvenanceWitnessed — the label that
// keeps it visibly distinct from a modeled estimate in the same score column
// (#2498).
func CoverageEntries(coverage map[string]float64) map[string]ScoreEntry {
	entries := make(map[string]ScoreEntry, len(coverage))
	for module, pct := range coverage {
		entries[module] = ScoreEntry{Score: pct, Provenance: ProvenanceWitnessed}
	}
	return entries
}
