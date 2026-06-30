package hooks

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// gate_tierdeclared.go — the whole-tree gate that catches drift between internal/<leaf>
// packages and the architest tier table. internal/architest's
// TestEveryPackageDeclaresTier already fails when a package on disk is missing from its
// `tier` map, or when the map names a removed package, but that test only reds the
// trunk AFTER the drift has been committed and CI runs. This gate surfaces the same gap
// one boundary earlier, in `fak hygiene`, so a contributor sees the missing or stale
// tier row before the shared trunk goes red (#962, #1145).
//
// It is the architest twin of laneaudit.go's dos.toml lane check: both answer "which
// real leaf drifted in without its declaration?" — one for the LANE taxonomy, one for
// the TIER taxonomy — by reading the single source of truth and comparing it to the
// internal/<pkg> dirs that actually hold a Go package. The tier source of truth is the
// `tier` map literal in internal/architest/architest_test.go; this gate parses its keys
// rather than maintaining a second copy, so it can never become a rival authority.

// tierTableFile is the single source of truth for the tier taxonomy.
const tierTableFile = "internal/architest/architest_test.go"

// tierKeyRE matches a `"<name>": <int>` tier entry. The entries are packed several
// per line (e.g. `"accounts": 1, "appversion": 1, "blob": 1, ...`), so the pattern is
// NOT line-anchored — every `"name": <digit>` on a line is a declared key. Requiring
// the trailing integer keeps it from matching a quoted string in a trailing comment.
var tierKeyRE = regexp.MustCompile(`"([A-Za-z0-9][\w.\-]*)"\s*:\s*\d`)

// declaredTiers parses the tier-map keys out of the architest tier table read from the
// tracked tree. ok is false when the table cannot be read (the gate then fails open via
// ErrCouldNotRun, never a false TIER_DECLARED on an unreadable source).
func declaredTiers(t *TrackedTree) (map[string]bool, bool) {
	body, exists := t.FileBytes(tierTableFile)
	if !exists {
		return nil, false
	}
	declared := map[string]bool{}
	inTable := false
	for _, line := range strings.Split(string(body), "\n") {
		if !inTable {
			if strings.Contains(line, "var tier = map[string]int{") {
				inTable = true
			}
			continue
		}
		// The table closes at the first line that is just `}` (the map literal's brace).
		if strings.TrimSpace(line) == "}" {
			break
		}
		for _, m := range tierKeyRE.FindAllStringSubmatch(line, -1) {
			declared[strings.ToLower(m[1])] = true
		}
	}
	if len(declared) == 0 {
		return nil, false // the marker moved or the file shape changed — fail open
	}
	return declared, true
}

// gateTierDeclaredTree emits a TIER_DECLARED finding for every internal/<leaf> package
// that holds a non-test .go file but is absent from the tier table, and for every tier
// table row whose package no longer exists in the tracked tree. architest excludes
// itself from its own on-disk scan, so this gate excludes it too. Returns ErrCouldNotRun
// when the tier table cannot be parsed (fail open, exit 2 → the architest TEST still
// catches it in CI as the backstop).
func gateTierDeclaredTree(t *TrackedTree) ([]Finding, error) {
	declared, ok := declaredTiers(t)
	if !ok {
		return nil, ErrCouldNotRun
	}
	// Collect the internal/<pkg> dirs that hold at least one non-test .go file, from the
	// tracked-path set (so the gate sees exactly what git tracks, like every other gate).
	hasGo := map[string]bool{}
	for _, p := range t.Paths {
		seg := strings.Split(p, "/")
		if len(seg) < 3 || seg[0] != "internal" {
			continue
		}
		if !strings.HasSuffix(seg[2], ".go") || strings.HasSuffix(seg[2], "_test.go") {
			continue
		}
		hasGo[strings.ToLower(seg[1])] = true
	}

	var findings []Finding
	for pkg := range hasGo {
		if pkg == "architest" { // architest excludes itself from its own tier scan
			continue
		}
		if declared[pkg] {
			continue
		}
		findings = append(findings, Finding{
			Gate: "TIER_DECLARED",
			File: "internal/" + pkg + "/",
			Detail: "internal/" + pkg + " has no tier declaration — add a row to the `tier` map in " +
				tierTableFile + " at the LOWEST tier whose role it fits (or run `python tools/new_leaf.py " +
				pkg + " --tier <tier>`), so its `(fak " + pkg + ")` ship-stamp binds to a declared layer.",
		})
	}
	for pkg := range declared {
		if pkg == "architest" { // keep symmetric with the on-disk scan above
			continue
		}
		if hasGo[pkg] {
			continue
		}
		if packageExistsOnDisk(t.Root, pkg) {
			continue
		}
		findings = append(findings, Finding{
			Gate: "TIER_DECLARED",
			File: "internal/" + pkg + "/",
			Detail: "tier table declares internal/" + pkg + ", but no tracked non-test Go package exists " +
				"(stale row — remove it from " + tierTableFile + ").",
		})
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].File < findings[j].File })
	return findings, nil
}

func packageExistsOnDisk(root, pkg string) bool {
	if root == "" {
		return false
	}
	entries, err := os.ReadDir(filepath.Join(root, "internal", pkg))
	if err != nil {
		return false
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
			return true
		}
	}
	return false
}
