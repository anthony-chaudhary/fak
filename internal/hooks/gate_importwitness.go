package hooks

import (
	"bufio"
	"bytes"
	"fmt"
	"path"
	"regexp"
	"strings"
)

// gate_importwitness.go — the IMPORT_WITNESS commit-boundary gate.
//
// Catching import of an uncommitted package at commit time:
// A committed .go file imports a module-local package whose directory has zero
// tracked or staged non-test .go files (a forgotten `git add`).
//
// ADVISORY by default (DefaultMode "warn"): names the uncommitted package and the
// `git add <pkg>/` fix without breaking the author's workflow unexpectedly. Set
// FLEET_IMPORT_WITNESS_GUARD=block to hard-enforce it, or ALLOW_UNCOMMITTED_IMPORT=1
// to skip it once.

const modulePkgPrefix = "github.com/anthony-chaudhary/fak/"

var importSpecRE = regexp.MustCompile(`^\s*(?:_\s+|[A-Za-z_][A-Za-z0-9_]*\s+)?"([^"]+)"\s*(?://.*)?$`)

func gateImportWitness(d *StagedDiff) ([]Finding, error) {
	if d == nil {
		return nil, nil
	}

	// 1. Gather all package directories that have at least one tracked or staged non-test .go file.
	deleted := map[string]bool{}
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		if _, ok := d.FileBytes(p); !ok {
			deleted[p] = true
		}
	}

	hasSource := map[string]bool{}
	for _, raw := range d.IndexPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		if strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go") && !deleted[p] {
			hasSource[path.Dir(p)] = true
		}
	}
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		if strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go") {
			if _, ok := d.FileBytes(p); ok {
				hasSource[path.Dir(p)] = true
			}
		}
	}

	var findings []Finding
	judged := 0

	// 2. Inspect each staged .go file (excluding _test.go files since they don't break `go build`).
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			continue
		}
		content, ok := d.FileBytes(p)
		if !ok {
			continue
		}
		judged++

		scanner := bufio.NewScanner(bytes.NewReader(content))
		lineNum := 0
		seenInFile := map[string]bool{}
		for scanner.Scan() {
			lineNum++
			m := importSpecRE.FindStringSubmatch(scanner.Text())
			if m == nil {
				continue
			}
			imp := m[1]
			if !strings.HasPrefix(imp, modulePkgPrefix) {
				continue
			}
			pkgDir := strings.TrimPrefix(imp, modulePkgPrefix)
			pkgDir = strings.TrimSuffix(pkgDir, "/")
			if seenInFile[pkgDir] {
				continue
			}
			seenInFile[pkgDir] = true

			if !hasSource[pkgDir] {
				findings = append(findings, Finding{
					Gate:   "IMPORT_WITNESS",
					File:   p,
					Line:   lineNum,
					Detail: fmt.Sprintf("imports %s — dir %s/ has zero tracked non-test .go files; `git add %s/` (or revert the importer until the package lands)", imp, pkgDir, pkgDir),
				})
			}
		}
	}

	d.NoteCandidates("IMPORT_WITNESS", judged, "files")
	return findings, nil
}
