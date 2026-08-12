package hooks

import (
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// CheckDesktopPopup shifts the popup floor from push time to the candidate-index
// boundary. A newly-created or edited helper is refused before it can become a
// commit (and therefore before background automation can consume it).
//
// Scan the COMPLETE staged file, not only added lines: whether an exec is safe can
// be expressed elsewhere in the file by a shared windowgate constructor or
// ConfigureBackgroundCommand call. FileBytes reads the candidate index, keeping a
// peer's unstaged/untracked work outside this commit's verdict.
func CheckDesktopPopup(d *StagedDiff) ([]Finding, error) {
	paths := popupSourcePaths(d.StagedPaths)
	added := make(map[string]bool, len(d.AddedPaths))
	for _, rel := range d.AddedPaths {
		added[filepath.ToSlash(rel)] = true
	}
	d.NoteCandidates("DESKTOP_POPUP_REGRESSION", len(paths), "staged .go/.py/.ps1 helper files")
	var findings []Finding
	for _, rel := range paths {
		b, ok := d.FileBytes(rel)
		if !ok {
			continue
		}
		r := popupFileReport(rel, string(b))
		// New helper files have no compatibility debt: literal console-tool
		// candidates are blocking at admission even outside the legacy hard-path
		// ratchet. Existing files retain the established hard-violation scope.
		if added[rel] {
			switch strings.ToLower(filepath.Ext(rel)) {
			case ".go":
				r.GoExecs = appendUnique(r.GoExecs, windowgate.GoExecCandidates(rel, string(b))...)
			case ".py":
				r.PySpawns = appendUnique(r.PySpawns, windowgate.PySpawnCandidates(rel, string(b))...)
			}
		}
		rows := append(append(append([]string{}, r.PSInstallers...), r.PSStartProcesses...), r.PySpawns...)
		rows = append(rows, r.GoExecs...)
		for _, row := range rows {
			findings = append(findings, Finding{Gate: "DESKTOP_POPUP_REGRESSION", File: rel, Line: popupFindingLine(row), Detail: row})
		}
	}
	return findings, nil
}

func popupSourcePaths(paths []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, rel := range paths {
		rel = filepath.ToSlash(rel)
		switch strings.ToLower(filepath.Ext(rel)) {
		case ".go", ".py", ".ps1":
			if !seen[rel] {
				seen[rel] = true
				out = append(out, rel)
			}
		}
	}
	sort.Strings(out)
	return out
}

func popupFileReport(rel, src string) windowgate.Report {
	r := windowgate.Report{}
	switch strings.ToLower(filepath.Ext(rel)) {
	case ".go":
		r.GoExecs = windowgate.GoExecViolations(rel, src)
	case ".py":
		r.PySpawns = windowgate.PySpawnViolations(rel, src)
	case ".ps1":
		if row, bad := windowgate.PSInstallerViolation(rel, src); bad {
			r.PSInstallers = append(r.PSInstallers, row)
		}
		r.PSStartProcesses = windowgate.PSStartProcessViolations(rel, src)
	}
	return r
}

func appendUnique(dst []string, rows ...string) []string {
	seen := make(map[string]bool, len(dst)+len(rows))
	for _, row := range dst {
		seen[row] = true
	}
	for _, row := range rows {
		if !seen[row] {
			seen[row] = true
			dst = append(dst, row)
		}
	}
	return dst
}

func popupFindingLine(row string) int {
	first := strings.IndexByte(row, ':')
	if first < 0 {
		return 0
	}
	rest := row[first+1:]
	second := strings.IndexByte(rest, ':')
	if second < 0 {
		return 0
	}
	line, _ := strconv.Atoi(rest[:second])
	return line
}
