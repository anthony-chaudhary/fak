package dispatchaudit

import (
	"os"
	"path/filepath"
	"strings"
)

// filer.go holds the dedup substrate for `--file-issues`: a finding is filed
// only when its fingerprint is NOT already marked (a .audit-filed/<fp> marker)
// AND its title is not already an open issue title. The marker write and the gh
// calls live in the cmd shell; the PURE dedup decision (NewFindings) is here so
// it is unit-testable.

// FiledMarkerDir is the per-runsDir directory holding one empty marker file per
// already-filed fingerprint.
func FiledMarkerDir(runsDir string) string {
	return filepath.Join(runsDir, ".audit-filed")
}

// AlreadyFiled reports whether a fingerprint already has a marker on disk.
func AlreadyFiled(runsDir, fingerprint string) bool {
	_, err := os.Stat(filepath.Join(FiledMarkerDir(runsDir), fingerprint))
	return err == nil
}

// MarkFiled writes the empty marker for a fingerprint (idempotent).
func MarkFiled(runsDir, fingerprint string) error {
	dir := FiledMarkerDir(runsDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, fingerprint), nil, 0o644)
}

// NewFindings is the PURE dedup fold: given all findings, the set of
// fingerprints already marked filed, and the set of existing open-issue titles,
// it returns only the findings that are genuinely new (no marker AND no
// title-collision). Deterministic — order-preserving over the input.
func NewFindings(findings []Finding, filedFingerprints, openTitles map[string]bool) []Finding {
	titles := map[string]bool{}
	for t := range openTitles {
		titles[strings.TrimSpace(strings.ToLower(t))] = true
	}
	var out []Finding
	for _, f := range findings {
		if filedFingerprints[f.Fingerprint] {
			continue
		}
		if titles[strings.TrimSpace(strings.ToLower(f.Title))] {
			continue
		}
		out = append(out, f)
	}
	return out
}
