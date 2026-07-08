package dispatchaudit

import (
	"os"
	"path/filepath"
	"sort"
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

// severityRank orders the fileable OUTCOME findings worst-first (lower = higher
// priority) so a per-run cap keeps the highest-signal waste and drops the noise.
// A provider hard-wall and a spawn burned into a capped backend are the most
// actionable (gate/rotate the account); a banner-only NO_OP is the least. An
// unranked outcome sorts last.
func severityRank(o Outcome) int {
	switch o {
	case OutcomeQuotaWalled:
		return 0
	case OutcomeWastedSpawn:
		return 1
	case OutcomeRetryStorm:
		return 2
	case OutcomeErrored:
		return 3
	case OutcomeNoOp:
		return 4
	default:
		return 5
	}
}

// rankFinding is the unified worst-first key (lower = higher priority) that the
// --max-issues cap sorts by. It MUST bridge the TWO finding taxonomies that
// share the fileable set — `fak dispatch audit` merges Fold() outcome findings
// AND ScanDirSignatures() log-signature findings (see cmd/fak/dispatchaudit.go).
// A signature finding carries Outcome=="" and a SignatureClass, so it would fall
// through severityRank's default arm and sort BELOW a banner-only NO_OP — the
// exact inversion that would make the cap withhold a worker panic / auth-wall /
// hook storm first. Ranking signatures by their position in the canonical
// worst-first SignatureClasses() order keeps this consistent with the detector
// table (no drift) and guarantees a crash/storm/auth signal ranks 0..4, never
// below NO_OP's 4.
func rankFinding(f Finding) int {
	if f.Outcome != "" {
		return severityRank(f.Outcome)
	}
	if f.SignatureClass != "" {
		for i, c := range SignatureClasses() {
			if c == f.SignatureClass {
				return i
			}
		}
	}
	return 5
}

// SelectFindingsToFile is the PURE "what to file this run" decision that the
// scheduled feeder (register_dispatch_session_audit.ps1 → `fak dispatch audit
// --file-issues --max-issues N`) relies on to stay storm-safe on its FIRST live
// run over a large historical backlog. It dedups (NewFindings), orders the
// survivors worst-first (severityRank, then fingerprint for determinism), and
// caps to max. It returns the selected findings plus the count withheld by the
// cap, so the caller can report the truncation rather than hide it. max <= 0
// means no cap. Deterministic: same inputs → same selection and same withheld.
func SelectFindingsToFile(findings []Finding, filedFingerprints, openTitles map[string]bool, max int) (selected []Finding, withheld int) {
	fresh := NewFindings(findings, filedFingerprints, openTitles)
	sort.SliceStable(fresh, func(i, j int) bool {
		ri, rj := rankFinding(fresh[i]), rankFinding(fresh[j])
		if ri != rj {
			return ri < rj
		}
		return fresh[i].Fingerprint < fresh[j].Fingerprint
	})
	if max > 0 && len(fresh) > max {
		return fresh[:max], len(fresh) - max
	}
	return fresh, 0
}
