package safecommit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// ReasonCheckerTampered ("CHECKER_TAMPERED") is the refusal a grade/close arm reports when a
// run's DECLARED checker files changed bytes between declare-time (when PinCheckers fingerprinted
// them) and grade-time (when GuardCheckerPin re-checks them). It closes the "grade the grader"
// hole: a run whose own test/checker file can be edited between declaration and grading can make
// a red grade go green by mutating the checker rather than the code, so a grade computed against
// a moved checker is not evidence of anything. When the checker drifted the arm refuses with THIS
// structured reason instead of ratifying the grade.
//
// It is registered as first-class refusal vocabulary in the repo dos.toml [reasons.CHECKER_TAMPERED]
// table, so `dos_check_reason CHECKER_TAMPERED` resolves it (known, refusal = true) and
// `dos_refuse_reasons` lists it — the "no" is a citable value from a closed set, not free-text prose.
//
// It is deliberately NOT a member of RefusalReasons() (the commit-clean operator set surfaced in
// the commit-clean skill doc): a tampered checker is a GRADE-arm integrity refusal, not a pre-commit
// pathspec/hygiene gate, so — following the package convention that a reason const lives in the file
// with its guard — it lives here beside PinCheckers/VerifyCheckers and is surfaced by whichever close
// arm consults GuardCheckerPin.
const ReasonCheckerTampered = "CHECKER_TAMPERED"

// Fingerprint sentinels for a declared checker path. A real fingerprint is a 64-char hex sha256, so
// neither sentinel can collide with the fingerprint of any file's bytes.
const (
	checkerAbsent     = "absent"     // the declared path did not exist when fingerprinted
	checkerUnreadable = "unreadable" // the path exists but its bytes could not be read (fail-closed → drift)
)

// CheckerBaseline is the pinned content fingerprint of a run's declared checker files, captured at
// declare-time by PinCheckers: a slash-normalized repo-relative path mapped to its fingerprint
// (hex sha256 of the file's bytes, or checkerAbsent for a path that did not exist at pin time). A
// nil or empty baseline pins nothing, so VerifyCheckers passes it through as un-drifted.
type CheckerBaseline map[string]string

// CheckerDrift is the typed grade-time verdict returned by VerifyCheckers. Tampered is true iff any
// pinned checker's current fingerprint differs from its declare-time value; Reason then carries
// ReasonCheckerTampered. The three slices partition the drifted paths by HOW they drifted so a close
// arm can render an actionable refusal (which checker, and whether it changed / vanished / appeared).
// Each slice is sorted, so the verdict is deterministic for a given tree and baseline.
type CheckerDrift struct {
	Tampered bool     `json:"tampered"`
	Reason   string   `json:"reason,omitempty"`
	Changed  []string `json:"changed,omitempty"`  // pinned present, bytes differ now (or unreadable now)
	Missing  []string `json:"missing,omitempty"`  // pinned present, absent now (deleted)
	Appeared []string `json:"appeared,omitempty"` // pinned absent, present now (injected grader)
}

// PinCheckers captures the declare-time baseline for a run's declared checker paths (each repo-
// relative to root): path -> content fingerprint. A path that does not yet exist is pinned as
// checkerAbsent, so a checker that APPEARS after declare-time is itself drift — an injected grader is
// as untrustworthy as a mutated one. Keys are slash-normalized so a baseline is portable across OSes
// (a Windows pin verifies on Linux and vice versa). It errors if a declared checker exists but its
// bytes cannot be read: a bad declaration surfaces at declare-time rather than being silently
// mis-graded later.
func PinCheckers(root string, paths []string) (CheckerBaseline, error) {
	base := make(CheckerBaseline, len(paths))
	for _, rel := range paths {
		fp, err := fingerprint(root, rel)
		if err != nil {
			return nil, fmt.Errorf("pin checker %q: %w", rel, err)
		}
		base[filepath.ToSlash(rel)] = fp
	}
	return base, nil
}

// VerifyCheckers recomputes the fingerprint of every pinned path under root and compares it to the
// baseline, returning the typed drift verdict. It is pure with respect to its inputs (it reads only
// the named files) and deterministic: the same tree and baseline always yield the same verdict, with
// each drifted-path slice sorted. It is fail-closed — a checker that can no longer be read is treated
// as drift, never as a pass. A nil or empty baseline is a clean pass: nothing was pinned, so nothing
// can drift.
func VerifyCheckers(root string, baseline CheckerBaseline) CheckerDrift {
	var d CheckerDrift
	keys := make([]string, 0, len(baseline))
	for k := range baseline {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, rel := range keys {
		want := baseline[rel]
		got, err := fingerprint(root, rel)
		if err != nil {
			got = checkerUnreadable // fail-closed: an unreadable checker cannot be trusted to match
		}
		if got == want {
			continue
		}
		switch {
		case want != checkerAbsent && got == checkerAbsent:
			d.Missing = append(d.Missing, rel)
		case want == checkerAbsent && got != checkerAbsent:
			d.Appeared = append(d.Appeared, rel)
		default:
			d.Changed = append(d.Changed, rel)
		}
	}
	if len(d.Changed)+len(d.Missing)+len(d.Appeared) > 0 {
		d.Tampered = true
		d.Reason = ReasonCheckerTampered
	}
	return d
}

// GuardCheckerPin is the close-arm consult: the single call a witness/grade arm makes right before
// it accepts a run. It returns (ReasonCheckerTampered, true) when the run's declared checkers drifted
// from their pinned baseline — the arm should refuse and surface the structured reason instead of
// ratifying a grade computed against a moved checker — and ("", false) to pass an un-drifted run
// through untouched.
func GuardCheckerPin(root string, baseline CheckerBaseline) (reason string, refused bool) {
	if d := VerifyCheckers(root, baseline); d.Tampered {
		return d.Reason, true
	}
	return "", false
}

// fingerprint returns the content fingerprint of the repo-relative path rel under root: hex sha256 of
// the file's bytes for a readable file, or checkerAbsent when the path does not exist. A read error
// other than not-exist is returned alongside checkerUnreadable so the caller decides its meaning —
// PinCheckers rejects it at declare-time, VerifyCheckers folds it into drift at grade-time.
func fingerprint(root, rel string) (string, error) {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		if os.IsNotExist(err) {
			return checkerAbsent, nil
		}
		return checkerUnreadable, err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
