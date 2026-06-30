// Package corelockaudit is a read-only fold that maps changed paths to candidate
// core-lock classes and reports, per class, the witness that would clear it.
//
// It consumes the shipped declarative taxonomy in internal/corelocks (issue
// #1681) as-is: corelocks owns the class/glob/reason vocabulary; this package
// only classifies a set of changed paths against that taxonomy and folds the
// result into a closed-schema Report. The motivation (issue #1680): existing
// gates catch specific cases after the fact, but there is no single audit-only
// view that says which coherence-bearing surface a change touched and what
// witness command would clear it.
//
// In THIS phase the audit is measurement-only. Verdicts are advisory:
//   - an open-leaf path (no declared glob claims it) is "ok";
//   - a coherence-bearing class (hard-self / serial-core / soft-contract) is
//     "warn", carrying its data-only reason token plus the witness that would
//     clear it;
//   - shadow-learn is an observed/learning surface: classified, but advisory
//     "ok" (it raises no hard coherence demand in this phase).
//
// A "warn" never fails: Audit returns no error for a warning, and the Report's
// boolean OK() reflects only that no class crossed into "refuse" (no class does
// in this phase). The report copies read paths and DECLARED witnesses only — it
// never copies private or operator-only evidence.
package corelockaudit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/corelocks"
)

// Verdict is the closed advisory verdict vocabulary for a finding. In this
// measurement-only phase the audit never emits VerdictRefuse — it is part of
// the closed schema so a later enforcement phase can use it without a schema
// change.
type Verdict string

const (
	// VerdictOK is an advisory pass: the path raises no coherence demand.
	VerdictOK Verdict = "ok"
	// VerdictWarn is an advisory warning: the path touches a coherence-bearing
	// class. It carries a reason token and the witness that would clear it. A
	// warn does NOT fail the audit in this phase.
	VerdictWarn Verdict = "warn"
	// VerdictRefuse is reserved for a later enforcement phase; this phase never
	// emits it.
	VerdictRefuse Verdict = "refuse"
)

// witnessByClass maps each coherence-bearing lock class to the DECLARED witness
// that would clear a warning on it, plus a short source note. These are the
// public, declared clearing commands — never private or operator-only evidence.
// open-leaf and shadow-learn are intentionally absent: they raise no warning.
var witnessByClass = map[string]struct {
	witnesses []string
	note      string
}{
	"hard-self": {
		witnesses: []string{"dos commit-audit HEAD", "dos review origin/main..HEAD"},
		note:      "core-lock taxonomy/internal/adjudicator: a self-modifying surface; the diff-witnessed commit audit must confirm the claim.",
	},
	"serial-core": {
		witnesses: []string{"dos arbitrate (serial lane held)", "dos verify <plan> <phase>"},
		note:      "serial-core surface (dos.toml / internal/resume): must be taken on a single serial lane; the arbiter proves no concurrent holder.",
	},
	"soft-contract": {
		witnesses: []string{"go test ./internal/corelockaudit/", "dos verify <plan> <phase>"},
		note:      "soft-contract surface (internal/canon / internal/covmatrix): a contract witness (the package's own tests) must pass.",
	},
}

// Finding is one closed-schema row of the audit: a lock class touched by the
// changed set, the paths that mapped onto it, the reason token it raises, the
// declared witnesses that would clear it, the advisory verdict, and a source
// note. It carries only read paths and declared witnesses — never private
// evidence.
type Finding struct {
	// LockID is a stable identifier for the finding within a report: the lock
	// class name. (Class is the human label; LockID is what a later phase keys
	// on. They coincide today because there is one finding per class.)
	LockID string `json:"lock_id"`
	// Class is the corelocks class name the paths mapped onto.
	Class string `json:"class"`
	// Paths are the changed repo-relative paths that mapped onto this class,
	// sorted for determinism.
	Paths []string `json:"paths"`
	// ReasonToken is the data-only corelocks reason token for the class (empty
	// for open-leaf).
	ReasonToken string `json:"reason_token"`
	// RequiredWitnesses are the DECLARED commands that would clear a warning on
	// this class. Empty for an "ok" verdict.
	RequiredWitnesses []string `json:"required_witnesses"`
	// Verdict is the advisory verdict (ok|warn|refuse). This phase emits only
	// ok or warn.
	Verdict Verdict `json:"verdict"`
	// SourceNote is a short, public note/doc-link explaining the class. Never
	// private or operator-only evidence.
	SourceNote string `json:"source_note"`
}

// Report is the deterministic fold over a changed-path set: the findings (one
// per touched class, sorted by class name) plus rolled-up counts. The schema is
// closed and JSON-stable.
type Report struct {
	// Findings is one row per lock class touched by the changed set, sorted by
	// class name for determinism.
	Findings []Finding `json:"findings"`
	// Changed is the count of changed paths classified.
	Changed int `json:"changed"`
	// Warnings is the count of findings with verdict=warn.
	Warnings int `json:"warnings"`
	// Refusals is the count of findings with verdict=refuse (always 0 in this
	// measurement-only phase).
	Refusals int `json:"refusals"`
}

// OK reports whether the audit found nothing that would fail. In this
// measurement-only phase warnings never fail, so OK is true unless a finding
// crossed into VerdictRefuse (which no finding does in this phase). A later
// enforcement phase can tighten this without a schema change.
func (r Report) OK() bool { return r.Refusals == 0 }

// Audit classifies each changed path against the supplied taxonomy and folds the
// result into a Report. It is pure: no I/O, no error. Pass the taxonomy from
// corelocks.LoadFixture (or any parsed *corelocks.Taxonomy). A nil taxonomy
// yields an empty report.
//
// Each path is mapped to its class via taxonomy.Classify. Paths are grouped by
// class; one Finding is emitted per touched class. open-leaf and shadow-learn
// findings are advisory "ok"; hard-self / serial-core / soft-contract findings
// are advisory "warn" with the declared witness that would clear them.
func Audit(taxonomy *corelocks.Taxonomy, changedPaths []string) Report {
	r := Report{Findings: []Finding{}}
	if taxonomy == nil {
		return r
	}

	type bucket struct {
		paths  map[string]bool
		reason string
	}
	byClass := map[string]*bucket{}

	for _, p := range changedPaths {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		class, reason := taxonomy.Classify(p)
		b := byClass[class]
		if b == nil {
			b = &bucket{paths: map[string]bool{}, reason: reason}
			byClass[class] = b
		}
		b.paths[p] = true
		r.Changed++
	}

	classes := make([]string, 0, len(byClass))
	for c := range byClass {
		classes = append(classes, c)
	}
	sort.Strings(classes)

	for _, class := range classes {
		b := byClass[class]
		paths := make([]string, 0, len(b.paths))
		for p := range b.paths {
			paths = append(paths, p)
		}
		sort.Strings(paths)

		f := Finding{
			LockID:            class,
			Class:             class,
			Paths:             paths,
			ReasonToken:       b.reason,
			RequiredWitnesses: []string{},
			Verdict:           VerdictOK,
		}

		if w, ok := witnessByClass[class]; ok {
			f.Verdict = VerdictWarn
			f.RequiredWitnesses = append([]string{}, w.witnesses...)
			f.SourceNote = w.note
			r.Warnings++
		}

		r.Findings = append(r.Findings, f)
	}

	return r
}

// JSON renders the report as deterministic, indented JSON. The same report
// always yields the same bytes (findings and paths are pre-sorted).
func (r Report) JSON() ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return nil, err
	}
	// json.Encoder.Encode appends a trailing newline; trim it so callers get
	// exactly the document bytes.
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// String renders a concise, human-readable summary of the report.
func (r Report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "core-lock audit: %d changed path(s), %d warning(s), %d refusal(s)\n",
		r.Changed, r.Warnings, r.Refusals)
	if len(r.Findings) == 0 {
		b.WriteString("  (no paths classified)\n")
		return b.String()
	}
	for _, f := range r.Findings {
		fmt.Fprintf(&b, "  [%s] %s (%d path(s))", strings.ToUpper(string(f.Verdict)), f.Class, len(f.Paths))
		if f.ReasonToken != "" {
			fmt.Fprintf(&b, " reason=%s", f.ReasonToken)
		}
		b.WriteString("\n")
		for _, p := range f.Paths {
			fmt.Fprintf(&b, "      %s\n", p)
		}
		if len(f.RequiredWitnesses) > 0 {
			fmt.Fprintf(&b, "      witness to clear: %s\n", strings.Join(f.RequiredWitnesses, "; "))
		}
	}
	return b.String()
}
