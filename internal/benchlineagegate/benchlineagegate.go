// Package benchlineagegate is the durable enforcement gate for issue #9: every
// benchmark emitter must stamp the four lineage axes (version / utc / git_commit /
// machine) onto the report artifact it writes, so a result is always traceable to
// the exact build that produced it. A new cmd/*bench* that writes a report without
// routing through benchcli's lineage stamper reds the trunk via the live test in
// this package.
//
// It is the bench-lineage sibling of internal/pythongate (the new-Python ratchet)
// and internal/conflationscore (the metric-conflation floor): a Go gate that scans
// the tracked tree and refuses a structural defect, run both standalone in `make
// hygiene` and automatically under `go test ./...`.
//
// The gate is deliberately a SHAPE check, not a semantic one: it confirms a report
// emitter references benchcli's stamper, not that the emitted lineage is correct at
// runtime (benchcli's own tests own that). That keeps it false-positive-free while
// making a lineage-free emitter mechanically un-shippable.
package benchlineagegate

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ReasonUnstampedEmitter is the closed-vocabulary refusal code for a bench source
// that writes a JSON report artifact but never stamps lineage onto it.
const ReasonUnstampedEmitter = "BENCH_EMITTER_UNSTAMPED"

// exemptMarker, anywhere in a file (it is a comment in practice), declares that the
// file intentionally writes JSON that is NOT a single benchmark result artifact —
// e.g. a JSONL stream of input fixtures — so a lineage block would be wrong. The
// marker must be paired with a human reason on the same line.
const exemptMarker = "lineage:exempt"

// Verdict classifies one bench source file's relationship to the lineage contract.
type Verdict int

const (
	// NotEmitter writes no JSON report artifact, so lineage is not required of it.
	NotEmitter Verdict = iota
	// Exempt declares lineage:exempt — an intentional, reasoned opt-out.
	Exempt
	// Stamped writes a report AND references benchcli's lineage stamper: compliant.
	Stamped
	// Unstamped writes a report but never references the stamper: the offense.
	Unstamped
)

// fileSinkSignals mark a write of bytes to a file (a persisted artifact).
var fileSinkSignals = []string{
	"os.WriteFile(",
	"benchcli.WriteFile(",
	"benchcli.WriteReport(",
}

// marshalSignals mark the production of JSON report bytes. "json.Marshal" matches
// json.MarshalIndent too and, importantly, does NOT match json.Unmarshal (so a file
// that only DECODES input is not mistaken for an emitter).
var marshalSignals = []string{
	"json.Marshal",
	".JSON()",
	"benchcli.MarshalReport(",
	"benchcli.WriteReport(",
}

// lineageSignals mark that a file routes through benchcli's lineage stamper.
var lineageSignals = []string{
	"benchcli.WriteReport(",
	"benchcli.MarshalReport(",
	"benchcli.Stamp(",
	"lineage_schema",
}

// Classify is the pure gate core: it decides a file's verdict from its source. A
// file is an "emitter" when it both produces JSON report bytes and writes bytes to
// a file; an emitter must reference benchcli's stamper unless it is exempt.
//
// Signal detection runs over the source with line comments stripped, so a doc
// comment mentioning os.WriteFile or json.Marshal cannot fabricate an emitter — but
// the exempt marker is sought in the RAW text, because it lives in a comment.
func Classify(src string) Verdict {
	if strings.Contains(src, exemptMarker) {
		return Exempt
	}
	code := stripLineComments(src)
	emits := strmatch.ContainsAny(code, fileSinkSignals...) && strmatch.ContainsAny(code, marshalSignals...)
	if !emits {
		return NotEmitter
	}
	if strmatch.ContainsAny(code, lineageSignals...) {
		return Stamped
	}
	return Unstamped
}

// Offense is one bench source file that writes a report without stamping lineage.
type Offense struct {
	Path string // repo-relative, e.g. "cmd/foobench/main.go"
}

// String renders the offense as a one-line fix report carrying the reason code.
func (o Offense) String() string {
	return fmt.Sprintf("%s writes a benchmark report without lineage; emit it through "+
		"benchcli.WriteReport / benchcli.MarshalReport (or mark %q with a reason) (%s)",
		o.Path, exemptMarker, ReasonUnstampedEmitter)
}

// ScanTree lists the tracked bench sources under repoRoot (cmd/*bench*/*.go plus
// internal/bench/*.go, test files excluded) and returns one Offense per file that
// writes a report without lineage, sorted for stable output.
func ScanTree(repoRoot string) ([]Offense, error) {
	paths, err := trackedBenchSources(repoRoot)
	if err != nil {
		return nil, err
	}
	var offenses []Offense
	for _, rel := range paths {
		b, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(rel)))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", rel, err)
		}
		if Classify(string(b)) == Unstamped {
			offenses = append(offenses, Offense{Path: rel})
		}
	}
	sort.Slice(offenses, func(i, j int) bool { return offenses[i].Path < offenses[j].Path })
	return offenses, nil
}

// trackedBenchSources shells out to git for the authoritative tracked-file list (so
// an untracked scratch .go is ignored — only files that would ship count) and keeps
// the non-test .go sources that are bench emitters: anything under internal/bench/,
// and anything under a cmd/<dir> whose directory name contains "bench".
func trackedBenchSources(repoRoot string) ([]string, error) {
	cmd := exec.Command("git", "ls-files", "cmd", "internal/bench")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files in %s: %w", repoRoot, err)
	}
	var paths []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		p := strings.ReplaceAll(strings.TrimSpace(line), "\\", "/")
		if p == "" || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			continue
		}
		if isBenchSource(p) {
			paths = append(paths, p)
		}
	}
	return paths, nil
}

// isBenchSource reports whether a repo-relative path is a bench emitter source in
// scope: under internal/bench/, or under a cmd/<dir-with-"bench"-in-its-name>/.
func isBenchSource(p string) bool {
	if strings.HasPrefix(p, "internal/bench/") {
		return true
	}
	parts := strings.Split(p, "/")
	return len(parts) >= 2 && parts[0] == "cmd" && strings.Contains(parts[1], "bench")
}

// stripLineComments removes // ... EOL from each line so signal detection ignores
// doc/comment text. It is intentionally naive (it does not parse string literals);
// the only effect of the rare "//" inside a string is to under-detect a signal,
// which can never turn a clean file into a false offense.
func stripLineComments(src string) string {
	var b strings.Builder
	for _, line := range strings.Split(src, "\n") {
		if i := strings.Index(line, "//"); i >= 0 {
			line = line[:i]
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String()
}
