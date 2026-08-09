package breath

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// BaselineFile is the accepted counted floor's tracked, repo-relative location.
// It lives beside the package that parses it so the command and checker cannot
// silently point at different authority files.
const BaselineFile = "internal/promptlint/breath/baseline.txt"

// The ratchet. fak's doc corpus predates the contract, so a gate that denied on any
// finding would deny on almost every page and would be wrong far more often than right.
// A ratchet is the strongest TRUE statement available — "this class is not growing" — and
// it is what makes the gate safe to run on a tree that still has the debt in it.
//
// Four clauses, the same contract internal/pythongate runs:
//
//	(a) key findings by something STABLE UNDER EDITING — Finding.Key is KIND<TAB>path, so
//	    inserting a sentence above a finding does not renumber it;
//	(b) store COUNTS, not line numbers, so fixing one of two findings tightens the floor
//	    on regeneration and adding a third is still caught;
//	(c) HARD-FAIL on an unparseable baseline row, naming the line — a lenient parser that
//	    skipped a malformed row would read as a floor of ZERO for that key, so this
//	    package's own bug would present as a fresh finding in someone else's diff;
//	(d) only ever claim "not growing", never "clean".

// Baseline is the frozen floor: Finding.Key -> the number of findings grandfathered for
// that key.
type Baseline map[string]int

// BaselineHeader is the comment block a regenerated baseline carries, so the file itself
// says what it is and which half of the contract produced it.
const BaselineHeader = "# fak breath ratchet baseline — KIND<TAB>path<TAB>count.\n" +
	"# Regenerate with: fak breath --emit-baseline > " + BaselineFile + "\n" +
	"# This is a FLOOR, not an allowlist: a key may shrink freely, and any count ABOVE\n" +
	"# its floor is a NEW finding. A green ratchet claims only that this class is not\n" +
	"# growing — never that the corpus is clean.\n" +
	"# " + ScopeNotice + "\n"

// ParseBaseline reads a `KIND<TAB>path<TAB>count` baseline.
//
// Blank lines and `#` comments are skipped. EVERY other line that does not parse is a
// hard error naming the 1-based line number — see clause (c) above. That strictness is
// the whole reason this parser is worth having.
func ParseBaseline(r io.Reader) (Baseline, error) {
	out := Baseline{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for n := 1; sc.Scan(); n++ {
		line := sc.Text()
		if t := strings.TrimSpace(line); t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		f := strings.Split(line, "\t")
		if len(f) != 3 {
			return nil, fmt.Errorf("breath baseline line %d: want 3 tab-separated fields "+
				"(KIND<TAB>path<TAB>count), got %d in %q", n, len(f), line)
		}
		kind, path, count := strings.TrimSpace(f[0]), strings.TrimSpace(f[1]), strings.TrimSpace(f[2])
		if kind == "" || path == "" {
			return nil, fmt.Errorf("breath baseline line %d: empty kind or path in %q", n, line)
		}
		if !knownKind(kind) {
			return nil, fmt.Errorf("breath baseline line %d: %q is not a breath finding kind "+
				"(known: %s)", n, kind, strings.Join(kindNames(), ", "))
		}
		c, err := strconv.Atoi(count)
		if err != nil || c < 0 {
			return nil, fmt.Errorf("breath baseline line %d: count %q is not a non-negative integer", n, count)
		}
		out[kind+"\t"+path] += c
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("breath baseline: read: %w", err)
	}
	return out, nil
}

func knownKind(s string) bool {
	for _, k := range Kinds() {
		if string(k) == s {
			return true
		}
	}
	return false
}

func kindNames() []string {
	out := make([]string, 0, len(Kinds()))
	for _, k := range Kinds() {
		out = append(out, string(k))
	}
	return out
}

// Counts folds findings into per-key counts — the shape the baseline stores.
func Counts(findings []Finding) map[string]int {
	out := map[string]int{}
	for _, f := range findings {
		out[f.Key()]++
	}
	return out
}

// FormatBaseline renders findings as a regenerated baseline, sorted so a diff is stable.
func FormatBaseline(findings []Finding) string {
	counts := Counts(findings)
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteString(BaselineHeader)
	for _, k := range keys {
		fmt.Fprintf(&b, "%s\t%d\n", k, counts[k])
	}
	return b.String()
}

// Ratchet returns the findings ABOVE the baseline floor: for each key, the trailing
// (count - floor) findings. Returning the trailing ones rather than "the second one" is
// the honest rendering — two findings of the same kind on the same page have no stable
// distinguishing identity, which is exactly why the baseline counts instead of pinning
// lines.
//
// BreathScanFloor is never ratcheted away: a run that examined too little to have an
// opinion is a defect in the gate, not grandfathered doc debt.
func Ratchet(findings []Finding, base Baseline) []Finding {
	seen := map[string]int{}
	var out []Finding
	for _, f := range findings {
		if f.Kind == BreathScanFloor {
			out = append(out, f)
			continue
		}
		seen[f.Key()]++
		if seen[f.Key()] > base[f.Key()] {
			out = append(out, f)
		}
	}
	return out
}
