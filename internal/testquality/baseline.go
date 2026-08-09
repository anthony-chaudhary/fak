package testquality

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// BaselineFile is the accepted floor's tracked location, repo-relative. It lives
// beside the package that reads it so the gate and the tool can never end up
// pointing at two different files.
const BaselineFile = "internal/testquality/baseline.txt"

// Baseline is the accepted floor: how many candidates of each kind the tree is
// already known to carry, keyed by Finding.Key(). A finding is NEW only when its
// key's running count EXCEEDS the floor.
//
// A ratchet and not a refusal, because a finding is a CANDIDATE: some of them are
// correct as written (a table whose expectation is asserted through a helper, an
// error deliberately dropped in a teardown path). A gate that denied on any
// candidate would be wrong often enough to be switched off, and a switched-off
// checker reports zero findings — which is byte-identical to a clean tree. "This
// class is not growing" is the strongest TRUE statement available here.
type Baseline map[string]int

// A COUNT rather than a per-finding line, because two findings of the same code
// in the same function have no stable distinguishing identity: the second one is
// only "the second one", and inserting a line above it would renumber it.
// Counting means fixing one of two still tightens the floor on regeneration, and
// adding a third is still caught.

// ParseBaseline reads the tab-separated baseline format:
//
//	CODE<TAB>path/to/file_test.go<TAB>TestFuncName<TAB>count
//
// Blank lines and `#` comments are skipped. EVERYTHING else that does not parse
// is an error naming the line number, never a skipped line. A baseline that
// silently dropped a malformed entry would read as a floor of ZERO for that key,
// so the ratchet's own bug would present as a fresh finding in somebody else's
// diff; and a typo'd key that parsed anyway would permanently absorb a real
// finding. Both directions are silent, which is why neither is tolerated.
func ParseBaseline(src []byte) (Baseline, error) {
	b := Baseline{}
	for i, ln := range strings.Split(string(src), "\n") {
		ln = strings.TrimSuffix(ln, "\r")
		s := strings.TrimSpace(ln)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		f := strings.Split(ln, "\t")
		if len(f) != 4 {
			return nil, fmt.Errorf("baseline line %d: want 4 tab-separated fields "+
				"(CODE, file, func, count), got %d: %q", i+1, len(f), ln)
		}
		code, file, fn := strings.TrimSpace(f[0]), strings.TrimSpace(f[1]), strings.TrimSpace(f[2])
		if !knownCode(code) {
			return nil, fmt.Errorf("baseline line %d: unknown code %q (known: %s)",
				i+1, code, strings.Join(Codes, ", "))
		}
		if file == "" || fn == "" {
			return nil, fmt.Errorf("baseline line %d: empty file or func field: %q", i+1, ln)
		}
		n, err := strconv.Atoi(strings.TrimSpace(f[3]))
		if err != nil || n < 1 {
			return nil, fmt.Errorf("baseline line %d: count must be a positive integer, got %q",
				i+1, f[3])
		}
		key := code + "\t" + file + "\t" + fn
		if _, dup := b[key]; dup {
			return nil, fmt.Errorf("baseline line %d: duplicate entry for %s %s %s "+
				"(two rows for one key means one of them is dead and the floor it names is a guess)",
				i+1, code, file, fn)
		}
		b[key] = n
	}
	return b, nil
}

// baselineHeader is the preamble every regenerated baseline carries. It exists
// because the file's most likely reader is somebody who just had a finding
// reported at them and is about to "fix" a row here.
const baselineHeader = `# internal/testquality baseline — the accepted FLOOR, not a list of approved defects.
#
# Format: CODE<TAB>file<TAB>func<TAB>count. Regenerate with:
#   fak test-quality --write-baseline
#
# Every row is a CANDIDATE: a test whose SHAPE means it can pass with the code
# under test broken. Some are correct as written, so nothing here should be
# "fixed" without reading it first. This gate's whole contract is that the list
# does not GROW; it never claims the tree is clean.
#
# The key is (code, file, func) and the value is a COUNT, never a line number, so
# the floor survives the file being reformatted or the finding moving.
`

// FormatBaseline renders findings as a baseline file, sorted so regenerating it
// on an unchanged tree produces byte-identical output. A baseline whose line
// order depended on a map walk would show up as a diff in every commit that
// touched it, and a noisy generated file stops being read.
func FormatBaseline(findings []Finding) []byte {
	counts := map[string]int{}
	for _, f := range findings {
		counts[f.Key()]++
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var sb strings.Builder
	sb.WriteString(baselineHeader)
	for _, k := range keys {
		fmt.Fprintf(&sb, "%s\t%d\n", k, counts[k])
	}
	return []byte(sb.String())
}

// NewFindings splits findings into those the baseline already accounts for and
// those it does not, and reports which baseline keys are now LOOSER than the tree
// (a fix landed, so the floor can be lowered).
//
// findings must already be in a deterministic order — Analyze sorts by line and
// Scan sorts by path — so which of two same-key findings is called "new" is
// stable rather than whichever the walk happened to reach first.
func NewFindings(findings []Finding, b Baseline) (fresh []Finding, slack map[string]int) {
	seen := map[string]int{}
	for _, f := range findings {
		k := f.Key()
		seen[k]++
		if seen[k] > b[k] {
			fresh = append(fresh, f)
		}
	}
	slack = map[string]int{}
	for k, n := range b {
		if d := n - seen[k]; d > 0 {
			slack[k] = d
		}
	}
	return fresh, slack
}

// CountsByCode folds findings into a per-code tally, for the whole-tree report.
func CountsByCode(findings []Finding) map[string]int {
	out := map[string]int{}
	for _, c := range Codes {
		out[c] = 0
	}
	for _, f := range findings {
		out[f.Code]++
	}
	return out
}
