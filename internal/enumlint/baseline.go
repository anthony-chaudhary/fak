package enumlint

import (
	"bufio"
	_ "embed"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// baseline.go — the COUNTED RATCHET that makes this tool safe to land on a tree
// that still carries debt.
//
// #5935's non-goal is explicit: do not refuse on a finding at landing time. The
// tree has real non-exhaustive sites today (the named-SHA census is emitted by
// TestScanActuallyReadTheTree), and a gate that reddened
// the trunk the hour it landed would be reverted the same hour — after which the
// sites go back to being invisible, which is strictly worse than before. So the
// gate makes the strongest statement that is TRUE: this class is not growing.
//
// Four clauses, and each one is load-bearing:
//
//  1. Key findings by something STABLE UNDER EDITING. Finding.Key is
//     rule+package+owner+type and carries no line number, so inserting a
//     paragraph above a site does not present as a fresh finding in a diff that
//     never touched it.
//  2. Store COUNTS, not locations. Two holes in one function are a floor of 2;
//     fixing one and regenerating tightens the floor to 1 and it can never go
//     back up.
//  3. HARD-FAIL on an unparseable baseline row. A lenient parser that skipped a
//     malformed line would read that key's floor as ZERO, so this package's own
//     bug would present as a fresh finding in somebody else's diff — the exact
//     failure mode a ratchet exists to avoid.
//  4. Only ever claim "not growing", never "clean". Ratchet returns the excess;
//     it never returns the baseline itself as a pass.
//
// fak already runs one ratchet of this shape and it is load-bearing —
// internal/pythongate (file names) and internal/boundarylint (file names). This
// is the counted form: a file-name ratchet cannot tell "one more hole in a file
// that already had one" from "no change", and for enums that is the whole
// question.

//go:embed baseline.txt
var baselineText string

// Baseline is a floor per finding key.
type Baseline struct {
	floors map[string]int
	// Header is the leading comment block, preserved across a regeneration so
	// the file's own instructions do not have to be re-typed.
	Header []string
}

// Floor returns the recorded count for a key; 0 when the key is unknown, which
// makes an unrecorded site a finding.
func (b *Baseline) Floor(key string) int {
	if b == nil {
		return 0
	}
	return b.floors[key]
}

// Keys returns the baseline's keys, sorted.
func (b *Baseline) Keys() []string {
	out := make([]string, 0, len(b.floors))
	for k := range b.floors {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Total is the sum of every floor — the debt the baseline records.
func (b *Baseline) Total() int {
	n := 0
	for _, v := range b.floors {
		n += v
	}
	return n
}

// LoadBaseline parses the in-tree baseline.txt. It is embedded rather than read
// from disk so callers behave identically from any working directory —
// a ratchet whose floor depends on the caller's cwd is a ratchet that fires at
// random.
func LoadBaseline() (*Baseline, error) {
	return ParseBaseline(strings.NewReader(baselineText))
}

// ParseBaseline reads the counted format:
//
//	<rule>\t<pkg>\t<owner>\t<type>\t<count>
//
// Blank lines and lines beginning with '#' are skipped. ANYTHING ELSE THAT DOES
// NOT PARSE IS AN ERROR, never a skipped line — see clause 3 above.
func ParseBaseline(r io.Reader) (*Baseline, error) {
	b := &Baseline{floors: map[string]int{}}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	lineNo := 0
	inHeader := true
	for sc.Scan() {
		lineNo++
		raw := strings.TrimRight(sc.Text(), "\r")
		if strings.TrimSpace(raw) == "" {
			if inHeader {
				b.Header = append(b.Header, raw)
			}
			continue
		}
		if strings.HasPrefix(strings.TrimSpace(raw), "#") {
			if inHeader {
				b.Header = append(b.Header, raw)
			}
			continue
		}
		inHeader = false
		parts := strings.Split(raw, "\t")
		if len(parts) != 5 {
			return nil, fmt.Errorf("enumlint: baseline line %d has %d tab-separated field(s), want 5 "+
				"(<rule>\\t<pkg>\\t<owner>\\t<type>\\t<count>): %q — a baseline row that does not parse is an "+
				"ERROR, never a skipped line: skipping it would read as a floor of zero for that key and turn "+
				"this package's bug into a denial in somebody else's diff", lineNo, len(parts), raw)
		}
		rule := parts[0]
		if rule != RuleSwitch && rule != RuleLiteral {
			return nil, fmt.Errorf("enumlint: baseline line %d names unknown rule %q (have %s)",
				lineNo, rule, strings.Join(AllRules(), ", "))
		}
		n, err := strconv.Atoi(parts[4])
		if err != nil {
			return nil, fmt.Errorf("enumlint: baseline line %d has a non-numeric count %q: %w", lineNo, parts[4], err)
		}
		if n < 1 {
			return nil, fmt.Errorf("enumlint: baseline line %d has count %d; a floor below 1 is a row that "+
				"should have been deleted", lineNo, n)
		}
		key := strings.Join(parts[:4], "\t")
		if _, dup := b.floors[key]; dup {
			return nil, fmt.Errorf("enumlint: baseline line %d repeats key %q; counts must be folded into one row",
				lineNo, strings.ReplaceAll(key, "\t", ":"))
		}
		b.floors[key] = n
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("enumlint: reading baseline: %w", err)
	}
	return b, nil
}

// RatchetResult is what a gate reads.
type RatchetResult struct {
	// New are the findings ABOVE the floor — the growth the ratchet refuses.
	New []Finding
	// Held is how many findings were absorbed by the baseline.
	Held int
	// Slack lists keys whose live count is BELOW their floor, with the surplus.
	// Not a failure: it is the burn-down that a regeneration will bank, and
	// naming it is how a maintainer knows there is something to bank.
	Slack map[string]int
}

// Ratchet compares a scan's findings against a baseline.
//
// A key with floor N absorbs its first N findings and reports the rest. The
// excess is chosen by the findings' own sort order (file, then line), so the
// same tree always reports the same site.
func Ratchet(findings []Finding, b *Baseline) RatchetResult {
	byKey := map[string][]Finding{}
	var order []string
	for _, f := range findings {
		k := f.Key()
		if _, seen := byKey[k]; !seen {
			order = append(order, k)
		}
		byKey[k] = append(byKey[k], f)
	}
	res := RatchetResult{Slack: map[string]int{}}
	for _, k := range order {
		fs := byKey[k]
		floor := b.Floor(k)
		if len(fs) <= floor {
			res.Held += len(fs)
			if len(fs) < floor {
				res.Slack[k] = floor - len(fs)
			}
			continue
		}
		res.Held += floor
		res.New = append(res.New, fs[floor:]...)
	}
	// A key in the baseline that produced nothing at all is pure slack too.
	for _, k := range b.Keys() {
		if _, live := byKey[k]; !live {
			res.Slack[k] = b.Floor(k)
		}
	}
	sort.Slice(res.New, func(i, j int) bool {
		a, c := res.New[i], res.New[j]
		if a.File != c.File {
			return a.File < c.File
		}
		if a.Line != c.Line {
			return a.Line < c.Line
		}
		return a.Msg < c.Msg
	})
	return res
}

// baselineHeader is the prose written at the top of a regenerated baseline. It
// lives here rather than only in the file so a regeneration cannot silently drop
// the instructions for regenerating.
const baselineHeader = `# internal/enumlint baseline — the COUNTED floor for #5935.
#
# Format: <rule>\t<pkg>\t<owner>\t<type>\t<count>
# Blank lines and '#' comments are skipped. Any other line that does not parse
# is a hard ERROR, never a skipped line: a skipped row reads as a floor of zero
# for that key, which turns this package's bug into a denial in someone else's
# diff.
#
# This file records the non-exhaustive sites that EXISTED when the gate landed.
# It is SHRINK-ONLY in spirit: fix a site, regenerate, and the floor tightens
# and can never go back up.
#
# Regenerate with FormatBaseline over a whole-tree Scan; TestBaselineIsParseableAndTight
#
# When the ratchet reds on a NEW site, do NOT paste it in here. The finding
# names the file, the line, and every missing member with its own declaration
# line — pay it at the source, in the lane that owns it, in one of two ways:
#   * cover the member (add the case / the map row / the slice element), or
#   * decide about it explicitly: give the switch a default, or write
#     //enumlint:exempt <reason> at the site, or add a reasoned entry to
#     exemptions in internal/enumlint/exempt.go.
`

// FormatBaseline renders findings as a baseline file, sorted and folded to one
// row per key.
func FormatBaseline(findings []Finding) string {
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
	return sb.String()
}
