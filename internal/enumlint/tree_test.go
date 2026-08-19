package enumlint

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// repoRoot walks up from this package to the module root.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("no go.mod above %s", dir)
	return ""
}

// censusReferenceSHA names the committed tree on which the landing census was
// captured. The baseline is the mechanical ratchet; this immutable name makes
// the published counts independently reproducible rather than "whatever HEAD
// meant when the test happened to run".
const censusReferenceSHA = "e8f9a8054539ce520f6d133930d6b7a5baf30724"

func scanTree(t *testing.T) Report {
	t.Helper()
	rep, err := Scan(repoRoot(t), Config{IncludeTestFiles: true, IncludeTopDirs: []string{"internal"}})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	return rep
}

// TestTreeEnumExhaustivenessNotGrowing is the deliverable gate, and it is a
// RATCHET rather than a refusal.
//
// #5935's non-goal is explicit: do not refuse on a finding at landing time. The
// tree carries real non-exhaustive sites today — the census is printed by
// TestScanActuallyReadTheTree and written up in
// docs/notes/ENUM-EXHAUSTIVENESS-CENSUS-2026-08-08.md — and a gate that reddened
// the trunk the hour it landed would be reverted the same hour, after which the
// sites go back to being invisible. So this fails only on GROWTH past the
// counted floor in baseline.txt.
//
// When it reds, the named site is almost never in this package: a new
// non-exhaustive switch is written under cmd/… or internal/<other>/…, whose
// author runs their own lane's suite and never sees this gate. Pay it at the
// source file, in the lane that owns it. Pasting the finding into baseline.txt
// is the one move the ratchet forbids — it is shrink-only, and grandfathering a
// NEW site retires the signal instead of paying it.
func TestTreeEnumExhaustivenessNotGrowing(t *testing.T) {
	rep := scanTree(t)
	base, err := LoadBaseline()
	if err != nil {
		t.Fatalf("LoadBaseline: %v", err)
	}
	res := Ratchet(rep.Findings, base)
	for _, f := range res.New {
		t.Errorf("NEW non-exhaustive site (above the recorded floor): %s", f.Msg)
	}
	t.Logf("%d finding(s) total; baseline floor %d, held %d, above floor %d, %d key(s) with slack",
		len(rep.Findings), base.Total(), res.Held, len(res.New), len(res.Slack))
}

// TestScanActuallyReadTheTree is the control that stops the gate above passing
// because it looked at nothing.
//
// A probe that reports NOTHING and is read as reporting CLEAN is the one failure
// an in-sweep control cannot catch, so "not growing" is only meaningful once it
// is established that the scan parsed a substantial tree, discovered the
// enumerations that are known to be there, and READ consumption sites. Sites is
// the sharpest of the four: enums without sites means the rules recognised
// nothing, which is the shape a refactor of resolveSwitch could silently cause.
func TestScanActuallyReadTheTree(t *testing.T) {
	rep := scanTree(t)
	if rep.Packages < 100 {
		t.Errorf("scan found enumerations in only %d package(s); this tree has 500+ internal packages, "+
			"so the walk or the discovery is broken and a not-growing verdict means nothing", rep.Packages)
	}
	if rep.Enums < 200 {
		t.Errorf("scan discovered %d enumeration(s); expected many more", rep.Enums)
	}
	if rep.Members < 1000 {
		t.Errorf("scan discovered %d constant(s) across those enumerations; expected many more", rep.Members)
	}
	if rep.Sites < 200 {
		t.Errorf("scan READ only %d consumption site(s); with this many enumerations the rules are "+
			"recognising almost nothing, so a clean verdict is a blind one", rep.Sites)
	}
	for _, u := range rep.Unparsed {
		// Not fatal: this tree is shared by ~20 live sessions and a peer's
		// half-written file is routine. Failing on someone else's editor state
		// would make this gate the fleet's problem rather than the tree's. But a
		// skipped file must never read as a checked one.
		t.Logf("unparsed (NOT checked): %s", u)
	}
	byRule := rep.CountByRule()
	t.Logf("CENSUS at named SHA %s: %s", censusReferenceSHA, rep.Census())
	t.Logf("CENSUS at named SHA %s: findings switch=%d literal=%d total=%d", censusReferenceSHA,
		byRule[RuleSwitch], byRule[RuleLiteral], len(rep.Findings))
}

// TestKnownVocabulariesAreDiscovered pins the specific closed vocabularies #5935
// names, by enumeration rather than by count.
//
// A census number alone is satisfiable by discovering two hundred of the wrong
// types. These are the ticket's subjects: internal/wipref's CensusClass is the
// one place in the tree that hand-checks exhaustiveness today
// (TestClassifyVocabulary), and the rest are the packages the ticket lists as
// re-implementing that coverage by hand or not at all. If a refactor moves or
// renames one, this says so instead of the tree-wide gate quietly stopping
// covering it.
func TestKnownVocabulariesAreDiscovered(t *testing.T) {
	root := repoRoot(t)
	cases := []struct {
		dir     string
		typ     string
		atLeast int
		why     string
	}{
		{"internal/wipref", "CensusClass", 5,
			"the ticket's exemplar: the ONE place the tree hand-checks exhaustiveness (TestClassifyVocabulary)"},
		{"internal/turnkind", "Kind", 4,
			"an iota enumeration — under-counting it makes every assertion over it vacuous"},
		{"internal/adjudicator", "ReversibilityClass", 2,
			"a closed verdict vocabulary in the package the ticket names"},
		{"internal/policy", "AmendmentClass", 2,
			"policy's closed amendment vocabulary"},
	}
	for _, c := range cases {
		t.Run(c.dir+"."+c.typ, func(t *testing.T) {
			p, unparsed, err := LoadPackage(root, filepath.Join(root, filepath.FromSlash(c.dir)), true)
			if err != nil {
				t.Fatalf("LoadPackage: %v", err)
			}
			if len(unparsed) > 0 {
				t.Logf("unparsed in %s: %s", c.dir, strings.Join(unparsed, "; "))
			}
			if p == nil {
				t.Fatalf("no enumerations discovered in %s at all (%s)", c.dir, c.why)
			}
			e, ok := p.Enums[c.typ]
			if !ok {
				names := p.EnumNames()
				sort.Strings(names)
				t.Fatalf("%s.%s was not discovered (%s); the package yielded: %s",
					c.dir, c.typ, c.why, strings.Join(names, ", "))
			}
			if len(e.Members) < c.atLeast {
				t.Errorf("%s.%s: discovered %d constant(s), want >= %d (%s) — an under-count here "+
					"makes every exhaustiveness assertion over it vacuous",
					c.dir, c.typ, len(e.Members), c.atLeast, c.why)
			}
			t.Logf("%s.%s: %d constant(s): %s", c.dir, c.typ, len(e.Members), strings.Join(e.Names(), ", "))
		})
	}
}

// TestNoStaleExemptions keeps the one hand-maintained list in this package
// honest in the direction it can actually go wrong.
//
// exemptions fails CLOSED — a site with no entry is a finding — so it cannot go
// wrong by being short. It goes wrong by being LONG: a site is fixed or deleted,
// its exemption stays, and whatever lands under that name next is silently
// unchecked forever. So every key must still name a site the linter WOULD
// otherwise report, proven by re-scanning with exemptions disabled.
func TestNativeBenchAlternativesExemptionExplainsPartialVocabulary(t *testing.T) {
	key := "literal|internal/nativebench|contracts.Alternatives"
	reason := ExemptionReason(key)
	if !strings.Contains(reason, "applicable comparison classes") {
		t.Fatalf("exemption %q lacks benchmark-specific reason: %q", key, reason)
	}
}

func TestNoStaleExemptions(t *testing.T) {
	for _, k := range ExemptionKeys() {
		if strings.TrimSpace(ExemptionReason(k)) == "" {
			t.Errorf("%v", ErrBlankReason(k))
		}
	}
	if len(ExemptionKeys()) == 0 {
		t.Log("no exemptions recorded; #5935 ships its debt as a counted baseline rather than as " +
			"exemptions, so an empty table is the expected landing state")
		return
	}
	rep, err := Scan(repoRoot(t), Config{IncludeTestFiles: true, IncludeTopDirs: []string{"internal"}, Exempt: NoExemptions})
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	found := map[string]bool{}
	for _, f := range rep.Findings {
		found[exemptKey(f.Rule, f.Pkg, f.Owner)] = true
	}
	for _, k := range ExemptionKeys() {
		if !found[k] {
			t.Errorf("stale exemption %q: nothing at that site is reported any more. Delete the entry — "+
				"an exemption outliving its site silently un-checks whatever lands there next.", k)
		}
	}
}

// TestBaselineIsParseableAndTight guards the ratchet's own file. A baseline that
// will not parse is a hard error everywhere else in this package, so it must not
// be possible to commit one; and a floor recorded for a key that no longer fires
// is slack that a regeneration should have banked.
func TestBaselineIsParseableAndTight(t *testing.T) {
	base, err := LoadBaseline()
	if err != nil {
		t.Fatalf("baseline.txt does not parse: %v", err)
	}
	rep := scanTree(t)
	live := map[string]int{}
	for _, f := range rep.Findings {
		live[f.Key()]++
	}
	stale := 0
	for _, k := range base.Keys() {
		if live[k] == 0 {
			stale++
			t.Logf("baseline key with no live finding (regenerate to bank it): %s", strings.ReplaceAll(k, "\t", ":"))
		}
	}
	t.Logf("baseline: %d key(s), floor total %d; live findings %d; %d key(s) fully burnt down",
		len(base.Keys()), base.Total(), len(rep.Findings), stale)
}
