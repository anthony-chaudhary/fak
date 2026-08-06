package orphanscan

import (
	"reflect"
	"strings"
	"testing"
)

// TestGateEndToEnd is the DoD witness for #3167: over one synthetic package the gate
// must fail on a NEW orphan and stay silent for the two sanctioned suppressions — the
// local //orphanscan:keep directive and a checked-in allowlist entry. All three funcs
// are equally unreferenced, so only the suppression tier distinguishes them, which is
// exactly the discrimination the gate is claimed to make.
func TestGateEndToEnd(t *testing.T) {
	dir := t.TempDir()
	writeGo(t, dir, "handlers.go", `package fixture

// freshOrphan was just written and wired to nothing — the gate must catch this.
func freshOrphan() {}

//orphanscan:keep dispatched by name through a reflection table
func directiveKept() {}

// listedKept is name-dispatched from another package's table; the allowlist covers it.
func listedKept() {}
`)

	keep, err := ParseKeepAllowlist("cmd/fixture listedKept # name-dispatched from a table the AST cannot see\n")
	if err != nil {
		t.Fatalf("ParseKeepAllowlist: %v", err)
	}

	rep, err := ScanDirReport(dir, "cmd/fixture")
	if err != nil {
		t.Fatalf("ScanDirReport: %v", err)
	}
	if len(rep.Unparsed) != 0 {
		t.Fatalf("fixture should parse cleanly, got unparsed %v", rep.Unparsed)
	}
	// The directive is honored inside the scan, so it never even reaches the gate.
	if got := orphanNames(rep.Orphans); !reflect.DeepEqual(got, []string{"freshOrphan", "listedKept"}) {
		t.Fatalf("scan = %v, want [freshOrphan listedKept] (directiveKept is suppressed by ScanDir)", got)
	}

	unexpected, unused := Gate([]string{"cmd/fixture"}, rep.Orphans, keep)
	if got := orphanNames(unexpected); !reflect.DeepEqual(got, []string{"freshOrphan"}) {
		t.Errorf("gate flagged %v, want only [freshOrphan]", got)
	}
	if len(unused) != 0 {
		t.Errorf("unused = %v, want none (the allowlisted func is still an orphan)", unused)
	}
	// The refusal has to be actionable, not just a "no".
	if hint := SuppressionHint(unexpected[0]); !strings.Contains(hint, "cmd/fixture freshOrphan") {
		t.Errorf("SuppressionHint = %q, want the exact allowlist line to add", hint)
	}
}

// TestGateUnusedEntryOnlyForScannedPkg pins the partial-scan rule: an allowlist entry is
// judged stale only when its package was actually walked. Without that scoping a caller
// scanning one package would be told every other package's entries are dead.
func TestGateUnusedEntryOnlyForScannedPkg(t *testing.T) {
	keep, err := ParseKeepAllowlist(
		"cmd/fak wiredNow # baseline: was an orphan, since wired\n" +
			"internal/gateway notLookedAt # baseline: this run never scanned gateway\n")
	if err != nil {
		t.Fatalf("ParseKeepAllowlist: %v", err)
	}
	// Scanned cmd/fak only, and it came back clean.
	unexpected, unused := Gate([]string{"cmd/fak"}, nil, keep)
	if len(unexpected) != 0 {
		t.Errorf("unexpected = %v, want none", unexpected)
	}
	if len(unused) != 1 || unused[0].Key() != "cmd/fak:wiredNow" {
		t.Fatalf("unused = %+v, want only cmd/fak:wiredNow", unused)
	}
}

// TestGateMatchesOnPkgNotFile: the exemption key is package+func, so a func that moved
// to another file in the same package — or merely drifted down its file — stays exempt.
// Keying on file:line would make every neighbouring edit silently lapse a suppression.
func TestGateMatchesOnPkgNotFile(t *testing.T) {
	keep, err := ParseKeepAllowlist("cmd/fak movedFunc # baseline: pinned by package, not location\n")
	if err != nil {
		t.Fatalf("ParseKeepAllowlist: %v", err)
	}
	moved := []Orphan{{Name: "movedFunc", File: "cmd/fak/somewhere_else.go", Line: 900}}
	if unexpected, _ := Gate([]string{"cmd/fak"}, moved, keep); len(unexpected) != 0 {
		t.Errorf("gate flagged %v after the func moved file/line; the entry should still hold", unexpected)
	}
}

func TestParseKeepAllowlistFormat(t *testing.T) {
	src := "# a comment\n" +
		"\n" +
		"   # an indented comment\n" +
		"internal/gateway zebra # reason z\n" +
		"cmd\\fak alpha   #   reason a  \n" // backslashes normalise; reason is trimmed
	got, err := ParseKeepAllowlist(src)
	if err != nil {
		t.Fatalf("ParseKeepAllowlist: %v", err)
	}
	want := []KeepEntry{
		{Pkg: "cmd/fak", Func: "alpha", Reason: "reason a"},
		{Pkg: "internal/gateway", Func: "zebra", Reason: "reason z"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("entries = %+v, want %+v (sorted by key)", got, want)
	}
	if line := got[0].String(); line != "cmd/fak alpha # reason a" {
		t.Errorf("String() = %q, want the on-disk line format", line)
	}
}

// TestParseKeepAllowlistRejects: every malformed shape is an error naming its line. A
// reason is mandatory — an unexplained suppression is how a ratchet becomes a rubber
// stamp — and a duplicate key would let one edit silently shadow another.
func TestParseKeepAllowlistRejects(t *testing.T) {
	cases := map[string]string{
		"no reason marker": "cmd/fak lonely\n",
		"empty reason":     "cmd/fak lonely #   \n",
		"one field":        "cmd/fak # reason but no func\n",
		"three fields":     "cmd/fak a b # too many\n",
		"duplicate key":    "cmd/fak dup # first\ncmd/fak dup # second\n",
	}
	for name, src := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseKeepAllowlist(src); err == nil {
				t.Fatalf("ParseKeepAllowlist(%q) = nil error, want a rejection", src)
			} else if !strings.Contains(err.Error(), "keep_allowlist.txt:") {
				t.Errorf("error %q should name the offending line", err)
			}
		})
	}
}

// TestCheckedInAllowlistParses: the embedded file is the gate's own input, so a bad edit
// to it must fail here rather than at the moment the gate needs to make a decision.
func TestCheckedInAllowlistParses(t *testing.T) {
	keep, err := KeepAllowlist()
	if err != nil {
		t.Fatalf("KeepAllowlist: %v", err)
	}
	if len(keep) == 0 {
		t.Skip("allowlist is empty — nothing to validate (a legitimately clean tree)")
	}
	for _, k := range keep {
		if k.Pkg == "" || k.Func == "" || k.Reason == "" {
			t.Errorf("entry %+v has an empty field", k)
		}
		if strings.HasPrefix(k.Pkg, "/") || strings.Contains(k.Pkg, "\\") {
			t.Errorf("entry %s: Pkg must be a repo-relative slash path", k.Key())
		}
		if !isUnexported(k.Func) {
			t.Errorf("entry %s: only unexported funcs are ever scan candidates, so an "+
				"exported name can never match", k.Key())
		}
	}
}

// TestScanDirReportRecordsUnparsed: the caveat a gate needs. A file that will not parse
// contributes no references, so it may hide the only call site of a live func — the scan
// must say so rather than let the caller read a partial result as a complete one.
func TestScanDirReportRecordsUnparsed(t *testing.T) {
	dir := t.TempDir()
	writeGo(t, dir, "good.go", `package fixture

func lonely() {}
`)
	writeGo(t, dir, "broken.go", "package fixture\n\nfunc (((( syntax error\n")
	rep, err := ScanDirReport(dir, "internal/fixture")
	if err != nil {
		t.Fatalf("ScanDirReport: %v", err)
	}
	if want := []string{"internal/fixture/broken.go"}; !reflect.DeepEqual(rep.Unparsed, want) {
		t.Errorf("Unparsed = %v, want %v", rep.Unparsed, want)
	}
	if got := orphanNames(rep.Orphans); !reflect.DeepEqual(got, []string{"lonely"}) {
		t.Errorf("Orphans = %v, want [lonely]", got)
	}
}

// orphanNames is the compact form the assertions compare on.
func orphanNames(orphans []Orphan) []string {
	var names []string
	for _, o := range orphans {
		names = append(names, o.Name)
	}
	return names
}
