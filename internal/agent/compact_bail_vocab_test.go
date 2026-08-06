package agent

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// compactReasonConstants parses every non-test .go file in dir and returns the wire token of
// each `CompactReason*` string constant, keyed by the Go constant name. It reads the SOURCE
// rather than the compiled package because Go constants are untyped strings with no runtime
// reflection surface: there is no way to enumerate them from inside the running test, which
// is exactly why the vocabulary could drift from its two consumers unnoticed (#5441).
//
// It is deliberately parameterised on dir. TestCompactBailReasonsRegistered points it at this
// package (the live check), and TestUnregisteredCompactBailReasonIsDetected points it at a
// byte-copy of this package plus one added reason (the check that the detector actually
// fires). Nothing is sampled or truncated — every file in dir is parsed and every matching
// constant is returned.
func compactReasonConstants(t *testing.T, dir string) map[string]string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	out := map[string]string{}
	fset := token.NewFileSet()
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, spec := range gd.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for i, ident := range vs.Names {
					if !strings.HasPrefix(ident.Name, "CompactReason") || i >= len(vs.Values) {
						continue
					}
					lit, ok := vs.Values[i].(*ast.BasicLit)
					if !ok || lit.Kind != token.STRING {
						continue
					}
					v, err := strconv.Unquote(lit.Value)
					if err != nil {
						t.Fatalf("%s: unquote %s: %v", name, ident.Name, err)
					}
					out[ident.Name] = v
				}
			}
		}
	}
	return out
}

// unregisteredCompactReasons returns the Go constant names in consts whose wire token is a
// bail (non-empty) but is absent from the registered vocabulary. CompactReasonNone is the
// FIRED outcome, not a bail, so its empty token is skipped rather than reported.
func unregisteredCompactReasons(consts map[string]string, registered []string) []string {
	known := map[string]bool{}
	for _, r := range registered {
		known[r] = true
	}
	var missing []string
	for name, wire := range consts {
		if wire == "" || known[wire] {
			continue
		}
		missing = append(missing, name+"="+strconv.Quote(wire))
	}
	sort.Strings(missing)
	return missing
}

// TestCompactBailReasonsRegistered is the live gate that #5441 asks for: every CompactReason*
// bail constant this package declares must be registered in compactBailReasonPreEligible.
// Adding a reason to the const block without registering it fails HERE, before it can reach a
// consumer that spells the vocabulary out by hand.
//
// It also checks the reverse direction, so the registry cannot accumulate ghosts: a reason
// registered here with no constant behind it would silently widen the rendered HELP's closed
// set with a label nothing can ever emit.
func TestCompactBailReasonsRegistered(t *testing.T) {
	consts := compactReasonConstants(t, ".")
	if len(consts) == 0 {
		t.Fatalf("scanned this package and found no CompactReason* constants — the scanner is broken, not the vocabulary")
	}
	if _, ok := consts["CompactReasonNone"]; !ok {
		t.Fatalf("scanned this package and did not find CompactReasonNone — the scanner is broken, not the vocabulary")
	}

	registered := CompactBailReasons()
	if missing := unregisteredCompactReasons(consts, registered); len(missing) > 0 {
		t.Fatalf("CompactReason* constant(s) not registered in compactBailReasonPreEligible: %v\n"+
			"registered vocabulary: %v\n"+
			"Add each token to compactBailReasonPreEligible in compact_bail_vocab.go, choosing the\n"+
			"side by WHERE the compactor decided: true if it returns from the opening eligibility\n"+
			"check (no compactible span existed yet), false if a real candidate was declined or\n"+
			"aborted. Leaving it out makes the rendered Prometheus HELP's closed-set claim false\n"+
			"and drops the reason from the alertable candidate-bail denominator.", missing, registered)
	}

	declared := map[string]bool{}
	for _, tok := range consts {
		declared[tok] = true
	}
	var ghosts []string
	for _, r := range registered {
		if !declared[r] {
			ghosts = append(ghosts, r)
		}
	}
	if len(ghosts) > 0 {
		t.Fatalf("registered bail reason(s) with no CompactReason* constant behind them: %v — the rendered HELP would advertise a label nothing can emit", ghosts)
	}
}

// TestUnregisteredCompactBailReasonIsDetected proves the gate above actually fires, by running
// the same scanner over a byte-copy of this package with ONE unregistered reason added. Without
// this, TestCompactBailReasonsRegistered passing would be indistinguishable from a scanner that
// silently matches nothing.
//
// The fixture is built in t.TempDir() from the real sources, so the corpus under test is
// production plus exactly one added constant.
func TestUnregisteredCompactBailReasonIsDetected(t *testing.T) {
	dir := t.TempDir()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	copied := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), src, 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
		copied++
	}
	if copied == 0 {
		t.Fatalf("copied no sources — the fixture would prove nothing")
	}

	// The clean copy must be as clean as the real package: if it were not, the positive
	// result below could come from the copy rather than from the added reason.
	registered := CompactBailReasons()
	if missing := unregisteredCompactReasons(compactReasonConstants(t, dir), registered); len(missing) > 0 {
		t.Fatalf("the unmodified copy already reports %v — the fixture cannot isolate the added reason", missing)
	}

	const probe = "package agent\n\n// A reason added to the const block and NOT registered — what this test exists to catch.\nconst CompactReasonProbeUnregistered = \"probe_unregistered\"\n"
	if err := os.WriteFile(filepath.Join(dir, "zz_probe_reason.go"), []byte(probe), 0o600); err != nil {
		t.Fatalf("write probe: %v", err)
	}

	missing := unregisteredCompactReasons(compactReasonConstants(t, dir), registered)
	want := `CompactReasonProbeUnregistered="probe_unregistered"`
	if len(missing) != 1 || missing[0] != want {
		t.Fatalf("unregistered reasons = %v, want exactly [%s] — an unregistered reason must red TestCompactBailReasonsRegistered", missing, want)
	}
	if CompactBailPreEligible("probe_unregistered") {
		t.Fatalf("an unregistered reason must fail OPEN to eligible, so the derived rate can only read conservatively high")
	}
}

// TestCompactBailReasonPartition pins WHICH side each registered reason sits on. The split is
// by where the compactor decided, not by how benign the outcome was: the late structural
// aborts (splice_failed, redecode_failed, prefix_mismatch, malformed_body) each abort a REAL
// candidate and must stay in the rate's denominator, while decode_failed — a fault, not an
// idle — sits on the pre-eligibility side because it returns from the same opening check.
func TestCompactBailReasonPartition(t *testing.T) {
	preEligible := []string{
		CompactReasonNonJSON,
		CompactReasonNoMsgsKey,
		CompactReasonDecodeFailed,
		CompactReasonTooFewMsgs,
	}
	eligible := []string{
		CompactReasonUnderBudget,
		CompactReasonNoBreakpoint,
		CompactReasonCachedSpan,
		CompactReasonWindowNoDrop,
		CompactReasonBurstUnprofitable,
		CompactReasonSpliceFailed,
		CompactReasonRedecodeFail,
		CompactReasonPrefixMismatch,
		CompactReasonMalformedBody,
		CompactReasonPinEvictRefused,
	}
	for _, r := range preEligible {
		if !CompactBailPreEligible(r) {
			t.Errorf("%q must be pre-eligibility: it returns before any compactible span exists, so counting it as a decline pins the rate near 1.0", r)
		}
	}
	for _, r := range eligible {
		if CompactBailPreEligible(r) {
			t.Errorf("%q must stay in the eligible half: a real candidate was declined or aborted, which is exactly what the rate measures", r)
		}
	}
	if n, want := len(CompactBailReasons()), len(preEligible)+len(eligible); n != want {
		t.Fatalf("CompactBailReasons() has %d members, want %d — this test enumerates both halves, so a new reason must be added to one of them", n, want)
	}

	// Fail OPEN: never registered, and never CompactReasonNone (the fired outcome).
	if CompactBailPreEligible("some_reason_from_the_future") {
		t.Errorf("an unknown reason must count as a candidate so the rate reads conservatively high")
	}
	if CompactBailPreEligible(CompactReasonNone) {
		t.Errorf("CompactReasonNone is the FIRED outcome, not a bail, and must never be classified")
	}
}

// TestCompactBailReasonsIsSortedAndOwnsItsSlice guards the two properties the render path
// depends on: a deterministic HELP string across scrapes, and a caller that cannot corrupt the
// process-global registry by sorting or appending to a shared backing array.
func TestCompactBailReasonsIsSortedAndOwnsItsSlice(t *testing.T) {
	got := CompactBailReasons()
	if !sort.StringsAreSorted(got) {
		t.Fatalf("CompactBailReasons() = %v, want sorted so the rendered HELP is byte-stable", got)
	}
	if len(got) == 0 {
		t.Fatalf("CompactBailReasons() is empty")
	}
	got[0] = "clobbered"
	if again := CompactBailReasons(); again[0] == "clobbered" {
		t.Fatalf("CompactBailReasons() handed out the registry's own backing array")
	}
	if strings.Contains(strings.Join(CompactBailReasons(), "|"), "||") {
		t.Fatalf("an empty member would render an empty label in the HELP's closed set")
	}
}
