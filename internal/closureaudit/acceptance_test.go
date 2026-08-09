package closureaudit

import (
	"fmt"
	"strings"
	"testing"
)

const bodyWithAcceptance = "## Problem\n" +
	"The spill path is not chosen anywhere. See `internal/hostplacement/spill.go` in the problem prose.\n" +
	"\n" +
	"## Acceptance\n" +
	"- `ResolveHostSpill` decides the spill target.\n" +
	"- the refusal is declared as `OVERLAY_WOULD_GATE`\n" +
	"- it is reachable from `cmd/fak/dispatch_tick_preflight.go`\n" +
	"```\n" +
	"go test ./internal/hostplacement/ -run `TestNeverExtracted`\n" +
	"```\n" +
	"- prose with `two words` and `ab` and `git` stay out\n" +
	"\n" +
	"## Notes\n" +
	"`NeverExtractedFromNotes` lives past the region.\n"

func TestExtractAcceptanceOnlyMinesTheAcceptanceRegion(t *testing.T) {
	acc := ExtractAcceptance(bodyWithAcceptance)
	if !acc.Found || acc.Section != "Acceptance" {
		t.Fatalf("region: found=%v section=%q", acc.Found, acc.Section)
	}
	got := map[string]Needle{}
	for _, n := range acc.Needles {
		got[n.Text] = n
	}
	for _, want := range []struct {
		text string
		kind NeedleKind
	}{
		{"ResolveHostSpill", NeedleSymbol},
		{"OVERLAY_WOULD_GATE", NeedleToken},
		{"cmd/fak/dispatch_tick_preflight.go", NeedlePath},
	} {
		n, ok := got[want.text]
		if !ok {
			t.Fatalf("needle %q missing; got %+v", want.text, acc.Needles)
		}
		if n.Kind != want.kind {
			t.Fatalf("needle %q kind=%q want %q", want.text, n.Kind, want.kind)
		}
	}
	// Outside the region (problem prose, a later section) and inside a fenced block
	// are all invisible: intent and repro commands never resolve an issue.
	for _, never := range []string{"internal/hostplacement/spill.go", "NeverExtractedFromNotes", "TestNeverExtracted"} {
		if _, ok := got[never]; ok {
			t.Fatalf("%q must not be extracted (outside region or fenced): %+v", never, acc.Needles)
		}
	}
	declined := map[string]string{}
	for _, d := range acc.Declined {
		declined[d.Text] = d.Reason
	}
	for text, want := range map[string]string{
		"two words": DeclineNotIdentifier,
		"ab":        DeclineTooShort,
		"git":       DeclineGeneric,
	} {
		if declined[text] != want {
			t.Fatalf("decline %q = %q, want %q (all: %+v)", text, declined[text], want, acc.Declined)
		}
	}
}

func TestExtractAcceptanceDeclinesRatherThanGuesses(t *testing.T) {
	// No acceptance section at all -> UNKNOWN, never SHIPPED.
	acc := ExtractAcceptance("## Problem\nSomething is broken. `ChannelCentral` is mentioned in prose.\n")
	if acc.Found || acc.Reason != ReasonNoAcceptanceSection {
		t.Fatalf("no-section: %+v", acc)
	}
	res := Resolve("", acc, nil)
	if res.Verdict != AcceptanceUnknown || !strings.Contains(res.Reason, ReasonNoAcceptanceSection) {
		t.Fatalf("no-section verdict: %+v", res)
	}

	// A section that names nothing backticked -> UNKNOWN(NO_NAMED_SYMBOL).
	acc = ExtractAcceptance("## Acceptance\n- it works end to end and the operator is happy\n")
	if !acc.Found || acc.Reason != ReasonNoNamedSymbol {
		t.Fatalf("no-symbol: %+v", acc)
	}
	if res := Resolve("", acc, nil); res.Verdict != AcceptanceUnknown {
		t.Fatalf("no-symbol verdict: %+v", res)
	}

	// A bare all-lowercase word is prose-shaped and declines rather than becoming a
	// grep needle that would match half the tree.
	acc = ExtractAcceptance("## Acceptance\n- `hostplacement` is used\n")
	if len(acc.Needles) != 0 || acc.Declined[0].Reason != DeclineNotSymbolShape {
		t.Fatalf("lowercase word: %+v", acc)
	}
}

func TestResolveVerdicts(t *testing.T) {
	acc := ExtractAcceptance(bodyWithAcceptance)
	full := []Presence{
		{Needle: "ResolveHostSpill", Found: true, DefFile: "internal/hostplacement/spill.go",
			Files: []string{"internal/hostplacement/spill.go", "internal/hostplacement/spill_test.go", "cmd/fak/dispatch_tick_preflight.go"}},
		{Needle: "OVERLAY_WOULD_GATE", Found: true, Files: []string{"dos.toml"}},
		{Needle: "cmd/fak/dispatch_tick_preflight.go", Found: true, Files: []string{"cmd/fak/dispatch_tick_preflight.go"}},
	}
	if res := Resolve("origin/main", acc, full); res.Verdict != AcceptanceShipped {
		t.Fatalf("shipped: %+v", res)
	}

	// The caller-count discriminator: identical presence, but the only referents are
	// the symbol's own package, its own tests, and architest.
	unwired := append([]Presence(nil), full...)
	unwired[0].Files = []string{
		"internal/hostplacement/spill.go",
		"internal/hostplacement/spill_test.go",
		"internal/architest/architest_test.go",
	}
	res := Resolve("origin/main", acc, unwired)
	if res.Verdict != AcceptancePartial {
		t.Fatalf("unwired verdict = %q want PARTIAL: %+v", res.Verdict, res)
	}
	if !strings.Contains(res.Remaining, "wire it into cmd/fak/dispatch_tick_preflight.go") {
		t.Fatalf("remaining must name the seam: %q", res.Remaining)
	}
	if !res.SeamNamed || res.Seam != "cmd/fak/dispatch_tick_preflight.go" {
		t.Fatalf("seam=%q named=%v", res.Seam, res.SeamNamed)
	}
	if len(res.Callers) != 1 || !res.Callers[0].Unwired ||
		res.Callers[0].Production != 0 || res.Callers[0].OwnPackage != 1 ||
		res.Callers[0].Tests != 2 || res.Callers[0].Architest != 0 {
		t.Fatalf("caller count: %+v", res.Callers)
	}

	// Nothing present -> OPEN, and the remaining work is the build itself.
	none := []Presence{
		{Needle: "ResolveHostSpill"}, {Needle: "OVERLAY_WOULD_GATE"},
		{Needle: "cmd/fak/dispatch_tick_preflight.go"},
	}
	res = Resolve("origin/main", acc, none)
	if res.Verdict != AcceptanceOpen || !strings.Contains(res.Remaining, "build it") {
		t.Fatalf("open: %+v", res)
	}

	// Some present -> PARTIAL naming what is still missing.
	half := []Presence{
		{Needle: "ResolveHostSpill", Found: true, DefFile: "internal/hostplacement/spill.go",
			Files: []string{"internal/hostplacement/spill.go", "cmd/fak/dispatch_tick_preflight.go"}},
		{Needle: "OVERLAY_WOULD_GATE"},
		{Needle: "cmd/fak/dispatch_tick_preflight.go", Found: true, Files: []string{"cmd/fak/dispatch_tick_preflight.go"}},
	}
	res = Resolve("origin/main", acc, half)
	if res.Verdict != AcceptancePartial || !strings.Contains(res.Remaining, "OVERLAY_WOULD_GATE") {
		t.Fatalf("half: %+v", res)
	}
}

func TestResolveNeverGuessesWhenEvidenceIsMissing(t *testing.T) {
	acc := ExtractAcceptance(bodyWithAcceptance)

	// A probe that was never run must not read as absent.
	res := Resolve("origin/main", acc, []Presence{{Needle: "ResolveHostSpill", Found: true}})
	if res.Verdict != AcceptanceUnknown || !strings.Contains(res.Reason, ReasonProbeMissing) {
		t.Fatalf("probe missing: %+v", res)
	}

	// A probe that errored must not read as absent either.
	res = Resolve("origin/main", acc, []Presence{
		{Needle: "ResolveHostSpill", Err: "fatal: bad revision origin/main"},
		{Needle: "OVERLAY_WOULD_GATE"}, {Needle: "cmd/fak/dispatch_tick_preflight.go"},
	})
	if res.Verdict != AcceptanceUnknown || !strings.Contains(res.Reason, ReasonProbeFailed) {
		t.Fatalf("probe failed: %+v", res)
	}
}

// TestResolveAbstainsOnGenericPresenceOnly reproduces #5822's false positive
// (#6002): every behaviour-shaped done condition declines at extraction
// (whitespace), the only surviving needle is the module-wide identifier
// `Affected`, and presence that generic must abstain — never resolve SHIPPED.
func TestResolveAbstainsOnGenericPresenceOnly(t *testing.T) {
	body := "## Done condition\n" +
		"- a post-restart drain reports `Outcome.Status == applied` and `Outcome.Affected == 0`\n" +
		"- `Affected` counts run-state changes, not raw writes\n"
	acc := ExtractAcceptance(body)
	if len(acc.Needles) != 1 || acc.Needles[0].Text != "Affected" || acc.Needles[0].Kind != NeedleSymbol {
		t.Fatalf("fixture must survive only the bare identifier: %+v", acc)
	}
	if len(acc.Declined) != 2 {
		t.Fatalf("the behaviour clauses must stay visible as declined: %+v", acc.Declined)
	}
	// The measured shape of the #5822 run: `Affected` present in 34 unrelated Go
	// files on the ref, declared at top level by none of them.
	files := make([]string, 34)
	for i := range files {
		files[i] = fmt.Sprintf("internal/pkg%02d/pkg.go", i)
	}
	res := Resolve("origin/main", acc, []Presence{{Needle: "Affected", Found: true, Files: files}})
	if res.Verdict == AcceptanceShipped {
		t.Fatalf("generic vocabulary presence resolved SHIPPED — the #5822 false positive: %+v", res)
	}
	if res.Verdict != AcceptanceUnknown || !strings.Contains(res.Reason, ReasonGenericPresenceOnly) {
		t.Fatalf("want UNKNOWN(%s): verdict=%q reason=%q", ReasonGenericPresenceOnly, res.Verdict, res.Reason)
	}
	if res.Remaining == "" {
		t.Fatal("an abstain must say how to corroborate")
	}

	// Positive control: the same generic needle corroborated by a seam path the
	// acceptance also names still ships.
	body = "## Done condition\n" +
		"- `Affected` counts run-state changes in `internal/fleetbus/drain.go`\n"
	acc = ExtractAcceptance(body)
	res = Resolve("origin/main", acc, []Presence{
		{Needle: "Affected", Found: true, Files: files},
		{Needle: "internal/fleetbus/drain.go", Found: true, Files: []string{"internal/fleetbus/drain.go"}},
	})
	if res.Verdict != AcceptanceShipped {
		t.Fatalf("a seam-specific path must keep corroborated resolutions SHIPPED: %+v", res)
	}
}

// TestResolveBareSymbolSpreadBoundary pins the narrowness cut: a bare symbol
// found in at most maxBareSymbolSpread files corroborates on its own; one file
// past that is module vocabulary and abstains.
func TestResolveBareSymbolSpreadBoundary(t *testing.T) {
	acc := ExtractAcceptance("## Acceptance\n- `ResolveHostSpill` decides the spill target\n")
	spread := func(n int) []Presence {
		files := make([]string, n)
		for i := range files {
			files[i] = fmt.Sprintf("cmd/fak/site%02d.go", i)
		}
		return []Presence{{Needle: "ResolveHostSpill", Found: true, Files: files}}
	}
	if res := Resolve("origin/main", acc, spread(maxBareSymbolSpread)); res.Verdict != AcceptanceShipped {
		t.Fatalf("a narrowly-spread bare symbol must still ship: %+v", res)
	}
	if res := Resolve("origin/main", acc, spread(maxBareSymbolSpread+1)); res.Verdict != AcceptanceUnknown ||
		!strings.Contains(res.Reason, ReasonGenericPresenceOnly) {
		t.Fatalf("one past the spread cap must abstain: %+v", res)
	}
}

func TestResolveEnforcesPackageQualifier(t *testing.T) {
	acc := ExtractAcceptance("## Acceptance\n- `computeadmit.Decide` admits the submit\n")
	if len(acc.Needles) != 1 || acc.Needles[0].Pkg != "computeadmit" || acc.Needles[0].Grep != "Decide" {
		t.Fatalf("qualified needle: %+v", acc.Needles)
	}
	// Present, but declared in a DIFFERENT package -> not a match, so OPEN, not SHIPPED.
	res := Resolve("origin/main", acc, []Presence{
		{Needle: "Decide", Found: true, DefFile: "internal/adjudicator/decide.go",
			Files: []string{"internal/adjudicator/decide.go", "cmd/fak/guard.go"}},
	})
	if res.Verdict != AcceptanceOpen {
		t.Fatalf("qualifier mismatch must not ship: %+v", res)
	}
	res = Resolve("origin/main", acc, []Presence{
		{Needle: "Decide", Found: true, DefFile: "internal/computeadmit/admit.go",
			Files: []string{"internal/computeadmit/admit.go", "internal/kernel/submit.go"}},
	})
	if res.Verdict != AcceptanceShipped {
		t.Fatalf("qualifier match should ship: %+v", res)
	}
}

func TestNameSeamFallsBackHonestly(t *testing.T) {
	acc := ExtractAcceptance("## Acceptance\n- `ShellRack` exists in `internal/toolsandbox/rack.go`\n")
	seam, named := nameSeam(acc, "internal/toolsandbox")
	if seam != DefaultSeam || named {
		t.Fatalf("a path inside the primitive's own package is not a seam: seam=%q named=%v", seam, named)
	}
}

func TestCountCallersPartitions(t *testing.T) {
	cc := CountCallers("Foo", "internal/foo/foo.go", []string{
		"internal/foo/foo.go",
		"internal/foo/other.go",
		"internal/foo/foo_test.go",
		"internal/architest/architest_test.go",
		"internal/architest/roles.go",
		"cmd/fak/verb.go",
	})
	if cc.OwnPackage != 2 || cc.Tests != 2 || cc.Architest != 1 || cc.Production != 1 || cc.Unwired {
		t.Fatalf("partition: %+v", cc)
	}
}

func TestAttachResolutionsAndStaleOpens(t *testing.T) {
	rep := Build("ws",
		[]Issue{{Number: 5319, State: "OPEN", Title: "shipped but open"}, {Number: 9, State: "OPEN", Title: "really open"}},
		nil, nil, "")
	AttachResolutions(&rep, map[int]Resolution{
		5319: {Verdict: AcceptanceShipped},
		9:    {Verdict: AcceptanceOpen},
	})
	if rep.AcceptanceCounts[AcceptanceShipped] != 1 || rep.AcceptanceCounts[AcceptanceOpen] != 1 {
		t.Fatalf("counts=%v", rep.AcceptanceCounts)
	}
	stale := StaleOpens(rep)
	if len(stale) != 1 || stale[0] != 5319 {
		t.Fatalf("stale opens=%v want [5319] — the issue a subject-grep audit would miss", stale)
	}
}
