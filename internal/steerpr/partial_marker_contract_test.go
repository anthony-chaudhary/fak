package steerpr

// partial_marker_contract_test.go — the drift witness FanoutMarker's doc comment
// promises (#5027).
//
// The denominator in partial.go is derived by substring-matching a marker key
// this package SPELLS ITSELF ("fanout-<leaf>-"), against bodies another package
// MINTS (issuefanout stamps `fanout-<leaf>-<slug>` into every filed child). Two
// independent spellings of one contract drift silently, and the drift lands
// exactly on this issue's acceptance gate: if the minted key stops matching the
// prefix, DeriveExpected still finds the spine but counts ZERO children, returns
// a confident M = 1, and a one-commit unit renders COMPLETE — the M = N
// inversion #5027 exists to prevent, arriving with no error anywhere.
//
// WHY THIS FILE AND NOT THE LEAF. internal/steerpr's non-test source must import
// NOTHING internal: the overlay fold is fenced stdlib-only so it can never grow
// a gate (architest OVERLAY_WOULD_GATE, pinned by
// TestSteerOverlayLeafStaysPureAndGitFree). That fence is enforced by an AST scan
// that parses the package's non-test files ONLY — its parser.ParseDir filter is
// `!strings.HasSuffix(fi.Name(), "_test.go")`, and the tier layering scan in
// internal/architest/architest_test.go skips test files by the same rule. So a
// _test.go here may import the producer to assert against it while partial.go
// itself stays pure, and the SHIPPED fold gains no import at all. The sibling
// loop_test.go already imports internal/loopgate on exactly that basis.
//
// WHAT MAKES IT A REAL BINDING. Every assertion goes through issuefanout's OWN
// minting path — Build mints Candidate.Key, LiveBody stamps that key into the
// filed body — never through a third hand-written copy of the literal. (The
// existing fanoutBody helper in partial_test.go IS such a third copy: it
// re-spells "fanout-"+leaf+"-"+slug by hand, so it agrees with a drifted
// producer forever. That is the hole this file closes.)
//
// A REDUNDANT CARRIER, FOUND WHILE PROVING THIS TEST BITES — read before
// strengthening the LiveBody case. A filed body carries the marker prefix in TWO
// places: the stamped `<!-- fak-issuefanout-key: ... -->` comment, and the
// boilerplate Batch policy line, which every candidate renders verbatim as
// "marker-key (fanout-<leaf>-*) dedupe against existing issues before filing".
// DeriveExpected's scan is a plain substring match by design (prose mentions
// count), so the prose carrier alone keeps the count correct even when the
// minted KEY has drifted. An end-to-end LiveBody -> DeriveExpected assertion is
// therefore NOT a key-drift sensor: it stays green through the drift. The
// key-drift sensors here are deliberately the two that isolate the key —
// markerPrefixOfEveryMintedKey and the key-only body case.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// markerContractPlan builds a real fan-out plan for one leaf through
// issuefanout's own entry point, so every key under test is MINTED rather than
// typed here.
func markerContractPlan(t *testing.T, leaf string) issuefanout.Plan {
	t.Helper()
	plan, err := issuefanout.Build(issuefanout.Input{
		Title:    "the " + leaf + " working spine",
		Leaf:     leaf,
		SpineRef: "#5027",
	})
	if err != nil {
		t.Fatalf("issuefanout.Build(leaf=%q): %v — the producer's own entry point must plan, or this contract test is scanning nothing", leaf, err)
	}
	// Non-vacuity: an empty plan would make every assertion below pass without
	// examining a single minted key.
	if len(plan.Candidates) < issuefanout.MinFanout {
		t.Fatalf("issuefanout.Build(leaf=%q) planned %d candidates, want at least the fan-out floor %d — a vacuous plan would let this drift test pass while proving nothing",
			leaf, len(plan.Candidates), issuefanout.MinFanout)
	}
	return plan
}

// markerContractIssues assembles the gathered issue graph DeriveExpected reads:
// the spine (carrying no marker of its own) plus one child per minted candidate,
// with each child's body produced by body.
func markerContractIssues(spineNumber int, plan issuefanout.Plan, body func(string) string) []IntentIssue {
	issues := []IntentIssue{{
		Number: spineNumber,
		Body:   "the spine issue itself, which carries no fanout marker of its own",
	}}
	for i, c := range plan.Candidates {
		issues = append(issues, IntentIssue{Number: 90000 + i, Body: body(c.Key)})
	}
	return issues
}

// TestPartialDenominatorMatchesTheRealFanoutMarkerContract is the witness
// FanoutMarker's doc comment names. It ties this package's re-spelled prefix to
// the key issuefanout ACTUALLY mints, at three points:
//
//   - every minted Candidate.Key starts with FanoutMarker(leaf);
//   - DeriveExpected, fed bodies that carry ONLY the minted key, counts them ALL
//     (M = 1 + children) — the consumer-side sensor, isolated from the Batch
//     policy prose that would otherwise mask a drifted key; and
//   - the resulting one-commit unit is FORMING, never COMPLETE.
//
// PROVEN TO BITE: replacing issuefanout's minted key shape ("fanout-"+leaf+"-"
// -> "fanout/"+leaf+"/") via `go test -overlay` reds this test with 15 minted-key
// mismatches, M = 1 against a want of 16, and the unit rendering COMPLETE on one
// commit. Under that same mutation every pre-existing #5027 test in
// partial_test.go stays GREEN — which is precisely why this file has to exist.
func TestPartialDenominatorMatchesTheRealFanoutMarkerContract(t *testing.T) {
	const leaf = "steerpr"
	const spineRef = "#5027"
	const spineNumber = 5027

	marker := FanoutMarker(leaf)
	if marker == "" {
		t.Fatalf("FanoutMarker(%q) is empty — the consumer's expectation must be a real, assertable value", leaf)
	}

	plan := markerContractPlan(t, leaf)

	// Point 1: the minted key carries the prefix the consumer scans for.
	for _, c := range plan.Candidates {
		if !strings.HasPrefix(c.Key, marker) {
			t.Errorf("issuefanout mints key %q, which does NOT start with steerpr's expected marker %q — the two spellings of one contract have DRIFTED. DeriveExpected would find the spine, count ZERO children, and report a confident M = 1, rendering a one-commit unit COMPLETE (the M = N inversion #5027 exists to prevent). Re-spell FanoutMarker in internal/steerpr/partial.go to match what issuefanout mints, or restore the minted key shape.",
				c.Key, marker)
		}
	}

	// Point 2: the CONSUMER, driven by the minted key and nothing else. Bodies
	// here carry the key ALONE, so the Batch policy prose cannot stand in for a
	// drifted key and this assertion measures what it claims to measure.
	keyOnly := markerContractIssues(spineNumber, plan, func(key string) string {
		return "<!-- fak-issuefanout-key: " + key + " -->\n\n## Lane\n\n" + leaf + "\n"
	})
	exp, ok := DeriveExpected(leaf, spineRef, keyOnly)
	if !ok {
		t.Fatalf("DeriveExpected(%q, %q, <spine + %d minted keys>) returned NOT derivable — the denominator cannot be read from the keys issuefanout actually mints", leaf, spineRef, len(plan.Candidates))
	}
	want := 1 + len(plan.Candidates)
	if exp.Total != want {
		t.Errorf("DeriveExpected over %d MINTED issuefanout keys derived M = %d, want %d (spine + %d children). The marker steerpr spells (%q) no longer matches the key issuefanout mints, so children are invisible to the denominator.",
			len(plan.Candidates), exp.Total, want, len(plan.Candidates), marker)
	}
	if exp.Source != SourceFanout {
		t.Errorf("derived source = %q, want %q — a fanout-derived denominator must report its real provenance", exp.Source, SourceFanout)
	}

	// Point 3: the inversion itself, asserted at the surface an operator reads.
	// One commit against a real multi-child fanout is FORMING, never COMPLETE. If
	// the marker drifts, exp.Total collapses to 1 and this flips to complete —
	// the #5027 acceptance gate failing in the exact shape the gate describes.
	p := NewPartial(1, exp, ok)
	if p.Complete {
		t.Errorf("a unit with 1 landed commit against a %d-member fanout renders COMPLETE (Partial=%+v) — the M = N inversion. An operator would read 'done' at the moment the whole budget to redirect was still unspent.", want, p)
	}
	if !p.Forming() {
		t.Errorf("a unit with 1 landed commit against a %d-member fanout is not FORMING (Partial=%+v) — the forming state is the one that is still cheap to steer, and it must be visible.", want, p)
	}
	if line := p.Annotate(); !strings.Contains(line, "forming") {
		t.Errorf("Annotate() = %q, want the forming rendering — a forming unit must not read like a finished one", line)
	}
}

// TestDenominatorCountsTheRealFiledBody pins that a child filed by
// `fak issue fanout --live` — the WHOLE rendered body, not an excerpt — is
// countable by the denominator. This is the integration half: it proves the
// marker survives LiveBody's stamping and that the substring scan finds it in
// the artifact that actually lands on the tracker.
//
// It is deliberately NOT the key-drift sensor, and must not be mistaken for one:
// the body's boilerplate Batch policy line renders "marker-key (fanout-<leaf>-*)
// dedupe ..." for every candidate, so this count stays correct even if the
// stamped key drifts. The sensors live in
// TestPartialDenominatorMatchesTheRealFanoutMarkerContract above.
func TestDenominatorCountsTheRealFiledBody(t *testing.T) {
	const leaf = "steerpr"
	const spineRef = "#5027"

	plan := markerContractPlan(t, leaf)
	issues := markerContractIssues(5027, plan, func(string) string { return "" })
	for i, c := range plan.Candidates {
		body := issuefanout.LiveBody(c)
		if !strings.Contains(body, c.Key) {
			t.Fatalf("issuefanout.LiveBody dropped the minted key %q from the filed body — the dedupe contract the denominator reads is not actually stamped, so no marker scan could ever count this child", c.Key)
		}
		issues[i+1].Body = body
	}

	exp, ok := DeriveExpected(leaf, spineRef, issues)
	if !ok {
		t.Fatalf("DeriveExpected(%q, %q, <spine + %d real filed bodies>) returned NOT derivable — a fan-out that was actually filed must yield a denominator", leaf, spineRef, len(plan.Candidates))
	}
	if want := 1 + len(plan.Candidates); exp.Total != want {
		t.Errorf("DeriveExpected over %d REAL issuefanout bodies derived M = %d, want %d — a filed fan-out is not countable end to end, so a real forming unit would report the wrong denominator.",
			len(plan.Candidates), exp.Total, want)
	}
}

// TestFanoutMarkerIsLeafScopedAgainstRealMintedKeys pins the other half of the
// contract: the marker must be scoped to ITS leaf, so a sibling leaf's real
// fan-out never inflates this intent's denominator. A degenerate marker (the
// bare "fanout-" prefix, or an empty string, which strings.Contains matches
// against every body) would over-count M — a fabricated denominator arriving by
// the opposite route from the M = N trap, and equally silent.
func TestFanoutMarkerIsLeafScopedAgainstRealMintedKeys(t *testing.T) {
	const mine = "steerpr"
	const spineRef = "#5027"

	foreign := markerContractPlan(t, "gateway")
	issues := []IntentIssue{{Number: 5027, Body: "the spine issue"}}
	for i, c := range foreign.Candidates {
		issues = append(issues, IntentIssue{Number: 91000 + i, Body: issuefanout.LiveBody(c)})
	}

	exp, ok := DeriveExpected(mine, spineRef, issues)
	if !ok {
		t.Fatalf("DeriveExpected(%q, %q, <spine + another leaf's real fanout>) is not derivable — the spine IS present, so the denominator must resolve to the spine alone", mine, spineRef)
	}
	if exp.Total != 1 {
		t.Errorf("another leaf's %d real fanout children inflated %q's denominator to M = %d, want 1 (the spine alone) — FanoutMarker(%q)=%q is not leaf-scoped against the keys issuefanout mints, so M is fabricated from work that can never land in this unit.",
			len(foreign.Candidates), mine, exp.Total, mine, FanoutMarker(mine))
	}

	// Empty leaf yields an empty marker, and an empty marker must never be handed
	// to strings.Contains as a denominator scan — it matches every body.
	if got := FanoutMarker("   "); got != "" {
		t.Errorf("FanoutMarker(blank) = %q, want \"\" — a blank leaf has no marker; a non-empty one would match bodies it has no claim to", got)
	}
}
