package main

// steer_prs_partial_test.go — issue #5027's OPERATOR surface. internal/steerpr
// proves the partial fold in the pure leaf (partial_test.go); this file proves
// the half the operator actually reads: that `fak steer prs` routes each unit to
// a declared denominator source, exposes expected/landed/complete on the
// `fak.steerpr.v1` payload, and renders a forming unit differently from a
// finished one.
//
// It also puts the first load on steerPRsIntentGather, which was declared as a
// seam so "a test run never reaches the network" but which no test overrode —
// meaning every steer-prs test until now shelled out to a real `gh issue list`
// and the #5027 wiring was covered only by whatever that call happened to
// return (nothing, from a t.TempDir() that is not a repo). Stubbing the seam is
// what makes the derivable cases reachable at all: with a live gh you can prove
// unknown by accident, but you can never prove "1 of 4".

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// The payload is emitted indented, so raw-key assertions must tolerate the
// space after the colon. Matching the compact form only would make the NEGATIVE
// assertions below pass vacuously — which is the same class of false green this
// issue is about.
var (
	steerPartialExpectedNull = regexp.MustCompile(`"expected":\s*null`)
	steerPartialCompleteTrue = regexp.MustCompile(`"complete":\s*true`)
)

// The four members of the partial fixture, one per denominator outcome.
const (
	steerPartialGatewaySHA = "1111111111111111111111111111111111111111"
	steerPartialModelSHA   = "2222222222222222222222222222222222222222"
	steerPartialModelFixSH = "3333333333333333333333333333333333333333"
	steerPartialBenchSHA   = "4444444444444444444444444444444444444444"
)

// steerPartialLog is a three-unit range covering all three partial states at
// once, because the operator value of #5027 is telling them apart in ONE view:
//   - gateway: 1 landed against a 4-member declared intent  → FORMING
//   - model:   2 landed against a 2-member declared intent  → COMPLETE
//   - bench:   bound to an intent absent from the graph     → UNKNOWN
const steerPartialLog = "\x1e" + steerPartialBenchSHA + "\x1ffeat(bench): add the inventory row (#9999) (fak bench)\x1f\x1f\ninternal/bench/b.go\n" +
	"\x1e" + steerPartialModelFixSH + "\x1ffix(model): unbreak the decode path (fak model)\x1f\x1f\ninternal/model/m.go\n" +
	"\x1e" + steerPartialModelSHA + "\x1ffeat(model): land the decode spine (#6001) (fak model)\x1f\x1f\ninternal/model/decode.go\n" +
	"\x1e" + steerPartialGatewaySHA + "\x1ffeat(gateway): land the cold-tool spine (#5015) (fak gateway)\x1f\x1f\ninternal/gateway/g.go\n"

// steerPartialFanoutBody mirrors what issuefanout stamps into a filed child, so
// the denominator is derived from the REAL marker-key contract rather than from
// a shape invented for the test.
func steerPartialFanoutBody(leaf, slug string) string {
	return "<!-- fak-issuefanout-key: fanout-" + leaf + "-" + slug + " -->\n\n## Lane\n\n" + leaf
}

// steerPartialIssues is the gathered issue graph: a gateway spine with three
// fanout children (M=4), a model spine with one (M=2), and NO #9999 — so the
// bench unit's intent cannot be counted and must report unknown.
func steerPartialIssues() []steerpr.IntentIssue {
	return []steerpr.IntentIssue{
		{Number: 5015, Body: "the gateway spine"},
		{Number: 5027, Body: steerPartialFanoutBody("gateway", "partial-bundle")},
		{Number: 5028, Body: steerPartialFanoutBody("gateway", "ack")},
		{Number: 5031, Body: steerPartialFanoutBody("gateway", "pause")},
		{Number: 6001, Body: "the model spine"},
		{Number: 6002, Body: steerPartialFanoutBody("model", "decode-followon")},
	}
}

// withSteerPartialFakes stubs the three seams the view build reads: git log, the
// verdict map, and the issue-graph gather. Every member is witnessed so the
// bands stay uniform and what the assertions move is the PARTIAL axis alone —
// the partial state must be orthogonal to the band, not a recolouring of it.
func withSteerPartialFakes(t *testing.T, issues []steerpr.IntentIssue) {
	t.Helper()
	origGit, origVerdicts, origGather := releasePRPlanGit, steerPRsVerdicts, steerPRsIntentGather
	releasePRPlanGit = prPlanFakeGit(steerPartialLog)
	steerPRsVerdicts = func(_, _, _ string) map[string]steerpr.Verdict {
		return map[string]steerpr.Verdict{
			steerPartialGatewaySHA: steerpr.VerdictWitnessed,
			steerPartialModelSHA:   steerpr.VerdictWitnessed,
			steerPartialModelFixSH: steerpr.VerdictWitnessed,
			steerPartialBenchSHA:   steerpr.VerdictWitnessed,
		}
	}
	steerPRsIntentGather = func(string) []steerpr.IntentIssue { return issues }
	t.Cleanup(func() {
		releasePRPlanGit, steerPRsVerdicts, steerPRsIntentGather = origGit, origVerdicts, origGather
	})
}

// steerPartialPayload is the `fak.steerpr.v1` contract #5027 adds. Expected is a
// POINTER so the test can tell an explicit null from a zero — the distinction
// the whole ticket turns on.
type steerPartialPayload struct {
	FormingCount         int `json:"forming_count"`
	UnknownExpectedCount int `json:"unknown_expected_count"`
	Units                []struct {
		Leaf    string `json:"leaf"`
		Partial *struct {
			Landed   int    `json:"landed"`
			Expected *int   `json:"expected"`
			Complete bool   `json:"complete"`
			Source   string `json:"source"`
		} `json:"partial"`
	} `json:"units"`
}

// runSteerPartialJSON drives the real CLI entry point and decodes the payload,
// returning the raw bytes too so a test can assert on key PRESENCE (which a
// decode into a pointer would hide).
func runSteerPartialJSON(t *testing.T) (steerPartialPayload, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--json", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var payload steerPartialPayload
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	return payload, stdout.String()
}

// TestSteerPRsPartialPayloadExposesExpectedLandedComplete is the payload half of
// the done condition: a unit bound to an intent with fanout children reports N
// of M, a fully-landed intent reports complete, and a unit whose intent is not
// in the graph reports expected: unknown — all three in one view, from the one
// bounded gather.
func TestSteerPRsPartialPayloadExposesExpectedLandedComplete(t *testing.T) {
	withSteerPartialFakes(t, steerPartialIssues())
	payload, raw := runSteerPartialJSON(t)

	byLeaf := map[string]*struct {
		Landed   int    `json:"landed"`
		Expected *int   `json:"expected"`
		Complete bool   `json:"complete"`
		Source   string `json:"source"`
	}{}
	for _, u := range payload.Units {
		if u.Partial == nil {
			t.Fatalf("unit %q carries no partial — unknown must be EXPLICIT, not an absent object", u.Leaf)
		}
		byLeaf[u.Leaf] = u.Partial
	}
	if len(byLeaf) != 3 {
		t.Fatalf("units = %v, want gateway, model and bench", payload.Units)
	}

	// FORMING: 1 of 4, derived from the spine plus its three fanout children.
	gw := byLeaf["gateway"]
	if gw.Expected == nil || *gw.Expected != 4 || gw.Landed != 1 || gw.Complete {
		t.Errorf("gateway partial = %+v (expected=%v), want landed 1 of 4, not complete", gw, gw.Expected)
	}
	if gw.Source != steerpr.SourceFanout {
		t.Errorf("gateway source = %q, want %q — the denominator must name where it came from", gw.Source, steerpr.SourceFanout)
	}

	// COMPLETE: 2 of 2. A finished intent is allowed to say so.
	md := byLeaf["model"]
	if md.Expected == nil || *md.Expected != 2 || md.Landed != 2 || !md.Complete {
		t.Errorf("model partial = %+v (expected=%v), want landed 2 of 2, complete", md, md.Expected)
	}

	// UNKNOWN: the acceptance gate. #9999 is absent from the graph, so there is
	// no denominator — and one landed commit must NOT become "1 of 1 complete".
	bn := byLeaf["bench"]
	if bn.Expected != nil {
		t.Errorf("bench expected = %d, want null — a denominator was fabricated for an intent absent from the graph", *bn.Expected)
	}
	if bn.Complete {
		t.Error("bench rendered complete on an unknown denominator — M = N was silently fabricated")
	}
	if bn.Landed != 1 {
		t.Errorf("bench landed = %d, want 1", bn.Landed)
	}

	// The key must be PRESENT and null, never omitted: an omitted key lets a
	// machine consumer default it to 0 and compute completeness itself.
	if !steerPartialExpectedNull.MatchString(raw) {
		t.Errorf("payload carries no explicit \"expected\": null:\n%s", raw)
	}

	if payload.FormingCount != 1 {
		t.Errorf("forming_count = %d, want 1 (gateway)", payload.FormingCount)
	}
	if payload.UnknownExpectedCount != 1 {
		t.Errorf("unknown_expected_count = %d, want 1 (bench)", payload.UnknownExpectedCount)
	}
}

// TestSteerPRsPartialRenderDistinguishesFormingFromComplete is the render half
// of the done condition: the three states must not read alike in the human
// view. A forming unit that renders like a finished one is the failure — it
// tells an operator the budget to redirect is spent while it is still there.
func TestSteerPRsPartialRenderDistinguishesFormingFromComplete(t *testing.T) {
	withSteerPartialFakes(t, steerPartialIssues())

	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()

	for _, want := range []string{
		// The split, posted up front so the operator sees it before any unit.
		"1 unit(s) still FORMING",
		"1 carry no derivable denominator (expected: unknown)",
		// Forming names its outstanding count and that it is still cheap to act.
		"forming: 1 of 4 expected commits landed (3 outstanding)",
		"still cheap to steer",
		// Complete is a distinct string, not the forming one with new numbers.
		"complete: 2 of 2 expected commits landed",
		// Unknown NAMES its own ignorance rather than going quiet; silence would
		// read as completeness to an operator scanning the overlay.
		"expected: unknown — 1 landed, no declared denominator",
		"not rendered complete",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}

	// The unknown unit must not pick up the complete rendering anywhere.
	if strings.Contains(out, "complete: 1 of 1") {
		t.Fatalf("an unknown denominator rendered as 1 of 1 complete:\n%s", out)
	}
}

// TestSteerPRsPartialUnknownWhenGraphUnreadable pins the degraded path the
// wiring's comment claims but nothing proved: when the gather yields nothing
// (gh absent, wrong repo, a failed scan) EVERY unit reports unknown and NONE
// reports complete.
//
// This is the fabricated-denominator failure in its most likely real costume.
// The gather is best-effort by design, so an empty graph is the case that will
// actually occur in production — and it is exactly the case where deriving
// "M = 1, therefore complete" from a spine nobody found would tell an operator
// that every in-flight intent had finished.
func TestSteerPRsPartialUnknownWhenGraphUnreadable(t *testing.T) {
	withSteerPartialFakes(t, nil)
	payload, raw := runSteerPartialJSON(t)

	if payload.FormingCount != 0 {
		t.Errorf("forming_count = %d, want 0 — nothing is countable without a graph", payload.FormingCount)
	}
	if payload.UnknownExpectedCount != 3 {
		t.Errorf("unknown_expected_count = %d, want 3 (every unit)", payload.UnknownExpectedCount)
	}
	for _, u := range payload.Units {
		if u.Partial == nil {
			t.Fatalf("unit %q dropped its partial — unknown must be explicit", u.Leaf)
		}
		if u.Partial.Expected != nil {
			t.Errorf("unit %q derived M = %d from an empty graph — a fabricated denominator", u.Leaf, *u.Partial.Expected)
		}
		if u.Partial.Complete {
			t.Errorf("unit %q rendered complete with no readable graph — the M = N trap", u.Leaf)
		}
	}
	// Guard the guard: the fixture must really have produced units, or the
	// negative assertions above would all hold over an empty view.
	if len(payload.Units) != 3 {
		t.Fatalf("units = %d, want 3 — the negative assertions must run against a real view", len(payload.Units))
	}
	if steerPartialCompleteTrue.MatchString(raw) {
		t.Errorf("some unit is complete with no derivable denominator:\n%s", raw)
	}
	if !steerPartialExpectedNull.MatchString(raw) {
		t.Errorf("no unit reported an explicit unknown denominator:\n%s", raw)
	}
}

// TestSteerPRsPartialGatherIsSeamed proves the view actually CONSULTS the
// gather seam (and does so once for the whole view, not once per unit — an
// un-hoisted gather would shell out to gh per unit, which is the second
// GitHub-client cost the issue's coordination note forbids).
func TestSteerPRsPartialGatherIsSeamed(t *testing.T) {
	withSteerPartialFakes(t, steerPartialIssues())

	calls := 0
	inner := steerPRsIntentGather
	steerPRsIntentGather = func(root string) []steerpr.IntentIssue {
		calls++
		return inner(root)
	}

	if _, err := buildSteerPRsView(t.TempDir(), "baseref", "headref"); err != nil {
		t.Fatalf("buildSteerPRsView: %v", err)
	}
	if calls != 1 {
		t.Fatalf("issue graph gathered %d time(s), want exactly 1 for the whole view", calls)
	}
}
