package issuededup

import "testing"

// The fixture pair mirrors the #1756 witness pattern (and the real #2401/#2417
// near-twin pair that motivated #2504): one open backlog issue, one candidate
// that restates it in different words, and one candidate in the same lane that
// is genuinely new work. Bodies deliberately carry the shared issue-template
// skeleton (marker comment + section headings) the producers all render, so
// these tests also witness that the skeleton alone cannot cross the threshold.
var fixtureBacklog = []BacklogIssue{
	{
		Number: 2401,
		Title:  "feat(gateway): treat same-tick ready guard as positive readiness signal",
		Body: "<!-- fak-dogfood-key: gateway-same-tick-ready -->\n" +
			"## Current state\n" +
			"The gateway readiness fold reports a guard that became ready inside the current tick as still pending, " +
			"so the dispatch loop waits one extra tick before admitting work the guard already cleared.\n" +
			"## In scope\n" +
			"Count a guard whose ready transition lands in the same tick as the poll as ready now, in the readiness fold only.\n" +
			"## Out of scope\n" +
			"Changing guard polling cadence or the dispatch admission order.\n" +
			"## Done condition\n" +
			"A guard flipping ready within the poll tick admits work in that tick, witnessed by the readiness fold test.\n" +
			"## Witness\n" +
			"go test ./internal/gateway -run TestSameTickReady\n",
	},
	{
		Number: 1900,
		Title:  "fix(gateway): rotate the usage ledger before the JSONL exceeds the retention cap",
		Body: "<!-- fak-dogfood-key: gateway-usage-rotation -->\n" +
			"## Current state\n" +
			"The usage ledger JSONL grows without bound; the retention cap is only enforced at startup.\n" +
			"## In scope\n" +
			"Rotate the ledger file when an append would exceed the cap.\n" +
			"## Done condition\n" +
			"An append past the cap rotates first, witnessed by the rotation test.\n" +
			"## Witness\n" +
			"go test ./internal/gateway -run TestLedgerRotation\n",
	},
	{
		Number: 2100,
		Title:  "docs(model): document the Metal q4k fused-path fallbacks",
		Body: "<!-- fak-dogfood-key: model-q4k-docs -->\n" +
			"## Current state\n" +
			"The q4k fused path silently falls back to the reference GEMM on older Metal families and nothing documents when.\n" +
			"## In scope\n" +
			"One explainer page naming each fallback trigger.\n" +
			"## Done condition\n" +
			"The explainer lists every fallback with its trigger, checked by the docs lint.\n" +
			"## Witness\n" +
			"make docs-lint\n",
	},
}

// TestParaphrasedTwinFlagged — the done condition's first half: a candidate that
// restates open issue #2401 in different words is flagged with the original's
// number, and the verdict is auditable (number + title + axis, never a bare
// score).
func TestParaphrasedTwinFlagged(t *testing.T) {
	ix := NewIndex(fixtureBacklog)
	twin := Candidate{
		Title: "feat(gateway): count a guard that turns ready within the current tick as already ready",
		Body: "<!-- fak-idea-scout-key: gateway-ready-same-tick -->\n" +
			"## Current state\n" +
			"When a guard becomes ready during the tick being polled, the gateway readiness fold still counts it as pending, " +
			"and the dispatch loop holds the work for an extra tick even though the guard has cleared.\n" +
			"## In scope\n" +
			"Treat a ready transition landing in the poll's own tick as ready immediately; touch only the readiness fold.\n" +
			"## Out of scope\n" +
			"Guard polling cadence and dispatch admission ordering stay as they are.\n" +
			"## Done condition\n" +
			"Work behind a guard that flips ready inside the poll tick is admitted in that same tick.\n" +
			"## Witness\n" +
			"go test ./internal/gateway -run TestSameTickReady\n",
	}
	got := ix.Check(twin, 0, 0)
	if len(got) == 0 {
		t.Fatalf("paraphrased twin of #2401 passed clean; want a dup-risk verdict")
	}
	v := got[0]
	if v.IssueNumber != 2401 {
		t.Fatalf("top verdict = %+v, want issue_number 2401", v)
	}
	if v.Similarity < DefaultThreshold {
		t.Fatalf("similarity %.3f below threshold %.2f in a returned verdict", v.Similarity, DefaultThreshold)
	}
	if v.MatchedOn != MatchedOnTitle && v.MatchedOn != MatchedOnTitleBody {
		t.Fatalf("matched_on = %q, want %q or %q", v.MatchedOn, MatchedOnTitle, MatchedOnTitleBody)
	}
	if v.Title == "" {
		t.Fatalf("verdict lost the matched issue's title: %+v", v)
	}
	for _, other := range got {
		if other.IssueNumber == 2100 {
			t.Fatalf("unrelated backlog issue #2100 flagged alongside the twin: %+v", got)
		}
	}
}

// TestUnrelatedCandidatePasses — the done condition's second half: an unrelated
// candidate in the same (gateway) lane, carrying the same template skeleton,
// passes clean.
func TestUnrelatedCandidatePasses(t *testing.T) {
	ix := NewIndex(fixtureBacklog)
	unrelated := Candidate{
		Title: "feat(gateway): expose per-account token spend in the accounts summary",
		Body: "<!-- fak-idea-scout-key: gateway-account-spend -->\n" +
			"## Current state\n" +
			"The accounts summary shows request counts but not token spend, so an operator cannot see which account burns budget.\n" +
			"## In scope\n" +
			"Fold per-account input/output token totals into the summary table.\n" +
			"## Out of scope\n" +
			"Pricing, billing exports, and any new ledger file.\n" +
			"## Done condition\n" +
			"The summary lists token spend per account, witnessed by the summary render test.\n" +
			"## Witness\n" +
			"go test ./internal/gateway -run TestAccountsSummary\n",
	}
	if got := ix.Check(unrelated, 0, 0); len(got) != 0 {
		t.Fatalf("unrelated same-lane candidate flagged: %+v", got)
	}
}

// TestSelfMatchExcluded — re-reviewing an issue that is already in the backlog
// (the `fak issue contract --from-issues` path) must not flag the issue as its
// own twin.
func TestSelfMatchExcluded(t *testing.T) {
	ix := NewIndex(fixtureBacklog)
	self := Candidate{Number: 2401, Title: fixtureBacklog[0].Title, Body: fixtureBacklog[0].Body}
	for _, v := range ix.Check(self, 0, 0) {
		if v.IssueNumber == 2401 {
			t.Fatalf("issue #2401 flagged itself: %+v", v)
		}
	}
}

// TestRepeatedBacklogRowReplaces — overlapping paged gh reads repeat rows; the
// index must hold one entry per issue (the simhash Add-replaces fix, #2504) so
// one twin cannot occupy two TopK slots.
func TestRepeatedBacklogRowReplaces(t *testing.T) {
	dup := append(append([]BacklogIssue(nil), fixtureBacklog...), fixtureBacklog[0])
	ix := NewIndex(dup)
	if ix.Len() != len(fixtureBacklog) {
		t.Fatalf("index len = %d, want %d (repeated row must replace)", ix.Len(), len(fixtureBacklog))
	}
}

// TestParseBacklogGhListShape — ParseBacklog accepts exactly what
// `gh issue list --json number,title,body` prints, including a PowerShell BOM,
// and rejects non-array shapes with a diagnosable error.
func TestParseBacklogGhListShape(t *testing.T) {
	rows, err := ParseBacklog([]byte("\ufeff[{\"number\":7,\"title\":\"t\",\"body\":\"b\"}]"))
	if err != nil || len(rows) != 1 || rows[0].Number != 7 {
		t.Fatalf("rows=%+v err=%v, want the one row back", rows, err)
	}
	if _, err := ParseBacklog([]byte(`{"number":7}`)); err == nil {
		t.Fatalf("object shape parsed; want an error naming the expected gh list array")
	}
}
