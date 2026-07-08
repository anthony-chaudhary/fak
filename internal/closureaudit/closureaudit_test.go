package closureaudit

import (
	"reflect"
	"testing"
)

func TestClassifyRefs(t *testing.T) {
	cases := []struct {
		name    string
		subject string
		body    string
		want    map[int]Kind
	}{
		{
			name:    "close verb in body is resolving",
			subject: "fix(tools): port the closure audit",
			body:    "Closes #1406\nBuilds on #1300",
			// #1406 resolved by verb; #1300 has a "Builds on" dependency tail so it is
			// a bare mention, not resolving.
			want: map[int]Kind{1406: Resolving, 1300: Mention},
		},
		{
			name:    "subject ref is resolving even without a verb",
			subject: "fix(gateway): treat same-tick ready as positive (#812)",
			body:    "see #99 for background",
			want:    map[int]Kind{812: Resolving, 99: Mention},
		},
		{
			name:    "issue noun form resolves the whole run",
			subject: "test(dispatch): cover the fold",
			body:    "issues #10, #11 and #12",
			want:    map[int]Kind{10: Resolving, 11: Resolving, 12: Resolving},
		},
		{
			name:    "dependency tail disqualifies the noun-form run",
			subject: "docs: note the plan",
			body:    "issue #55 blocks #56",
			// The "blocks" dependency tail follows the "issue #55" run, so the whole
			// noun-form match is skipped and neither ref becomes resolving.
			want: map[int]Kind{55: Mention, 56: Mention},
		},
		{
			name:    "glued token is not a ref",
			subject: "chore: rotate creds",
			body:    "token xoxb-abc#118 rotated",
			want:    map[int]Kind{},
		},
		{
			name:    "resolving wins over mention for the same issue",
			subject: "fix: resolve #7",
			body:    "see #7 again",
			want:    map[int]Kind{7: Resolving},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyRefs(tc.subject, tc.body)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ClassifyRefs(%q,%q) = %v, want %v", tc.subject, tc.body, got, tc.want)
			}
		})
	}
}

func TestGradeBuckets(t *testing.T) {
	ref := func(sha string, k Kind) CommitRef { return CommitRef{SHA: sha, Kind: k} }
	diff := Audit{Verdict: "OK", Witness: "diff-witnessed"}
	data := Audit{Verdict: "OK", Witness: "data-witnessed"}
	unwit := Audit{Verdict: "OK", Witness: "claim-unwitnessed"}
	audits := map[string]Audit{
		"aaaaaaa1": diff,
		"bbbbbbb2": data,
		"ccccccc3": unwit,
	}
	cases := []struct {
		name  string
		issue Issue
		refs  []CommitRef
		want  string
	}{
		{"closed + diff-witnessed", Issue{Number: 1, State: "CLOSED"}, []CommitRef{ref("aaaaaaa1", Resolving)}, TrueResolved},
		{"closed + data-witnessed", Issue{Number: 2, State: "CLOSED"}, []CommitRef{ref("bbbbbbb2", Resolving)}, DataResolved},
		{"closed + unwitnessed", Issue{Number: 3, State: "CLOSED"}, []CommitRef{ref("ccccccc3", Resolving)}, ClaimedClosed},
		{"closed not planned", Issue{Number: 4, State: "CLOSED", StateReason: "not_planned"}, []CommitRef{ref("aaaaaaa1", Resolving)}, ClosedNotPlanned},
		{"open + witnessed", Issue{Number: 5, State: "OPEN"}, []CommitRef{ref("aaaaaaa1", Resolving)}, OpenWitnessed},
		{"open + only mention", Issue{Number: 6, State: "OPEN"}, []CommitRef{ref("aaaaaaa1", Mention)}, Open},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := Grade(tc.issue, tc.refs, audits)
			if g.Bucket != tc.want {
				t.Fatalf("Grade bucket = %s, want %s", g.Bucket, tc.want)
			}
		})
	}
}

func TestBuildClosureRateAndVerdict(t *testing.T) {
	issues := []Issue{
		{Number: 1, State: "CLOSED"},                             // TRUE_RESOLVED
		{Number: 2, State: "CLOSED"},                             // CLAIMED_CLOSED
		{Number: 3, State: "OPEN"},                               // OPEN_WITNESSED
		{Number: 4, State: "CLOSED", StateReason: "not_planned"}, // CLOSED_NOT_PLANNED
	}
	refs := map[int][]CommitRef{
		1: {{SHA: "aaaaaaa1", Kind: Resolving}},
		2: {{SHA: "ccccccc3", Kind: Resolving}}, // unwitnessed
		3: {{SHA: "aaaaaaa1", Kind: Resolving}},
		4: {{SHA: "aaaaaaa1", Kind: Resolving}},
	}
	audits := map[string]Audit{
		"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"},
		"ccccccc3": {Verdict: "OK", Witness: "claim-unwitnessed"},
	}
	rep := Build("/ws", issues, refs, audits, "")

	if rep.Schema != Schema {
		t.Fatalf("schema = %q, want %q", rep.Schema, Schema)
	}
	// closure_rate = TRUE_RESOLVED / (TRUE_RESOLVED + CLAIMED_CLOSED) = 1/2 = 0.5.
	if rep.ClosureRate == nil || *rep.ClosureRate != 0.5 {
		t.Fatalf("closure_rate = %v, want 0.5", rep.ClosureRate)
	}
	// A CLAIMED_CLOSED issue exists, so the audit is not OK and demands ACTION.
	if rep.OK || rep.Verdict != "ACTION" || rep.Finding != "claimed_closed" {
		t.Fatalf("verdict = %+v, want ACTION/claimed_closed/ok=false", rep)
	}
	if got := WitnessedOpenNumbers(rep); !reflect.DeepEqual(got, []int{3}) {
		t.Fatalf("WitnessedOpenNumbers = %v, want [3]", got)
	}
	if rep.Counts[TrueResolved] != 1 || rep.Counts[ClaimedClosed] != 1 ||
		rep.Counts[OpenWitnessed] != 1 || rep.Counts[ClosedNotPlanned] != 1 {
		t.Fatalf("counts = %v", rep.Counts)
	}
	// Actionable buckets sort first: CLAIMED_CLOSED then OPEN_WITNESSED.
	if rep.Issues[0].Bucket != ClaimedClosed || rep.Issues[1].Bucket != OpenWitnessed {
		t.Fatalf("sort order = %s,%s", rep.Issues[0].Bucket, rep.Issues[1].Bucket)
	}
}

func TestBuildAllWitnessedOK(t *testing.T) {
	issues := []Issue{{Number: 1, State: "CLOSED"}}
	refs := map[int][]CommitRef{1: {{SHA: "aaaaaaa1", Kind: Resolving}}}
	audits := map[string]Audit{"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"}}
	rep := Build("/ws", issues, refs, audits, "")
	if !rep.OK || rep.Verdict != "OK" || rep.Finding != "closures_witnessed" {
		t.Fatalf("clean audit = %+v, want OK/closures_witnessed", rep)
	}
	if rep.ClosureRate == nil || *rep.ClosureRate != 1.0 {
		t.Fatalf("closure_rate = %v, want 1.0", rep.ClosureRate)
	}
}

func TestRefsFromCommitsDeterministic(t *testing.T) {
	commits := []Commit{
		{SHA: "sha0002", Subject: "fix: resolve #1"},
		{SHA: "sha0001", Subject: "test: cover #1"},
	}
	refs := RefsFromCommits(commits)
	got := refs[1]
	if len(got) != 2 || got[0].SHA != "sha0001" || got[1].SHA != "sha0002" {
		t.Fatalf("refs[1] not sorted by sha: %+v", got)
	}
	if got[0].Kind != Resolving {
		t.Fatalf("subject ref should be resolving: %+v", got[0])
	}
}

func TestBuildAuditError(t *testing.T) {
	rep := Build("/ws", nil, nil, nil, "gh returned no issues")
	if rep.OK || rep.Verdict != "AUDIT_ERROR" || rep.Reason != "gh returned no issues" {
		t.Fatalf("audit error not surfaced: %+v", rep)
	}
}
