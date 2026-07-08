package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/closureaudit"
)

// withClosureAuditSeams installs fake gh/git/dos seams and restores them.
func withClosureAuditSeams(t *testing.T,
	issues []closureaudit.Issue,
	commits []closureaudit.Commit,
	audits map[string]closureaudit.Audit,
) {
	t.Helper()
	oi, oc, oa := closureAuditFetchIssues, closureAuditReadCommits, closureAuditCommitAudit
	t.Cleanup(func() {
		closureAuditFetchIssues, closureAuditReadCommits, closureAuditCommitAudit = oi, oc, oa
	})
	closureAuditFetchIssues = func(_ string, _ int) ([]closureaudit.Issue, error) { return issues, nil }
	closureAuditReadCommits = func(_ string, _ int) ([]closureaudit.Commit, error) { return commits, nil }
	closureAuditCommitAudit = func(_, sha string) closureaudit.Audit { return audits[sha] }
}

func TestRunDispatchClosureAuditJSON(t *testing.T) {
	withClosureAuditSeams(t,
		[]closureaudit.Issue{
			{Number: 100, State: "OPEN", Title: "still open, shipped"},
			{Number: 200, State: "CLOSED", Title: "closed, unwitnessed"},
		},
		[]closureaudit.Commit{
			{SHA: "aaaaaaa1", Subject: "fix(x): resolve #100"},
			{SHA: "ccccccc3", Subject: "chore: closes #200"},
		},
		map[string]closureaudit.Audit{
			"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"},
			"ccccccc3": {Verdict: "OK", Witness: "claim-unwitnessed"},
		},
	)

	var stdout, stderr bytes.Buffer
	code := runDispatchClosureAudit(&stdout, &stderr, []string{"--json"})
	// A CLAIMED_CLOSED issue exists -> not OK -> exit 1.
	if code != 1 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var rep closureaudit.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout.String())
	}
	if rep.Schema != closureaudit.Schema {
		t.Fatalf("schema=%q", rep.Schema)
	}
	if rep.Verdict != "ACTION" || rep.Finding != "claimed_closed" {
		t.Fatalf("verdict=%s finding=%s", rep.Verdict, rep.Finding)
	}
	if rep.Counts[closureaudit.OpenWitnessed] != 1 || rep.Counts[closureaudit.ClaimedClosed] != 1 {
		t.Fatalf("counts=%v", rep.Counts)
	}
	if rep.ClosureRate == nil || *rep.ClosureRate != 0 {
		t.Fatalf("closure_rate=%v want 0 (0 true / 1 claimed)", rep.ClosureRate)
	}
}

func TestRunDispatchClosureAuditHumanAndMarkdown(t *testing.T) {
	withClosureAuditSeams(t,
		[]closureaudit.Issue{{Number: 1, State: "CLOSED", Title: "done"}},
		[]closureaudit.Commit{{SHA: "aaaaaaa1", Subject: "fix: resolve #1"}},
		map[string]closureaudit.Audit{"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"}},
	)
	var stdout, stderr bytes.Buffer
	if code := runDispatchClosureAudit(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("human exit=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "issue-closure audit: OK") {
		t.Fatalf("human render missing OK header:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runDispatchClosureAudit(&stdout, &stderr, []string{"--markdown"}); code != 0 {
		t.Fatalf("markdown exit=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "### issue-closure audit") || !strings.Contains(stdout.String(), "| bucket | count |") {
		t.Fatalf("markdown render malformed:\n%s", stdout.String())
	}
}

func TestRunDispatchClosureAuditRejectsBothFormats(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDispatchClosureAudit(&stdout, &stderr, []string{"--json", "--markdown"}); code != 2 {
		t.Fatalf("want exit 2 for conflicting formats, got %d", code)
	}
}

func TestParseClosureAuditLog(t *testing.T) {
	raw := "sha1" + commitLinkFieldSep + "fix: resolve #1" + commitLinkFieldSep + "body line\nCloses #2" + commitLinkRecordSep +
		"sha2" + commitLinkFieldSep + "docs: note" + commitLinkFieldSep + "" + commitLinkRecordSep
	commits := parseClosureAuditLog(raw)
	if len(commits) != 2 {
		t.Fatalf("got %d commits, want 2: %+v", len(commits), commits)
	}
	if commits[0].SHA != "sha1" || commits[0].Subject != "fix: resolve #1" || !strings.Contains(commits[0].Body, "Closes #2") {
		t.Fatalf("commit0 parse: %+v", commits[0])
	}
}

func TestParseCommitAuditRecord(t *testing.T) {
	got := parseCommitAuditRecord([]byte(`[{"sha":"deadbeef","verdict":"OK","witness":"diff-witnessed","claim_kind":"code_effect"}]`))
	if got.Verdict != "OK" || got.Witness != "diff-witnessed" || got.ClaimKind != "code_effect" {
		t.Fatalf("array parse: %+v", got)
	}
	got = parseCommitAuditRecord([]byte(`{"verdict":"CLAIM_UNWITNESSED","witness":""}`))
	if got.Verdict != "CLAIM_UNWITNESSED" {
		t.Fatalf("object parse: %+v", got)
	}
	if (parseCommitAuditRecord([]byte("not json"))) != (closureaudit.Audit{}) {
		t.Fatalf("bad json should be zero Audit")
	}
}
