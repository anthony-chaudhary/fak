package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/closureaudit"
)

// withClosureAuditSeams installs fake gh/git/dos seams and restores them.
func withClosureAuditSeams(t *testing.T,
	issues []closureaudit.Issue,
	commits []closureaudit.Commit,
	audits map[string]closureaudit.Audit,
) {
	t.Helper()
	oi, oc, oa, ob, ot := closureAuditFetchIssues, closureAuditReadCommits, closureAuditCommitAudit, closureAuditCommitAudits, closureAuditTotalCommits
	t.Cleanup(func() {
		closureAuditFetchIssues, closureAuditReadCommits, closureAuditCommitAudit, closureAuditCommitAudits, closureAuditTotalCommits = oi, oc, oa, ob, ot
	})
	closureAuditFetchIssues = func(_ string, _ int) ([]closureaudit.Issue, error) { return issues, nil }
	closureAuditReadCommits = func(_ string, _ int) ([]closureaudit.Commit, error) { return commits, nil }
	closureAuditCommitAudit = func(_, sha string) closureaudit.Audit { return audits[sha] }
	// Default: the scanned window equals the whole history, so coverage is COMPLETE
	// unless a test narrows a cap. Keeps the git rev-list I/O out of unit tests.
	total := len(commits)
	closureAuditTotalCommits = func(_ string) *int { return &total }
}

func TestClosureAuditCommitAuditsBounded(t *testing.T) {
	original := closureAuditCommitAudit
	t.Cleanup(func() { closureAuditCommitAudit = original })

	var active atomic.Int32
	var maxActive atomic.Int32
	closureAuditCommitAudit = func(_, sha string) closureaudit.Audit {
		now := active.Add(1)
		for old := maxActive.Load(); now > old && !maxActive.CompareAndSwap(old, now); old = maxActive.Load() {
		}
		time.Sleep(10 * time.Millisecond)
		active.Add(-1)
		return closureaudit.Audit{Verdict: "OK", Witness: sha}
	}

	shas := make([]string, closureAuditWorkers*2)
	for i := range shas {
		shas[i] = string(rune('a' + i))
	}
	got := closureAuditCommitAuditsBounded(".", shas)
	if len(got) != len(shas) {
		t.Fatalf("audited %d SHA(s), want %d", len(got), len(shas))
	}
	if maxActive.Load() <= 1 {
		t.Fatalf("max concurrent audits=%d, want >1", maxActive.Load())
	}
	if maxActive.Load() > closureAuditWorkers {
		t.Fatalf("max concurrent audits=%d, cap=%d", maxActive.Load(), closureAuditWorkers)
	}
	for _, sha := range shas {
		if got[sha].Witness != sha {
			t.Fatalf("audit[%q]=%+v, want matching witness", sha, got[sha])
		}
	}
}

func TestCollectClosureAuditFreezesCoverageBeforeWitnesses(t *testing.T) {
	oi, oc, ob, ot := closureAuditFetchIssues, closureAuditReadCommits, closureAuditCommitAudits, closureAuditTotalCommits
	t.Cleanup(func() {
		closureAuditFetchIssues, closureAuditReadCommits, closureAuditCommitAudits, closureAuditTotalCommits = oi, oc, ob, ot
	})

	var order []string
	closureAuditFetchIssues = func(_ string, _ int) ([]closureaudit.Issue, error) {
		return []closureaudit.Issue{{Number: 1, State: "CLOSED"}}, nil
	}
	closureAuditTotalCommits = func(_ string) *int {
		order = append(order, "total")
		total := 1
		return &total
	}
	closureAuditReadCommits = func(_ string, _ int) ([]closureaudit.Commit, error) {
		order = append(order, "commits")
		return []closureaudit.Commit{{SHA: "aaaaaaa1", Subject: "fix: resolve #1"}}, nil
	}
	closureAuditCommitAudits = func(_ string, _ []string) map[string]closureaudit.Audit {
		order = append(order, "audits")
		return map[string]closureaudit.Audit{"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"}}
	}

	rep := collectDispatchClosureAudit(".", 10, 10)
	if !rep.Coverage.Complete {
		t.Fatalf("coverage=%+v, want complete", rep.Coverage)
	}
	if got := strings.Join(order, ","); got != "total,commits,audits" {
		t.Fatalf("I/O order=%q, want total,commits,audits", got)
	}
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

// TestRunDispatchClosureAuditWindowTruncated pins the ported coverage-truncation
// warning: when the git-log window is narrower than the repo history, a resolving
// commit could fall outside the window, so the audit must flag AUDIT_WINDOW_TRUNCATED
// rather than let a narrowed audit present as complete coverage.
func TestRunDispatchClosureAuditWindowTruncated(t *testing.T) {
	withClosureAuditSeams(t,
		[]closureaudit.Issue{{Number: 200, State: "CLOSED", Title: "closed, in-window"}},
		[]closureaudit.Commit{{SHA: "aaaaaaa1", Subject: "fix(x): resolve #200"}},
		map[string]closureaudit.Audit{"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"}},
	)
	// The repo has far more history than the scanned window.
	total := 100
	closureAuditTotalCommits = func(_ string) *int { return &total }

	var stdout, stderr bytes.Buffer
	runDispatchClosureAudit(&stdout, &stderr, []string{"--json", "--max-commits", "1"})
	var rep closureaudit.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout.String())
	}
	if rep.Coverage == nil {
		t.Fatal("coverage block missing from truncated audit")
	}
	if rep.Coverage.Verdict != closureaudit.CoverageIncomplete ||
		rep.Coverage.Warning != closureaudit.AuditWindowTruncated {
		t.Fatalf("coverage verdict=%q warning=%q; want incomplete + truncated",
			rep.Coverage.Verdict, rep.Coverage.Warning)
	}
	if !rep.Coverage.CommitsTruncated || rep.Coverage.Complete {
		t.Fatalf("expected commits_truncated + incomplete, got %+v", rep.Coverage)
	}
	if rep.Coverage.Recommended.MaxCommits != total+1000 {
		t.Fatalf("recommended max_commits=%d want %d", rep.Coverage.Recommended.MaxCommits, total+1000)
	}
	// The warning token must reach the machine payload verbatim.
	if !strings.Contains(stdout.String(), closureaudit.AuditWindowTruncated) {
		t.Fatalf("json missing %s token:\n%s", closureaudit.AuditWindowTruncated, stdout.String())
	}
}

// TestRunDispatchClosureAuditCoverageComplete is the companion: a window that
// covers the whole history reports COVERAGE_COMPLETE with no warning.
func TestRunDispatchClosureAuditCoverageComplete(t *testing.T) {
	withClosureAuditSeams(t,
		[]closureaudit.Issue{{Number: 100, State: "OPEN", Title: "open, shipped"}},
		[]closureaudit.Commit{{SHA: "aaaaaaa1", Subject: "fix(x): resolve #100"}},
		map[string]closureaudit.Audit{"aaaaaaa1": {Verdict: "OK", Witness: "diff-witnessed"}},
	)
	var stdout, stderr bytes.Buffer
	runDispatchClosureAudit(&stdout, &stderr, []string{"--json"})
	var rep closureaudit.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout.String())
	}
	if rep.Coverage == nil || !rep.Coverage.Complete ||
		rep.Coverage.Verdict != closureaudit.CoverageComplete || rep.Coverage.Warning != "" {
		t.Fatalf("expected complete coverage, got %+v", rep.Coverage)
	}
}
