package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestIssueAuditCommandEmitsVerifiedReceipt(t *testing.T) {
	manifestPath := filepath.Join(t.TempDir(), "author.json")
	manifest := modelroute.AuthorManifest{
		Schema:         modelroute.CrossAuditAuthorSchema,
		Author:         modelroute.ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-5.4", Harness: "codex"},
		SourceEvidence: []modelroute.EvidenceRef{{Kind: "session", Ref: "codex:session"}},
	}
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, b, 0o600); err != nil {
		t.Fatal(err)
	}

	fetcher := modelroute.IssueAuditFetcherFunc(func(context.Context, int) (modelroute.IssueAuditEvidence, error) {
		return modelroute.IssueAuditEvidence{
			IssueNumber: 1185,
			IssueURL:    "https://github.com/anthony-chaudhary/fak/issues/1185",
			Title:       "review rung",
			Body:        "Done condition: review the diff.",
			State:       "CLOSED",
			ClosedAt:    "2026-06-29T05:13:50Z",
			CommitSHA:   "eb25512f57f1c717e5a53e3d7bde0582b9651bc0",
			Diff:        "diff --git a/a b/a\n+reviewed\n",
		}, nil
	})
	reviewer := modelroute.IssueAuditReviewerFunc(func(context.Context, modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
		return modelroute.IssueAuditReviewResult{Verdict: modelroute.CrossAuditPass, Reason: "done condition is covered", EvidenceRefs: []string{"test:review"}}, nil
	})
	var stdout, stderr bytes.Buffer
	code := runIssueAuditWith(&stdout, &stderr, []string{
		"--issue", "1185",
		"--author-manifest", manifestPath,
		"--auditor", "anthropic/claude/claude-opus-4-6",
		"--auditor-reasoning", "high",
		"--json",
	}, fetcher, reviewer)
	if code != 0 {
		t.Fatalf("runIssueAuditWith exit=%d stderr=%s", code, stderr.String())
	}
	parsed, err := modelroute.ParseIssueAuditReceipt(stdout.Bytes())
	if err != nil {
		t.Fatalf("parse emitted receipt: %v\n%s", err, stdout.String())
	}
	if parsed.Subject.IssueNumber != 1185 || parsed.Author.Family != "gpt" || parsed.Auditor.Family != "claude" || parsed.Verdict != modelroute.CrossAuditPass {
		t.Fatalf("emitted receipt = %+v", parsed)
	}
}

func TestIssueAuditFetcherNeverGuessesReferencedCommit(t *testing.T) {
	sha := "eb25512f57f1c717e5a53e3d7bde0582b9651bc0"
	runner := func(_ context.Context, name string, args ...string) ([]byte, error) {
		joined := name + " " + strings.Join(args, " ")
		switch {
		case joined == "gh repo view --json nameWithOwner":
			return []byte(`{"nameWithOwner":"anthony-chaudhary/fak"}`), nil
		case strings.HasPrefix(joined, "gh issue view 1185 "):
			return []byte(`{"number":1185,"title":"review rung","body":"done","state":"CLOSED","closedAt":"2026-06-29T05:13:50Z","url":"https://github.com/anthony-chaudhary/fak/issues/1185","closedByPullRequestsReferences":[]}`), nil
		case strings.HasPrefix(joined, "gh api repos/anthony-chaudhary/fak/issues/1185/events"):
			return []byte(fmt.Sprintf(`[{"event":"referenced","commit_id":%q,"created_at":"2026-06-29T05:13:20Z"}]`, sha)), nil
		default:
			return nil, fmt.Errorf("unexpected command %s", joined)
		}
	}
	fetcher := &githubIssueAuditFetcher{runner: runner}
	_, err := fetcher.FetchIssueAuditEvidence(context.Background(), 1185)
	if err == nil || !strings.Contains(err.Error(), "no unambiguous closed-event/PR commit") {
		t.Fatalf("referenced-only fetch error = %v, want ambiguity refusal", err)
	}
}

func TestIssueAuditFetcherBindsExplicitResolvingCommitAndCurrentRepo(t *testing.T) {
	sha := "eb25512f57f1c717e5a53e3d7bde0582b9651bc0"
	runner := func(_ context.Context, name string, args ...string) ([]byte, error) {
		joined := name + " " + strings.Join(args, " ")
		switch {
		case joined == "gh repo view --json nameWithOwner":
			return []byte(`{"nameWithOwner":"anthony-chaudhary/fak"}`), nil
		case strings.HasPrefix(joined, "gh issue view 1185 "):
			return []byte(`{"number":1185,"title":"review rung","body":"done","state":"CLOSED","closedAt":"2026-06-29T05:13:50Z","url":"https://github.com/anthony-chaudhary/fak/issues/1185","closedByPullRequestsReferences":[]}`), nil
		case joined == "git show -s --format=%s%x1f%b "+sha:
			return []byte("feat(loop): add scout review rung (#1185)\x1f\n"), nil
		case joined == "git show --format= --no-ext-diff --binary --no-renames "+sha:
			return []byte("diff --git a/a b/a\n+reviewed\n"), nil
		default:
			return nil, fmt.Errorf("unexpected command %s", joined)
		}
	}
	fetcher := &githubIssueAuditFetcher{repo: "anthony-chaudhary/fak", commitRef: sha, runner: runner}
	evidence, err := fetcher.FetchIssueAuditEvidence(context.Background(), 1185)
	if err != nil {
		t.Fatalf("FetchIssueAuditEvidence: %v", err)
	}
	if evidence.CommitSHA != sha || len(evidence.Evidence) != 1 || !strings.HasPrefix(evidence.Evidence[0].Ref, "author-manifest:") {
		t.Fatalf("evidence = %+v", evidence)
	}

	crossRepo := &githubIssueAuditFetcher{repo: "other/repo", commitRef: sha, runner: runner}
	if _, err := crossRepo.FetchIssueAuditEvidence(context.Background(), 1185); err == nil || !strings.Contains(err.Error(), "cross-repo git evidence is refused") {
		t.Fatalf("cross-repo error = %v", err)
	}
}

func TestParseIssueAuditReviewerOutputSupportsClaudeEnvelope(t *testing.T) {
	out := []byte(`{"type":"result","structured_output":{"verdict":"REFUTE","reason":"missing test","evidence_refs":["diff:a.go:4"]}}`)
	result, err := parseIssueAuditReviewerOutput(out)
	if err != nil || result.Verdict != modelroute.CrossAuditRefute || result.Reason != "missing test" {
		t.Fatalf("parsed=%+v err=%v", result, err)
	}
}
