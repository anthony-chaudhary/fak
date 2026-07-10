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
		case joined == "git rev-list --parents -n 1 "+sha:
			return []byte(sha + " 1111111111111111111111111111111111111111\n"), nil
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

func TestIssueAuditFetcherUsesFirstParentDiffForMergeCommit(t *testing.T) {
	merge := "cb41cc3e490074029ad3e0f46650537f5de47a84"
	firstParent := "1111111111111111111111111111111111111111"
	secondParent := "2222222222222222222222222222222222222222"
	wantDiff := []byte("diff --git a/a.go b/a.go\n+actual PR change\n")
	var calls []string
	fetcher := &githubIssueAuditFetcher{runner: func(_ context.Context, name string, args ...string) ([]byte, error) {
		joined := name + " " + strings.Join(args, " ")
		calls = append(calls, joined)
		switch joined {
		case "git rev-list --parents -n 1 " + merge:
			return []byte(merge + " " + firstParent + " " + secondParent + "\n"), nil
		case "git diff --no-ext-diff --binary --no-renames " + firstParent + " " + merge:
			return wantDiff, nil
		default:
			return nil, fmt.Errorf("unexpected command %s", joined)
		}
	}}
	got, err := fetcher.readClosingDiff(context.Background(), merge)
	if err != nil {
		t.Fatalf("readClosingDiff: %v", err)
	}
	if !bytes.Equal(got, wantDiff) {
		t.Fatalf("merge diff = %q, want %q", got, wantDiff)
	}
	for _, call := range calls {
		if strings.HasPrefix(call, "git show ") {
			t.Fatalf("merge diff used git show and can silently bind zero bytes: %v", calls)
		}
	}
}

func TestParseIssueAuditReviewerOutputSupportsClaudeEnvelope(t *testing.T) {
	out := []byte(`{"type":"result","structured_output":{"verdict":"REFUTE","reason":"missing test","evidence_refs":["diff:a.go:4"]}}`)
	result, err := parseIssueAuditReviewerOutput(out)
	if err != nil || result.Verdict != modelroute.CrossAuditRefute || result.Reason != "missing test" {
		t.Fatalf("parsed=%+v err=%v", result, err)
	}
}

func TestIssueAuditCLIDriversSendExactBoundPrompt(t *testing.T) {
	const boundPrompt = "BOUND SYSTEM POLICY\n\nBOUND UNTRUSTED EVIDENCE"
	req := modelroute.IssueAuditReviewRequest{Prompt: boundPrompt, PromptDigest: "sha256:test"}

	t.Run("claude", func(t *testing.T) {
		var gotStdin, gotName string
		var gotArgs []string
		reviewer := &issueAuditCLIReviewer{
			driver:   "claude",
			identity: modelroute.ModelIdentity{Model: "claude-opus-4-6"},
			runner: func(_ context.Context, stdin, name string, args ...string) ([]byte, error) {
				gotStdin, gotName, gotArgs = stdin, name, append([]string(nil), args...)
				return []byte(`{"structured_output":{"verdict":"PASS","reason":"ok","evidence_refs":[]}}`), nil
			},
		}
		if _, err := reviewer.ReviewIssue(context.Background(), req); err != nil {
			t.Fatalf("ReviewIssue: %v", err)
		}
		if gotStdin != boundPrompt || gotName != "claude" {
			t.Fatalf("claude sent stdin=%q name=%q", gotStdin, gotName)
		}
		joined := strings.Join(gotArgs, " ")
		if !strings.Contains(joined, "--json-schema") || !strings.Contains(joined, "--tools ") {
			t.Fatalf("claude argv = %v", gotArgs)
		}
	})

	t.Run("codex", func(t *testing.T) {
		var gotStdin, gotName string
		var gotArgs []string
		reviewer := &issueAuditCLIReviewer{
			driver:   "codex",
			identity: modelroute.ModelIdentity{Model: "gpt-5.6-sol", ReasoningPosture: "xhigh"},
			runner: func(_ context.Context, stdin, name string, args ...string) ([]byte, error) {
				gotStdin, gotName, gotArgs = stdin, name, append([]string(nil), args...)
				for i := range args {
					if args[i] == "-o" && i+1 < len(args) {
						return nil, os.WriteFile(args[i+1], []byte(`{"verdict":"PASS","reason":"ok","evidence_refs":[]}`), 0o600)
					}
				}
				return nil, fmt.Errorf("codex argv has no -o path: %v", args)
			},
		}
		if _, err := reviewer.ReviewIssue(context.Background(), req); err != nil {
			t.Fatalf("ReviewIssue: %v", err)
		}
		if gotStdin != boundPrompt || gotName != "codex" {
			t.Fatalf("codex sent stdin=%q name=%q", gotStdin, gotName)
		}
		joined := strings.Join(gotArgs, " ")
		for _, want := range []string{"exec", "--output-schema", "model_reasoning_effort=\"xhigh\"", "-"} {
			if !strings.Contains(joined, want) {
				t.Fatalf("codex argv %q missing %q", joined, want)
			}
		}
	})
}

func TestIssueAuditDriverIdentityRejectsObviousFamilyMismatch(t *testing.T) {
	tests := []struct {
		driver  string
		id      modelroute.ModelIdentity
		wantErr bool
	}{
		{"claude", modelroute.ModelIdentity{Provider: "anthropic", Family: "claude", Model: "claude-opus-4-6"}, false},
		{"claude", modelroute.ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-5.6-sol"}, true},
		{"codex", modelroute.ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-5.6-sol"}, false},
		{"codex", modelroute.ModelIdentity{Provider: "anthropic", Family: "claude", Model: "claude-opus-4-6"}, true},
	}
	for _, tt := range tests {
		err := validateIssueAuditDriverIdentity(tt.driver, tt.id)
		if (err != nil) != tt.wantErr {
			t.Errorf("driver=%s id=%+v err=%v wantErr=%v", tt.driver, tt.id, err, tt.wantErr)
		}
	}
}
