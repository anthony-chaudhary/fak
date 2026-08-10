package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestIssueAuditCommandEmitsVerifiedReceipt(t *testing.T) {
	manifestPath := filepath.Join(t.TempDir(), "author.json")
	manifest := modelroute.AuthorManifest{
		Schema: modelroute.CrossAuditAuthorSchema,
		Author: modelroute.ModelIdentity{
			Provider: "openai", Family: "gpt", Model: "gpt-author", WeightsRevision: "gpt-w54", Harness: "codex",
			EndpointClass: "remote", AccountClass: "subscription", ReasoningPosture: "xhigh", ProvenanceSource: "session:codex",
		},
		SourceEvidence: []modelroute.EvidenceRef{{Kind: "session", Ref: "codex:session"}},
	}
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, b, 0o600); err != nil {
		t.Fatal(err)
	}
	rosterPath := writeIssueAuditRoster(t, []modelroute.AuditIdentityAlias{
		{Alias: "gpt-author", CanonicalModel: "gpt-5.4", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w54", ProvenanceSource: "registry:gpt"},
		{Alias: "claude-review", CanonicalModel: "claude-opus-4-6", Provider: "anthropic", Family: "claude", WeightsRevision: "claude-w46", ProvenanceSource: "registry:claude"},
	})

	fetcher := modelroute.IssueAuditFetcherFunc(func(context.Context, int) (modelroute.IssueAuditEvidence, error) {
		patch := "diff --git a/a b/a\n+reviewed\n"
		return modelroute.IssueAuditEvidence{
			IssueNumber: 1185,
			IssueURL:    "https://github.com/anthony-chaudhary/fak/issues/1185",
			Title:       "review rung",
			Body:        "Done condition: review the diff.",
			State:       "CLOSED",
			ClosedAt:    "2026-06-29T05:13:50Z",
			CommitSHA:   "eb25512f57f1c717e5a53e3d7bde0582b9651bc0",
			Diff:        patch,
			ClosingCommits: []modelroute.IssueAuditClosingCommit{{
				SHA: "eb25512f57f1c717e5a53e3d7bde0582b9651bc0", FirstParentSHA: "1111111111111111111111111111111111111111",
				TreeOID: "tree-closing", FirstParentTreeOID: "tree-parent", Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"a", "a_test.go"},
			}},
		}, nil
	})
	reviewer := modelroute.IssueAuditReviewerFunc(func(context.Context, modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
		return modelroute.IssueAuditReviewResult{Verdict: modelroute.CrossAuditPass, Reason: "done condition is covered", EvidenceRefs: []string{"test:review"}}, nil
	})
	var stdout, stderr bytes.Buffer
	code := runIssueAuditWith(&stdout, &stderr, []string{
		"--issue", "1185",
		"--author-manifest", manifestPath,
		"--auditor", "anthropic/claude/claude-review",
		"--auditor-driver", "claude",
		"--auditor-reasoning", "high",
		"--identity-roster", rosterPath,
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

func TestIssueAuditBundleOnlyEmitsVerifiedBundleWithoutReviewerConfig(t *testing.T) {
	patch := "diff --git a/a.go b/a.go\n+safe\n"
	fetcher := modelroute.IssueAuditFetcherFunc(func(_ context.Context, issue int) (modelroute.IssueAuditEvidence, error) {
		return modelroute.IssueAuditEvidence{
			IssueNumber: issue, IssueURL: "https://example/issues/42", Title: "bundle only", Body: "## Done condition\nsafe", State: "CLOSED", ClosedAt: "2026-07-10T00:00:00Z", CommitSHA: "abcdef1", Diff: patch,
			ClosingCommits: []modelroute.IssueAuditClosingCommit{{SHA: "abcdef1", FirstParentSHA: "1234567", TreeOID: "tree-new", FirstParentTreeOID: "tree-old", Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"a.go", "a_test.go"}}},
			Tests:          []modelroute.EvidenceRef{{Kind: "test-path", Ref: "a_test.go"}},
		}, nil
	})
	reviewCalled := false
	reviewer := modelroute.IssueAuditReviewerFunc(func(context.Context, modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
		reviewCalled = true
		return modelroute.IssueAuditReviewResult{}, nil
	})
	var stdout, stderr bytes.Buffer
	code := runIssueAuditWith(&stdout, &stderr, []string{"--issue", "42", "--bundle-only"}, fetcher, reviewer)
	if code != 0 || stderr.Len() != 0 {
		t.Fatalf("bundle-only exit=%d stderr=%s", code, stderr.String())
	}
	bundle, err := modelroute.ParseIssueAuditBundle(stdout.Bytes())
	if err != nil || !bundle.Complete || bundle.Issue.Number != 42 {
		t.Fatalf("bundle-only output = %+v err=%v", bundle, err)
	}
	if reviewCalled {
		t.Fatal("bundle-only path called an auditor")
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
		case joined == "git diff --no-ext-diff --binary --no-renames 1111111111111111111111111111111111111111 "+sha:
			return []byte("diff --git a/a b/a\n+reviewed\n"), nil
		case joined == "git rev-parse "+sha+"^{tree}":
			return []byte("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"), nil
		case joined == "git rev-parse 1111111111111111111111111111111111111111^{tree}":
			return []byte("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"), nil
		case joined == "git diff --name-only -z --no-renames 1111111111111111111111111111111111111111 "+sha:
			return []byte("a\x00a_test.go\x00"), nil
		default:
			return nil, fmt.Errorf("unexpected command %s", joined)
		}
	}
	fetcher := &githubIssueAuditFetcher{repo: "anthony-chaudhary/fak", commitRef: sha, runner: runner}
	evidence, err := fetcher.FetchIssueAuditEvidence(context.Background(), 1185)
	if err != nil {
		t.Fatalf("FetchIssueAuditEvidence: %v", err)
	}
	if evidence.CommitSHA != sha || len(evidence.ClosingCommits) != 1 || evidence.ClosingCommits[0].FirstParentSHA == "" || len(evidence.Evidence) != 1 || !strings.HasPrefix(evidence.Evidence[0].Ref, "author-manifest:") {
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
		case "git rev-parse " + merge + "^{tree}":
			return []byte("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"), nil
		case "git rev-parse " + firstParent + "^{tree}":
			return []byte("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"), nil
		case "git diff --name-only -z --no-renames " + firstParent + " " + merge:
			return []byte("a.go\x00a_test.go\x00"), nil
		default:
			return nil, fmt.Errorf("unexpected command %s", joined)
		}
	}}
	got, err := fetcher.readClosingCommitEvidence(context.Background(), merge)
	if err != nil {
		t.Fatalf("readClosingCommitEvidence: %v", err)
	}
	if !bytes.Equal([]byte(got.Patch), wantDiff) || got.FirstParentSHA != firstParent || got.TreeOID == "" || got.FirstParentTreeOID == "" || !reflect.DeepEqual(got.ChangedPaths, []string{"a.go", "a_test.go"}) {
		t.Fatalf("merge closing evidence = %+v, want exact first-parent patch/trees/paths", got)
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
	patch := "diff --git a/a.go b/a.go\n+safe\n"
	bundle, err := modelroute.BuildIssueAuditBundle(modelroute.IssueAuditEvidence{
		IssueNumber: 42, IssueURL: "https://example/issues/42", Title: "bound", Body: "done", State: "CLOSED", ClosedAt: "2026-07-10T00:00:00Z", CommitSHA: "abcdef1", Diff: patch,
		ClosingCommits: []modelroute.IssueAuditClosingCommit{{SHA: "abcdef1", FirstParentSHA: "1234567", TreeOID: "tree-new", FirstParentTreeOID: "tree-old", Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"a.go"}}},
	}, modelroute.IssueAuditBundleOptions{})
	if err != nil {
		t.Fatal(err)
	}
	req, err := modelroute.NewIssueAuditReviewRequest(42, bundle.BundleDigest, bundle)
	if err != nil {
		t.Fatal(err)
	}
	boundPrompt := req.Prompt

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
	aliases := []modelroute.AuditIdentityAlias{
		{Alias: "claude-prod", CanonicalModel: "claude-opus-4-6", Provider: "anthropic", Family: "claude", WeightsRevision: "claude-w46", ProvenanceSource: "registry:claude"},
		{Alias: "gpt-prod", CanonicalModel: "gpt-5.6-sol", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w56", ProvenanceSource: "registry:gpt"},
	}
	tests := []struct {
		driver  string
		id      modelroute.ModelIdentity
		wantErr bool
	}{
		{"claude", modelroute.ModelIdentity{Provider: "anthropic", Family: "claude", Model: "claude-prod"}, false},
		{"claude", modelroute.ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-prod"}, true},
		{"codex", modelroute.ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-prod"}, false},
		{"codex", modelroute.ModelIdentity{Provider: "anthropic", Family: "claude", Model: "claude-prod"}, true},
		{"codex", modelroute.ModelIdentity{Provider: "openai", Family: "gpt-fake", Model: "gpt-prod"}, true},
		{"claude", modelroute.ModelIdentity{Provider: "anthropic", Family: "claude", Model: "unregistered"}, true},
		{"http", modelroute.ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-prod"}, false},
	}
	for _, tt := range tests {
		got, err := modelroute.ValidateAuditDriverIdentity(tt.driver, tt.id, aliases)
		if (err != nil) != tt.wantErr {
			t.Errorf("driver=%s id=%+v err=%v wantErr=%v", tt.driver, tt.id, err, tt.wantErr)
		}
		if err == nil && got.Driver != tt.driver {
			t.Errorf("driver=%s canonical identity driver=%q", tt.driver, got.Driver)
		}
	}
}

func TestIssueAuditHTTPReviewerCarriesUpstreamObservedModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode HTTP audit request: %v", err)
		}
		if len(request.Messages) != 2 || request.Messages[0].Role != "system" || request.Messages[1].Role != "user" || request.Messages[0].Content != modelroute.CrossAuditSystemPrompt {
			t.Errorf("HTTP audit channels = %+v", request.Messages)
		}
		if !strings.Contains(request.Messages[1].Content, modelroute.IssueAuditBundleSchema) {
			t.Errorf("HTTP untrusted channel does not carry bundle: %+v", request.Messages)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"claude-opus-4-6","choices":[{"message":{"role":"assistant","content":"{\"verdict\":\"PASS\",\"reason\":\"ok\",\"evidence_refs\":[]}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()
	reviewer := newIssueAuditHTTPReviewer(modelroute.AuditIdentity{Model: "gpt-5.6-sol"}, server.URL, "")
	result, err := reviewer.ReviewIssue(context.Background(), issueAuditTestReviewRequest(t))
	if err != nil {
		t.Fatalf("ReviewIssue: %v", err)
	}
	if result.ObservedAuditor == nil || result.ObservedAuditor.Model != "claude-opus-4-6" {
		t.Fatalf("observed auditor = %+v", result.ObservedAuditor)
	}
}

func TestIssueAuditCommandHTTPDeclaredFamilyMismatchFailsClosed(t *testing.T) {
	manifestPath := writeIssueAuditManifest(t, modelroute.AuthorManifest{
		Schema: modelroute.CrossAuditAuthorSchema,
		Author: modelroute.AuditIdentity{
			Model: "qwen-author", Provider: "local", Family: "qwen", WeightsRevision: "qwen-w35", Harness: "fak-local",
			EndpointClass: "local", AccountClass: "local", ReasoningPosture: "high", ProvenanceSource: "weights:qwen",
		},
	})
	rosterPath := writeIssueAuditRoster(t, []modelroute.AuditIdentityAlias{
		{Alias: "qwen-author", CanonicalModel: "qwen3.5-27b", Provider: "local", Family: "qwen", WeightsRevision: "qwen-w35", ProvenanceSource: "registry:qwen"},
		{Alias: "gpt-review", CanonicalModel: "gpt-5.6-sol", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w56", ProvenanceSource: "registry:gpt"},
		{Alias: "claude-served", CanonicalModel: "claude-opus-4-6", Provider: "anthropic", Family: "claude", WeightsRevision: "claude-w46", ProvenanceSource: "registry:claude"},
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"claude-served","choices":[{"message":{"role":"assistant","content":"{\"verdict\":\"PASS\",\"reason\":\"declared pass\",\"evidence_refs\":[]}"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	fetcher := modelroute.IssueAuditFetcherFunc(func(context.Context, int) (modelroute.IssueAuditEvidence, error) {
		patch := "diff"
		return modelroute.IssueAuditEvidence{
			IssueNumber: 42, IssueURL: "https://example/issues/42", Title: "t", Body: "b", State: "CLOSED", ClosedAt: "2026-07-10T00:00:00Z", CommitSHA: "abcdef1", Diff: patch,
			ClosingCommits: []modelroute.IssueAuditClosingCommit{{SHA: "abcdef1", FirstParentSHA: "1234567", TreeOID: "tree-new", FirstParentTreeOID: "tree-old", Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"a.go"}}},
		}, nil
	})
	var stdout, stderr bytes.Buffer
	code := runIssueAuditWith(&stdout, &stderr, []string{
		"--issue", "42", "--author-manifest", manifestPath,
		"--auditor", "openai/gpt/gpt-review", "--auditor-driver", "http",
		"--auditor-endpoint", server.URL,
		"--identity-roster", rosterPath, "--json",
	}, fetcher, nil)
	if code == 0 {
		t.Fatalf("HTTP mismatch unexpectedly passed: %s", stdout.String())
	}
	if stdout.Len() != 0 || !strings.Contains(stderr.String(), "observed auditor identity mismatches declared auditor") {
		t.Fatalf("HTTP mismatch output: stderr=%s stdout=%s", stderr.String(), stdout.String())
	}
}

func writeIssueAuditManifest(t *testing.T, manifest modelroute.AuthorManifest) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "author.json")
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func issueAuditTestReviewRequest(t *testing.T) modelroute.IssueAuditReviewRequest {
	t.Helper()
	patch := "diff --git a/a.go b/a.go\n+safe\n"
	bundle, err := modelroute.BuildIssueAuditBundle(modelroute.IssueAuditEvidence{
		IssueNumber: 42, IssueURL: "https://example/issues/42", Title: "audit", Body: "done", State: "CLOSED", ClosedAt: "2026-07-10T00:00:00Z", CommitSHA: "abcdef1", Diff: patch,
		ClosingCommits: []modelroute.IssueAuditClosingCommit{{SHA: "abcdef1", FirstParentSHA: "1234567", TreeOID: "tree-new", FirstParentTreeOID: "tree-old", Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"a.go"}}},
	}, modelroute.IssueAuditBundleOptions{})
	if err != nil {
		t.Fatal(err)
	}
	req, err := modelroute.NewIssueAuditReviewRequest(42, bundle.BundleDigest, bundle)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func writeIssueAuditRoster(t *testing.T, aliases []modelroute.AuditIdentityAlias) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "roster.json")
	b, err := json.Marshal(modelroute.AuditIdentityRoster{Schema: modelroute.AuditIdentityRosterSchema, Aliases: aliases})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestIssueAuditLedgerAppendDuplicateAndJSONCursor(t *testing.T) {
	manifestPath, rosterPath, fetcher, reviewer := issueAuditLedgerFixtures(t)
	ledger := filepath.Join(t.TempDir(), "receipts.jsonl")
	base := []string{
		"--issue", "42", "--author-manifest", manifestPath, "--identity-roster", rosterPath,
		"--auditor", "anthropic/claude/claude-review", "--auditor-weights", "claude-w46",
		"--auditor-driver", "claude", "--auditor-reasoning", "high", "--ledger", ledger,
	}
	var stdout, stderr bytes.Buffer
	if code := runIssueAuditWith(&stdout, &stderr, base, fetcher, reviewer); code != 0 {
		t.Fatalf("append code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "ledger: appended rows=1 head_hash=") {
		t.Fatalf("append output=%s", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := runIssueAuditWith(&stdout, &stderr, base, fetcher, reviewer); code != 0 {
		t.Fatalf("duplicate code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "ledger: duplicate rows=1 head_hash=") {
		t.Fatalf("duplicate output=%s", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := runIssueAuditWith(&stdout, &stderr, append(base, "--json"), fetcher, reviewer); code != 0 {
		t.Fatalf("json code=%d stderr=%s", code, stderr.String())
	}
	var payload issueAuditLedgerOutput
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if !payload.Ledger.Duplicate || payload.Ledger.Cursor.Rows != 1 || payload.Ledger.Cursor.HeadHash == "" {
		t.Fatalf("json ledger=%+v", payload.Ledger)
	}
	if v, err := modelroute.VerifyAuditReceiptLedger(ledger); err != nil || v.Rows != 1 {
		t.Fatalf("ledger verification=%+v err=%v", v, err)
	}
}

func TestIssueAuditLedgerSameKeyConflictFailsLoud(t *testing.T) {
	manifestPath, rosterPath, fetcher, reviewer := issueAuditLedgerFixtures(t)
	ledger := filepath.Join(t.TempDir(), "receipts.jsonl")
	args := []string{
		"--issue", "42", "--author-manifest", manifestPath, "--identity-roster", rosterPath,
		"--auditor", "anthropic/claude/claude-review", "--auditor-weights", "claude-w46",
		"--auditor-driver", "claude", "--auditor-reasoning", "high", "--ledger", ledger,
	}
	var stdout, stderr bytes.Buffer
	if code := runIssueAuditWith(&stdout, &stderr, args, fetcher, reviewer); code != 0 {
		t.Fatalf("seed: %s", stderr.String())
	}
	conflictReviewer := modelroute.IssueAuditReviewerFunc(func(context.Context, modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
		return modelroute.IssueAuditReviewResult{Verdict: modelroute.CrossAuditPass, Reason: "different verified review"}, nil
	})
	stdout.Reset()
	stderr.Reset()
	if code := runIssueAuditWith(&stdout, &stderr, args, fetcher, conflictReviewer); code == 0 {
		t.Fatalf("same-key conflict succeeded: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), modelroute.ErrAuditReceiptKeyConflict.Error()) {
		t.Fatalf("typed conflict missing: %s", stderr.String())
	}
}

func issueAuditLedgerFixtures(t *testing.T) (string, string, modelroute.IssueAuditFetcher, modelroute.IssueAuditReviewer) {
	t.Helper()
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "author.json")
	rosterPath := filepath.Join(dir, "roster.json")
	manifest := modelroute.AuthorManifest{
		Schema:         modelroute.CrossAuditAuthorSchema,
		Author:         modelroute.AuditIdentity{Harness: "codex", Provider: "openai", Family: "gpt", Model: "gpt-author", WeightsRevision: "gpt-w54", EndpointClass: "remote", AccountClass: "subscription", ReasoningPosture: "xhigh"},
		SourceEvidence: []modelroute.EvidenceRef{{Kind: "session", Ref: "author-session"}}, CommitRange: "abc123..def456",
	}
	mb, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, mb, 0o600); err != nil {
		t.Fatal(err)
	}
	roster := modelroute.AuditIdentityRoster{Schema: modelroute.AuditIdentityRosterSchema, Aliases: []modelroute.AuditIdentityAlias{
		{Alias: "gpt-author", CanonicalModel: "gpt-5.4", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w54", ProvenanceSource: "roster:author"},
		{Alias: "claude-review", CanonicalModel: "claude-opus-4-6", Provider: "anthropic", Family: "claude", WeightsRevision: "claude-w46", ProvenanceSource: "roster:auditor"},
	}}
	rb, err := json.Marshal(roster)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(rosterPath, rb, 0o600); err != nil {
		t.Fatal(err)
	}
	patch := "diff --git a/thing.go b/thing.go\n+fixed\n"
	fetcher := modelroute.IssueAuditFetcherFunc(func(context.Context, int) (modelroute.IssueAuditEvidence, error) {
		return modelroute.IssueAuditEvidence{
			IssueNumber: 42, IssueURL: "https://github.com/example/repo/issues/42", Title: "fix thing", Body: "done", State: "CLOSED", ClosedAt: "2026-07-10T00:00:00Z", CommitSHA: "def456", Diff: patch,
			ClosingCommits: []modelroute.IssueAuditClosingCommit{{SHA: "def456", FirstParentSHA: "abc123", TreeOID: "tree-new", FirstParentTreeOID: "tree-old", Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"thing.go"}}},
			Tests:          []modelroute.EvidenceRef{{Kind: "test-path", Ref: "thing_test.go"}}, CI: []modelroute.EvidenceRef{{Kind: "check", Ref: "ci/unit"}}, DOS: []modelroute.EvidenceRef{{Kind: "dos-commit-audit", Ref: "commit:def456"}}, Evidence: []modelroute.EvidenceRef{{Kind: "issue-event", Ref: "referenced:def456"}},
		}, nil
	})
	reviewer := modelroute.IssueAuditReviewerFunc(func(context.Context, modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
		return modelroute.IssueAuditReviewResult{Verdict: modelroute.CrossAuditPass, Reason: "verified review"}, nil
	})
	return manifestPath, rosterPath, fetcher, reviewer
}
