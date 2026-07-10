package modelroute

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestCrossAuditSpineRejectsSameFamilyAndBindsReceipt(t *testing.T) {
	manifest := AuthorManifest{
		Schema: CrossAuditAuthorSchema,
		Author: ModelIdentity{
			Harness:          "codex",
			Provider:         "openai",
			Family:           "gpt",
			Model:            "gpt-5.4",
			ReasoningPosture: "xhigh",
		},
		SourceEvidence: []EvidenceRef{{Kind: "session", Ref: "codex-session:abc"}},
		CommitRange:    "abc123..def456",
	}

	var fetchCalls, reviewCalls int
	fetcher := IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) {
		fetchCalls++
		return crossAuditFixtureEvidence(), nil
	})
	reviewer := IssueAuditReviewerFunc(func(_ context.Context, req IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
		reviewCalls++
		if req.SubjectDigest == "" || req.PromptDigest == "" || req.PromptVersion != CrossAuditPromptVersion {
			t.Fatalf("review request lost bound prompt/subject fields: %+v", req)
		}
		for _, want := range []string{"BEGIN_UNTRUSTED_ISSUE_BODY", "BEGIN_UNTRUSTED_DIFF", "subject_digest:"} {
			if !strings.Contains(req.Prompt, want) {
				t.Fatalf("review prompt missing %q", want)
			}
		}
		return IssueAuditReviewResult{
			Verdict:      CrossAuditPass,
			Reason:       "the closing diff implements the issue contract and carries a regression test",
			EvidenceRefs: []string{"test:TestThing"},
		}, nil
	})

	_, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber: 42,
		Author:      manifest,
		Auditor:     ModelIdentity{Provider: "azure-openai", Family: " GPT ", Model: "gpt-5.6-sol"},
	}, fetcher, reviewer)
	if !IsIndependenceRefusal(err) || !strings.Contains(err.Error(), "SAME_MODEL_FAMILY") {
		t.Fatalf("same-family error = %v, want typed SAME_MODEL_FAMILY refusal", err)
	}
	if fetchCalls != 0 || reviewCalls != 0 {
		t.Fatalf("same-family refusal happened after work: fetch=%d review=%d, want 0/0", fetchCalls, reviewCalls)
	}

	receipt, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber: 42,
		Author:      manifest,
		Auditor: ModelIdentity{
			Harness:          "fak issue audit",
			Provider:         "anthropic",
			Family:           "claude",
			Model:            "claude-opus-4-6",
			EndpointClass:    "hosted",
			ReasoningPosture: "high",
		},
	}, fetcher, reviewer)
	if err != nil {
		t.Fatalf("AuditIssue different-family: %v", err)
	}
	if fetchCalls != 1 || reviewCalls != 1 {
		t.Fatalf("different-family calls = fetch %d review %d, want 1/1", fetchCalls, reviewCalls)
	}
	if receipt.Schema != CrossAuditReceiptSchema || receipt.Subject.IssueNumber != 42 || receipt.Subject.CommitSHA != "def456" {
		t.Fatalf("receipt lost subject binding: %+v", receipt)
	}
	if receipt.Subject.DiffSHA256 == "" || receipt.Subject.Digest == "" || receipt.PromptDigest == "" || receipt.ReceiptDigest == "" {
		t.Fatalf("receipt has empty digest binding: %+v", receipt)
	}
	if receipt.Author.Family != "gpt" || receipt.Auditor.Family != "claude" || !receipt.Independence.Admitted || receipt.Independence.Reason != "DIFFERENT_MODEL_FAMILY" {
		t.Fatalf("identity/independence row = author %+v auditor %+v decision %+v", receipt.Author, receipt.Auditor, receipt.Independence)
	}
	if receipt.PolicyVersion != CrossAuditPolicyVersion || receipt.PromptVersion != CrossAuditPromptVersion || receipt.Verdict != CrossAuditPass {
		t.Fatalf("receipt policy/prompt/verdict = %+v", receipt)
	}
	if len(receipt.EvidenceRefs) < 5 {
		t.Fatalf("receipt evidence refs = %+v, want author, fetch, reviewer, issue, commit, diff refs", receipt.EvidenceRefs)
	}
	if err := receipt.Verify(); err != nil {
		t.Fatalf("receipt digest did not recompute: %v", err)
	}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseIssueAuditReceipt(b)
	if err != nil || parsed.ReceiptDigest != receipt.ReceiptDigest {
		t.Fatalf("receipt JSON round trip: parsed=%+v err=%v", parsed, err)
	}
	tampered := receipt
	tampered.Subject.CommitSHA = "poisoned"
	if err := tampered.Verify(); err == nil {
		t.Fatal("tampered receipt still verified")
	}
}

func TestCrossAuditSpineClassifiesReviewerFailuresWithoutFailOpen(t *testing.T) {
	base := IssueAuditRequest{
		IssueNumber: 42,
		Author: AuthorManifest{
			Schema: CrossAuditAuthorSchema,
			Author: ModelIdentity{Provider: "openai", Family: "gpt", Model: "gpt-5.4"},
		},
		Auditor: ModelIdentity{Provider: "anthropic", Family: "claude", Model: "claude-opus-4-6"},
	}
	fetcher := IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) {
		return crossAuditFixtureEvidence(), nil
	})

	t.Run("unavailable", func(t *testing.T) {
		receipt, err := AuditIssue(context.Background(), base, fetcher, IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
			return IssueAuditReviewResult{}, context.DeadlineExceeded
		}))
		if err != nil || receipt.Verdict != CrossAuditUnavailable || !strings.Contains(receipt.Reason, "deadline") {
			t.Fatalf("unavailable receipt = %+v err=%v", receipt, err)
		}
		if err := receipt.Verify(); err != nil {
			t.Fatalf("unavailable receipt did not bind: %v", err)
		}
	})

	t.Run("invalid-is-inconclusive", func(t *testing.T) {
		receipt, err := AuditIssue(context.Background(), base, fetcher, IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
			return IssueAuditReviewResult{Verdict: "MAYBE", Reason: "looks okay"}, nil
		}))
		if err != nil || receipt.Verdict != CrossAuditInconclusive || !strings.Contains(receipt.Reason, "out-of-vocabulary") {
			t.Fatalf("invalid receipt = %+v err=%v", receipt, err)
		}
	})
}

func crossAuditFixtureEvidence() IssueAuditEvidence {
	return IssueAuditEvidence{
		IssueNumber: 42,
		IssueURL:    "https://github.com/example/repo/issues/42",
		Title:       "fix the thing",
		Body:        "Done condition: the thing is fixed.",
		State:       "CLOSED",
		ClosedAt:    "2026-07-10T00:00:00Z",
		CommitSHA:   "def456",
		Diff:        "diff --git a/thing.go b/thing.go\n+fixed\n",
		Evidence:    []EvidenceRef{{Kind: "issue-event", Ref: "referenced:def456"}},
	}
}
