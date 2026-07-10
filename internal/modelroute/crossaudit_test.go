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
			WeightsRevision:  "gpt-w54",
			EndpointClass:    "remote",
			AccountClass:     "subscription",
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
		if req.PromptDigest != hashString(req.Prompt) {
			t.Fatalf("prompt digest %q does not bind exact sent payload %q", req.PromptDigest, hashString(req.Prompt))
		}
		for _, want := range []string{CrossAuditSystemPrompt, "BEGIN_UNTRUSTED_ISSUE_BODY", "BEGIN_UNTRUSTED_DIFF", "subject_digest:"} {
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
		IssueNumber:        42,
		Author:             manifest,
		Auditor:            fullAuditIdentity("gpt-review", "azure-openai", "gpt", "gpt-w56", "codex", "remote", "api", "xhigh", "session:gpt-review"),
		IndependencePolicy: crossAuditTestPolicy(),
	}, fetcher, reviewer)
	if !IsIndependenceRefusal(err) || !strings.Contains(err.Error(), string(AuditReasonRefuseSameFamily)) {
		t.Fatalf("same-family error = %v, want typed %s refusal", err, AuditReasonRefuseSameFamily)
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
			Model:            "claude-review",
			WeightsRevision:  "claude-w46",
			EndpointClass:    "hosted",
			AccountClass:     "subscription",
			ReasoningPosture: "high",
			ProvenanceSource: "session:claude-review",
		},
		IndependencePolicy: crossAuditTestPolicy(),
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
	if receipt.Author.Family != "gpt" || receipt.Auditor.Family != "claude" || !receipt.Independence.Admitted || receipt.Independence.Verdict != AuditIndependenceAdmit || receipt.Independence.Reason != string(AuditReasonAdmitIndependent) {
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
	contradictory := receipt
	contradictory.Independence.Admitted = false
	contradictory.ReceiptDigest = contradictory.recomputeDigest()
	if err := contradictory.Verify(); err == nil || !strings.Contains(err.Error(), "contradicts") {
		t.Fatalf("re-digested contradictory receipt error = %v", err)
	}
	policyMismatch := receipt
	policyMismatch.Independence.Rule = "other-policy/v1"
	policyMismatch.ReceiptDigest = policyMismatch.recomputeDigest()
	if err := policyMismatch.Verify(); err == nil || !strings.Contains(err.Error(), "differs") {
		t.Fatalf("re-digested policy mismatch error = %v", err)
	}
}

func TestCrossAuditSpineRefusesRelabeledSameFamilyBeforeInference(t *testing.T) {
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = []AuditIdentityAlias{
		{Alias: "author-frontier", CanonicalModel: "gpt-5.6-sol", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w56", ProvenanceSource: "registry:v1"},
		{Alias: "independent-reviewer", CanonicalModel: "gpt-5.6-sol", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w56", ProvenanceSource: "registry:v1"},
	}
	var fetchCalls, reviewCalls int
	_, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber:        42,
		Author:             AuthorManifest{Schema: CrossAuditAuthorSchema, Author: AuditIdentity{Model: "author-frontier"}},
		Auditor:            AuditIdentity{Model: "independent-reviewer"},
		IndependencePolicy: policy,
	}, IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) {
		fetchCalls++
		return crossAuditFixtureEvidence(), nil
	}), IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
		reviewCalls++
		return IssueAuditReviewResult{Verdict: CrossAuditPass}, nil
	}))
	if !IsIndependenceRefusal(err) || !strings.Contains(err.Error(), string(AuditReasonRefuseSameWeights)) {
		t.Fatalf("relabeled alias error = %v", err)
	}
	if fetchCalls != 0 || reviewCalls != 0 {
		t.Fatalf("alias laundering reached work: fetch=%d review=%d", fetchCalls, reviewCalls)
	}
}

func TestCrossAuditSpineRefusesHTTPDeclaredFamilyMismatch(t *testing.T) {
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = []AuditIdentityAlias{
		auditAlias("qwen-local", "qwen3.5-27b", "local", "qwen", "qwen-w35"),
		auditAlias("gpt-review", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
		auditAlias("claude-review", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
	}
	receipt, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber: 42,
		Author: AuthorManifest{Schema: CrossAuditAuthorSchema, Author: fullAuditIdentity(
			"qwen-local", "local", "qwen", "qwen-w35", "fak-local", "local", "local", "high", "weights:qwen",
		)},
		Auditor: func() AuditIdentity {
			id := fullAuditIdentity("gpt-review", "openai", "gpt", "gpt-w56", "http", "remote", "api", "high", "registry:gpt")
			id.Driver = "http"
			return id
		}(),
		IndependencePolicy: policy,
	}, IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) {
		return crossAuditFixtureEvidence(), nil
	}), IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
		return IssueAuditReviewResult{
			Verdict:         CrossAuditPass,
			Reason:          "declared pass",
			ObservedAuditor: &AuditIdentity{Model: "claude-review"},
		}, nil
	}))
	if err != nil {
		t.Fatalf("AuditIssue: %v", err)
	}
	if receipt.Independence.Admitted || receipt.Independence.Verdict != AuditIndependenceRefuse || receipt.Independence.Reason != string(AuditReasonRefuseObservedMismatch) {
		t.Fatalf("HTTP mismatch independence = %+v", receipt.Independence)
	}
	if receipt.Verdict != CrossAuditInconclusive || receipt.ObservedAuditor == nil || receipt.ObservedAuditor.Family != "claude" {
		t.Fatalf("HTTP mismatch receipt = %+v", receipt)
	}
	if err := receipt.Verify(); err == nil || !strings.Contains(err.Error(), "mismatches declared auditor") {
		t.Fatalf("HTTP mismatch receipt verification error = %v", err)
	}
}

func TestCrossAuditSpineHTTPDriverCannotOptOutOfObservedIdentity(t *testing.T) {
	policy := crossAuditHTTPPolicy()
	auditor := fullAuditIdentity("gpt-review", "openai", "gpt", "gpt-w56", "openai-compatible-http", "remote", "api", "high", "")
	auditor.Driver = "http"
	receipt, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber:        42,
		Author:             AuthorManifest{Schema: CrossAuditAuthorSchema, Author: fullAuditIdentity("qwen-local", "local", "qwen", "qwen-w35", "fak-local", "local", "local", "high", "weights:qwen")},
		Auditor:            auditor,
		IndependencePolicy: policy,
		// Deliberately false: the HTTP driver's capability is authoritative.
		RequireObservedAuditorIdentity: false,
	}, IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) {
		return crossAuditFixtureEvidence(), nil
	}), IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
		return IssueAuditReviewResult{Verdict: CrossAuditPass, Reason: "caller tried to bypass readback"}, nil
	}))
	if err != nil {
		t.Fatalf("AuditIssue: %v", err)
	}
	if receipt.Verdict == CrossAuditPass || receipt.Independence.Admitted || receipt.Independence.Verdict != AuditIndependenceUnknown {
		t.Fatalf("HTTP opt-out bypass receipt = %+v", receipt)
	}
	if err := receipt.Verify(); err == nil || !strings.Contains(err.Error(), "unresolved") {
		t.Fatalf("HTTP missing-observation receipt verification error = %v", err)
	}
}

func TestIssueAuditReceiptVerifyRejectsTamperedHTTPObservation(t *testing.T) {
	policy := crossAuditHTTPPolicy()
	auditor := fullAuditIdentity("gpt-review", "openai", "gpt", "gpt-w56", "openai-compatible-http", "remote", "api", "high", "")
	auditor.Driver = "http"
	receipt, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber:        42,
		Author:             AuthorManifest{Schema: CrossAuditAuthorSchema, Author: fullAuditIdentity("qwen-local", "local", "qwen", "qwen-w35", "fak-local", "local", "local", "high", "weights:qwen")},
		Auditor:            auditor,
		IndependencePolicy: policy,
	}, IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) {
		return crossAuditFixtureEvidence(), nil
	}), IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
		return IssueAuditReviewResult{Verdict: CrossAuditPass, Reason: "matched", ObservedAuditor: &AuditIdentity{Model: "gpt-review"}}, nil
	}))
	if err != nil {
		t.Fatalf("AuditIssue: %v", err)
	}
	if err := receipt.Verify(); err != nil {
		t.Fatalf("valid HTTP receipt: %v", err)
	}

	tampered := receipt
	tampered.ObservedAuditor = nil
	tampered.ReceiptDigest = tampered.recomputeDigest()
	if err := tampered.Verify(); err == nil || !strings.Contains(err.Error(), "requires observed") {
		t.Fatalf("re-digested missing observation error = %v", err)
	}

	tampered = receipt
	other := *tampered.ObservedAuditor
	other.Model = "forged-model"
	tampered.ObservedAuditor = &other
	tampered.ReceiptDigest = tampered.recomputeDigest()
	if err := tampered.Verify(); err == nil || !strings.Contains(err.Error(), "mismatches declared auditor") {
		t.Fatalf("re-digested mismatched observation error = %v", err)
	}
}

func TestCrossAuditSpineClassifiesReviewerFailuresWithoutFailOpen(t *testing.T) {
	base := IssueAuditRequest{
		IssueNumber: 42,
		Author: AuthorManifest{
			Schema: CrossAuditAuthorSchema,
			Author: fullAuditIdentity("gpt-author", "openai", "gpt", "gpt-w54", "codex", "remote", "subscription", "xhigh", "session:gpt"),
		},
		Auditor:            fullAuditIdentity("claude-review", "anthropic", "claude", "claude-w46", "claude-code", "remote", "subscription", "high", "session:claude"),
		IndependencePolicy: crossAuditTestPolicy(),
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

	t.Run("unavailable-observed-identity-remains-unknown", func(t *testing.T) {
		req := base
		req.RequireObservedAuditorIdentity = true
		receipt, err := AuditIssue(context.Background(), req, fetcher, IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
			return IssueAuditReviewResult{}, context.DeadlineExceeded
		}))
		if err != nil || receipt.Verdict != CrossAuditUnavailable || receipt.Independence.Admitted || receipt.Independence.Verdict != AuditIndependenceUnknown || receipt.Independence.Reason != string(AuditReasonUnknownObservedIdentity) {
			t.Fatalf("transport-error receipt = %+v err=%v", receipt, err)
		}
		if err := receipt.Verify(); err != nil {
			t.Fatalf("transport-error receipt did not bind: %v", err)
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

func crossAuditTestPolicy() AuditIndependencePolicy {
	policy := DefaultAuditIndependencePolicy()
	gpt54 := auditAlias("gpt-5.4", "gpt-5.4", "openai", "gpt", "gpt-w54")
	gptAuthor := auditAlias("gpt-author", "gpt-5.4", "openai", "gpt", "gpt-w54")
	gptAuthor.ProvenanceSource = gpt54.ProvenanceSource
	policy.Aliases = []AuditIdentityAlias{
		gpt54,
		gptAuthor,
		auditAlias("gpt-review", "gpt-5.6-sol", "azure-openai", "gpt", "gpt-w56"),
		auditAlias("claude-review", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
	}
	return policy
}

func crossAuditHTTPPolicy() AuditIndependencePolicy {
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = []AuditIdentityAlias{
		auditAlias("qwen-local", "qwen3.5-27b", "local", "qwen", "qwen-w35"),
		auditAlias("gpt-review", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
		auditAlias("claude-review", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
	}
	return policy
}
