package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

func TestAuditVerifyDispatchesAllChainSchemas(t *testing.T) {
	t.Run("decision journal", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "decision.jsonl")
		j, err := journal.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		j.Emit(abi.Event{Kind: abi.EvDecide, Call: &abi.ToolCall{SeqNo: 1, Tool: "read"}, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})
		if err := j.Close(); err != nil {
			t.Fatal(err)
		}
		var out, stderr bytes.Buffer
		if code := runAuditVerify(&out, &stderr, path); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, stderr.String())
		}
		if !strings.Contains(out.String(), "chain intact") {
			t.Fatalf("out=%s", out.String())
		}
	})

	t.Run("usage log", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "usage.jsonl")
		row := []byte(`{"schema":"` + usagelog.SchemaV1 + `","seq":1,"ts_unix_nano":1,"prev_hash":"","hash":"bad"}` + "\n")
		if err := os.WriteFile(path, row, 0o644); err != nil {
			t.Fatal(err)
		}
		if got := auditVerifySchema(path); got != usagelog.SchemaV1 {
			t.Fatalf("schema=%q", got)
		}
		var out, stderr bytes.Buffer
		if code := runAuditVerify(&out, &stderr, path); code != 1 {
			t.Fatalf("code=%d out=%s", code, out.String())
		}
		if !strings.Contains(stderr.String(), "TAMPERED/BROKEN") {
			t.Fatalf("stderr=%s", stderr.String())
		}
	})

	t.Run("cross audit receipt", func(t *testing.T) {
		path := validAuditReceiptLedger(t)
		var out, stderr bytes.Buffer
		if code := runAuditVerify(&out, &stderr, path); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, stderr.String())
		}
		for _, want := range []string{"cross-audit receipt", "unique_audits=1", "PASS:1", "head_hash="} {
			if !strings.Contains(out.String(), want) {
				t.Fatalf("out missing %q: %s", want, out.String())
			}
		}

		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		b = bytes.Replace(b, []byte(`"verdict":"PASS"`), []byte(`"verdict":"REFUTE"`), 1)
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		out.Reset()
		stderr.Reset()
		if code := runAuditVerify(&out, &stderr, path); code != 1 {
			t.Fatalf("mutated code=%d out=%s", code, out.String())
		}
		if !strings.Contains(stderr.String(), "line 1") || !strings.Contains(stderr.String(), "TAMPERED/BROKEN") {
			t.Fatalf("mutation not localized: %s", stderr.String())
		}
	})
}

func validAuditReceiptLedger(t *testing.T) string {
	t.Helper()
	patch := "diff --git a/thing.go b/thing.go\n+fixed\n"
	evidence := modelroute.IssueAuditEvidence{
		IssueNumber: 42, IssueURL: "https://github.com/example/repo/issues/42", Title: "fix the thing",
		Body: "Done condition: fixed.", State: "CLOSED", ClosedAt: "2026-07-10T00:00:00Z",
		CommitSHA: "def456", Diff: patch,
		ClosingCommits: []modelroute.IssueAuditClosingCommit{{
			SHA: "def456", FirstParentSHA: "abc123", TreeOID: "tree-new", FirstParentTreeOID: "tree-old",
			Patch: patch, PatchSHA256: modelroute.IssueAuditContentDigest(patch), ChangedPaths: []string{"thing.go"},
		}},
		Tests:    []modelroute.EvidenceRef{{Kind: "test-path", Ref: "thing_test.go"}},
		CI:       []modelroute.EvidenceRef{{Kind: "check", Ref: "ci/unit"}},
		DOS:      []modelroute.EvidenceRef{{Kind: "dos-commit-audit", Ref: "commit:def456"}},
		Evidence: []modelroute.EvidenceRef{{Kind: "issue-event", Ref: "referenced:def456"}},
	}
	author := modelroute.AuditIdentity{
		Harness: "codex", Provider: "openai", Family: "gpt", Model: "gpt-author", WeightsRevision: "gpt-w54",
		EndpointClass: "remote", AccountClass: "subscription", ReasoningPosture: "xhigh", ProvenanceSource: "session:author",
	}
	auditor := modelroute.AuditIdentity{
		Harness: "fak issue audit", Provider: "anthropic", Family: "claude", Model: "claude-review", WeightsRevision: "claude-w46",
		EndpointClass: "hosted", AccountClass: "subscription", ReasoningPosture: "high", ProvenanceSource: "session:auditor",
	}
	policy := modelroute.DefaultAuditIndependencePolicy()
	policy.Aliases = []modelroute.AuditIdentityAlias{
		{Alias: "gpt-author", CanonicalModel: "gpt-5.4", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w54", ProvenanceSource: "session:author"},
		{Alias: "claude-review", CanonicalModel: "claude-opus-4-6", Provider: "anthropic", Family: "claude", WeightsRevision: "claude-w46", ProvenanceSource: "session:auditor"},
	}
	receipt, err := modelroute.AuditIssue(context.Background(), modelroute.IssueAuditRequest{
		IssueNumber: 42,
		Author:      modelroute.AuthorManifest{Schema: modelroute.CrossAuditAuthorSchema, Author: author, SourceEvidence: []modelroute.EvidenceRef{{Kind: "session", Ref: "author"}}, CommitRange: "abc123..def456"},
		Auditor:     auditor, IndependencePolicy: policy,
	}, modelroute.IssueAuditFetcherFunc(func(context.Context, int) (modelroute.IssueAuditEvidence, error) { return evidence, nil }),
		modelroute.IssueAuditReviewerFunc(func(context.Context, modelroute.IssueAuditReviewRequest) (modelroute.IssueAuditReviewResult, error) {
			return modelroute.IssueAuditReviewResult{Verdict: modelroute.CrossAuditPass, Reason: "evidence bound", EvidenceRefs: []string{"test:thing"}}, nil
		}))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "receipts.jsonl")
	if _, err := modelroute.AppendAuditReceiptLedger(path, receipt); err != nil {
		t.Fatal(err)
	}
	if v, err := modelroute.VerifyAuditReceiptLedger(path); err != nil || v.Rows != 1 {
		t.Fatalf("fixture invalid: %+v %v", v, err)
	}
	return path
}
