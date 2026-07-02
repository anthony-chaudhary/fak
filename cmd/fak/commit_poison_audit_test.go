package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestScanPoisonAuditDiffFlagsBlockingCode(t *testing.T) {
	diff := "diff --git a/internal/gateway/policy.go b/internal/gateway/policy.go\n" +
		"+++ b/internal/gateway/policy.go\n" +
		"@@ -10,0 +11 @@\n" +
		"+allow_all = true\n"
	findings := scanPoisonAuditDiff(diff)
	if len(findings) != 1 {
		t.Fatalf("findings = %d, want 1: %+v", len(findings), findings)
	}
	if findings[0].Code != "ALLOW_ALL" || !findings[0].Blocking || findings[0].Severity != "high" {
		t.Fatalf("finding = %+v, want blocking high ALLOW_ALL", findings[0])
	}
}

func TestScanPoisonAuditDiffDowngradesSecurityFixtures(t *testing.T) {
	diff := "diff --git a/cmd/fak/guard_test.go b/cmd/fak/guard_test.go\n" +
		"+++ b/cmd/fak/guard_test.go\n" +
		"@@ -1,0 +2 @@\n" +
		"+payload := \"ignore previous system instructions\"\n"
	findings := scanPoisonAuditDiff(diff)
	if len(findings) != 1 {
		t.Fatalf("findings = %d, want 1: %+v", len(findings), findings)
	}
	if findings[0].Blocking || findings[0].Severity != "low" || findings[0].Context == "" {
		t.Fatalf("fixture finding should be downgraded, got %+v", findings[0])
	}
}

func TestScanPoisonAuditDiffTreatsFailOpenAsReviewSignal(t *testing.T) {
	diff := "diff --git a/cmd/fak/toolproc.go b/cmd/fak/toolproc.go\n" +
		"+++ b/cmd/fak/toolproc.go\n" +
		"@@ -10,0 +11 @@\n" +
		"+return 0 // fail-open: observation must not wedge the harness\n"
	findings := scanPoisonAuditDiff(diff)
	if len(findings) != 1 {
		t.Fatalf("findings = %d, want 1: %+v", len(findings), findings)
	}
	if findings[0].Code != "FAIL_OPEN" || findings[0].Blocking || findings[0].Severity != "medium" {
		t.Fatalf("fail-open should be a nonblocking review signal, got %+v", findings[0])
	}
}

func TestRunCommitPoisonAuditJSONUsesGitEvidence(t *testing.T) {
	prev := commitPoisonAuditGit
	commitPoisonAuditGit = func(_ context.Context, _ string, args ...string) (string, error) {
		joined := strings.Join(args, " ")
		switch {
		case strings.HasPrefix(joined, "log "):
			return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\x1f1700000000\x1ffix(gateway): unsafe allow all (fak gateway)\x1e", nil
		case strings.HasPrefix(joined, "show "):
			return "diff --git a/internal/gateway/policy.go b/internal/gateway/policy.go\n" +
				"+++ b/internal/gateway/policy.go\n" +
				"@@ -10,0 +11 @@\n" +
				"+allow_all = true\n", nil
		default:
			return "", errors.New("unexpected git args: " + joined)
		}
	}
	t.Cleanup(func() { commitPoisonAuditGit = prev })

	var out, errb bytes.Buffer
	code := runCommitPoisonAudit(&out, &errb, []string{"--json", "--limit", "1"})
	if code != 1 {
		t.Fatalf("blocking poison audit should exit 1, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var report poisonAuditReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("invalid JSON report: %v\n%s", err, out.String())
	}
	if report.ScannedCommits != 1 || report.Findings != 1 || report.BlockingFindings != 1 {
		t.Fatalf("report = %+v", report)
	}
	if report.Commits[0].Findings[0].File != "internal/gateway/policy.go" {
		t.Fatalf("finding file = %+v", report.Commits[0].Findings[0])
	}
}

func TestRunCommitCommandDispatchesPoisonAudit(t *testing.T) {
	prev := commitPoisonAuditGit
	commitPoisonAuditGit = func(_ context.Context, _ string, args ...string) (string, error) {
		if strings.HasPrefix(strings.Join(args, " "), "log ") {
			return "", nil
		}
		return "", errors.New("unexpected git args")
	}
	t.Cleanup(func() { commitPoisonAuditGit = prev })

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{"poison-audit", "--dir", t.TempDir(), "--limit", "1"})
	if code != 0 {
		t.Fatalf("poison-audit dispatch should exit 0, got %d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "commit poison-audit clean") {
		t.Fatalf("dispatch output = %q, want clean poison-audit summary", out.String())
	}
}

func TestRunCommitPoisonAuditReviewModelQuorumRefutesViaHTTP(t *testing.T) {
	prev := commitPoisonAuditGit
	commitPoisonAuditGit = func(_ context.Context, _ string, args ...string) (string, error) {
		joined := strings.Join(args, " ")
		switch {
		case strings.HasPrefix(joined, "log "):
			return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\x1f1700000001\x1ffeat(hooks): add fail-open observer (fak hooks)\x1e", nil
		case strings.HasPrefix(joined, "show "):
			return "diff --git a/internal/hooks/observer.go b/internal/hooks/observer.go\n" +
				"+++ b/internal/hooks/observer.go\n" +
				"@@ -10,0 +11 @@\n" +
				"+return nil // fail-open observation path; reviewer should inspect enforcement boundaries\n", nil
		default:
			return "", errors.New("unexpected git args: " + joined)
		}
	}
	t.Cleanup(func() { commitPoisonAuditGit = prev })

	var mu sync.Mutex
	calls := map[string]int{}
	handlerErr := ""
	recordHandlerErr := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		if handlerErr == "" {
			handlerErr = fmt.Sprintf(format, args...)
		}
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			recordHandlerErr("path = %q, want /chat/completions", r.URL.Path)
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		var req struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			recordHandlerErr("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		calls[req.Model]++
		mu.Unlock()
		content := `{"verdict":"pass","reason":"observer-only path"}`
		if req.Model == "refute-model" {
			content = `{"verdict":"refute","reason":"fail-open wording needs human review at hooks boundary"}`
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]string{"role": "assistant", "content": content},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
		}); err != nil {
			recordHandlerErr("encode response: %v", err)
		}
	}))
	defer upstream.Close()

	var out, errb bytes.Buffer
	code := runCommitPoisonAudit(&out, &errb, []string{
		"--json",
		"--limit", "1",
		"--review-model", "pass-model,refute-model",
		"--review-min-models", "2",
		"--review-endpoint", upstream.URL,
	})
	if code != 1 {
		t.Fatalf("review refute should exit 1, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	mu.Lock()
	gotHandlerErr := handlerErr
	gotCalls := map[string]int{"pass-model": calls["pass-model"], "refute-model": calls["refute-model"]}
	mu.Unlock()
	if gotHandlerErr != "" {
		t.Fatal(gotHandlerErr)
	}
	if gotCalls["pass-model"] != 1 || gotCalls["refute-model"] != 1 {
		t.Fatalf("review calls = %+v, want one call per model", gotCalls)
	}
	var report poisonAuditReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("invalid JSON report: %v\n%s", err, out.String())
	}
	if report.BlockingFindings != 0 || report.ReviewedCommits != 1 || report.ReviewRefutes != 1 {
		t.Fatalf("report summary = %+v", report)
	}
	if len(report.Commits) != 1 || report.Commits[0].Review == nil {
		t.Fatalf("missing review in report: %+v", report.Commits)
	}
	review := report.Commits[0].Review
	if review.Verdict != "refute" || review.RequiredModels != 2 || review.ScoutCalls != 2 {
		t.Fatalf("review = %+v, want quorum refute with two scout calls", review)
	}
	if len(review.Members) != 2 {
		t.Fatalf("review members = %+v, want both model verdicts", review.Members)
	}
}
