package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunDispatchDoneClaimAuditFindsUnsupportedClaim(t *testing.T) {
	old := doneClaimAuditCommand
	t.Cleanup(func() { doneClaimAuditCommand = old })
	doneClaimAuditCommand = func(name string, args ...string) ([]byte, error) {
		if name == "gh" {
			return []byte(`[{"number":42,"title":"claim","state":"CLOSED","url":"https://example/42","comments":[{"body":"Shipped in deadbee.","url":"https://example/c"}]}]`), nil
		}
		return nil, &testCommandError{}
	}
	var stdout, stderr bytes.Buffer
	if code := runDispatchDoneClaimAudit(&stdout, &stderr, []string{"--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"number": 42`) || !strings.Contains(stdout.String(), `"verdict": "ACTION"`) {
		t.Fatalf("stdout=%s", stdout.String())
	}
}

func TestRunDispatchDoneClaimAuditAcceptsCommitWithPaths(t *testing.T) {
	old := doneClaimAuditCommand
	t.Cleanup(func() { doneClaimAuditCommand = old })
	doneClaimAuditCommand = func(name string, args ...string) ([]byte, error) {
		if name == "gh" {
			return []byte(`[{"number":43,"title":"claim","state":"CLOSED","comments":[{"body":"Landed in abcdef1."}]}]`), nil
		}
		return []byte("internal/x/x.go\n"), nil
	}
	var stdout, stderr bytes.Buffer
	if code := runDispatchDoneClaimAudit(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "done-claim-audit OK") {
		t.Fatalf("stdout=%s", stdout.String())
	}
}

func TestRunDispatchDoneClaimAuditRejectsBadLimit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDispatchDoneClaimAudit(&stdout, &stderr, []string{"--limit", "0"}); code != 2 {
		t.Fatalf("code=%d", code)
	}
}

type testCommandError struct{}

func (*testCommandError) Error() string { return "not found" }
