package devcmd

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFanoutExistingFixture(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "existing.json")
	fixture := fmt.Sprintf(`[{"number": 42, "body": %q}]`, body)
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestIssueFanoutLiveFilesUnseenAndRerunFilesZero(t *testing.T) {
	// The fixture tracker already carries the qa-edge-sweep marker key, so the
	// first live run files the other two qa candidates and skips it.
	fixture := writeFanoutExistingFixture(t, "carries fanout-fanoutlivetest-qa-edge-sweep already")
	var calls [][]string
	gh := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return fmt.Sprintf("https://github.com/o/r/issues/%d", 900+len(calls)), "", true
	}
	argv := []string{
		"--title", "fanout live test", "--leaf", "fanoutlivetest", "--spine", "deadbeef",
		"--areas", "qa", "--parent-issue", "36", "--parent-baseline-points", "100", "--target-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--witnessed-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--live", "--existing-json", fixture,
	}
	var out, errOut bytes.Buffer
	if code := runIssueFanoutWith(&out, &errOut, argv, gh); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "filed 2, skipped 1, failed 0") {
		t.Fatalf("first run output missing filed/skipped fold:\n%s", out.String())
	}
	if len(calls) != 2 {
		t.Fatalf("gh create calls = %d, want 2", len(calls))
	}

	// Rerun against a tracker carrying every key: files zero, spams nothing.
	all := writeFanoutExistingFixture(t,
		"fanout-fanoutlivetest-qa-edge-sweep fanout-fanoutlivetest-qa-failure-paths fanout-fanoutlivetest-qa-determinism")
	calls = nil
	out.Reset()
	argv[len(argv)-1] = all
	if code := runIssueFanoutWith(&out, &errOut, argv, gh); code != 0 {
		t.Fatalf("rerun exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "filed 0, skipped 3, failed 0") || len(calls) != 0 {
		t.Fatalf("rerun must file zero (gh calls = %d):\n%s", len(calls), out.String())
	}
}

func TestIssueFanoutLiveGhFailureExitsOne(t *testing.T) {
	fixture := writeFanoutExistingFixture(t, "no keys here")
	gh := func(args []string) (string, string, bool) { return "", "gh: boom", false }
	var out, errOut bytes.Buffer
	code := runIssueFanoutWith(&out, &errOut, []string{
		"--title", "t", "--leaf", "fanoutlivetest", "--spine", "s",
		"--areas", "qa", "--parent-issue", "36", "--parent-baseline-points", "100", "--target-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--witnessed-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--live", "--existing-json", fixture,
	}, gh)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 on gh failure\n%s", code, out.String())
	}
}

func TestIssueFanoutLiveRefusesUnboundedDedupe(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runIssueFanoutWith(&out, &errOut, []string{
		"--title", "t", "--leaf", "l", "--spine", "s", "--live", "--dedupe-cap", "0",
	}, nil)
	if code != 2 || !strings.Contains(errOut.String(), "--dedupe-cap") {
		t.Fatalf("exit = %d stderr = %q, want 2 + dedupe-cap refusal", code, errOut.String())
	}
}

func TestIssueFanoutOfflineDefaultUnchangedByLiveFlags(t *testing.T) {
	// The offline default path must not consult gh or mention live filing.
	gh := func(args []string) (string, string, bool) {
		t.Fatalf("offline run must not invoke gh, got %v", args)
		return "", "", false
	}
	var out, errOut bytes.Buffer
	if code := runIssueFanoutWith(&out, &errOut, []string{
		"--title", "t", "--leaf", "fanoutlivetest", "--spine", "s", "--areas", "qa",
	}, gh); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.HasPrefix(out.String(), "fanout: 3 contract-ready follow-ons") {
		t.Fatalf("offline output changed:\n%s", out.String())
	}
}
