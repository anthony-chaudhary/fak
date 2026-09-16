package devcmd

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// nearDupBacklog is the fake bounded backlog: one open issue whose title is a
// near-twin of the candidate the refusal tests file.
const nearDupBacklog = `[{"number":1596,"title":"fix(dispatch): the preflight lane-lock gate ignores contract-lease TTL","body":"the preflight lane-lock gate reads the contract lease but ignores its ttl"}]`

// withIssueCreateDedupeFetcher swaps the package fetcher seam for one test.
func withIssueCreateDedupeFetcher(t *testing.T, fetch issueCreateDedupeFunc) {
	t.Helper()
	prev := issueCreateDedupeFetcher
	issueCreateDedupeFetcher = fetch
	t.Cleanup(func() { issueCreateDedupeFetcher = prev })
}

func TestIssueCreateDedupeGateRefusesNearDuplicate(t *testing.T) {
	fetch := func(cap int) ([]byte, error) { return []byte(nearDupBacklog), nil }
	result := &issueCreateResult{}
	var out, errb bytes.Buffer
	code, refused := issueCreateRunDedupeGate(&out, &errb, result, fetch, true, 0, 0, false,
		"fix(dispatch): the preflight lane-lock gate ignores contract-lease TTL",
		"the preflight lane-lock gate reads the contract lease but ignores its ttl", false)
	if !refused || code != 3 {
		t.Fatalf("code=%d refused=%v, want 3/true; stderr=%s", code, refused, errb.String())
	}
	if !strings.Contains(errb.String(), issueCreateDedupeReason) || !strings.Contains(errb.String(), "#1596") {
		t.Fatalf("stderr missing refusal + twin pointer: %s", errb.String())
	}
	if result.Dedupe == nil || !result.Dedupe.Refused {
		t.Fatalf("result.Dedupe not marked refused: %+v", result.Dedupe)
	}
}

func TestIssueCreateDedupeGateFailsOpenOnFetchError(t *testing.T) {
	fetch := func(cap int) ([]byte, error) { return nil, errors.New("gh: not found") }
	result := &issueCreateResult{}
	var out, errb bytes.Buffer
	code, refused := issueCreateRunDedupeGate(&out, &errb, result, fetch, true, 0, 0, false,
		"fix(dispatch): the preflight lane-lock gate ignores contract-lease TTL", "body", false)
	if refused || code != 0 {
		t.Fatalf("code=%d refused=%v, want 0/false (fail open)", code, refused)
	}
	if !strings.Contains(errb.String(), "failing open") {
		t.Fatalf("stderr missing fail-open notice: %s", errb.String())
	}
}

func TestIssueCreateDedupeGateWarnOnly(t *testing.T) {
	fetch := func(cap int) ([]byte, error) { return []byte(nearDupBacklog), nil }
	result := &issueCreateResult{}
	var out, errb bytes.Buffer
	code, refused := issueCreateRunDedupeGate(&out, &errb, result, fetch, true, 0, 0, true,
		"fix(dispatch): the preflight lane-lock gate ignores contract-lease TTL",
		"the preflight lane-lock gate reads the contract lease but ignores its ttl", false)
	if refused || code != 0 {
		t.Fatalf("code=%d refused=%v, want 0/false (warn only)", code, refused)
	}
	if !strings.Contains(errb.String(), "WARNING") {
		t.Fatalf("stderr missing WARNING: %s", errb.String())
	}
}

func TestIssueCreateDedupeGateDisarmed(t *testing.T) {
	fetch := func(cap int) ([]byte, error) {
		t.Fatalf("fetch must never run while the gate is disarmed")
		return nil, nil
	}
	result := &issueCreateResult{}
	var out, errb bytes.Buffer
	code, refused := issueCreateRunDedupeGate(&out, &errb, result, fetch, false, 0, 0, false, "t", "b", false)
	if refused || code != 0 {
		t.Fatalf("code=%d refused=%v, want 0/false (disarmed)", code, refused)
	}
	if result.Dedupe != nil {
		t.Fatalf("disarmed gate attached evidence: %+v", result.Dedupe)
	}
}

func TestIssueCreateWriteBackFiledMarkerAppends(t *testing.T) {
	path := filepath.Join(t.TempDir(), "draft.md")
	if err := os.WriteFile(path, []byte("## Core through-line\nkeep me\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var errb bytes.Buffer
	issueCreateWriteBackFiledMarker(&errb, path, "", "https://github.com/anthony-chaudhary/fak-private/issues/1633")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "github_issue: anthony-chaudhary/fak-private#1633") {
		t.Fatalf("marker not appended:\n%s", b)
	}
	if !strings.Contains(string(b), "keep me") {
		t.Fatalf("write-back clobbered the draft:\n%s", b)
	}
}

func TestIssueCreateWriteBackFiledMarkerReplacesExisting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "draft.md")
	if err := os.WriteFile(path, []byte("body\ngithub_issue: anthony-chaudhary/fak-private#1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var errb bytes.Buffer
	issueCreateWriteBackFiledMarker(&errb, path, "", "https://github.com/anthony-chaudhary/fak-private/issues/1633")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	if strings.Count(text, "github_issue:") != 1 {
		t.Fatalf("expected exactly one marker line, got %d:\n%s", strings.Count(text, "github_issue:"), text)
	}
	if !strings.Contains(text, "github_issue: anthony-chaudhary/fak-private#1633") {
		t.Fatalf("marker not updated:\n%s", text)
	}
	if strings.Contains(text, "#1\n") {
		t.Fatalf("stale marker survived:\n%s", text)
	}
}

func TestIssueCreateDedupeGateViaRunIssueCreateWith(t *testing.T) {
	withIssueCreateDedupeFetcher(t, func(cap int) ([]byte, error) { return []byte(nearDupBacklog), nil })
	body := "## Parent context\n#1\n\n## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nNo extras." + validIssueCreateProblemFrame("Core")
	var out, errb bytes.Buffer
	code := runIssueCreateWithCleanScrub(&out, &errb, []string{
		"--title", "fix(dispatch): the preflight lane-lock gate ignores contract-lease TTL",
		"--body", body,
		"--estimate-points", "1",
		"--parent-baseline-points", "1",
		"--target-envelope", "- paths: >= 1 command",
		"--witnessed-envelope", "- paths: 1 command",
		"--dedupe-checked",
		"--dry-run",
	}, nil)
	if code != 3 {
		t.Fatalf("code=%d want 3; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), issueCreateDedupeReason) {
		t.Fatalf("stderr missing typed refusal: %s", errb.String())
	}
}
