package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// issueDraftTestRE is the frozen marker-key grammar the draft must satisfy.
var issueDraftTestRE = regexp.MustCompile(`<!--\s*fak-[A-Za-z0-9_-]+-key:\s*([^>\s]+)\s*-->`)

// withIssueDraftRunner swaps the gh runner seam for the duration of a test.
func withIssueDraftRunner(t *testing.T, run issueCreateRunner) {
	t.Helper()
	prev := issueDraftGHRunner
	issueDraftGHRunner = run
	t.Cleanup(func() { issueDraftGHRunner = prev })
}

func TestIssueDraftMintsWellFormedMarker(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dispatch", "--slug", "my-key", "--out", dir,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	body, err := os.ReadFile(filepath.Join(dir, "my-key.md"))
	if err != nil {
		t.Fatalf("read draft: %v", err)
	}
	m := issueDraftTestRE.FindStringSubmatch(string(body))
	if m == nil {
		t.Fatalf("no marker matching grammar in body:\n%s", body)
	}
	if got := m[0]; got != "<!-- fak-dispatch-key: my-key -->" {
		t.Fatalf("marker=%q", got)
	}
	if !strings.HasPrefix(string(body), "<!-- fak-dispatch-key: my-key -->") {
		t.Fatalf("marker is not the first line:\n%s", body)
	}
}

func TestIssueDraftRequiredContractHeadings(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dispatch", "--slug", "headings", "--out", dir,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	body, err := os.ReadFile(filepath.Join(dir, "headings.md"))
	if err != nil {
		t.Fatalf("read draft: %v", err)
	}
	text := string(body)
	for _, heading := range []string{"## Core through-line", "## Gold-plating boundary", "## Definition of done", "## Witness", "## Acceptance gate", "## Lane"} {
		if !strings.Contains(text, heading) {
			t.Fatalf("missing %q in body:\n%s", heading, text)
		}
	}
}

func TestIssueDraftDedupeRefusesDuplicateKey(t *testing.T) {
	dir := t.TempDir()
	backlog, _ := json.Marshal([]map[string]any{
		{"number": 42, "body": "<!-- fak-dispatch-key: dup-key -->\n\n## Core through-line\n"},
	})
	withIssueDraftRunner(t, func(args []string) (string, string, bool) {
		return string(backlog), "", true
	})
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dispatch", "--slug", "dup-key", "--dedupe-checked", "--out", dir,
	})
	if code != 3 {
		t.Fatalf("code=%d want 3; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "ISSUE_DUPLICATE_KEY") {
		t.Fatalf("stderr missing typed reason: %s", errb.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "dup-key.md")); !os.IsNotExist(err) {
		t.Fatalf("draft was written despite refusal")
	}
}

func TestIssueDraftDedupeFailsOpen(t *testing.T) {
	dir := t.TempDir()
	withIssueDraftRunner(t, func(args []string) (string, string, bool) {
		return "", "gh: not found", false
	})
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dispatch", "--slug", "offline-key", "--dedupe-checked", "--out", dir,
	})
	if code != 0 {
		t.Fatalf("code=%d want 0 (fail open); stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "warning") {
		t.Fatalf("expected a fail-open warning, got: %s", errb.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "offline-key.md")); err != nil {
		t.Fatalf("draft not written on fail-open: %v", err)
	}
}

func TestIssueDraftDryRunWritesNothing(t *testing.T) {
	dir := t.TempDir()
	withIssueDraftRunner(t, func(args []string) (string, string, bool) {
		t.Fatalf("dry-run must not invoke gh; args=%v", args)
		return "", "", false
	})
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dispatch", "--slug", "dry", "--dry-run", "--dedupe-checked", "--out", dir,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "<!-- fak-dispatch-key: dry -->") {
		t.Fatalf("dry-run stdout missing marker: %s", out.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "dry.md")); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote a file")
	}
}

func TestIssueDraftLaneSanitized(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dis patch/../x", "--slug", "s", "--out", dir,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	body, err := os.ReadFile(filepath.Join(dir, "s.md"))
	if err != nil {
		t.Fatalf("read draft: %v", err)
	}
	if !strings.Contains(string(body), "<!-- fak-dis-patch-x-key: s -->") {
		t.Fatalf("lane not sanitized as expected:\n%s", body)
	}
}

func TestIssueDraftJSONResult(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runIssueDraft(&out, &errb, []string{
		"--lane", "dispatch", "--slug", "j", "--out", dir, "--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var res issueDraftResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out.String())
	}
	if res.Schema != "fak.issue-draft.v1" || res.Lane != "dispatch" || res.Key != "fak-dispatch-key" {
		t.Fatalf("unexpected result: %+v", res)
	}
	if !res.Wrote || res.Path == "" {
		t.Fatalf("expected wrote path: %+v", res)
	}
}
