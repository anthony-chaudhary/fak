package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func trunkRedClaimRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q", "-b", "main"},
		{"config", "user.email", "t@example.invalid"},
		{"config", "user.name", "t"},
		{"config", "commit.gpgsign", "false"},
	} {
		if out, err := exec.Command("git", append([]string{"-C", dir}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"add", "a.txt"}, {"commit", "-q", "-m", "base"}} {
		if out, err := exec.Command("git", append([]string{"-C", dir}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

// TestTrunkRedClaimVerbAndGateAdvisory: session A claims the red package; the commit gate's
// advisory for the next session names A's claim; B's own claim is refused; after A releases,
// the advisory goes back to suggesting a claim.
func TestTrunkRedClaimVerbAndGateAdvisory(t *testing.T) {
	t.Setenv("FAK_SESSION_ID", "")
	t.Setenv("CLAUDE_SESSION_ID", "")
	dir := trunkRedClaimRepo(t)
	head, err := gitRevParse(dir, "HEAD")
	if err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := runTrunkRed(&out, &errb, []string{"claim", "--repo", dir, "--holder", "A", "internal/ctxmmu"}); code != 0 {
		t.Fatalf("A claim exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "claimed repair: internal/ctxmmu") {
		t.Fatalf("claim output: %q", out.String())
	}

	adv := formatRedRepairAdvisory(dir, head, []string{"internal/ctxmmu"})
	if !strings.Contains(adv, "repair already CLAIMED: internal/ctxmmu by A") {
		t.Fatalf("gate advisory must surface A's claim:\n%s", adv)
	}

	out.Reset()
	errb.Reset()
	if code := runTrunkRed(&out, &errb, []string{"claim", "--repo", dir, "--holder", "B", "internal/ctxmmu"}); code != 3 {
		t.Fatalf("B duplicate claim exit %d, want 3: %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "INTENT_COLLISION") || !strings.Contains(errb.String(), "A") {
		t.Fatalf("B refusal must name the incumbent: %q", errb.String())
	}

	out.Reset()
	if code := runTrunkRed(&out, &errb, []string{"release", "--repo", dir, "internal/ctxmmu"}); code != 0 {
		t.Fatalf("release exit %d: %s", code, errb.String())
	}
	adv = formatRedRepairAdvisory(dir, head, []string{"internal/ctxmmu"})
	if strings.Contains(adv, "CLAIMED") || !strings.Contains(adv, "fak trunk-red claim internal/ctxmmu") {
		t.Fatalf("after release the advisory must suggest a claim:\n%s", adv)
	}
}
