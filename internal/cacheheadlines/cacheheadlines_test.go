package cacheheadlines

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func initGitRepo(t *testing.T, dir string) {
	t.Helper()
	runGit(t, dir, "init", "-q")
	runGit(t, dir, "config", "user.name", "cache-headline-test")
	runGit(t, dir, "config", "user.email", "cache-headline-test@example.invalid")
	runGit(t, dir, "config", "core.autocrlf", "false")
}

func commitAll(t *testing.T, dir string, msg string) {
	t.Helper()
	runGit(t, dir, "add", "-A")
	runGit(t, dir, "commit", "-q", "-m", msg)
}

func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v failed in %s: %v\nOutput: %s", args, dir, err, string(out))
	}
	return string(out)
}

func scanTree(t *testing.T, files map[string]string) []Finding {
	t.Helper()
	d := t.TempDir()
	initGitRepo(t, d)
	for name, body := range files {
		p := filepath.Join(d, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatalf("mkdir failed: %v", err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatalf("write file %s: %v", name, err)
		}
	}
	runGit(t, d, "add", "-A")
	findings, err := AuditTree(d)
	if err != nil {
		t.Fatalf("AuditTree failed: %v", err)
	}
	return findings
}

// 1. Bare "99% cache" headline without a plane label must FAIL.
func TestFlagsBare99PctCache(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"bad.md": "We hit a 99% cache — the biggest win of the quarter!\n",
	})
	if len(findings) == 0 {
		t.Fatalf("expected bare '99%% cache' headline to fail, got 0 findings")
	}
	if findings[0].Line != 1 || !strings.Contains(findings[0].Text, "99% cache") {
		t.Errorf("unexpected finding: %+v", findings[0])
	}
}

// 2. Bare "cache win" headline without a plane label must FAIL.
func TestFlagsBareCacheWin(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"b.md": "Ship it: this is a huge cache win.\n",
	})
	if len(findings) == 0 {
		t.Fatalf("expected bare 'cache win' headline to fail, got 0 findings")
	}
	if findings[0].Line != 1 || !strings.Contains(findings[0].Text, "cache win") {
		t.Errorf("unexpected finding: %+v", findings[0])
	}
}

// 3. "cache is 99% of the story" without a plane label must FAIL.
func TestFlagsCacheIs99PctOfTheStory(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"s.md": "Honestly the cache is 99% of the speedup story here.\n",
	})
	if len(findings) == 0 {
		t.Fatalf("expected 'cache is 99%% of the story' to fail, got 0 findings")
	}
	if findings[0].Line != 1 {
		t.Errorf("unexpected line number: %d", findings[0].Line)
	}
}

// 4. Headline naming the provider plane + OBSERVED provenance passes.
func TestPassesProviderLabeled(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"ok.md": "provider prompt-cache rebate: 99% cache read, OBSERVED (cost/latency only, not fak-owned reuse)\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected provider-labeled headline to pass, got %+v", findings)
	}
}

// 5. Headline naming the kernel plane + WITNESSED provenance passes.
func TestPassesKernelLabeled(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"ok.md": "kernel KV reuse was the cache win here — WITNESSED, bit-identical to a full re-prefill\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected kernel-labeled headline to pass, got %+v", findings)
	}
}

// 6. Headline naming the context plane passes.
func TestPassesContextLabeled(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"ok.md": "O(1) context saved 99% cache prefill work (WITNESSED resident-view elision)\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected context-labeled headline to pass, got %+v", findings)
	}
}

// 7. Line quoting the legacy phrasing to REMOVE it passes (carve-out).
func TestPassesLegacyQuoteToRemove(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"legacy.md": "- \"cache win\" as a bare headline is legacy language to remove.\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected legacy quote to remove to pass, got %+v", findings)
	}
}

// 8. Word boundary: "cache window" / "windowed" must not be matched as "cache win".
func TestPassesCacheWindowWordBoundary(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"w.md": "the bounded cache window is argmax-identical to the full-cache windowed decode\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected cache window to pass, got %+v", findings)
	}
}

// 9. Hyphenated hit-rate stat "99%-cache-hit" is out of scope and passes.
func TestPassesHyphenatedHitRateStat(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"h.md": "the corpus had two ~99%-cache-hit sessions in the tail\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected hyphenated hit rate to pass, got %+v", findings)
	}
}

// 10. Clean text without any headline keywords passes with 0 findings.
func TestCleanTreeIsZero(t *testing.T) {
	findings := scanTree(t, map[string]string{
		"plain.md": "fak treats the model like an untrusted program.\n",
	})
	if len(findings) != 0 {
		t.Fatalf("expected plain file to pass, got %+v", findings)
	}
}

// 11. Staged mode: flags newly added bad line.
func TestAuditStagedFlagsAddedBadLine(t *testing.T) {
	d := t.TempDir()
	initGitRepo(t, d)

	badFile := filepath.Join(d, "bad.md")
	if err := os.WriteFile(badFile, []byte("plain line\n"), 0644); err != nil {
		t.Fatalf("write seed: %v", err)
	}
	commitAll(t, d, "seed")

	if err := os.WriteFile(badFile, []byte("plain line\na fresh 99% cache win, the quarter's best\n"), 0644); err != nil {
		t.Fatalf("write modified: %v", err)
	}
	runGit(t, d, "add", "bad.md")

	findings, err := AuditStaged(d)
	if err != nil {
		t.Fatalf("AuditStaged failed: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(findings), findings)
	}
	if findings[0].Line != 2 {
		t.Errorf("expected line 2, got %d", findings[0].Line)
	}
	if !strings.Contains(findings[0].Text, "99% cache win") {
		t.Errorf("unexpected text: %s", findings[0].Text)
	}
}

// 12. Staged mode: ignores pre-existing bad line when only clean lines are staged.
func TestAuditStagedIgnoresPreexistingBadLine(t *testing.T) {
	d := t.TempDir()
	initGitRepo(t, d)

	legacyFile := filepath.Join(d, "legacy.md")
	if err := os.WriteFile(legacyFile, []byte("a bare 99% cache win headline\n"), 0644); err != nil {
		t.Fatalf("write seed: %v", err)
	}
	commitAll(t, d, "seed")

	if err := os.WriteFile(legacyFile, []byte("a bare 99% cache win headline\nnew clean line\n"), 0644); err != nil {
		t.Fatalf("write modified: %v", err)
	}
	runGit(t, d, "add", "legacy.md")

	findings, err := AuditStaged(d)
	if err != nil {
		t.Fatalf("AuditStaged failed: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings for staged addition of clean line, got %+v", findings)
	}
}

// 13. Direct unit tests for CheckLine across all patterns and permutations.
func TestCheckLine(t *testing.T) {
	tests := []struct {
		name          string
		line          string
		wantViolation bool
	}{
		// Violations (legacy shapes, no label)
		{"bare cache win", "This is a big cache win for the quarter", true},
		{"bare cache-win", "Reported a cache-win", true},
		{"bare cache wins", "The system cache wins here", true},
		{"bare 99% cache", "We hit 99% cache in testing", true},
		{"bare 99 % cache", "We hit 99 % cache in testing", true},
		{"bare 50% cache", "Reached 50% cache today", true},
		{"bare cache is 99%", "The cache is 99% of total speedup", true},
		{"bare 99% of the story", "Cache hit is 99% of the story", true},
		{"case-insensitive CACHE WIN", "HUGE CACHE WIN", true},
		{"case-insensitive 99% CACHE", "WE HIT 99% CACHE", true},

		// Honest labeled headlines (must pass)
		{"labeled provider", "provider prompt-cache: 99% cache achieved", false},
		{"labeled kernel", "kernel KV cache win", false},
		{"labeled context", "O(1) context 99% cache prefill", false},
		{"labeled forecast", "forecasted 99% cache win", false},
		{"labeled observed", "observed 99% cache win", false},
		{"labeled witnessed", "witnessed cache win", false},
		{"labeled measured", "measured 99% cache", false},
		{"labeled modeled", "modeled 99% cache win", false},
		{"labeled hypothesis", "hypothesis: 99% cache", false},
		{"labeled decision", "decision: cache win", false},
		{"labeled vcache", "vcache 99% cache", false},
		{"labeled radixkv", "radixkv cache win", false},
		{"labeled pagedattention", "pagedattention 99% cache", false},
		{"labeled sglang", "sglang 99% cache", false},
		{"labeled vllm", "vllm 99% cache", false},
		{"labeled rebate", "rebate 99% cache", false},
		{"labeled reuse", "reuse cache win", false},
		{"labeled kv", "kv cache win", false},

		// Carve-outs and allow patterns (must pass)
		{"allow legacy", "legacy assumption: 99% cache", false},
		{"allow remove", "remove 99% cache claims", false},
		{"allow mislead", "misleading 99% cache headline", false},
		{"allow hide", "hides which mechanism fired: 99% cache", false},
		{"allow blend", "blended 99% cache headline", false},
		{"allow fails review", "99% cache fails review without labels", false},
		{"allow omit", "omits plane label: 99% cache", false},

		// Quoted headline references (must pass)
		{"quoted double quotes", `Reviewing the "cache win" pattern`, false},
		{"quoted single quotes", "Reviewing the '99% cache' pattern", false},
		{"quoted backticks", "Reviewing the `cache is 99% of the story` pattern", false},

		// Word boundaries and non-matches (must pass)
		{"cache window", "the cache window is 4k tokens", false},
		{"hyphenated hit rate", "session had 99%-cache-hit rate", false},
		{"unrelated", "database queries are optimized", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotBad, gotMatched := CheckLine(tc.line)
			if gotBad != tc.wantViolation {
				t.Errorf("CheckLine(%q) violation = %v (matched %q), want %v", tc.line, gotBad, gotMatched, tc.wantViolation)
			}
		})
	}
}

// 14. ScanText tests line numbering and truncated text.
func TestScanText(t *testing.T) {
	content := "Line 1 is clean\nLine 2 is a cache win\nLine 3 is clean\nLine 4 is a 99% cache\n"
	findings := ScanText(content, "test.md")
	if len(findings) != 2 {
		t.Fatalf("expected 2 findings, got %d", len(findings))
	}
	if findings[0].Line != 2 || findings[0].File != "test.md" {
		t.Errorf("unexpected finding[0]: %+v", findings[0])
	}
	if findings[1].Line != 4 || findings[1].File != "test.md" {
		t.Errorf("unexpected finding[1]: %+v", findings[1])
	}
}

// 15. Skip prefixes and basenames.
func TestShouldSkip(t *testing.T) {
	if !ShouldSkip("docs/releases/v0.10.0.md") {
		t.Errorf("expected docs/releases/ to be skipped")
	}
	if !ShouldSkip("vendor/github.com/pkg/file.md") {
		t.Errorf("expected vendor/ to be skipped")
	}
	if !ShouldSkip("node_modules/pkg/README.md") {
		t.Errorf("expected node_modules/ to be skipped")
	}
	if !ShouldSkip("llms-full.txt") {
		t.Errorf("expected llms-full.txt to be skipped")
	}
	if !ShouldSkip("check_cache_headlines.py") {
		t.Errorf("expected check_cache_headlines.py to be skipped")
	}
	if !ShouldSkip("cacheheadlines.go") {
		t.Errorf("expected cacheheadlines.go to be skipped")
	}
	if ShouldSkip("docs/overview.md") {
		t.Errorf("expected docs/overview.md not to be skipped")
	}
}
