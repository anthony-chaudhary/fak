package kimik3page

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func pageRoot(t *testing.T) string {
	t.Helper()
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(here), "..", "..", "docs", "kimi-k3"))
}

func TestMarketingPageContract(t *testing.T) {
	root := pageRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "index.html"))
	if err != nil {
		t.Fatal(err)
	}
	page := string(raw)

	required := []string{
		`<!doctype html>`,
		`<meta name="viewport"`,
		`Kimi K3 for Claude Code, through fak`,
		`reasoning_effort`,
		`Native max reasoning`,
		`claude-kimi-k3.sh`,
		`claude-kimi-k3.ps1`,
		`DOGFOOD-CLAUDE.md#moonshot-kimi-k3`,
		`@media (max-width: 580px)`,
		`prefers-reduced-motion`,
		`aria-label="Request path"`,
		`social-card.svg`,
	}
	for _, want := range required {
		if !strings.Contains(page, want) {
			t.Errorf("page missing %q", want)
		}
	}
	forbidden := []string{"fastest", "cheapest", "best model", "x faster", "× faster"}
	lower := strings.ToLower(page)
	for _, phrase := range forbidden {
		if strings.Contains(lower, phrase) {
			t.Errorf("unwitnessed marketing claim %q", phrase)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "social-card.svg")); err != nil {
		t.Errorf("social card: %v", err)
	}
}

func TestMarketingPageRepositoryLinksResolve(t *testing.T) {
	_, here, _, _ := runtime.Caller(0)
	repo := filepath.Clean(filepath.Join(filepath.Dir(here), "..", ".."))
	for _, rel := range []string{
		"DOGFOOD-CLAUDE.md",
		filepath.Join("scripts", "claude-kimi-k3.sh"),
		filepath.Join("scripts", "claude-kimi-k3.ps1"),
		filepath.Join("internal", "agent", "kimi_k3.go"),
		filepath.Join("visuals", "brand", "fak-mark.svg"),
		filepath.Join("visuals", "brand", "fak-favicon.svg"),
	} {
		if _, err := os.Stat(filepath.Join(repo, rel)); err != nil {
			t.Errorf("linked artifact %s: %v", rel, err)
		}
	}
}
