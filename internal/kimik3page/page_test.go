package kimik3page

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func pageRoot(tb testing.TB) string {
	tb.Helper()
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("resolve test path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(here), "..", "..", "docs", "kimi-k3"))
}

func TestMarketingPageContract(t *testing.T) {
	root := pageRoot(t)
	if err := ValidatePage(root); err != nil {
		t.Fatalf("ValidatePage contract failed: %v", err)
	}

	spec := DefaultSpec()
	if spec.PageTitle == "" || spec.Description == "" {
		t.Errorf("incomplete DefaultSpec: %+v", spec)
	}

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

	verified, err := AuditLinks(repo)
	if err != nil {
		t.Fatalf("AuditLinks failed: %v", err)
	}
	if len(verified) == 0 {
		t.Errorf("expected verified links, got none")
	}

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

func TestValidatePageErrors(t *testing.T) {
	tmp := t.TempDir()
	// Guard: fail-closed on missing index.html
	if err := ValidatePage(tmp); err == nil {
		t.Error("expected error for empty dir without index.html")
	}

	// Missing required substring
	if err := os.WriteFile(filepath.Join(tmp, "index.html"), []byte("<html></html>"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePage(tmp); err == nil {
		t.Error("expected error for missing required substrings")
	}

	// Unwitnessed marketing claim
	badContent := "<!doctype html><meta name=\"viewport\">Kimi K3 for Claude Code, through fak reasoning_effort Native max reasoning claude-kimi-k3.sh claude-kimi-k3.ps1 DOGFOOD-CLAUDE.md#moonshot-kimi-k3 @media (max-width: 580px) prefers-reduced-motion aria-label=\"Request path\" social-card.svg fastest"
	if err := os.WriteFile(filepath.Join(tmp, "index.html"), []byte(badContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePage(tmp); err == nil {
		t.Error("expected error for forbidden claim 'fastest'")
	}
}

func BenchmarkValidatePage(b *testing.B) {
	root := pageRoot(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidatePage(root); err != nil {
			b.Fatal(err)
		}
	}
}
