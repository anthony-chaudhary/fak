package memoryread

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fact(name, desc, body string) string {
	return "---\nname: " + name + "\ndescription: " + desc + "\nmetadata:\n  type: project\n---\n\n" + body + "\n"
}

func buildStore(t *testing.T, dir string) {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "MEMORY.md"),
		"- [First fact](first-fact.md) - the hook one\n"+
			"- [Second fact](second-fact.md) - the hook two\n"+
			"- [Archive index](MEMORY_archive.md) - cold tier\n")
	mustWrite(t, filepath.Join(dir, "first-fact.md"), fact("first-fact", "desc one", "BODY-ONE is the durable fact."))
	mustWrite(t, filepath.Join(dir, "second-fact.md"), fact("second-fact", "desc two", "BODY-TWO is the other fact."))
	mustWrite(t, filepath.Join(dir, "MEMORY_archive.md"), "- [old](old.md) - should not expand\n")
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestParseIndex(t *testing.T) {
	got := ParseIndex("- [A](a.md) - h\n- [B](b.md) - h\n- [A again](a.md) - dup\n- [Idx](MEMORY.md) - self\n- [Sub](sub/c.md) - path\n")
	want := [][2]string{{"A", "a.md"}, {"B", "b.md"}}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("ParseIndex=%v, want %v", got, want)
	}
}

func TestStripFrontmatter(t *testing.T) {
	if got := StripFrontmatter("---\nname: x\n---\n\nhello\n"); got != "hello\n" {
		t.Fatalf("StripFrontmatter=%q", got)
	}
	if got := StripFrontmatter("just text\n"); got != "just text\n" {
		t.Fatalf("passthrough=%q", got)
	}
}

func TestRenderDigestAbsentStore(t *testing.T) {
	out := RenderDigest(filepath.Join(t.TempDir(), "nope"), false, 60000)
	if !strings.Contains(out, "no committed memory mirror") || strings.Contains(out, "BODY-ONE") {
		t.Fatalf("unexpected absent digest:\n%s", out)
	}
}

func TestRenderDigestFull(t *testing.T) {
	dir := t.TempDir()
	buildStore(t, dir)
	out := RenderDigest(dir, false, 60000)
	for _, want := range []string{"committed mirror", "First fact", "BODY-ONE is the durable fact.", "BODY-TWO is the other fact."} {
		if !strings.Contains(out, want) {
			t.Fatalf("digest missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "name: first-fact") || strings.Contains(out, "should not expand") {
		t.Fatalf("digest leaked frontmatter or expanded non-fact:\n%s", out)
	}
}

func TestRenderDigestIndexOnly(t *testing.T) {
	dir := t.TempDir()
	buildStore(t, dir)
	out := RenderDigest(dir, true, 60000)
	if !strings.Contains(out, "First fact") || strings.Contains(out, "BODY-ONE") {
		t.Fatalf("index-only digest mismatch:\n%s", out)
	}
}

func TestRenderDigestMaxBytesOmission(t *testing.T) {
	dir := t.TempDir()
	buildStore(t, dir)
	out := RenderDigest(dir, false, 1)
	// The first fact is always emitted (emitted>0 guard); the second overflows the
	// 1-byte budget. The over-budget fact is NAMED under a typed MEMORY_INDEX_OVERFLOW
	// advisory (#2430) — never an anonymous "N omitted" count.
	if !strings.Contains(out, "BODY-ONE") || strings.Contains(out, "BODY-TWO") {
		t.Fatalf("bounded digest mismatch:\n%s", out)
	}
	if !strings.Contains(out, OverflowReason) || !strings.Contains(out, "Second fact (second-fact.md)") {
		t.Fatalf("over-budget fact not named under %s:\n%s", OverflowReason, out)
	}
}

func TestRenderDigestZeroOverflowNoAdvisory(t *testing.T) {
	dir := t.TempDir()
	buildStore(t, dir)
	out := RenderDigest(dir, false, 60000)
	if strings.Contains(out, OverflowReason) {
		t.Fatalf("in-budget digest emitted an overflow advisory:\n%s", out)
	}
}

func TestDiscoverStore(t *testing.T) {
	// 1. Empty root has no store
	emptyRoot := t.TempDir()
	if got := DiscoverStore(emptyRoot); got != "" {
		t.Fatalf("expected empty store for empty root, got %q", got)
	}

	// 2. Discover .fak/memory
	fakRoot := t.TempDir()
	fakDir := filepath.Join(fakRoot, ".fak", "memory")
	if err := os.MkdirAll(fakDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakDir, "MEMORY.md"), []byte("# Fak Memory\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverStore(fakRoot); got != fakDir {
		t.Fatalf("DiscoverStore(.fak/memory) = %q, want %q", got, fakDir)
	}

	// 3. Discover .claude/memory
	claudeRoot := t.TempDir()
	claudeDir := filepath.Join(claudeRoot, ".claude", "memory")
	if err := os.MkdirAll(claudeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(claudeDir, "MEMORY.md"), []byte("# Claude Memory\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverStore(claudeRoot); got != claudeDir {
		t.Fatalf("DiscoverStore(.claude/memory) = %q, want %q", got, claudeDir)
	}

	// 4. Discover root MEMORY.md
	memRoot := t.TempDir()
	memFile := filepath.Join(memRoot, "MEMORY.md")
	if err := os.WriteFile(memFile, []byte("# Root Memory\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverStore(memRoot); got != memFile {
		t.Fatalf("DiscoverStore(root MEMORY.md) = %q, want %q", got, memFile)
	}

	// 5. Precedence: .fak/memory wins over .claude/memory and root MEMORY.md
	multiRoot := t.TempDir()
	multiFakDir := filepath.Join(multiRoot, ".fak", "memory")
	multiClaudeDir := filepath.Join(multiRoot, ".claude", "memory")
	_ = os.MkdirAll(multiFakDir, 0o755)
	_ = os.WriteFile(filepath.Join(multiFakDir, "MEMORY.md"), []byte("# Fak\n"), 0o644)
	_ = os.MkdirAll(multiClaudeDir, 0o755)
	_ = os.WriteFile(filepath.Join(multiClaudeDir, "MEMORY.md"), []byte("# Claude\n"), 0o644)
	_ = os.WriteFile(filepath.Join(multiRoot, "MEMORY.md"), []byte("# Root\n"), 0o644)
	if got := DiscoverStore(multiRoot); got != multiFakDir {
		t.Fatalf("precedence (.fak over all) = %q, want %q", got, multiFakDir)
	}

	// 6. Precedence: .claude/memory wins over root MEMORY.md
	claudeAndRoot := t.TempDir()
	crClaudeDir := filepath.Join(claudeAndRoot, ".claude", "memory")
	_ = os.MkdirAll(crClaudeDir, 0o755)
	_ = os.WriteFile(filepath.Join(crClaudeDir, "MEMORY.md"), []byte("# Claude\n"), 0o644)
	_ = os.WriteFile(filepath.Join(claudeAndRoot, "MEMORY.md"), []byte("# Root\n"), 0o644)
	if got := DiscoverStore(claudeAndRoot); got != crClaudeDir {
		t.Fatalf("precedence (.claude over root) = %q, want %q", got, crClaudeDir)
	}
}

func TestResolveStore(t *testing.T) {
	root := t.TempDir()

	// Empty explicit falls back to discovery
	if got := ResolveStore(root, ""); got != "" {
		t.Fatalf("ResolveStore(empty) = %q, want empty", got)
	}

	// Explicit absolute path
	customAbs := filepath.Join(t.TempDir(), "custom-store")
	if got := ResolveStore(root, customAbs); got != customAbs {
		t.Fatalf("ResolveStore(customAbs) = %q, want %q", got, customAbs)
	}

	// Explicit relative path
	if got := ResolveStore(root, "relative/store"); got != filepath.Join(root, "relative", "store") {
		t.Fatalf("ResolveStore(relative) = %q, want %q", got, filepath.Join(root, "relative", "store"))
	}
}

func TestRenderDigestDirectFile(t *testing.T) {
	dir := t.TempDir()
	buildStore(t, dir)
	memFile := filepath.Join(dir, "MEMORY.md")
	out := RenderDigest(memFile, false, 60000)
	if !strings.Contains(out, "First fact") || !strings.Contains(out, "BODY-ONE is the durable fact.") {
		t.Fatalf("RenderDigest(file) failed:\n%s", out)
	}
}
