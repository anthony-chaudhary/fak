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
	if !strings.Contains(out, "BODY-ONE") || strings.Contains(out, "BODY-TWO") || !strings.Contains(out, "omitted") {
		t.Fatalf("bounded digest mismatch:\n%s", out)
	}
}
