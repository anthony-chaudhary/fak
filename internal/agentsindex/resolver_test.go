package agentsindex

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func writeEffectiveFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func effectiveFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeEffectiveFile(t, root, "AGENTS.md", "root instructions\n")
	writeEffectiveFile(t, root, "nested/AGENTS.md", "nested canonical\n")
	writeEffectiveFile(t, root, "nested/AGENTS.override.md", "nested override\n")
	writeEffectiveFile(t, root, "nested/FALLBACK.md", "nested fallback\n")
	writeEffectiveFile(t, root, "nested/deeper/FALLBACK.md", "deep fallback\n")
	writeEffectiveFile(t, root, "nested/deeper/work.go", "package work\n")
	return root
}

func TestResolveEffectivePrecedenceProvenanceAndDigest(t *testing.T) {
	root := effectiveFixture(t)
	opts := ResolveOptions{Fallbacks: []string{"FALLBACK.md"}, MaxBytes: 4096, Trusted: true}
	got := ResolveEffective(root, "nested/deeper/work.go", opts)
	if got.Status != StatusComplete {
		t.Fatalf("status=%s diagnostics=%+v", got.Status, got.Diagnostics)
	}
	if got.Target != "nested/deeper/work.go" || got.TargetKind != "file" {
		t.Fatalf("target=%q kind=%q", got.Target, got.TargetKind)
	}
	if got.Instructions != "root instructions\nnested override\ndeep fallback\n" {
		t.Fatalf("instructions=%q", got.Instructions)
	}
	wantPaths := []string{
		"AGENTS.md:selected",
		"nested/AGENTS.override.md:selected",
		"nested/AGENTS.md:higher_precedence_selected",
		"nested/FALLBACK.md:higher_precedence_selected",
		"nested/deeper/FALLBACK.md:selected",
	}
	var paths []string
	for _, source := range got.Sources {
		paths = append(paths, source.Path+":"+source.Reason)
		if strings.Contains(source.Path, "\\") {
			t.Fatalf("source path is not normalized: %q", source.Path)
		}
		if source.Bytes > 0 && (len(source.SHA256) != 64 || source.Content == "") { //boundarylint:ignore CHANGE_DETECTOR_TEST SHA-256 provenance digests are exactly 64 hexadecimal characters
			t.Fatalf("incomplete provenance: %+v", source)
		}
		if source.Included && (source.Span == nil || source.Span.End-source.Span.Start != source.Bytes) {
			t.Fatalf("bad source span: %+v", source)
		}
	}
	if !reflect.DeepEqual(paths, wantPaths) {
		t.Fatalf("sources=%v want=%v", paths, wantPaths)
	}
	if got.EffectiveSHA256 == "" {
		t.Fatal("missing effective digest")
	}
	again := ResolveEffective(root, filepath.FromSlash("nested/deeper"), opts)
	if again.Status != StatusComplete || again.Instructions != got.Instructions || again.EffectiveSHA256 != got.EffectiveSHA256 {
		t.Fatalf("directory/second resolution diverged:\nfirst=%+v\nsecond=%+v", got, again)
	}

	// The digest binds length-delimited relative path/content pairs, not the absolute
	// fixture location or ambiguous concatenation.
	copyRoot := effectiveFixture(t)
	copyResult := ResolveEffective(copyRoot, filepath.Join(copyRoot, "nested", "deeper", "work.go"), opts)
	if copyResult.EffectiveSHA256 != got.EffectiveSHA256 {
		t.Fatalf("digest depends on absolute root: %s != %s", copyResult.EffectiveSHA256, got.EffectiveSHA256)
	}
}

func TestResolveEffectiveBudgetTrustAndUnknownFailClosed(t *testing.T) {
	root := effectiveFixture(t)
	exactBytes := int64(len("root instructions\nnested override\ndeep fallback\n"))
	exact := ResolveEffective(root, "nested/deeper", ResolveOptions{Fallbacks: []string{"FALLBACK.md"}, MaxBytes: exactBytes, Trusted: true})
	if exact.Status != StatusComplete || int64(exact.Bytes) != exactBytes {
		t.Fatalf("exact budget result=%+v", exact)
	}
	over := ResolveEffective(root, "nested/deeper", ResolveOptions{Fallbacks: []string{"FALLBACK.md"}, MaxBytes: exactBytes - 1, Trusted: true})
	if over.Status != StatusTruncated || over.Instructions != "" || over.EffectiveSHA256 != "" {
		t.Fatalf("over-budget result must fail closed: %+v", over)
	}
	for _, source := range over.Sources {
		if source.Reason == "byte_budget_exceeded" && source.Content != "" {
			t.Fatalf("over-budget source leaked payload: %+v", source)
		}
	}
	untrusted := ResolveEffective(root, "nested/deeper", ResolveOptions{Fallbacks: []string{"FALLBACK.md"}, Trusted: false})
	if untrusted.Status != StatusUntrusted || untrusted.Instructions != "" || untrusted.EffectiveSHA256 != "" {
		t.Fatalf("untrusted result must fail closed: %+v", untrusted)
	}
	empty := t.TempDir()
	unknown := ResolveEffective(empty, ".", ResolveOptions{Trusted: true})
	if unknown.Status != StatusUnknown || unknown.Instructions != "" {
		t.Fatalf("unknown result=%+v", unknown)
	}
}

func TestResolveEffectiveMutationDeletionFallbackAndCWDInvariance(t *testing.T) {
	root := effectiveFixture(t)
	opts := ResolveOptions{Fallbacks: []string{"FALLBACK.md"}, Trusted: true}
	before := ResolveEffective(root, "nested/deeper/work.go", opts)
	writeEffectiveFile(t, root, "nested/AGENTS.override.md", "mutated override\n")
	mutated := ResolveEffective(root, "nested/deeper/work.go", opts)
	if mutated.EffectiveSHA256 == before.EffectiveSHA256 || !strings.Contains(mutated.Instructions, "mutated override") {
		t.Fatalf("mutation did not change identity: before=%s after=%s", before.EffectiveSHA256, mutated.EffectiveSHA256)
	}
	if err := os.Remove(filepath.Join(root, "nested", "AGENTS.override.md")); err != nil {
		t.Fatal(err)
	}
	promoted := ResolveEffective(root, "nested/deeper/work.go", opts)
	if !strings.Contains(promoted.Instructions, "nested canonical") || strings.Contains(promoted.Instructions, "mutated override") {
		t.Fatalf("deletion did not promote canonical: %q", promoted.Instructions)
	}
	if err := os.Remove(filepath.Join(root, "nested", "AGENTS.md")); err != nil {
		t.Fatal(err)
	}
	fallbackPromoted := ResolveEffective(root, "nested/deeper/work.go", opts)
	if !strings.Contains(fallbackPromoted.Instructions, "nested fallback") || fallbackPromoted.EffectiveSHA256 == promoted.EffectiveSHA256 {
		t.Fatalf("canonical deletion did not promote fallback: %+v", fallbackPromoted)
	}
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	other := t.TempDir()
	if err := os.Chdir(other); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
	afterCWD := ResolveEffective(root, "nested/deeper/work.go", opts)
	if afterCWD.EffectiveSHA256 != fallbackPromoted.EffectiveSHA256 || afterCWD.Instructions != fallbackPromoted.Instructions {
		t.Fatalf("cwd changed resolution: before=%+v after=%+v", fallbackPromoted, afterCWD)
	}
}

func TestResolveEffectiveBoundaryAndSymlinkEscape(t *testing.T) {
	root := effectiveFixture(t)
	outside := t.TempDir()
	writeEffectiveFile(t, outside, "work.go", "package outside\n")
	got := ResolveEffective(root, filepath.Join(outside, "work.go"), ResolveOptions{Trusted: true})
	if got.Status != StatusOutsideRoot || got.Instructions != "" {
		t.Fatalf("outside target result=%+v", got)
	}

	if runtime.GOOS == "windows" {
		// Developer Mode/admin policy controls symlink creation on Windows. Exercise it
		// when available and leave the boundary test above as the portable witness.
	}
	link := filepath.Join(root, "escape")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	viaLink := ResolveEffective(root, filepath.Join(link, "work.go"), ResolveOptions{Trusted: true})
	if viaLink.Status != StatusOutsideRoot {
		t.Fatalf("symlink escape status=%s", viaLink.Status)
	}

	outsideInstructions := filepath.Join(outside, "outside-agents.md")
	if err := os.WriteFile(outsideInstructions, []byte("outside instructions\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sourceLink := filepath.Join(root, "nested", "deeper", "AGENTS.override.md")
	if err := os.Symlink(outsideInstructions, sourceLink); err != nil {
		t.Skipf("file symlink unavailable: %v", err)
	}
	sourceEscape := ResolveEffective(root, "nested/deeper", ResolveOptions{Fallbacks: []string{"FALLBACK.md"}, Trusted: true})
	if sourceEscape.Status != StatusOutsideRoot || sourceEscape.Instructions != "" {
		t.Fatalf("instruction symlink escape result=%+v", sourceEscape)
	}
}

func TestResolveEffectiveSourceDigestIsExactBytes(t *testing.T) {
	root := t.TempDir()
	body := "# contract\r\nbyte exact\r\n"
	writeEffectiveFile(t, root, "AGENTS.md", body)
	got := ResolveEffective(root, ".", ResolveOptions{Trusted: true})
	if got.Status != StatusComplete || len(got.Sources) != 1 {
		t.Fatalf("result=%+v", got)
	}
	h := sha256.Sum256([]byte(body))
	if got.Sources[0].SHA256 != hex.EncodeToString(h[:]) || got.Sources[0].Content != body {
		t.Fatalf("source identity=%+v", got.Sources[0])
	}
}
