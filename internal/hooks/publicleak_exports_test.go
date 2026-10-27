package hooks

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestScanRangePublicLeakResolvesSidecarFromSelectedRevision(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init", "-q", "-b", "main")
	gitFixture(t, root, "config", "user.email", "fixture@example.com")
	gitFixture(t, root, "config", "user.name", "Fixture")

	needle := "10.11.12.13"
	sidecar := filepath.Join(root, filepath.FromSlash(privateNeedlesRel))
	if err := os.MkdirAll(filepath.Dir(sidecar), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sidecar, []byte(`{"export_audit_needles":["`+needle+`"]}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("tools/_registry/\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", ".gitignore")
	gitFixture(t, root, "add", "-f", privateNeedlesRel)
	gitFixture(t, root, "commit", "-qm", "seed")
	base := gitFixture(t, root, "rev-parse", "HEAD")

	if err := os.WriteFile(filepath.Join(root, "node.json"), []byte(`{"ip":"`+needle+`"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", "node.json")
	gitFixture(t, root, "commit", "-qm", "leak")

	findings, err := ScanRangePublicLeak(root, base+"..HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 || findings[0].File != "node.json" || !strings.Contains(findings[0].Detail, needle) {
		t.Fatalf("range findings = %+v, want selected-revision sidecar needle in node.json", findings)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestAuditPublicLeakTreeMatchesCachedScan(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitFixture(t, root, "init", "-q")
	alias := privateGPUAliasFixture()
	files := map[string]string{
		"text.txt": "\nNEEDLE\r\n\n" + strings.Repeat("x", 70<<10) + "needle\n" +
			gcpSAFixture() + "\n" + userPathFixture(`\work`) + "\n" +
			"xoxb-" + strings.Repeat("1", 8) + "-" + strings.Repeat("2", 8) + "-" + strings.Repeat("a", 16),
		"unicode.txt":                          "\xfféNEEDLE\r\n\n",
		"empty.txt":                            "",
		"docs/" + alias + ".bin":               alias + "\nneedle\n\x00",
		"docs/space in name.txt":               alias + "\n",
		"PUBLIC-SCRUB-POLICY.md":               alias + "\nneedle\n",
		"docs/research/inventory/sample.jsonl": "/home/" + "upstream-contributor/work\n" + userPathFixture(`\work`),
		"deleted.txt":                          alias,
		privateNeedlesRel:                      "{}",
	}
	for path, body := range files {
		writeFile(t, root, path, body)
	}
	gitFixture(t, root, "add", ".")
	if err := os.Remove(filepath.Join(root, "deleted.txt")); err != nil {
		t.Fatal(err)
	}
	for _, sidecar := range []struct {
		name, body, mode string
	}{
		{"full", `{"audit_needles":["ignored"],"export_audit_needles":["needle","é","","needle"]}`, "full"},
		{"malformed", "{", "shape-only"},
		{"absent", "", "shape-only"},
	} {
		t.Run(sidecar.name, func(t *testing.T) {
			if sidecar.name == "absent" {
				if err := os.Remove(filepath.Join(root, filepath.FromSlash(privateNeedlesRel))); err != nil {
					t.Fatal(err)
				}
			} else {
				writeFile(t, root, privateNeedlesRel, sidecar.body)
			}
			tree, err := ReadTrackedTree(root)
			if err != nil {
				t.Fatal(err)
			}
			want, mode := legacyPublicLeakTreeScan(tree)
			got, err := AuditPublicLeakTree(root)
			if err != nil {
				t.Fatal(err)
			}
			if len(want) == 0 || mode != sidecar.mode || got.Mode != mode ||
				!reflect.DeepEqual(got.Findings, want) || !reflect.DeepEqual(got.Misses, want) ||
				got.Count != len(want) || got.OK != (len(want) == 0) {
				t.Fatalf("audit = %+v; want mode %q, exact findings %+v", got, mode, want)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPublicLeakTreeReaderReleasesOnlyAfterLastUse(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeFile(t, root, privateNeedlesRel, "first sidecar")
	writeFile(t, root, "repeat.txt", "first body")
	writeFile(t, root, "single.txt", "one body")
	tree := &TrackedTree{Root: root, Paths: []string{
		"repeat.txt", "single.txt", "repeat.txt", "missing.txt", "missing.txt", privateNeedlesRel,
	}, fileCache: map[string]fileEntry{}}
	read := oneShotPublicLeakTreeReader(tree)
	check := func(path, want string, exists, retained bool) {
		t.Helper()
		body, ok := read(path)
		if string(body) != want || ok != exists {
			t.Fatalf("read %q = (%q, %v), want (%q, %v)", path, body, ok, want, exists)
		}
		if _, cached := tree.fileCache[path]; cached != retained {
			t.Fatalf("read %q retained = %v, want %v", path, cached, retained)
		}
	}
	check(privateNeedlesRel, "first sidecar", true, true)
	writeFile(t, root, privateNeedlesRel, "changed sidecar")
	check("repeat.txt", "first body", true, true)
	writeFile(t, root, "repeat.txt", "changed body")
	check("single.txt", "one body", true, false)
	check("repeat.txt", "first body", true, false)
	check("missing.txt", "", false, true)
	writeFile(t, root, "missing.txt", "created after first observation")
	check("missing.txt", "", false, false)
	check(privateNeedlesRel, "first sidecar", true, false)
	if len(tree.fileCache) != 0 {
		t.Fatalf("one-shot scan retained %d entries", len(tree.fileCache))
	}

	// The shared-reader contract remains memoized across complete gate calls.
	shared := &TrackedTree{Root: root, Paths: []string{"repeat.txt"}, fileCache: map[string]fileEntry{}}
	scanPublicLeakTree(shared)
	writeFile(t, root, "repeat.txt", "changed again")
	if body, ok := shared.FileBytes("repeat.txt"); !ok || string(body) != "changed body" {
		t.Fatalf("shared reader lost its first observation: (%q, %v)", body, ok)
	}
}

// legacyPublicLeakTreeScan is the pre-optimization scanner, kept as the exact
// decision/order oracle and matched benchmark baseline, not a production path.
func legacyPublicLeakTreeScan(t *TrackedTree) ([]Finding, string) {
	mode := "shape-only"
	var needles []string
	if b, ok := t.FileBytes(privateNeedlesRel); ok {
		needles = treeExportAuditNeedles(b)
		if len(needles) > 0 {
			mode = "full"
		}
	}
	var findings []Finding
	for _, f := range t.Paths {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfReferentialLeak[norm] {
			continue
		}
		findings = append(findings, publicLeakLineFindings(norm, 0, norm, needles)...)
		body, ok := t.FileBytes(f)
		if !ok {
			continue
		}
		if strings.ContainsRune(string(body), '\x00') {
			continue
		}
		for i, line := range strings.Split(string(body), "\n") {
			findings = append(findings, publicLeakLineFindings(line, i+1, f, needles)...)
		}
	}
	return findings, mode
}

func BenchmarkPublicLeakTreeRetention(b *testing.B) {
	root := b.TempDir()
	const fileCount = 16
	body := []byte(strings.Repeat(strings.Repeat("x", 127)+"\n", 2048))
	paths := make([]string, fileCount)
	for i := range paths {
		paths[i] = fmt.Sprintf("file-%02d.txt", i)
		if err := os.WriteFile(filepath.Join(root, paths[i]), body, 0o600); err != nil {
			b.Fatal(err)
		}
	}
	for _, candidate := range []bool{false, true} {
		name := "cached-split"
		if candidate {
			name = "last-use-splitseq"
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(fileCount * len(body)))
			for b.Loop() {
				tree := &TrackedTree{Root: root, Paths: paths, fileCache: map[string]fileEntry{}}
				var findings []Finding
				if candidate {
					findings, _ = scanPublicLeakTreeFiles(paths, oneShotPublicLeakTreeReader(tree))
				} else {
					findings, _ = legacyPublicLeakTreeScan(tree)
				}
				if len(findings) != 0 {
					b.Fatalf("unexpected findings: %+v", findings)
				}
				retained := 0
				for _, entry := range tree.fileCache {
					retained += len(entry.data)
				}
				b.ReportMetric(float64(retained), "retained-B")
			}
		})
	}
}
