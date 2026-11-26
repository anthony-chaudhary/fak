package devcmd

import (
	"bytes"
	"go/build"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastradius"
)

// The fixture's files get explicit modification times so a fingerprint change is
// something the test states, never something it waits for.
var (
	blastFixtureCommitA = time.Unix(1_700_000_000, 0)
	blastFixtureCommitB = blastFixtureCommitA.Add(time.Hour)
)

func writeBlastFixtureFile(t *testing.T, root, rel, content string, modTime time.Time) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, modTime, modTime); err != nil {
		t.Fatal(err)
	}
}

// newBlastFixtureModule lays out a module that exercises every rule the scan shares
// with `go list ./...`: test and external-test imports, a constraint-excluded file,
// and the pruned trees (testdata, vendor, dot and underscore directories, a nested
// module).
func newBlastFixtureModule(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":                 "module example.com/m\n\ngo 1.22\n",
		"root.go":                "package m\n\nimport _ \"example.com/m/a\"\n",
		"a/a.go":                 "package a\n",
		"b/b.go":                 "package b\n\nimport (\n\t_ \"example.com/m/a\"\n\t_ \"fmt\"\n)\n",
		"b/b_test.go":            "package b\n\nimport _ \"example.com/m/c\"\n",
		"c/c.go":                 "package c\n",
		"c/c_x_test.go":          "package c_test\n\nimport _ \"example.com/m/d\"\n",
		"d/d.go":                 "package d\n",
		"d/d_never.go":           "//go:build blast_fixture_never\n\npackage d\n\nimport _ \"example.com/m/a\"\n",
		"e/testdata/x/x.go":      "package x\n\nimport _ \"example.com/m/a\"\n",
		"_skip/s.go":             "package s\n\nimport _ \"example.com/m/a\"\n",
		".hidden/h.go":           "package h\n\nimport _ \"example.com/m/a\"\n",
		"vendor/v/v.go":          "package v\n\nimport _ \"example.com/m/a\"\n",
		"nested/go.mod":          "module example.com/nested\n\ngo 1.22\n",
		"nested/n.go":            "package nested\n\nimport _ \"example.com/m/a\"\n",
		"cmd/tool/main.go":       "package main\n\nimport _ \"example.com/m/b\"\n\nfunc main() {}\n",
		"docs/readme.md":         "not go\n",
		"cmd/tool/_ignored.go":   "package main\n\nimport _ \"example.com/m/d\"\n",
		"cmd/tool/notes/note.md": "not go\n",
	}
	for rel, content := range files {
		writeBlastFixtureFile(t, root, rel, content, blastFixtureCommitA)
	}
	return root
}

// blastFixturePackageDirs is every package directory newBlastFixtureModule
// produces, in the sorted order countBlastScans reports.
var blastFixturePackageDirs = []string{".", "a", "b", "c", "cmd/tool", "d"}

// requireBlastFullScan fails unless a rescan visited exactly the fixture's package
// directories: no package skipped, and no non-package directory such as tools/ or
// the cache itself scanned.
func requireBlastFullScan(t *testing.T, scanned []string, what string) {
	t.Helper()
	if !reflect.DeepEqual(scanned, blastFixturePackageDirs) {
		t.Fatalf("%s scanned %v, want exactly the package dirs %v", what, scanned, blastFixturePackageDirs)
	}
}

// countBlastScans swaps the package-scan seam for one that records which
// directories were rescanned, relative to root.
func countBlastScans(t *testing.T, root string) func() []string {
	t.Helper()
	var (
		mu      sync.Mutex
		scanned []string
	)
	original := blastScanPackage
	blastScanPackage = func(ctxt *build.Context, absDir, modPath string) blastPackageScan {
		rel, err := filepath.Rel(root, absDir)
		if err != nil {
			rel = absDir
		}
		mu.Lock()
		scanned = append(scanned, filepath.ToSlash(rel))
		mu.Unlock()
		return original(ctxt, absDir, modPath)
	}
	t.Cleanup(func() { blastScanPackage = original })
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		out := append([]string(nil), scanned...)
		scanned = nil
		sort.Strings(out)
		return out
	}
}

func TestScanDirGraphFollowsGoListPackageRules(t *testing.T) {
	root := newBlastFixtureModule(t)
	edges, modPath, total, _, err := scanDirGraph(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if modPath != "example.com/m" {
		t.Errorf("modPath = %q, want example.com/m", modPath)
	}
	want := map[string][]string{
		".":        {"a"},
		"b":        {"a", "c"},
		"c":        {"d"},
		"cmd/tool": {"b"},
	}
	if !reflect.DeepEqual(edges, want) {
		t.Errorf("edges = %v, want %v", edges, want)
	}
	if total != 6 {
		t.Errorf("total = %d, want 6 package dirs (., a, b, c, cmd/tool, d)", total)
	}
}

// A directory whose Go files are all excluded by build constraints (a generator
// script tagged ignore, a package for another platform) is left out of ./... by the
// go command: it is no graph node, it is not counted, and an import of it is
// dropped. One buildable file makes it a package again.
func TestScanDirGraphLeavesOutIgnoreOnlyDirectories(t *testing.T) {
	root := newBlastFixtureModule(t)
	cachePath := filepath.Join(t.TempDir(), "graph.json")
	writeBlastFixtureFile(t, root, "tools/gen.go", "//go:build ignore\n\npackage main\n\nimport _ \"example.com/m/a\"\n\nfunc main() {}\n", blastFixtureCommitA)
	writeBlastFixtureFile(t, root, "elsewhere/only.go", "//go:build blast_fixture_never\n\npackage elsewhere\n\nimport _ \"example.com/m/a\"\n", blastFixtureCommitA)
	writeBlastFixtureFile(t, root, "a/uses_tools.go", "package a\n\nimport _ \"example.com/m/tools\"\n", blastFixtureCommitA)

	for _, label := range []string{"cold", "warm"} {
		edges, _, total, _, err := scanDirGraph(root, cachePath)
		if err != nil {
			t.Fatal(err)
		}
		if total != 6 {
			t.Errorf("%s: total = %d, want 6 (tools is not a package)", label, total)
		}
		if deps, ok := edges["tools"]; ok {
			t.Errorf("%s: ignore-only directory has edges %v", label, deps)
		}
		if deps, ok := edges["a"]; ok {
			t.Errorf("%s: an import of the ignore-only directory should be dropped, got edges[a] = %v", label, deps)
		}
	}

	writeBlastFixtureFile(t, root, "tools/doc.go", "package tools\n", blastFixtureCommitB)
	edges, _, total, _, err := scanDirGraph(root, cachePath)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"tools"}; total != 7 || !reflect.DeepEqual(edges["a"], want) {
		t.Errorf("after adding a real file: total = %d, edges[a] = %v; want 7 and %v", total, edges["a"], want)
	}
}

func runBlastEstimateJSON(t *testing.T, target string) string {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if rc := RunBlast(&stdout, &stderr, []string{"estimate", target, "--json"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, stderr.String())
	}
	return stdout.String()
}

func useBlastFixtureLeases(t *testing.T) {
	t.Helper()
	original := blastLeaseSource
	blastLeaseSource = func(string, time.Time) ([]blastradius.Lease, error) {
		return []blastradius.Lease{
			{Lane: "lane-b", TreeGlobs: []string{"b/**"}},
			{Lane: "lane-d", TreeGlobs: []string{"d/**"}},
		}, nil
	}
	t.Cleanup(func() { blastLeaseSource = original })
}

// The estimate is the same bytes whether the graph came from an uncached scan, a
// scan that filled the cache, or the cache alone, and the warm run rescans nothing.
func TestBlastEstimateCachedMatchesUncached(t *testing.T) {
	root := newBlastFixtureModule(t)
	t.Chdir(root)
	t.Setenv(blastGraphEnvMode, "")
	useBlastFixtureLeases(t)
	scans := countBlastScans(t, root)

	t.Setenv(blastGraphEnvCache, "off")
	uncached := runBlastEstimateJSON(t, "a")
	requireBlastFullScan(t, scans(), "uncached run")
	if !strings.Contains(uncached, `"cmd/tool"`) || !strings.Contains(uncached, `"lane-b"`) {
		t.Fatalf("uncached estimate is missing the expected radius or held lease:\n%s", uncached)
	}

	t.Setenv(blastGraphEnvCache, t.TempDir())
	cold := runBlastEstimateJSON(t, "a")
	requireBlastFullScan(t, scans(), "cold cached run")
	warm := runBlastEstimateJSON(t, "a")
	if got := scans(); len(got) != 0 {
		t.Fatalf("warm cached run rescanned %v, want none", got)
	}
	if cold != uncached {
		t.Errorf("cold cached estimate differs from uncached:\n%s\nvs\n%s", cold, uncached)
	}
	if warm != uncached {
		t.Errorf("warm cached estimate differs from uncached:\n%s\nvs\n%s", warm, uncached)
	}
}

// A checkout that moves to another commit rewrites, adds, and removes files. The
// next build rescans exactly the directories whose Go files changed and agrees with
// an uncached scan of the new tree.
func TestBlastGraphCacheInvalidatesWhenCheckoutMoves(t *testing.T) {
	root := newBlastFixtureModule(t)
	cachePath := filepath.Join(t.TempDir(), "graph.json")
	scans := countBlastScans(t, root)

	before, _, _, _, err := scanDirGraph(root, cachePath)
	if err != nil {
		t.Fatal(err)
	}
	scans()
	if _, ok := before["cmd/tool"]; !ok {
		t.Fatalf("fixture graph should hold cmd/tool before the move: %v", before)
	}

	writeBlastFixtureFile(t, root, "c/c.go", "package c\n\nimport _ \"example.com/m/a\"\n", blastFixtureCommitB)
	writeBlastFixtureFile(t, root, "f/f.go", "package f\n\nimport _ \"example.com/m/c\"\n", blastFixtureCommitB)
	if err := os.RemoveAll(filepath.Join(root, "cmd")); err != nil {
		t.Fatal(err)
	}

	after, _, total, stats, err := scanDirGraph(root, cachePath)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := scans(), []string{"c", "f"}; !reflect.DeepEqual(got, want) {
		t.Errorf("rescanned %v after the move, want exactly %v", got, want)
	}
	if !stats.Wrote {
		t.Error("a changed tree should rewrite the cache")
	}
	fresh, _, freshTotal, _, err := scanDirGraph(root, "")
	if err != nil {
		t.Fatal(err)
	}
	scans()
	if !reflect.DeepEqual(after, fresh) || total != freshTotal {
		t.Errorf("cached graph after the move = %v (total %d), uncached = %v (total %d)", after, total, fresh, freshTotal)
	}
	want := map[string][]string{
		".": {"a"},
		"b": {"a", "c"},
		"c": {"a", "d"},
		"f": {"c"},
	}
	if !reflect.DeepEqual(after, want) {
		t.Errorf("graph after the move = %v, want %v", after, want)
	}

	if _, _, _, stats, err = scanDirGraph(root, cachePath); err != nil {
		t.Fatal(err)
	}
	if got := scans(); len(got) != 0 || stats.Wrote {
		t.Errorf("an unchanged tree rescanned %v (wrote=%t), want a pure lookup", got, stats.Wrote)
	}
}

// go.mod names the module every import is matched against, so a new go.mod drops
// the whole cache rather than trusting entries resolved under the old one.
func TestBlastGraphCacheDroppedWhenGoModChanges(t *testing.T) {
	root := newBlastFixtureModule(t)
	cachePath := filepath.Join(t.TempDir(), "graph.json")
	scans := countBlastScans(t, root)
	if _, _, _, _, err := scanDirGraph(root, cachePath); err != nil {
		t.Fatal(err)
	}
	scans()

	writeBlastFixtureFile(t, root, "go.mod", "module example.com/m\n\ngo 1.23\n", blastFixtureCommitB)
	if _, _, _, _, err := scanDirGraph(root, cachePath); err != nil {
		t.Fatal(err)
	}
	requireBlastFullScan(t, scans(), "rescan after a go.mod change")
}

func TestBlastGraphCacheIgnoresCorruptAndOversizedFiles(t *testing.T) {
	root := newBlastFixtureModule(t)
	want, _, _, _, err := scanDirGraph(root, "")
	if err != nil {
		t.Fatal(err)
	}

	corrupt := filepath.Join(t.TempDir(), "graph.json")
	if err := os.WriteFile(corrupt, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, _, _, stats, err := scanDirGraph(root, corrupt)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) || stats.Scanned != 6 {
		t.Errorf("corrupt cache: graph = %v scanned = %d, want %v with a full rescan", got, stats.Scanned, want)
	}

	oversized := filepath.Join(t.TempDir(), "graph.json")
	if _, _, _, _, err := scanDirGraph(root, oversized); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(oversized, blastGraphCacheMaxBytes+1); err != nil {
		t.Fatal(err)
	}
	if entries := readBlastGraphCache(oversized, "any"); entries != nil {
		t.Errorf("an oversized cache file was read: %d entries", len(entries))
	}
}

// A directory that could not be read completely is answered from what was read but
// never cached, so the next run reads it again instead of trusting a partial result.
func TestBlastGraphCacheSkipsIncompleteScans(t *testing.T) {
	root := newBlastFixtureModule(t)
	cachePath := filepath.Join(t.TempDir(), "graph.json")
	original := blastScanPackage
	t.Cleanup(func() { blastScanPackage = original })
	var mu sync.Mutex
	reads := map[string]int{}
	blastScanPackage = func(ctxt *build.Context, absDir, modPath string) blastPackageScan {
		scan := original(ctxt, absDir, modPath)
		if filepath.Base(absDir) == "b" {
			scan.complete = false
		}
		mu.Lock()
		reads[filepath.Base(absDir)]++
		mu.Unlock()
		return scan
	}
	for run := 0; run < 2; run++ {
		edges, _, _, _, err := scanDirGraph(root, cachePath)
		if err != nil {
			t.Fatal(err)
		}
		if want := []string{"a", "c"}; !reflect.DeepEqual(edges["b"], want) {
			t.Errorf("run %d: edges[b] = %v, want %v", run, edges["b"], want)
		}
	}
	if reads["b"] != 2 || reads["a"] != 1 {
		t.Errorf("reads = %v, want the incomplete dir b read on both runs and a read once", reads)
	}
}

func TestScanPackageImportsMarksUnparsedSourceIncomplete(t *testing.T) {
	root := newBlastFixtureModule(t)
	ctxt, _ := blastBuildContext("example.com/m", nil)
	if scan := scanPackageImports(&ctxt, filepath.Join(root, "b"), "example.com/m"); !scan.complete {
		t.Error("a readable package should scan complete")
	}
	if scan := scanPackageImports(&ctxt, filepath.Join(root, "docs"), "example.com/m"); !scan.complete {
		t.Error("a directory with no Go files is a stable result and should scan complete")
	}
	writeBlastFixtureFile(t, root, "b/broken.go", "package b\n\nimport \"unterminated\n", blastFixtureCommitB)
	if scan := scanPackageImports(&ctxt, filepath.Join(root, "b"), "example.com/m"); scan.complete {
		t.Error("a package with a file go/build could not parse should not scan complete")
	}
}

func TestPruneBlastGraphCacheKeepsNewestFiles(t *testing.T) {
	dir := t.TempDir()
	cacheFiles := []string{"00000000000000aa.json", "00000000000000bb.json", "00000000000000cc.json.tmp-123", "00000000000000dd.json", "00000000000000ee.json"}
	for i, name := range cacheFiles {
		writeBlastFixtureFile(t, dir, name, "{}", blastFixtureCommitA.Add(time.Duration(i)*time.Minute))
	}
	writeBlastFixtureFile(t, dir, "keep.txt", "not a cache file", blastFixtureCommitA.Add(-time.Hour))
	writeBlastFixtureFile(t, dir, "receipt.json", "someone else's file", blastFixtureCommitA.Add(-time.Hour))

	pruneBlastGraphCache(dir, 3)

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	if want := []string{"00000000000000cc.json.tmp-123", "00000000000000dd.json", "00000000000000ee.json", "keep.txt", "receipt.json"}; !reflect.DeepEqual(names, want) {
		t.Errorf("after pruning to 3: %v, want %v", names, want)
	}
}

func TestGoflagsBuildTags(t *testing.T) {
	cases := []struct {
		goflags string
		want    []string
	}{
		{"", nil},
		{"-mod=mod", nil},
		{"-tags=a,b", []string{"a", "b"}},
		{"-mod=mod --tags c", []string{"c"}},
		{"-tags=a -tags=b", []string{"b"}},
	}
	for _, tc := range cases {
		if got := goflagsBuildTags(tc.goflags); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("goflagsBuildTags(%q) = %v, want %v", tc.goflags, got, tc.want)
		}
	}
}

func TestBlastGraphCachePathHonorsOverrideAndOff(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(blastGraphEnvCache, dir)
	one, two := blastGraphCachePath(filepath.Join(dir, "repo-one")), blastGraphCachePath(filepath.Join(dir, "repo-two"))
	if filepath.Dir(one) != dir || one == two {
		t.Errorf("cache paths %q and %q should be distinct files under %q", one, two, dir)
	}
	t.Setenv(blastGraphEnvCache, "off")
	if got := blastGraphCachePath(dir); got != "" {
		t.Errorf("cache path with the cache off = %q, want empty", got)
	}
}

// Parity with the reference implementation on a real checkout. It spawns `go list`
// over the whole module, which takes minutes on a busy host, so it runs only when
// FAK_BLAST_GOLIST_PARITY names the module root to compare.
func TestScanDirGraphMatchesGoListOnRealModule(t *testing.T) {
	root := strings.TrimSpace(os.Getenv("FAK_BLAST_GOLIST_PARITY"))
	if root == "" {
		t.Skip("set FAK_BLAST_GOLIST_PARITY=<module root> to compare the scan against go list")
	}
	wantEdges, wantMod, wantTotal, err := goListDirGraph(root)
	if err != nil {
		t.Fatal(err)
	}
	gotEdges, gotMod, gotTotal, _, err := scanDirGraph(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if gotMod != wantMod || gotTotal != wantTotal {
		t.Errorf("scan: module %q, %d package dirs; go list: module %q, %d", gotMod, gotTotal, wantMod, wantTotal)
	}
	for dir, want := range wantEdges {
		if got := gotEdges[dir]; !reflect.DeepEqual(got, want) {
			t.Errorf("edges[%s]\n scan:    %v\n go list: %v", dir, got, want)
		}
	}
	for dir, got := range gotEdges {
		if _, ok := wantEdges[dir]; !ok {
			t.Errorf("scan has edges for %s that go list does not: %v", dir, got)
		}
	}
}
