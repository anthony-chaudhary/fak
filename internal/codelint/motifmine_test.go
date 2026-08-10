package codelint

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- fixtures ---------------------------------------------------------------
//
// Positive fixtures carry the STRUCTURE and deliberately avoid the motif name;
// the negative fixture carries the NAME and the comment and nothing else. That
// pairing is the whole claim of the miner, so it is pinned by tests, not prose.

const motifSrcInspectEditVerify = `package p

import "os"

func rotate(target string) error {
	before, err := os.ReadFile(target)
	if err != nil {
		return err
	}
	if err := os.WriteFile(target, before, 0o600); err != nil {
		return err
	}
	after, err := os.ReadFile(target)
	if err != nil {
		return err
	}
	_ = after
	return nil
}
`

const motifSrcLeaseIsolateLand = `package p

import (
	"os"
	"sync"
)

func apply(mu *sync.Mutex, target string, body []byte) error {
	mu.Lock()
	defer mu.Unlock()
	return os.WriteFile(target, body, 0o600)
}
`

const motifSrcReproduceFixWitness = `package p

import (
	"path/filepath"
	"testing"
)

func roundTrip(t *testing.T) {
	dir := t.TempDir()
	seed := filepath.Join(dir, "seed.txt")
	got := load(seed)
	if got != "default" {
		t.Fatalf("got %q", got)
	}
}

func load(string) string { return "default" }
`

const motifSrcPlanFanoutReconcile = `package p

import "sync"

func spread(paths []string) []string {
	work := append([]string(nil), paths...)
	out := make([]string, len(work))
	var wg sync.WaitGroup
	for i, p := range work {
		wg.Add(1)
		go func(i int, p string) {
			defer wg.Done()
			out[i] = p
		}(i, p)
	}
	wg.Wait()
	return out
}
`

// motifSrcNamesOnly spells every motif out in identifiers and comments and does
// none of them. It must produce zero findings.
const motifSrcNamesOnly = `package p

import "testing"

// inspectEditVerify inspects the file, edits it, then verifies the result.
func inspectEditVerify(target string) string {
	return "inspect -> edit -> verify: " + target
}

// leaseIsolateLand takes the lane lease, isolates the work, and lands the diff.
func leaseIsolateLand() string {
	const doc = "lease -> isolate -> land"
	return doc
}

// TestReproduceFixWitness reproduces the bug, fixes it, and witnesses the fix.
func TestReproduceFixWitness(t *testing.T) {
	t.Log("reproduce -> fix -> witness")
}

// planFanoutReconcile plans the work, fans it out, and reconciles the results.
func planFanoutReconcile(items []string) int {
	return len(items)
}
`

func motifMine(t *testing.T, path, src string) []MotifFinding {
	t.Helper()
	got, err := MineMotifsSource(path, []byte(src))
	if err != nil {
		t.Fatalf("MineMotifsSource(%s): %v", path, err)
	}
	return got
}

func motifOnly(t *testing.T, path, src, wantCatalog string) MotifFinding {
	t.Helper()
	got := motifMine(t, path, src)
	if len(got) != 1 {
		t.Fatalf("want exactly 1 finding, got %d: %+v", len(got), got)
	}
	if got[0].CatalogID != wantCatalog {
		t.Fatalf("catalog=%q want %q", got[0].CatalogID, wantCatalog)
	}
	return got[0]
}

// --- true positives ---------------------------------------------------------

func TestMineMotifsDetectsInspectEditVerify(t *testing.T) {
	f := motifOnly(t, "rotate.go", motifSrcInspectEditVerify, MotifCatalogInspectEditVerify)
	if f.Symbol != "rotate" {
		t.Fatalf("symbol=%q", f.Symbol)
	}
	wantRoles := []string{"inspect", "edit", "verify"}
	for i, r := range wantRoles {
		if f.Evidence[i].Role != r {
			t.Fatalf("evidence[%d].role=%q want %q", i, f.Evidence[i].Role, r)
		}
	}
	if !strings.Contains(f.Evidence[1].Source, "os.WriteFile") {
		t.Fatalf("edit evidence=%q", f.Evidence[1].Source)
	}
	if f.Evidence[0].Line >= f.Evidence[1].Line || f.Evidence[1].Line >= f.Evidence[2].Line {
		t.Fatalf("evidence lines out of order: %+v", f.Evidence)
	}
}

func TestMineMotifsDetectsLeaseIsolateLand(t *testing.T) {
	f := motifOnly(t, "apply.go", motifSrcLeaseIsolateLand, MotifCatalogLeaseIsolateLand)
	if f.Symbol != "apply" {
		t.Fatalf("symbol=%q", f.Symbol)
	}
	if !strings.HasPrefix(f.Evidence[1].Source, "defer ") {
		t.Fatalf("isolate evidence=%q", f.Evidence[1].Source)
	}
	if !strings.Contains(f.Reason, "mu") {
		t.Fatalf("reason should name the shared receiver: %q", f.Reason)
	}
}

func TestMineMotifsDetectsReproduceFixWitness(t *testing.T) {
	// The subject function is called roundTrip, not TestX: the *testing.T
	// parameter type, not the name, is what admits it.
	f := motifOnly(t, "roundtrip_test.go", motifSrcReproduceFixWitness, MotifCatalogReproduceFixWitness)
	if f.Symbol != "roundTrip" {
		t.Fatalf("symbol=%q", f.Symbol)
	}
	if !strings.Contains(f.Evidence[2].Source, "t.Fatalf") {
		t.Fatalf("witness evidence=%q", f.Evidence[2].Source)
	}
	if !strings.Contains(f.Reason, "seed") || !strings.Contains(f.Reason, "got") {
		t.Fatalf("reason should name the data-flow chain: %q", f.Reason)
	}
}

func TestMineMotifsDetectsPlanFanoutReconcile(t *testing.T) {
	f := motifOnly(t, "spread.go", motifSrcPlanFanoutReconcile, MotifCatalogPlanFanoutReconcile)
	if f.Symbol != "spread" {
		t.Fatalf("symbol=%q", f.Symbol)
	}
	if !strings.Contains(f.Evidence[1].Source, "go statement") {
		t.Fatalf("fanout evidence=%q", f.Evidence[1].Source)
	}
	if !strings.Contains(f.Evidence[2].Source, "Wait") {
		t.Fatalf("reconcile evidence=%q", f.Evidence[2].Source)
	}
}

func TestMotifDetectorsCoverFourMotifsWithCatalogIDs(t *testing.T) {
	dets := MotifDetectors()
	if len(dets) < 4 {
		t.Fatalf("want >=4 detectors, got %d", len(dets))
	}
	seen := map[string]bool{}
	for _, d := range dets {
		if d.CatalogID == "" || d.Motif == "" || d.Rule == "" || len(d.Roles) < 3 {
			t.Fatalf("detector %+v is under-specified", d)
		}
		if d.Version != MotifMinerVersion {
			t.Fatalf("detector %s version=%q want %q", d.ID, d.Version, MotifMinerVersion)
		}
		if d.Confidence <= 0 || d.Confidence > 1 {
			t.Fatalf("detector %s confidence=%v", d.ID, d.Confidence)
		}
		if seen[d.CatalogID] {
			t.Fatalf("duplicate catalog id %q", d.CatalogID)
		}
		seen[d.CatalogID] = true
	}
}

// TestMineMotifsFindingsCarryProvenance pins the reporting contract: a finding
// copied out of the report must still be checkable on its own.
func TestMineMotifsFindingsCarryProvenance(t *testing.T) {
	srcs := map[string]string{
		"rotate.go":         motifSrcInspectEditVerify,
		"apply.go":          motifSrcLeaseIsolateLand,
		"roundtrip_test.go": motifSrcReproduceFixWitness,
		"spread.go":         motifSrcPlanFanoutReconcile,
	}
	for path, src := range srcs {
		for _, f := range motifMine(t, path, src) {
			switch {
			case f.Path != path:
				t.Fatalf("%s: path=%q", path, f.Path)
			case f.Detector == "" || f.Version == "":
				t.Fatalf("%s: detector provenance missing: %+v", path, f)
			case f.StartLine <= 0 || f.EndLine < f.StartLine:
				t.Fatalf("%s: bad range %d-%d", path, f.StartLine, f.EndLine)
			case f.Confidence <= 0 || f.Reason == "":
				t.Fatalf("%s: no confidence/reason: %+v", path, f)
			case len(f.Evidence) < 3:
				t.Fatalf("%s: want positive evidence per role, got %d", path, len(f.Evidence))
			}
			for _, e := range f.Evidence {
				if e.Role == "" || e.Source == "" || e.Line < f.StartLine || e.Line > f.EndLine {
					t.Fatalf("%s: evidence %+v outside %d-%d", path, e, f.StartLine, f.EndLine)
				}
			}
		}
	}
}

// --- false positives --------------------------------------------------------

func TestMineMotifsAbstainsOnNamesAndCommentsAlone(t *testing.T) {
	if got := motifMine(t, "namesonly.go", motifSrcNamesOnly); len(got) != 0 {
		t.Fatalf("names/comments alone produced %d findings: %+v", len(got), got)
	}
}

// TestMineMotifsAbstainsWithoutTheLinkage removes only the linkage from each
// positive fixture — the calls stay, the shared target/receiver/dependency goes.
func TestMineMotifsAbstainsWithoutTheLinkage(t *testing.T) {
	cases := []struct{ name, src string }{
		{"read and write different targets", `package p

import "os"

func rotate(a, b string) error {
	before, err := os.ReadFile(a)
	if err != nil {
		return err
	}
	if err := os.WriteFile(b, before, 0o600); err != nil {
		return err
	}
	_, err = os.ReadFile(a)
	return err
}
`},
		{"no re-read after the edit", `package p

import "os"

func rotate(target string) error {
	before, err := os.ReadFile(target)
	if err != nil {
		return err
	}
	return os.WriteFile(target, before, 0o600)
}
`},
		{"release deferred on a different receiver", `package p

import (
	"os"
	"sync"
)

func apply(mu, other *sync.Mutex, target string, body []byte) error {
	mu.Lock()
	defer other.Unlock()
	return os.WriteFile(target, body, 0o600)
}
`},
		{"unmatched acquire and release", `package p

import (
	"os"
	"sync"
)

func apply(mu *sync.Mutex, target string, body []byte) error {
	mu.Lock()
	defer mu.Close()
	return os.WriteFile(target, body, 0o600)
}
`},
		{"assertion on an unrelated value", `package p

import (
	"path/filepath"
	"testing"
)

func roundTrip(t *testing.T) {
	dir := t.TempDir()
	seed := filepath.Join(dir, "seed.txt")
	got := load(seed)
	_ = got
	if other() != "default" {
		t.Fatalf("bad")
	}
}

func load(string) string  { return "default" }
func other() string       { return "default" }
`},
		{"fanout without a join", `package p

func spread(paths []string) {
	work := append([]string(nil), paths...)
	for _, p := range work {
		go func(string) {}(p)
	}
}
`},
		{"loop over an unplanned identifier", `package p

import "sync"

func spread(paths []string) {
	var wg sync.WaitGroup
	for _, p := range paths {
		wg.Add(1)
		go func(string) { defer wg.Done() }(p)
	}
	wg.Wait()
}
`},
	}
	for _, tc := range cases {
		if got := motifMine(t, "case.go", tc.src); len(got) != 0 {
			t.Fatalf("%s: want abstain, got %+v", tc.name, got)
		}
	}
}

// --- malformed sources ------------------------------------------------------

func TestMineMotifsSourceRejectsMalformedGo(t *testing.T) {
	for _, src := range []string{"package p\nfunc Bad( {\n", "", "not go at all"} {
		if _, err := MineMotifsSource("bad.go", []byte(src)); err == nil {
			t.Fatalf("want parse error for %q", src)
		}
	}
}

// --- exclusion policy -------------------------------------------------------

func TestMotifExcludedPathPolicy(t *testing.T) {
	excluded := map[string]string{
		"vendor/x/dep.go":                 "vendor",
		"node_modules/a/b.go":             "vendor",
		"third_party/lib/x.go":            "vendor",
		".git/hooks/x.go":                 "hidden-or-private",
		".scratch/session/tmp.go":         "hidden-or-private",
		"private/notes.go":                "private",
		"internal/secrets/load.go":        "private",
		"tmp/probe.go":                    "scratch",
		"scratchpad/x.go":                 "scratch",
		"internal/api/api.pb.go":          "generated",
		"internal/api/zz_generated.go":    "generated",
		"internal/api/model_generated.go": "generated",
		"internal/api/types.gen.go":       "generated",
	}
	for path, want := range excluded {
		got, ok := MotifExcludedPath(path)
		if !ok || got != want {
			t.Fatalf("MotifExcludedPath(%q)=(%q,%v) want (%q,true)", path, got, ok, want)
		}
	}
	for _, path := range []string{
		"internal/codelint/motifmine.go",
		"cmd/fak/main.go",
		"internal/vendorless/x.go",
		"internal/gen/api.go",
		"internal/privatepath/privatepath.go",
	} {
		if got, ok := MotifExcludedPath(path); ok {
			t.Fatalf("MotifExcludedPath(%q) excluded as %q, want kept", path, got)
		}
	}
}

func motifWrite(t *testing.T, root, rel, src string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(src), 0o600); err != nil {
		t.Fatal(err)
	}
}

// motifFixtureTree plants one findable file plus one of every excluded shape,
// each carrying a real motif, so an exclusion miss shows up as an extra finding.
func motifFixtureTree(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	motifWrite(t, root, "good/rotate.go", motifSrcInspectEditVerify)
	motifWrite(t, root, "good/apply.go", motifSrcLeaseIsolateLand)
	motifWrite(t, root, "vendor/dep/rotate.go", motifSrcInspectEditVerify)
	motifWrite(t, root, ".scratch/rotate.go", motifSrcInspectEditVerify)
	motifWrite(t, root, "private/rotate.go", motifSrcInspectEditVerify)
	motifWrite(t, root, "tmp/rotate.go", motifSrcInspectEditVerify)
	motifWrite(t, root, "good/zz_generated.go", motifSrcInspectEditVerify)
	motifWrite(t, root, "good/emitted.go", "// Code generated by fak. DO NOT EDIT.\n\n"+motifSrcInspectEditVerify)
	motifWrite(t, root, "good/broken.go", "package p\nfunc Bad( {\n")
	motifWrite(t, root, "good/notes.md", "inspect edit verify lease isolate land")
	return root
}

func TestMineMotifsExcludesGeneratedVendorScratchAndPrivate(t *testing.T) {
	rep, err := MineMotifs(motifFixtureTree(t))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Scanned != 2 {
		t.Fatalf("scanned=%d want 2 (good/apply.go, good/rotate.go)", rep.Scanned)
	}
	for _, f := range rep.Findings {
		if !strings.HasPrefix(f.Path, "good/") || strings.Contains(f.Path, "generated") {
			t.Fatalf("finding from an excluded path: %+v", f)
		}
	}
	want := map[string]string{
		"vendor/":              "vendor",
		".scratch/":            "hidden-or-private",
		"private/":             "private",
		"tmp/":                 "scratch",
		"good/zz_generated.go": "generated",
		"good/emitted.go":      "generated",
		"good/broken.go":       "unparsable",
	}
	got := map[string]string{}
	for _, s := range rep.Skipped {
		got[s.Path] = s.Reason
	}
	for path, reason := range want {
		if got[path] != reason {
			t.Fatalf("skip[%q]=%q want %q (all: %+v)", path, got[path], reason, rep.Skipped)
		}
	}
	if len(rep.Findings) != 2 {
		t.Fatalf("findings=%d want 2: %+v", len(rep.Findings), rep.Findings)
	}
}

// --- determinism and ordering ----------------------------------------------

func motifSorted(t *testing.T, f []MotifFinding) {
	t.Helper()
	for i := 1; i < len(f); i++ {
		a, b := f[i-1], f[i]
		if a.Path > b.Path || (a.Path == b.Path && a.StartLine > b.StartLine) {
			t.Fatalf("unsorted at %d: %s:%d then %s:%d", i, a.Path, a.StartLine, b.Path, b.StartLine)
		}
	}
}

func TestMineMotifsIsByteIdenticalAcrossRuns(t *testing.T) {
	root := motifFixtureTree(t)
	first, err := MineMotifs(root)
	if err != nil {
		t.Fatal(err)
	}
	second, err := MineMotifs(root)
	if err != nil {
		t.Fatal(err)
	}
	a, err := MotifReportJSON(first)
	if err != nil {
		t.Fatal(err)
	}
	b, err := MotifReportJSON(second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("JSON differs across runs:\n%s\n---\n%s", a, b)
	}
	if !bytes.Contains(a, []byte(MotifCatalogInspectEditVerify)) {
		t.Fatalf("report lost its catalog id:\n%s", a)
	}
	if !bytes.Contains(a, []byte(`"detector_version"`)) {
		t.Fatalf("report lost detector provenance:\n%s", a)
	}
	motifSorted(t, first.Findings)
	motifSorted(t, second.Findings)
}

func TestMineMotifsOrdersFindingsByPathThenLine(t *testing.T) {
	root := t.TempDir()
	// Written out of order on purpose; the report must not depend on that.
	motifWrite(t, root, "z/spread.go", motifSrcPlanFanoutReconcile)
	motifWrite(t, root, "a/two.go", motifSrcInspectEditVerify+"\n"+motifSrcLeaseIsolateLand[len("package p\n"):])
	motifWrite(t, root, "m/apply.go", motifSrcLeaseIsolateLand)
	rep, err := MineMotifs(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) < 2 {
		t.Fatalf("findings=%d want at least 2 high-confidence motifs: %+v", len(rep.Findings), rep.Findings)
	}
	motifSorted(t, rep.Findings)
	if rep.Findings[0].Path != "m/apply.go" || rep.Findings[len(rep.Findings)-1].Path != "z/spread.go" {
		t.Fatalf("path order wrong: %+v", rep.Findings)
	}
}

// TestMineMotifsOverLiveTree is the live-tree half of the issue's witness: the
// miner must run over this repository's own committed source without erroring,
// find something, and produce byte-identical JSON twice.
func TestMineMotifsOverLiveTree(t *testing.T) {
	if testing.Short() {
		t.Skip("walks the whole repository")
	}
	root := filepath.Join("..", "..")
	first, err := MineMotifs(root)
	if err != nil {
		t.Fatalf("MineMotifs(live tree): %v", err)
	}
	if first.Scanned < 100 {
		t.Fatalf("scanned=%d — the live tree walk looks wrong", first.Scanned)
	}
	if len(first.Findings) == 0 {
		t.Fatal("no motifs mined from the live tree")
	}
	motifSorted(t, first.Findings)
	for _, f := range first.Findings {
		if reason, ex := MotifExcludedPath(f.Path); ex {
			t.Fatalf("finding from excluded path %q (%s)", f.Path, reason)
		}
	}
	second, err := MineMotifs(root)
	if err != nil {
		t.Fatal(err)
	}
	a, err := MotifReportJSON(first)
	if err != nil {
		t.Fatal(err)
	}
	b, err := MotifReportJSON(second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatal("live-tree JSON differs across runs")
	}
	t.Logf("live tree: scanned=%d findings=%d skipped=%d", first.Scanned, len(first.Findings), len(first.Skipped))
}
