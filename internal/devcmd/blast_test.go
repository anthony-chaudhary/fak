package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastradius"
)

// testBlastGraph is the same b->a, c->b, d->x chain the pure package uses, in the
// repo-relative-dir namespace the go-list fold produces. modPath "" means no import
// prefix to strip.
func testBlastGraph(string) (map[string][]string, string, int, error) {
	return map[string][]string{
		"internal/b": {"internal/a"},
		"internal/c": {"internal/b"},
		"internal/d": {"internal/x"},
	}, "", 4, nil
}

func withBlastSeams(t *testing.T, graph func(string) (map[string][]string, string, int, error), leases func(string, time.Time) ([]blastradius.Lease, error)) {
	t.Helper()
	og, ol := blastDirGraph, blastLeaseSource
	blastDirGraph, blastLeaseSource = graph, leases
	t.Cleanup(func() { blastDirGraph, blastLeaseSource = og, ol })
}

// blastJSON is the envelope the shell emits: the pure AffectedSet plus a schema stamp.
type blastJSON struct {
	Schema string   `json:"schema"`
	Broken string   `json:"broken"`
	Radius []string `json:"radius"`
	Leases []struct {
		Lane    string   `json:"lane"`
		Matched []string `json:"matched"`
	} `json:"leases"`
	Issues []struct {
		ID      string   `json:"id"`
		Matched []string `json:"matched"`
	} `json:"issues"`
	ExcludedLeases []string `json:"excluded_leases"`
	ExcludedIssues []string `json:"excluded_issues"`
}

func TestBlastEstimateJSONHeldAndExcluded(t *testing.T) {
	withBlastSeams(t, testBlastGraph, func(string, time.Time) ([]blastradius.Lease, error) {
		return []blastradius.Lease{
			{Lane: "lane-b", TreeGlobs: []string{"internal/b/**"}}, // dependent -> held
			{Lane: "lane-d", TreeGlobs: []string{"internal/d/**"}}, // disjoint -> excluded
		}, nil
	})

	var stdout, stderr bytes.Buffer
	if rc := RunBlast(&stdout, &stderr, []string{"estimate", "internal/a", "--json"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, stderr.String())
	}

	var got blastJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if got.Schema != blastradius.Schema {
		t.Errorf("schema = %q, want %q", got.Schema, blastradius.Schema)
	}
	if got.Broken != "internal/a" {
		t.Errorf("broken = %q, want internal/a", got.Broken)
	}
	if want := []string{"internal/a", "internal/b", "internal/c"}; !reflect.DeepEqual(got.Radius, want) {
		t.Errorf("radius = %v, want %v", got.Radius, want)
	}
	if len(got.Leases) != 1 || got.Leases[0].Lane != "lane-b" {
		t.Fatalf("held leases = %#v, want only lane-b", got.Leases)
	}
	if want := []string{"internal/b"}; !reflect.DeepEqual(got.Leases[0].Matched, want) {
		t.Errorf("lane-b matched = %v, want %v", got.Leases[0].Matched, want)
	}
	if want := []string{"lane-d"}; !reflect.DeepEqual(got.ExcludedLeases, want) {
		t.Errorf("excluded leases = %v, want %v", got.ExcludedLeases, want)
	}
}

// The --leases and --issues fixtures drive an offline estimate (the witness path):
// live-ledger read is bypassed, and both sides partition the same way.
func TestBlastEstimateWithFixtureFiles(t *testing.T) {
	withBlastSeams(t, testBlastGraph, func(string, time.Time) ([]blastradius.Lease, error) {
		t.Fatal("live lease source must not be read when --leases is given")
		return nil, nil
	})

	dir := t.TempDir()
	leasesPath := filepath.Join(dir, "leases.jsonl")
	if err := os.WriteFile(leasesPath, []byte(
		`{"lane":"held-c","tree_globs":["internal/c/**"]}`+"\n"+
			`{"lane":"free-x","tree_globs":["internal/x/**"]}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	issuesPath := filepath.Join(dir, "issues.jsonl")
	if err := os.WriteFile(issuesPath, []byte(
		`{"id":"7001","paths":["internal/a/fix.go"]}`+"\n"+
			`{"id":"7002","paths":["internal/z/other.go"]}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := RunBlast(&stdout, &stderr, []string{"estimate", "internal/a", "--json", "--leases", leasesPath, "--issues", issuesPath})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, stderr.String())
	}
	var got blastJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(got.Leases) != 1 || got.Leases[0].Lane != "held-c" {
		t.Errorf("held leases = %#v, want only held-c", got.Leases)
	}
	if want := []string{"free-x"}; !reflect.DeepEqual(got.ExcludedLeases, want) {
		t.Errorf("excluded leases = %v, want %v", got.ExcludedLeases, want)
	}
	if len(got.Issues) != 1 || got.Issues[0].ID != "7001" {
		t.Errorf("held issues = %#v, want only 7001", got.Issues)
	}
	if want := []string{"7002"}; !reflect.DeepEqual(got.ExcludedIssues, want) {
		t.Errorf("excluded issues = %v, want %v", got.ExcludedIssues, want)
	}
}

func TestBlastEstimateTextMode(t *testing.T) {
	withBlastSeams(t, testBlastGraph, func(string, time.Time) ([]blastradius.Lease, error) {
		return []blastradius.Lease{
			{Lane: "lane-b", TreeGlobs: []string{"internal/b/**"}},
			{Lane: "lane-d", TreeGlobs: []string{"internal/d/**"}},
		}, nil
	})
	var stdout, stderr bytes.Buffer
	if rc := RunBlast(&stdout, &stderr, []string{"estimate", "internal/a"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"blast radius of internal/a", "held", "lease lane-b", "excluded", "lease lane-d"} {
		if !strings.Contains(out, want) {
			t.Errorf("text output missing %q\n%s", want, out)
		}
	}
}

func TestBlastUsageErrors(t *testing.T) {
	withBlastSeams(t, testBlastGraph, func(string, time.Time) ([]blastradius.Lease, error) { return nil, nil })
	cases := [][]string{
		{},                     // no subcommand
		{"bogus"},              // unknown subcommand
		{"estimate"},           // estimate with no target
		{"estimate", "a", "b"}, // too many targets
	}
	for _, argv := range cases {
		var stdout, stderr bytes.Buffer
		if rc := RunBlast(&stdout, &stderr, argv); rc != 2 {
			t.Errorf("RunBlast(%v) rc = %d, want 2 (stderr=%s)", argv, rc, stderr.String())
		}
	}
}

// parseGoListDirs folds a go-list stream into a repo-relative-dir graph: intra-module
// imports become dir->dir edges, foreign imports (fmt) are dropped, and the module path
// is surfaced for prefix stripping.
func TestParseGoListDirs(t *testing.T) {
	stream := strings.Join([]string{
		`{"ImportPath":"m/internal/a","Dir":"root/internal/a","Module":{"Path":"m","Dir":"root"},"Imports":[]}`,
		`{"ImportPath":"m/internal/b","Dir":"root/internal/b","Imports":["m/internal/a","fmt"]}`,
		`{"ImportPath":"m/internal/c","Dir":"root/internal/c","TestImports":["m/internal/b"]}`,
	}, "\n")

	edges, modPath, total, err := parseGoListDirs(strings.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	if modPath != "m" {
		t.Errorf("modPath = %q, want m", modPath)
	}
	if total != 3 {
		t.Errorf("total = %d, want 3", total)
	}
	if want := []string{"internal/a"}; !reflect.DeepEqual(edges["internal/b"], want) {
		t.Errorf("edges[internal/b] = %v, want %v (fmt dropped)", edges["internal/b"], want)
	}
	if want := []string{"internal/b"}; !reflect.DeepEqual(edges["internal/c"], want) {
		t.Errorf("edges[internal/c] = %v, want %v", edges["internal/c"], want)
	}
}

func TestTrimModulePrefix(t *testing.T) {
	const mod = "github.com/anthony-chaudhary/fak"
	cases := []struct{ tree, want string }{
		{mod + "/internal/foo", "internal/foo"}, // full import path -> repo-relative dir
		{"internal/foo", "internal/foo"},        // already repo-relative
		{mod, "."},                              // bare module path -> root
	}
	for _, tc := range cases {
		if got := trimModulePrefix(tc.tree, mod); got != tc.want {
			t.Errorf("trimModulePrefix(%q) = %q, want %q", tc.tree, got, tc.want)
		}
	}
	if got := trimModulePrefix("internal/foo", ""); got != "internal/foo" {
		t.Errorf("empty modPath should pass through, got %q", got)
	}
}
