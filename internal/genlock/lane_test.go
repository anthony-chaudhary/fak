package genlock_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/genlock"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// repoRoot walks up from the test's working directory to the checkout root — the
// directory holding both go.mod and dos.toml. The lane assertion below is only worth
// anything against the REAL dos.toml, so a fixture would defeat its purpose.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("no checkout root with dos.toml above the test's working directory")
		}
		dir = parent
	}
}

func realTaxonomy(t *testing.T) laneadmit.Taxonomy {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repoRoot(t), "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v — it IS the path-lane map this test checks against", err)
	}
	tax := laneadmit.ParseTaxonomy(b)
	if !tax.Loaded || len(tax.Trees) == 0 {
		t.Fatal("parsed dos.toml but got no lane trees; the reader is broken, not the config")
	}
	return tax
}

// TestLockPathTakesANamedLaneNotTheGlobalCatchAll is the placement half of the ticket,
// and the reason the lock is not at the repo root.
//
// dos.toml's [lanes.trees] decides which path lease a file falls under. Every glob there
// roots at a named directory except one: `global = ["**/*"]`, which is declared
// EXCLUSIVE — it runs alone. A file no named tree covers therefore does not merely lack
// a label, it lands in the catch-all, and a worker touching it serializes the whole
// fleet behind itself.
//
// The trap is easy to walk into here because fak's docs lane enumerates its root files
// BY NAME (README.md, INDEX.md, llms.txt, llms-full.txt, llms-updates.txt) rather than
// by glob. A new loose file at the repo root — the obvious place to put a lock that
// describes root-level artifacts — matches none of them.
func TestLockPathTakesANamedLaneNotTheGlobalCatchAll(t *testing.T) {
	tax := realTaxonomy(t)

	// The catch-all is real and it is exclusive; the negative cases below cost what
	// this says they cost.
	if !tax.IsExclusive("global") {
		t.Fatal("dos.toml no longer declares `global` exclusive; re-derive this test's premise " +
			"before trusting its verdict")
	}
	if len(tax.Trees["global"]) == 0 {
		t.Fatal("dos.toml declares no tree for `global`; the catch-all this test reasons about is gone")
	}

	lock := genlock.PathFor("marketing-aeo")
	lane := laneadmit.LaneForPath(lock, tax, laneadmit.GranLeaf)
	if lane == "" {
		t.Fatalf("%s is covered by NO named lane in dos.toml, so it falls through to the "+
			"exclusive `global` catch-all and every write to it serializes the fleet. "+
			"Move it under a declared tree.", lock)
	}
	if lane == "global" {
		t.Fatalf("%s resolves to the `global` catch-all lane itself", lock)
	}
	if want := "tools"; lane != want {
		t.Fatalf("%s resolves to lane %q, want %q (dos.toml: tools = [\"tools/**\", \"scripts/**\"]). "+
			"If the lock moved on purpose, re-justify the new lane in the package doc.", lock, lane, want)
	}
	if tax.IsExclusive(lane) {
		t.Errorf("lane %q is EXCLUSIVE, so a lock write would still run alone. The point of "+
			"placing the lock deliberately is to land in a CONCURRENT lane.", lane)
	}
	if len(tax.TreeFor(lane)) == 0 {
		t.Errorf("lane %q has no [lanes.trees] entry; laneadmit.Decide falls back to an empty "+
			"tree, and an empty tree conservatively overlaps everything", lane)
	}

	// The negative controls. Each is a placement someone would plausibly reach for; the
	// first two are why the lock is where it is.
	for _, tc := range []struct {
		name, path, wantLane, why string
	}{
		{
			name:     "repo root",
			path:     "genlock.lock.json",
			wantLane: "",
			why: "the docs lane names its root files individually, so a NEW loose root file " +
				"matches no named tree and falls to the exclusive `global` catch-all",
		},
		{
			name:     "beside the artifacts it describes, at the root",
			path:     "llms-updates.lock.json",
			wantLane: "",
			why:      "same trap: llms-updates.txt is named in the docs lane, llms-updates.lock.json is not",
		},
		{
			name:     "under docs/",
			path:     "docs/genlock.lock.json",
			wantLane: "docs",
			why: "mechanically safe here (docs/** IS a glob), but it is still build state filed " +
				"as a document, and it puts the lock in the same lane as the artifacts it " +
				"describes — so a lock-only update would take the very lease a no-op run avoids",
		},
		{
			name:     "the chosen home",
			path:     genlock.Dir + "/anything.lock.json",
			wantLane: "tools",
			why:      "tools/** is a named concurrent lane",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := laneadmit.LaneForPath(tc.path, tax, laneadmit.GranLeaf); got != tc.wantLane {
				t.Errorf("LaneForPath(%q) = %q, want %q — %s", tc.path, got, tc.wantLane, tc.why)
			}
		})
	}
}

// TestLockIsCommittedNotIgnored pins the third rule. A gitignored freshness record
// answers "were these artifacts built from the tree as it stands?" for exactly one
// person: whoever last ran the tool on that machine. tools/ already carries ignored
// machine-local state (tools/_registry/, tools/_watchdog/, tools/.bin/), so landing
// under an ignore rule by accident is a live possibility rather than a hypothetical.
func TestLockIsCommittedNotIgnored(t *testing.T) {
	root := repoRoot(t)
	b, err := os.ReadFile(filepath.Join(root, ".gitignore"))
	if err != nil {
		t.Skipf("no .gitignore at %s", root)
	}
	lock := genlock.PathFor("marketing-aeo")
	for _, pat := range ignorePatterns(string(b)) {
		if matchesIgnore(pat, lock) {
			t.Errorf(".gitignore pattern %q would exclude %s. The lock has to be committed: its "+
				"whole job is to answer a question about the COMMITTED artifacts, which an "+
				"ignored file can only answer for the last person who ran the tool.", pat, lock)
		}
	}
}

func ignorePatterns(body string) []string {
	var out []string
	for _, line := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "!") {
			continue
		}
		out = append(out, line)
	}
	return out
}

// matchesIgnore is a deliberately conservative reading of a gitignore line: it fires on
// a directory prefix, an exact path, or a basename/suffix glob. It over-matches rather
// than under-matches, because a false alarm here costs a rename and a miss costs the
// property the whole package depends on.
func matchesIgnore(pat, path string) bool {
	pat = strings.TrimPrefix(pat, "/")
	if pat == "" {
		return false
	}
	dir := strings.TrimSuffix(pat, "/")
	if path == dir || strings.HasPrefix(path, dir+"/") {
		return true
	}
	if ok, _ := filepath.Match(pat, path); ok {
		return true
	}
	if ok, _ := filepath.Match(pat, filepath.Base(path)); ok {
		return true
	}
	return false
}

// TestMatchesIgnoreActuallyFires keeps the gate above from being a test that can never
// fail: a matcher that always returned false would leave TestLockIsCommittedNotIgnored
// green forever.
func TestMatchesIgnoreActuallyFires(t *testing.T) {
	for _, tc := range []struct {
		pat, path string
		want      bool
	}{
		{"tools/_registry/", "tools/_registry/x.json", true},
		{"tools/.bin/", "tools/.bin/fak", true},
		{"tools/genlock/", "tools/genlock/marketing-aeo.lock.json", true},
		{"*.lock.json", "tools/genlock/marketing-aeo.lock.json", true},
		{"/*_out.txt", "serve_out.txt", true},
		{"tools/_registry/", "tools/genlock/marketing-aeo.lock.json", false},
		{"docs/_audits/", "tools/genlock/marketing-aeo.lock.json", false},
		{"*.exe", "tools/genlock/marketing-aeo.lock.json", false},
	} {
		if got := matchesIgnore(tc.pat, tc.path); got != tc.want {
			t.Errorf("matchesIgnore(%q, %q) = %v, want %v", tc.pat, tc.path, got, tc.want)
		}
	}
}
