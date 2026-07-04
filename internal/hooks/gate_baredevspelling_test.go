package hooks

import (
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// treeFrom builds a TrackedTree straight from an in-memory {path: content} map, populating
// the file cache so the gate reads no disk. Paths are sorted like git ls-files.
func treeFrom(files map[string]string) *TrackedTree {
	t := &TrackedTree{fileCache: map[string]fileEntry{}}
	for p, c := range files {
		t.Paths = append(t.Paths, p)
		t.fileCache[p] = fileEntry{data: []byte(c), exists: true}
	}
	sort.Strings(t.Paths)
	return t
}

func findingSet(fs []Finding) map[string]bool {
	m := map[string]bool{}
	for _, f := range fs {
		m[f.File] = true
	}
	return m
}

// TestBareDevSpelling_ScopeAndClassification is the red/green witness the DoD names: the
// sweep reds on an injected bare dev spelling in every caller surface, and is silent on the
// migrated form, on frontdoor verbs, on out-of-scope files, and on pruned checkout copies.
func TestBareDevSpelling_ScopeAndClassification(t *testing.T) {
	// Guard against devindex drift: the sample verbs must actually be dev-tier, or the whole
	// gate is testing nothing. (commit/sweep/orient = dev; run = frontdoor.)
	for _, v := range []string{"commit", "sweep", "orient"} {
		if tier, ok := devindex.TierOf(v); !ok || tier != devindex.TierDev {
			t.Fatalf("precondition: TierOf(%q)=(%v,%v), want dev — devindex classification drifted", v, tier, ok)
		}
	}
	if tier, _ := devindex.TierOf("run"); tier != devindex.TierFrontdoor {
		t.Fatalf("precondition: expected 'run' to be a frontdoor verb, got %q", tier)
	}

	tree := treeFrom(map[string]string{
		// executable surfaces: any bare spelling is a caller (RED)
		"tools/foo.py":               "subprocess.run(['fak', 'commit'])  # fak commit\n",
		"tools/register_thing.ps1":   "fak sweep --now\n",
		"Makefile":                   "ci:\n\tfak orient --paths x\n",
		".github/workflows/nite.yml": "      - run: fak sweep\n",
		".claude/skills/x/SKILL.md":  "Then run `fak commit -m ...` to land it.\n",
		// the MIGRATED form is silent: the token after `fak` is `dev`, which is untiered
		"tools/migrated.py": "subprocess.run(['fak', 'dev', 'commit'])  # fak dev commit\n",
		// frontdoor verb is silent
		"tools/front.py": "os.system('fak run --live')\n",
		// docs: prose naming the old spelling is NOT a caller; only the fenced code block is
		"docs/guide.md": "Historically you typed fak commit by hand.\n\n```sh\nfak sweep --now\n```\n",
		// out of scope / pruned — all silent
		"README.md":                   "fak commit\n",
		"internal/hooks/notes.go":     "// fak commit\n",
		".fak/wt/tools/copy.py":       "fak commit\n",
		"docs/archive/old-runbook.md": "```sh\nfak commit\n```\n",
	})

	got := findingSet(bareDevSpellingFindings(tree, bareDevAllowlist{exact: map[string]bool{}}))

	wantRed := []string{
		"tools/foo.py", "tools/register_thing.ps1", "Makefile",
		".github/workflows/nite.yml", ".claude/skills/x/SKILL.md", "docs/guide.md",
	}
	for _, f := range wantRed {
		if !got[f] {
			t.Errorf("expected BARE_DEV_SPELLING finding for %s, got none", f)
		}
	}
	wantGreen := []string{
		"tools/migrated.py", "tools/front.py", "README.md",
		"internal/hooks/notes.go", ".fak/wt/tools/copy.py", "docs/archive/old-runbook.md",
	}
	for _, f := range wantGreen {
		if got[f] {
			t.Errorf("did NOT expect a finding for %s (migrated/frontdoor/out-of-scope/pruned), got one", f)
		}
	}
}

// TestBareDevSpelling_DocsProseNotFlaggedButCodeIs pins the code-block boundary: the SAME
// bare spelling is silent in prose and red inside a fence, in one docs file.
func TestBareDevSpelling_DocsProseNotFlaggedButCodeIs(t *testing.T) {
	tree := treeFrom(map[string]string{
		"docs/x.md": "prose fak sweep here\n```\nfak sweep in code\n```\nmore prose fak sweep\n",
	})
	fs := bareDevSpellingFindings(tree, bareDevAllowlist{exact: map[string]bool{}})
	if len(fs) != 1 {
		t.Fatalf("want exactly 1 finding (the fenced line), got %d: %+v", len(fs), fs)
	}
	if fs[0].Line != 3 {
		t.Errorf("finding should cite the fenced code line (3), got line %d", fs[0].Line)
	}
}

// TestBareDevSpelling_AllowlistSuppresses proves an entry on the allowlist (exact path and
// directory prefix) exempts its surface.
func TestBareDevSpelling_AllowlistSuppresses(t *testing.T) {
	tree := treeFrom(map[string]string{
		"tools/legacy.ps1":    "fak commit\n",          // in-scope, exact-allowlisted
		"docs/vendor/snap.md": "```\nfak sweep\n```\n", // in-scope code block, prefix-allowlisted
		"tools/other.py":      "fak commit\n",          // in-scope, NOT allowlisted
	})
	allow := parseBareDevAllowlist(strings.Join([]string{
		"# a comment",
		"tools/legacy.ps1   # historical registration snapshot",
		"docs/vendor/       # vendored marketing capture subtree",
		"",
	}, "\n"))
	got := findingSet(bareDevSpellingFindings(tree, allow))
	if got["tools/legacy.ps1"] {
		t.Errorf("allowlisted exact path tools/legacy.ps1 should be suppressed")
	}
	if got["docs/vendor/snap.md"] {
		t.Errorf("allowlisted prefix docs/vendor/ should suppress its subtree")
	}
	if !got["tools/other.py"] {
		t.Errorf("non-allowlisted tools/other.py should still red")
	}
}

// TestBareDevSpelling_EmbeddedAllowlistParses guards the shipped allowlist file: it must parse
// (every non-comment entry carries a reason is a human rule, but the parser must not panic and
// the gate registration must resolve).
func TestBareDevSpelling_EmbeddedAllowlistParses(t *testing.T) {
	_ = parseBareDevAllowlist(bareDevAllowlistRaw) // must not panic
	if HygieneGateByName("BARE_DEV_SPELLING") == nil {
		t.Fatal("BARE_DEV_SPELLING gate not registered in HygieneGates()")
	}
}

// knownDefaultOffGates is the explicit allowlist of gates permitted to land DefaultOff — each a
// deliberate migration-in-flight ratchet that runs only via `--gates <NAME>` until its tree is
// clean, then flips DefaultOff:false to become an enforcing gate. Any OTHER DefaultOff gate is an
// accident (a gate silently excluded from the default `make ci` sweep), which the test below
// still catches. Add a gate here only with the same intent BARE_DEV_SPELLING and DEAD_CODE carry.
var knownDefaultOffGates = map[string]bool{
	"BARE_DEV_SPELLING": true, // C4 spelling migration in flight (#2228, #2233)
	"DEAD_CODE":         true, // pre-existing dead symbols being retired by /slop-score
}

// TestBareDevSpelling_RegisteredDefaultOff pins the ratchet-in-flight contract: the gate is
// registered but DefaultOff, so `make ci`'s default sweep never reds the trunk against the
// not-yet-migrated tree, while `--gates BARE_DEV_SPELLING` can still run it. It also guards the
// general invariant that no gate is DefaultOff by accident — every DefaultOff gate must be a
// declared migration ratchet in knownDefaultOffGates.
func TestBareDevSpelling_RegisteredDefaultOff(t *testing.T) {
	var found, off bool
	for _, g := range HygieneGates() {
		if g.Name == "BARE_DEV_SPELLING" {
			found, off = true, g.DefaultOff
		}
		if g.DefaultOff && !knownDefaultOffGates[g.Name] {
			t.Errorf("gate %s is unexpectedly DefaultOff — only a declared migration ratchet "+
				"(knownDefaultOffGates) may be", g.Name)
		}
	}
	if !found {
		t.Fatal("BARE_DEV_SPELLING not in HygieneGates()")
	}
	if !off {
		t.Error("BARE_DEV_SPELLING must be DefaultOff while the C4 migration is in flight (flip on for the C5 gate)")
	}
}
