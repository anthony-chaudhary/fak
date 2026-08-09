package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

// policyRevPattern is the derived-version shape #2462 stamps into the modver
// report/ledger: the monotonic revision counter plus the last-touch commit, e.g.
// "r2+g04270cd98". The witness matches the SHAPE rather than a literal rev, because the
// rev of a live manifest legitimately climbs every time that manifest is edited.
var policyRevPattern = regexp.MustCompile(`\br[0-9]+\+g[0-9a-f]{7,}\b`)

// minimalPolicyManifest is the smallest manifest policy.LoadRuntime accepts, used by the
// arms that need a VALID manifest at a path of the test's choosing (outside the repo).
const minimalPolicyManifest = `{"version":"fak-policy/v1","allow":["read"]}`

// TestPolicyCheckSurfacesManifestRev is the #4311 done-condition witness: `fak policy
// --check examples/<x>.json` prints that manifest's derived modver rev BESIDE its path.
// It drives the real report formatter (checkPolicyFile) over a real tracked manifest in
// this checkout, so it fails if the rev is dropped, printed somewhere other than beside
// the path, or silently stops resolving.
func TestPolicyCheckSurfacesManifestRev(t *testing.T) {
	root := repoRootForPolicyRevTest(t)
	const manifest = "examples/dev-agent-policy.json"
	path := filepath.Join(root, filepath.FromSlash(manifest))
	if _, err := os.Stat(path); err != nil {
		t.Skipf("tracked manifest %s absent from this checkout: %v", manifest, err)
	}

	rev := policyManifestRev(path)
	if rev == "" {
		// No git, a shallow/exported tree, or a timed-out walk. The display is advisory
		// and MUST degrade rather than fail, so the absence is a skip here -- the
		// graceful-degradation arm below is what pins that behavior.
		t.Skip("no derived rev resolvable in this environment (no git history?); advisory display degrades by design")
	}
	if !policyRevPattern.MatchString(rev) {
		t.Fatalf("resolved rev %q does not match the derived r<rev>+g<sha> shape", rev)
	}

	report, err := checkPolicyFile(path, orgCheckOptions{})
	if err != nil {
		t.Fatalf("checkPolicyFile(%s) = %v, want a valid manifest", manifest, err)
	}
	// "beside its path": the rev must sit on the OK line, immediately after the path and
	// before the validity note -- asserting the exact join the formatter builds.
	want := "OK  " + path + "  " + rev + "  (manifest valid;"
	if !strings.Contains(report, want) {
		t.Fatalf("rev not surfaced beside the manifest path\n  want substring: %q\n  report:        %q", want, firstLine(report))
	}
}

// TestPolicyCheckRevGracefulOutsideRepo pins the #4311 graceful arm: a manifest that is
// not a tracked policy module of this checkout still validates, still prints, and prints
// NO rev -- the OK line stays byte-identical to its pre-#4311 self. This is the arm that
// keeps an advisory display from ever becoming a new way for --check to fail.
func TestPolicyCheckRevGracefulOutsideRepo(t *testing.T) {
	dir := t.TempDir() // outside the checkout, and not a git repo
	path := filepath.Join(dir, "outside-policy.json")
	if err := os.WriteFile(path, []byte(minimalPolicyManifest), 0o600); err != nil {
		t.Fatal(err)
	}

	if rev := policyManifestRev(path); rev != "" {
		t.Fatalf("manifest outside the checkout resolved a rev %q, want none", rev)
	}

	report, err := checkPolicyFile(path, orgCheckOptions{})
	if err != nil {
		t.Fatalf("checkPolicyFile outside a repo = %v, want the manifest to validate anyway", err)
	}
	want := "OK  " + path + "  (manifest valid; every deny cites a closed-vocabulary reason)"
	if !strings.Contains(report, want) {
		t.Fatalf("no-rev report drifted from its pre-#4311 wording\n  want substring: %q\n  report:        %q", want, firstLine(report))
	}
	if policyRevPattern.MatchString(firstLine(report)) {
		t.Fatalf("a manifest with no derived rev printed one anyway: %q", firstLine(report))
	}
}

// TestPolicyCheckRevRenderedShape drives the formatter through the policyManifestRevFn
// seam with a FIXED rev, so the rendered shape is pinned deterministically -- no git, no
// checkout layout, no dependence on how often the live manifest has been edited.
func TestPolicyCheckRevRenderedShape(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pinned.json")
	if err := os.WriteFile(path, []byte(minimalPolicyManifest), 0o600); err != nil {
		t.Fatal(err)
	}
	restore := policyManifestRevFn
	policyManifestRevFn = func(string) string { return "r2+g04270cd98" }
	t.Cleanup(func() { policyManifestRevFn = restore })

	report, err := checkPolicyFile(path, orgCheckOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := "OK  " + path + "  r2+g04270cd98  (manifest valid;"
	if !strings.Contains(report, want) {
		t.Fatalf("rendered rev shape drifted\n  want substring: %q\n  report:        %q", want, firstLine(report))
	}
}

// TestPolicyModuleKey pins the pure key rule that decides whether a --check path can
// carry a rev at all. It is the guard that keeps the verb affordable (a non-manifest
// path must short-circuit BEFORE git runs) and mirrors #2462's flat, file-keyed
// examples/ keyspace: only a top-level examples/<file>.json is a module.
func TestPolicyModuleKey(t *testing.T) {
	// Anchor the fixture root absolutely, on whatever volume the test runs on:
	// filepath.Rel cannot relate a volume-qualified path to a volume-less one (nor two
	// different Windows volumes), so a bare "/repo" would measure the drive rather than
	// the keyspace rule this test exists to pin.
	root := mustAbs(t, filepath.FromSlash("/repo"))
	cases := []struct {
		name string
		path string
		want string
	}{
		{"top-level manifest", filepath.Join(root, "examples", "dev-agent-policy.json"), "examples/dev-agent-policy.json"},
		{"nested demo manifest is not a module", filepath.Join(root, "examples", "compose-ollama", "policy.json"), ""},
		{"non-json under examples", filepath.Join(root, "examples", "README.md"), ""},
		{"outside the examples keyspace", filepath.Join(root, "internal", "policy", "x.json"), ""},
		{"outside the checkout", filepath.Join(root, "..", "elsewhere", "examples", "x.json"), ""},
		{"the examples root itself", filepath.Join(root, "examples"), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := policyModuleKey(root, tc.path); got != tc.want {
				t.Fatalf("policyModuleKey(%q, %q) = %q, want %q", root, tc.path, got, tc.want)
			}
		})
	}
}

// TestPolicyRevIn covers the pure matcher over a synthetic report: an exact module-name
// hit yields the derived version, and a miss (an uncommitted manifest, which has no
// history and so no rev) stays silent instead of guessing a neighbor's rev.
func TestPolicyRevIn(t *testing.T) {
	rep := modver.Report{Modules: []modver.Module{
		{Name: "examples/dev-agent-policy.json", Kind: "policy", Rev: 4, LastCommit: "805b0cc920"},
		{Name: "examples/coding-agent-safe.json", Kind: "policy", Rev: 1, LastCommit: "3b7fb1cf4b"},
	}}
	if got, want := policyRevIn(rep, "examples/dev-agent-policy.json"), "r4+g805b0cc920"; got != want {
		t.Fatalf("policyRevIn hit = %q, want %q", got, want)
	}
	if got := policyRevIn(rep, "examples/not-yet-committed.json"); got != "" {
		t.Fatalf("policyRevIn miss = %q, want no rev", got)
	}
}

// TestPolicyRevTag pins the additive-separator rule: a rev is joined with the two-space
// separator the OK line uses, and no rev contributes NOTHING -- not even a space, which
// is what keeps the no-rev line byte-identical to its pre-#4311 self.
func TestPolicyRevTag(t *testing.T) {
	if got, want := policyRevTag("r2+g04270cd98"), "  r2+g04270cd98"; got != want {
		t.Fatalf("policyRevTag = %q, want %q", got, want)
	}
	if got := policyRevTag(""); got != "" {
		t.Fatalf("policyRevTag(\"\") = %q, want the empty string", got)
	}
}

func mustAbs(t *testing.T, p string) string {
	t.Helper()
	abs, err := filepath.Abs(p)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// repoRootForPolicyRevTest walks up from the test's working directory to the checkout
// root (the directory holding go.mod), the same anchor repoRoot() resolves at runtime.
func repoRootForPolicyRevTest(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for dir := wd; ; {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("go.mod not found walking up from %s", wd)
		}
		dir = parent
	}
}
