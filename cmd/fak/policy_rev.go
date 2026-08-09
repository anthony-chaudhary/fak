package main

import (
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

// policyRevTimeout bounds the advisory rev lookup behind `fak policy --check` (#4311).
// The lookup is a whole-history modver.Snapshot and --check is a validation command an
// operator waits on, so a slow or wedged git must degrade the ADVISORY tag, never the
// validation the verb exists to perform. On expiry the tag is simply omitted.
const policyRevTimeout = 3 * time.Second

// policyManifestRevFn is the seam `fak policy --check` resolves its advisory rev
// through. It is a variable so the report formatter stays testable without a git repo:
// the default reads the real checkout, and a test swaps in a fixed rev to pin the
// rendered shape. Every arm is best-effort -- a resolver that cannot answer returns ""
// and the report keeps its pre-#4311 wording byte for byte.
var policyManifestRevFn = policyManifestRev

// policyManifestRev resolves the derived modver version ("r<rev>+g<sha>") of the policy
// manifest at path, or "" when that file is not a versioned policy module of THIS
// checkout. Advisory display only (#4311, the deferred display half of #2462): it feeds
// nothing but the printed line, so it never touches policy loading or enforcement and
// never fails a check.
//
// The cheap PURE key test runs FIRST, and that ordering is what keeps the verb
// affordable: only a top-level examples/<file>.json is a tracked policy module, so every
// other path -- a temp-dir manifest, a nested examples/<demo>/policy.json, a file
// outside the checkout -- short-circuits to "" BEFORE git is ever invoked. Without that
// guard every --check anywhere would pay a whole-history log walk only to learn it has
// no rev to show.
func policyManifestRev(path string) string {
	root := repoRoot()
	key := policyModuleKey(root, path)
	if key == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), policyRevTimeout)
	defer cancel()
	rep, err := modver.Snapshot(ctx, root, modver.RealRunner)
	if err != nil {
		// Not a git repo, git absent, or the walk timed out. The display is advisory,
		// so an unanswerable lookup stays silent rather than degrading the check.
		return ""
	}
	view, err := rep.View("examples/", "name", 0)
	if err != nil {
		return ""
	}
	return policyRevIn(view, key)
}

// policyRevIn is the pure matcher: the derived version of the module named key, or ""
// when the view carries no such module. A manifest that is added but not yet committed
// has no history and therefore no rev -- an absence the caller renders as silence, not
// as an error, because an uncommitted manifest is still a perfectly valid one.
func policyRevIn(rep modver.Report, key string) string {
	for _, m := range rep.Modules {
		if m.Name == key {
			return m.Version()
		}
	}
	return ""
}

// policyModuleKey maps a --check path to the modver module key that would carry its rev,
// or "" when no module can. It mirrors the examples/ keyspace rule #2462 shipped
// (modver.moduleOf): that keyspace is FLAT and file-keyed, so exactly a top-level
// examples/<file>.json is a module -- nested examples/<demo>/... paths are runnable demos
// and their fixtures, not deployable capability-floor manifests, and carry no rev.
//
// The path is resolved against the checkout root, so `fak policy --check` answers the
// same from any working directory and an absolute path resolves too. A path that escapes
// the root (a manifest outside this checkout, whose relative form leads with "..") is
// rejected here, and that rejection is what makes the outside-a-repo case graceful
// instead of a bogus match on a same-named file.
func policyModuleKey(root, path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return ""
	}
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return "" // e.g. a different Windows volume than the checkout
	}
	// Normalize the OS separator AND a literal backslash: filepath.ToSlash rewrites only
	// the OS separator (a no-op for a backslash on a Linux runner), so a Windows-authored
	// path would otherwise never prefix-match the slash-keyed module names.
	rel = strings.ReplaceAll(filepath.ToSlash(rel), `\`, "/")
	parts := strings.Split(rel, "/")
	if len(parts) != 2 || parts[0] != "examples" || !strings.HasSuffix(parts[1], ".json") {
		return ""
	}
	return rel
}

// policyRevTag renders a resolved rev for display beside the manifest path, or "" when
// there is none. Keeping the separator HERE rather than in the caller's format string is
// what guarantees a no-rev report stays byte-identical to its pre-#4311 self: the tag is
// additive, so its absence can never leave a stray double space behind.
func policyRevTag(rev string) string {
	if rev == "" {
		return ""
	}
	return "  " + rev
}
