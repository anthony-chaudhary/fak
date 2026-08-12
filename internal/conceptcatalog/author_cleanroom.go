package conceptcatalog

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// renderFromGitTree generates the tracked artifacts inside a clean-room export of
// treeish and returns their bytes keyed by tracked path. An empty treeish means the
// current index (git write-tree).
//
// overlay replaces files inside the EXPORTED concept corpus, keyed by base name, before
// the generator runs. That is what lets a planned-but-unstaged corpus edit be scored
// without ever handing the generator the worktree: the numbers still come from a tree
// that exists in git, plus exactly the one edit the caller is proposing.
func renderFromGitTree(root, treeish string, overlay map[string][]byte) (map[string][]byte, error) {
	tree, cleanup, err := materializeGitTree(root, treeish)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	if len(overlay) > 0 {
		dir := filepath.Join(tree, filepath.FromSlash(DataRel))
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}
		for name, b := range overlay {
			if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
				return nil, err
			}
		}
	}
	// Generate into scratch: generate() encodes the "exit 1 is an honest ACTION verdict
	// as long as every artifact exists" rule, so a real crash must not leave half-written
	// artifacts anywhere a caller will publish them.
	out, err := os.MkdirTemp("", "fak-concept-render-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(out)
	if err := generate(tree, out); err != nil {
		return nil, err
	}
	rendered := make(map[string][]byte, len(generatedArtifacts))
	for _, art := range generatedArtifacts {
		b, readErr := os.ReadFile(filepath.Join(out, art.Name))
		if readErr != nil {
			return nil, fmt.Errorf("read generated %s: %w", art.Name, readErr)
		}
		rendered[art.Tracked] = b
	}
	return rendered, nil
}

// addClassifyGeneratedArtifacts renders a classify plan's tracked artifacts from a
// committed git tree instead of from the worktree.
//
// The scorecard is a COMMITTED artifact whose numbers are derived by walking a whole
// workspace, so which workspace it walks is the whole question. generateShadow walks the
// repo root, and on a shared multi-session trunk that root carries every peer session's
// uncommitted symbols: one classification would publish a "discovered" count computed
// over a tree that exists in no commit, and the count would move again the moment those
// peers landed or abandoned their WIP (#6521). Rendering from HEAD plus the caller's own
// corpus edit publishes numbers any reader can reproduce, and leaves whatever the
// committer's pathspec adds on top to CONCEPT_FRESHNESS - whose cure already regenerates
// from that exact tree.
func addClassifyGeneratedArtifacts(c Catalog, plan Plan) (Plan, error) {
	root := filepath.Dir(filepath.Dir(c.Dir))
	if _, err := os.Stat(filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")); os.IsNotExist(err) {
		// No canonical generator in this workspace: there is nothing to regenerate. The
		// same no-op generateShadow takes, so in-memory catalogs still plan.
		return plan, nil
	}
	if !isRepoRoot(root) {
		// No repository of its own to score, and no peer session to be dirty against:
		// the worktree IS the tree.
		return AddGeneratedArtifacts(c, plan)
	}
	overlay := map[string][]byte{}
	for _, ch := range plan.Changes {
		if filepath.Clean(filepath.Dir(ch.Path)) == filepath.Clean(c.Dir) {
			overlay[filepath.Base(ch.Path)] = ch.Content
		}
	}
	// HEAD, not the index. `git write-tree` takes .git/index.lock, so on a shared trunk an
	// unrelated peer's staging turns a classification into a hard failure, and the index it
	// hashes would carry that peer's staged work as well. HEAD is a commit that outlives
	// every session.
	rendered, err := renderFromGitTree(root, "HEAD", overlay)
	if err != nil {
		return Plan{}, err
	}
	// Every generated artifact ages with the catalog: a fresh scorecard beside a stale
	// name index would answer one of the two questions from a retired catalog.
	for _, art := range generatedArtifacts {
		dst := filepath.Join(root, filepath.FromSlash(art.Tracked))
		plan.Changes = append(plan.Changes, Change{Path: dst, Content: rendered[art.Tracked]})
		plan.Files = append(plan.Files, filepath.ToSlash(dst))
	}
	sort.Strings(plan.Files)
	return plan, nil
}

// isRepoRoot reports whether root is the TOP LEVEL of a git work tree.
//
// "Inside a work tree" is not a strong enough question. Scratch fixtures routinely sit
// inside an unrelated enclosing checkout - go's own test temp directory does on this
// machine - and `git archive` run from a subdirectory exports that subdirectory's subtree,
// which for an untracked scratch path is nothing at all. The concept root is always a
// repository top level in production (conceptRoot resolves it with rev-parse
// --show-toplevel), so anything else is a fixture and belongs on the worktree path.
func isRepoRoot(root string) bool {
	b, err := git(root, "rev-parse", "--show-toplevel")
	if err != nil {
		return false
	}
	return sameDir(strings.TrimSpace(string(b)), root)
}

// sameDir compares two directory paths as the filesystem resolves them. git reports the
// top level with forward slashes and through resolved links, which need not match the
// caller's spelling of the same directory.
func sameDir(a, b string) bool {
	resolve := func(p string) string {
		p = filepath.Clean(filepath.FromSlash(p))
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return r
		}
		return p
	}
	x, y := resolve(a), resolve(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(x, y)
	}
	return x == y
}
