package devindex

// #5107: the HEAD-aware dead-link pass. The tier-1 detectors in freshness.go
// (DeadDocLinks / DeadLLMSLinks) resolve INDEX.md and llms.txt link targets against
// the WORKING TREE (os.Stat under c.Root) — fast, git-free local dev. That leaves
// exactly one blind spot: a link line committed at HEAD whose target exists only as
// an UNTRACKED working-tree file. os.Stat finds the file, the tier-1 check reads
// clean, yet every fresh `git clone` of HEAD has a dead link no gate ever flags.
// The commit-by-path additive sweep (INDEX.md staged whole-file) produces that
// state routinely: an index line lands ahead of its target.
//
// CheckFreshnessAgainstHEAD is the opt-in tier-2 answer: it reads INDEX.md and
// llms.txt AS COMMITTED AT HEAD (git show), resolves each local .md link against
// HEAD's committed tree (git ls-tree), and flags any target absent from that tree —
// exactly what a pristine checkout would see. It shells out to git, so it lives in
// this separate file, is never called by the tier-1 CheckFreshness fold, and errors
// (rather than degrades) when git/HEAD is unavailable: a caller opting into the
// HEAD view must not mistake "git failed" for "link-clean". The link filters below
// deliberately mirror their tier-1 twins line for line (and the Python reciprocal
// gate, tools/check_index_sync.py) so the two views can never disagree on what a
// checkable local link IS — only on which tree it is resolved against.

import (
	"fmt"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/docsearch"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	// DriftDeadDocLinkHEAD: an INDEX.md doc-map entry committed at HEAD whose target
	// path is absent from HEAD's tree (even when an untracked working-tree file of
	// that name exists — the blind spot the working-tree DriftDeadDocLink cannot see).
	DriftDeadDocLinkHEAD DriftKind = "dead-doc-link-head"
	// DriftDeadLLMSLinkHEAD: an llms.txt local .md link committed at HEAD whose
	// target is absent from HEAD's tree — the same HEAD-resolution applied to the
	// LLM-facing map.
	DriftDeadLLMSLinkHEAD DriftKind = "dead-llms-link-head"
)

// gitHEADOut runs one read-only git command against the repo at root and returns
// its stdout. Every window is suppressed (DESKTOP_POPUP_REGRESSION gate).
func gitHEADOut(root string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Output()
}

// headTreePaths returns the set of slash-separated file paths committed at HEAD,
// path.Clean-normalized for membership tests.
func headTreePaths(root string) (map[string]bool, error) {
	top, err := gitHEADOut(root, "rev-parse", "--show-toplevel")
	if err != nil {
		return nil, err
	}
	want, err := canonicalRepoPath(root)
	if err != nil {
		return nil, err
	}
	got, err := canonicalRepoPath(strings.TrimSpace(string(top)))
	if err != nil {
		return nil, err
	}
	if got != want {
		return nil, fmt.Errorf("git root %s does not match requested catalog root %s", got, want)
	}
	out, err := gitHEADOut(root, "ls-tree", "-r", "--name-only", "HEAD")
	if err != nil {
		return nil, err
	}
	tree := map[string]bool{}
	for _, line := range strings.Split(string(out), "\n") {
		p := strings.TrimSpace(line)
		if p == "" {
			continue
		}
		tree[path.Clean(p)] = true
	}
	return tree, nil
}

func canonicalRepoPath(root string) (string, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	return filepath.Clean(resolved), nil
}

// cleanLocalLinkTarget applies the shared local-link filter both HEAD-aware
// detectors use, mirroring the tier-1 twins exactly: an external http(s) / mailto
// target, an in-page "#anchor", and an absolute "/" path are not ours to check; a
// trailing #anchor or ?query is stripped so the tree lookup sees a real path. The
// second return is false when the target is not a checkable local path.
func cleanLocalLinkTarget(target string) (string, bool) {
	t := strings.TrimSpace(target)
	if t == "" || strings.HasPrefix(t, "#") || strings.HasPrefix(t, "/") ||
		strings.HasPrefix(t, "http://") || strings.HasPrefix(t, "https://") ||
		strings.HasPrefix(t, "mailto:") {
		return "", false
	}
	if i := strings.IndexAny(t, "#?"); i >= 0 {
		t = t[:i]
	}
	if t == "" {
		return "", false
	}
	return t, true
}

// CheckFreshnessAgainstHEAD resolves the INDEX.md doc-map bullets and llms.txt
// local .md links AS COMMITTED AT HEAD against HEAD's committed tree, and returns
// a Drift for every link whose target a fresh checkout would not have — the
// HEAD-only dead links the tier-1 working-tree pass is structurally blind to
// (#5107). Findings are sorted (kind, then subject) like CheckFreshness. A file
// absent from HEAD (no committed INDEX.md / llms.txt) contributes no finding — no
// committed map, no committed claim. It returns an error only when git itself
// cannot answer (no repo, no HEAD): the caller opted into the HEAD view, so a
// failure to read it must never masquerade as clean. Tier-1 CheckFreshness never
// calls this; it stays git-free.
func (c *Catalog) CheckFreshnessAgainstHEAD() ([]Drift, error) {
	tree, err := headTreePaths(c.Root)
	if err != nil {
		return nil, err
	}
	var out []Drift
	// INDEX.md as committed at HEAD: same bullet grammar the catalog parses
	// (docLineRE), same local-target filter as tier-1 DeadDocLinks.
	if idx, err := gitHEADOut(c.Root, "show", "HEAD:INDEX.md"); err == nil {
		seen := map[string]bool{}
		for _, raw := range strings.Split(string(idx), "\n") {
			title, target, _, ok := docsearch.ParseBullet(raw)
			if !ok {
				continue
			}
			clean, ok := cleanLocalLinkTarget(target)
			if !ok || seen[clean] {
				continue
			}
			seen[clean] = true
			if !tree[path.Clean(clean)] {
				out = append(out, Drift{
					Kind:    DriftDeadDocLinkHEAD,
					Subject: clean,
					Reason:  "INDEX.md at HEAD links " + title + " -> " + clean + " but HEAD's tree does not contain it (a fresh checkout has a dead link)",
				})
			}
		}
	}
	// llms.txt as committed at HEAD: same link scan + .md-only filter as tier-1
	// DeadLLMSLinks.
	if b, err := gitHEADOut(c.Root, "show", "HEAD:llms.txt"); err == nil {
		seen := map[string]bool{}
		for _, m := range llmsLinkRE.FindAllStringSubmatch(string(b), -1) {
			clean, ok := cleanLocalLinkTarget(m[1])
			if !ok || !strings.HasSuffix(clean, ".md") || seen[clean] {
				continue
			}
			seen[clean] = true
			if !tree[path.Clean(clean)] {
				out = append(out, Drift{
					Kind:    DriftDeadLLMSLinkHEAD,
					Subject: clean,
					Reason:  "llms.txt at HEAD links " + clean + " but HEAD's tree does not contain it (a fresh checkout has a dead link)",
				})
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Subject < out[j].Subject
	})
	return out, nil
}
