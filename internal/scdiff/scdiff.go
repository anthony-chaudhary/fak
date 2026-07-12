// Package scdiff is the shared diff-scoping seam for shift-left scorecards.
//
// The scorecard family's post-hoc cost is concentrated in whole-tree rescans:
// every card re-reads its entire corpus every run, so an agent editing one file
// pays a full-tree scan to learn "did my edit move the number?". That expense is
// why the heavy portfolio fold (internal/scorecardpane.Collect) is relegated to a
// cron/CI cadence instead of running at the origin of each edit.
//
// scdiff inverts that: given a git ref the caller based its work on, it reports
// exactly which repo-relative paths changed, so a card can (a) SKIP entirely when
// none of its corpus was touched (the holistic-card fast path) or (b) scan only
// the changed slice of its corpus (the per-file-card fast path). This is the same
// diff-scoped, attributable, at-origin model `fak dup guard --range` and
// `fak score negframe --since` already use, lifted into one reusable seam every
// card and the portfolio pane can adopt.
//
// The pure surface (Intersect, Filter, MatchAny, MatchGlob) is tested directly
// over path slices; ChangedPaths is the one git-touching shell (exec is contained
// here, the way negframe keeps its `git show` reads in the CLI layer).
package scdiff

import (
	"bytes"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ChangedPaths returns the repo-relative, slash-separated paths that differ
// between `since` and the current working tree, so a scorecard can scope its scan
// to what a change actually touched. It unions three sources, matching the
// "everything I've touched since I based off <since>" an at-origin agent means:
//
//   - `git diff --name-only <since>` — tracked files that differ from the ref
//     (two-dot: ref vs working tree, so staged AND unstaged edits are included),
//   - `git ls-files --others --exclude-standard` — new untracked files (a new
//     source file adds debt, so it must count as changed),
//
// deduped and sorted for determinism. A blank `since` returns (nil, nil) — the
// caller treats that as "no baseline, full scan". An unresolvable ref or a git
// failure returns the error so the caller can fall back to a full scan rather
// than silently scoping to nothing (which would falsely report "unchanged").
func ChangedPaths(root, since string) ([]string, error) {
	since = strings.TrimSpace(since)
	if since == "" {
		return nil, nil
	}
	set := map[string]struct{}{}

	diff, err := gitLines(root, "diff", "--name-only", since, "--")
	if err != nil {
		return nil, err
	}
	for _, p := range diff {
		set[normalizeRel(p)] = struct{}{}
	}
	// Untracked files never appear in `git diff`; add them explicitly. A failure
	// here is non-fatal (the tracked diff already succeeded) — new files are a
	// best-effort addition, not the load-bearing signal.
	if others, oerr := gitLines(root, "ls-files", "--others", "--exclude-standard"); oerr == nil {
		for _, p := range others {
			set[normalizeRel(p)] = struct{}{}
		}
	}

	out := make([]string, 0, len(set))
	for p := range set {
		if p != "" {
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out, nil
}

// Intersect returns the subset of corpus that also appears in changed, as an
// exact repo-relative slash-path set intersection. Used by holistic cards whose
// corpus is an ENUMERABLE, fixed file set (e.g. ui-quality's render sources): an
// empty result means "none of my inputs moved, skip the rescan". Both inputs are
// normalized, and the result preserves corpus order (deterministic, caller-facing).
func Intersect(corpus, changed []string) []string {
	if len(corpus) == 0 || len(changed) == 0 {
		return nil
	}
	want := make(map[string]struct{}, len(changed))
	for _, c := range changed {
		want[normalizeRel(c)] = struct{}{}
	}
	var out []string
	for _, c := range corpus {
		if _, ok := want[normalizeRel(c)]; ok {
			out = append(out, c)
		}
	}
	return out
}

// Filter returns the subset of changed paths that match any of the given corpus
// globs. Used by GLOB-corpus cards (docs/*, whole subtrees) and by the portfolio
// pane to decide whether a card's corpus could have moved without enumerating it.
// Order follows changed (deterministic). A nil/empty globs matches nothing.
func Filter(changed, globs []string) []string {
	if len(changed) == 0 || len(globs) == 0 {
		return nil
	}
	var out []string
	for _, c := range changed {
		if MatchAny(globs, c) {
			out = append(out, c)
		}
	}
	return out
}

// MatchAny reports whether p matches any glob in globs.
func MatchAny(globs []string, p string) bool {
	for _, g := range globs {
		if MatchGlob(g, p) {
			return true
		}
	}
	return false
}

// MatchGlob reports whether the repo-relative slash path p matches glob g, with
// just enough glob vocabulary for corpus declarations and no external dependency:
//
//   - a trailing "/" (e.g. "internal/uiquality/") is a directory-prefix match —
//     p matches if it is under that directory at any depth,
//   - "**" matches any run of characters including "/" (e.g. "tools/**_test.py"),
//     and the segment form "**/" matches zero or more leading path segments (so
//     "**/AGENTS.md" matches both "AGENTS.md" and "a/b/AGENTS.md"),
//   - a segment-local "*" matches within a single path segment (never crosses "/"),
//   - an exact string otherwise.
//
// Both g and p are normalized (backslashes to slashes, "./" stripped) first. The
// glob is compiled to an anchored regexp once and cached, since MatchGlob runs in
// changed×globs loops.
func MatchGlob(g, p string) bool {
	g = normalizeRel(g)
	p = normalizeRel(p)
	switch {
	case g == "":
		return false
	case strings.HasSuffix(g, "/"):
		return strings.HasPrefix(p, g) || p == strings.TrimSuffix(g, "/")
	default:
		re := compileGlob(g)
		return re != nil && re.MatchString(p)
	}
}

var globCache sync.Map // glob string -> *regexp.Regexp (nil sentinel on bad glob)

// compileGlob translates a corpus glob into an anchored regexp, caching the
// result. "**/" becomes an optional run of segments, a bare "**" becomes ".*"
// (crossing "/"), a single "*" becomes "[^/]*" (segment-local), and every other
// character is matched literally (regexp metacharacters escaped).
func compileGlob(g string) *regexp.Regexp {
	if v, ok := globCache.Load(g); ok {
		re, _ := v.(*regexp.Regexp)
		return re
	}
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(g); {
		switch {
		case g[i] == '*' && i+1 < len(g) && g[i+1] == '*':
			if i+2 < len(g) && g[i+2] == '/' {
				b.WriteString("(?:.*/)?") // "**/" — zero or more leading segments
				i += 3
			} else {
				b.WriteString(".*") // bare "**" — any run, crossing "/"
				i += 2
			}
		case g[i] == '*':
			b.WriteString("[^/]*") // segment-local wildcard
			i++
		default:
			b.WriteString(regexp.QuoteMeta(string(g[i])))
			i++
		}
	}
	b.WriteString("$")
	re, err := regexp.Compile(b.String())
	if err != nil {
		re = nil // authored globs shouldn't reach here; a bad one never matches
	}
	globCache.Store(g, re)
	return re
}

// normalizeRel canonicalizes a path to repo-relative slash form: backslashes to
// slashes, a leading "./" stripped, surrounding whitespace trimmed. It does not
// clean ".." (corpus globs and git output never carry it).
func normalizeRel(p string) string {
	p = strings.TrimSpace(p)
	p = strings.ReplaceAll(p, `\`, "/")
	p = strings.TrimPrefix(p, "./")
	return p
}

// gitLines runs `git -C root <args...>` and returns stdout split into trimmed,
// non-empty lines. A non-zero exit surfaces as an error carrying stderr's tail.
func gitLines(root string, args ...string) ([]string, error) {
	full := append([]string{"-C", root}, args...)
	cmd := exec.Command("git", full...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, &GitError{Args: args, Err: err, Stderr: strings.TrimSpace(stderr.String())}
	}
	var out []string
	for _, ln := range strings.Split(stdout.String(), "\n") {
		if s := strings.TrimSpace(ln); s != "" {
			out = append(out, s)
		}
	}
	return out, nil
}

// GitError carries the failing git invocation so a caller can distinguish an
// unresolvable ref (fall back to a full scan) from a real breakage.
type GitError struct {
	Args   []string
	Err    error
	Stderr string
}

func (e *GitError) Error() string {
	msg := "git " + strings.Join(e.Args, " ") + ": " + e.Err.Error()
	if e.Stderr != "" {
		msg += ": " + e.Stderr
	}
	return msg
}

func (e *GitError) Unwrap() error { return e.Err }
