package main

// wip_fence.go — `fak wip fence` / `fak wip unfence` (#4153): make the durable
// WIP-tag convention STRUCTURAL. AGENTS.md requires a not-yet-compiling .go file
// on the shared trunk to be fenced behind `//go:build wip_<slug>` so the DEFAULT
// build (`go build ./cmd/fak`) stays green for every peer and for CI — the exact
// invariant internal/buildwitness enforces. Today that fence is hand discipline;
// these verbs make it a one-command mechanical transform:
//
//	fak wip fence <file> [--feature <slug>] | --all-untracked [-C <repo>]
//	fak wip unfence <file> [-C <repo>]
//
// The fenced file stays on disk and builds under `go build -tags wip_<slug>`;
// dropping the fence once the defining symbol lands is `fak wip unfence`. The
// transforms (wipFenceSlug, fenceText, unfenceText) are pure and unit-tested; the
// two run* shells own only flag parsing and file I/O. Complements the checkpoint/
// restore spine in wip.go: checkpoint makes WIP durable, fence makes it harmless.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/buildoverlay"
)

// wipFenceSlug derives the fence slug (the tag is wip_<slug>) from a file path or
// name: base name, minus a trailing ".go", minus a trailing "_test", lowercased,
// every run of characters outside [a-z0-9] collapsed to a single underscore, then
// trimmed of leading/trailing underscores. An empty derivation falls back to
// "wip". Build-tag terms allow [A-Za-z0-9_.]; underscores keep it simple.
func wipFenceSlug(pathOrName string) string {
	// Normalize BOTH separators by hand rather than filepath.ToSlash: ToSlash only
	// rewrites the host OS's separator, so a Windows path fed to a Linux build (the
	// canonical WSL test runner) keeps its backslashes and path.Base returns the whole
	// string. A slug must not depend on which OS derived it.
	name := path.Base(strings.ReplaceAll(pathOrName, `\`, "/"))
	name = strings.TrimSuffix(name, ".go")
	name = strings.TrimSuffix(name, "_test")
	name = strings.ToLower(name)
	var b strings.Builder
	pendingSep := false
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			if pendingSep {
				b.WriteByte('_')
				pendingSep = false
			}
			b.WriteRune(r)
			continue
		}
		if b.Len() > 0 {
			pendingSep = true // collapse the run; emit one '_' only if more follows
		}
	}
	if b.Len() == 0 {
		return "wip"
	}
	return b.String()
}

// wipLeadingGoBuild scans src's LEADING build-constraint region — blank lines and
// `//` line comments from the top, ended by the first other line (the package
// clause, a /* block comment, code) — for a //go:build line. It returns that
// line's index and constraint expression when one is present. lines is src split
// on "\n"; a trailing "\r" (CRLF source) is tolerated on every line. Detection is
// deliberately lenient (leading whitespace before // is accepted) so fenceText can
// never stack a second constraint next to an almost-valid first one.
func wipLeadingGoBuild(lines []string) (idx int, expr string, found bool) {
	for i, ln := range lines {
		t := strings.TrimSpace(strings.TrimSuffix(ln, "\r"))
		if t == "" {
			continue // blank line: still in the leading region
		}
		if !strings.HasPrefix(t, "//") {
			return 0, "", false // first non-comment line: region over, no constraint
		}
		if strings.HasPrefix(t, "//go:build") {
			rest := t[len("//go:build"):]
			if rest == "" || rest[0] == ' ' || rest[0] == '\t' {
				return i, strings.TrimSpace(rest), true
			}
		}
	}
	return 0, "", false
}

// fenceText prepends the `//go:build wip_<slug>` fence — plus the blank line Go
// requires between a build constraint and the package clause — to src. Placing the
// constraint as the very first line satisfies Go's rule that it be preceded only
// by blank lines and line comments. Idempotent and stack-safe: a file whose
// leading region ALREADY carries any //go:build line (a previous fence or a real
// platform constraint) is returned unchanged (changed=false), because stacking a
// second constraint would corrupt the file.
func fenceText(src, slug string) (out string, changed bool) {
	if _, _, found := wipLeadingGoBuild(strings.Split(src, "\n")); found {
		return src, false
	}
	return "//go:build wip_" + slug + "\n\n" + src, true
}

// unfenceText removes a `//go:build wip_<slug>` fence from src's leading region —
// exactly that line plus the single immediately-following blank line, the pair
// fenceText added — and returns the recovered slug (the part after "wip_"). It
// refuses to touch a NON-wip constraint: `//go:build linux`, or any compound
// expression, is a real build gate rather than a fence, so it stays (changed=
// false). A file with no //go:build line is returned unchanged. unfence(fence(src))
// round-trips to src byte-identically.
func unfenceText(src string) (out, slug string, changed bool) {
	lines := strings.Split(src, "\n")
	idx, expr, found := wipLeadingGoBuild(lines)
	if !found {
		return src, "", false
	}
	fields := strings.Fields(expr)
	if len(fields) != 1 || !strings.HasPrefix(fields[0], "wip_") {
		return src, "", false // a real (non-wip / compound) constraint: not ours to remove
	}
	slug = strings.TrimPrefix(fields[0], "wip_")
	next := idx + 1
	if next < len(lines) && strings.TrimSpace(strings.TrimSuffix(lines[next], "\r")) == "" {
		next++ // also drop the single blank separator line the fence added
	}
	rest := append(append([]string{}, lines[:idx]...), lines[next:]...)
	return strings.Join(rest, "\n"), slug, true
}

// runWipFence is `fak wip fence`: fence .go file(s) behind //go:build wip_<slug>.
// Exit codes: 0 all ok, 1 a read/write error on some file, 2 usage error.
func runWipFence(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip fence", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	feature := fs.String("feature", "", "override the wip slug (tag becomes wip_<feature>)")
	allUntracked := fs.Bool("all-untracked", false, "fence every currently-untracked, not-already-fenced .go file (bulk rescue)")
	repo := fs.String("C", "", "repo dir (default: the enclosing module root for --all-untracked, cwd-relative paths otherwise)")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}

	var targets []string
	switch {
	case *allUntracked:
		root := *repo
		if root == "" {
			root = repoRoot()
		}
		files, err := buildoverlay.UntrackedGoFiles(root)
		if err != nil {
			fmt.Fprintf(stderr, "fak wip fence: listing untracked files: %v\n", err)
			return 1
		}
		for _, f := range files {
			if strings.HasSuffix(f, ".go") {
				targets = append(targets, filepath.Join(root, filepath.FromSlash(f)))
			}
		}
		if len(targets) == 0 {
			fmt.Fprintln(stdout, "no untracked .go files to fence")
			return 0
		}
	case fs.NArg() == 0:
		fmt.Fprintln(stderr, "fak wip fence: at least one <file> argument is required (or --all-untracked)")
		return 2
	default:
		for _, t := range fs.Args() {
			if !strings.HasSuffix(t, ".go") {
				fmt.Fprintf(stderr, "fak wip fence: %s is not a .go file (only .go files take a build fence)\n", t)
				return 2
			}
			targets = append(targets, wipFenceResolve(*repo, t))
		}
	}

	exit := 0
	for _, target := range targets {
		b, err := os.ReadFile(target)
		if err != nil {
			fmt.Fprintf(stderr, "fak wip fence: %v\n", err)
			exit = 1
			continue
		}
		slug := strings.TrimSpace(*feature)
		if slug == "" {
			slug = wipFenceSlug(target)
		}
		out, changed := fenceText(string(b), slug)
		if !changed {
			fmt.Fprintf(stdout, "already fenced (or has a build constraint): %s\n", target)
			continue
		}
		if err := wipFenceWrite(target, out); err != nil {
			fmt.Fprintf(stderr, "fak wip fence: %v\n", err)
			exit = 1
			continue
		}
		fmt.Fprintf(stdout, "fenced %s -> //go:build wip_%s\n", target, slug)
	}
	return exit
}

// runWipUnfence is `fak wip unfence`: remove a //go:build wip_<slug> fence once
// the defining symbol has landed. Exit codes mirror fence: 0 ok, 1 IO error, 2
// usage error. Removing a fence that is not there is a narrated no-op (exit 0).
func runWipUnfence(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip unfence", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "repo dir to resolve relative targets under (default: cwd)")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak wip unfence: at least one <file> argument is required")
		return 2
	}
	exit := 0
	for _, t := range fs.Args() {
		target := wipFenceResolve(*repo, t)
		b, err := os.ReadFile(target)
		if err != nil {
			fmt.Fprintf(stderr, "fak wip unfence: %v\n", err)
			exit = 1
			continue
		}
		out, slug, changed := unfenceText(string(b))
		if !changed {
			fmt.Fprintf(stdout, "no wip_ fence to remove: %s\n", target)
			continue
		}
		if err := wipFenceWrite(target, out); err != nil {
			fmt.Fprintf(stderr, "fak wip unfence: %v\n", err)
			exit = 1
			continue
		}
		fmt.Fprintf(stdout, "unfenced %s (was wip_%s)\n", target, slug)
	}
	return exit
}

// wipFenceResolve resolves a positional target against -C <repo>: a relative path
// is taken under the repo dir; an absolute path (or no -C) is used as given.
func wipFenceResolve(repo, target string) string {
	if repo == "" || filepath.IsAbs(target) {
		return target
	}
	return filepath.Join(repo, target)
}

// wipFenceWrite writes the transformed source back, preserving the file's existing
// permission bits (0644 only on the unreachable fresh-create path — the file was
// just read).
func wipFenceWrite(name, content string) error {
	perm := os.FileMode(0o644)
	if fi, err := os.Stat(name); err == nil {
		perm = fi.Mode().Perm()
	}
	return os.WriteFile(name, []byte(content), perm)
}
