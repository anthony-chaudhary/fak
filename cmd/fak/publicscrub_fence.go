package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/hooks/alphafence"
)

// fenceMaxFileBytes bounds one fenced or baseline file; larger files are generated artifacts,
// not hand-written workflow text.
const fenceMaxFileBytes = 4 << 20

// fenceSkip records a private input the fence does not cover, so an engineer is never left
// believing a file is protected when it is not.
type fenceSkip struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// runPublicScrubFence builds the alpha fence: salted hashes of the distinctive lines of the
// engineer's private files, written under the root where every public-scrub audit and the
// outbound scan pick it up. The private text never enters the fence or the output.
func runPublicScrubFence(stdout, stderr io.Writer, argv []string) int {
	fl := flag.NewFlagSet("public-scrub fence", flag.ContinueOnError)
	fl.SetOutput(stderr)
	root := fl.String("root", "", "repo root the fence protects (default: git toplevel from cwd)")
	labelPaths := fl.Bool("label-paths", false, "label entries with their file path instead of an opaque id")
	minHits := fl.Int("min-hits", alphafence.DefaultMinHits, "distinct fenced lines one file or payload must reproduce to be refused")
	minLineLen := fl.Int("min-line-len", alphafence.DefaultMinLineLen, "shortest normalized line worth fencing")
	asJSON := fl.Bool("json", false, "emit the build summary as JSON")
	if !parseFlags(fl, argv) {
		return 2
	}
	fail := func(format string, args ...any) int {
		fmt.Fprintf(stderr, "fak public-scrub fence: "+format+"\n", args...)
		return 2
	}
	if fl.NArg() == 0 {
		return fail("pass the private files or directories to fence")
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2
	}
	files, err := fenceSourceFiles(fl.Args())
	if err != nil {
		return fail("%v", err)
	}
	tracked, changed := fenceTrackedState(r)
	for _, path := range files {
		if _, isTracked := tracked[fenceKey(path)]; isTracked {
			return fail("%s is tracked in %s; a fenced file must stay out of the repository it is fenced from", path, r)
		}
	}
	var (
		sources []alphafence.Source
		skipped []fenceSkip
	)
	for i, path := range files {
		body, reason := fenceReadText(path)
		if reason != "" {
			skipped = append(skipped, fenceSkip{Path: path, Reason: reason})
			continue
		}
		label := fmt.Sprintf("alpha-%d", i+1)
		if *labelPaths {
			label = filepath.ToSlash(path)
		}
		sources = append(sources, alphafence.Source{Label: label, Body: body})
	}
	f, err := alphafence.Build(
		alphafence.Options{MinHits: *minHits, MinLineLen: *minLineLen},
		sources, fencePublicBaseline(tracked, changed),
	)
	if err != nil {
		return fail("%v", err)
	}
	for _, s := range skipped {
		fmt.Fprintf(stderr, "fak public-scrub fence: NOT fenced: %s (%s)\n", s.Path, s.Reason)
	}
	if f.Lines() == 0 {
		return fail("no distinctive private lines found; nothing would be protected")
	}
	raw, err := f.Marshal()
	if err != nil {
		return fail("%v", err)
	}
	out := filepath.Join(r, filepath.FromSlash(alphafence.DefaultRel))
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return fail("%v", err)
	}
	if err := os.WriteFile(out, raw, 0o600); err != nil {
		return fail("%v", err)
	}
	ignored := fenceEnsureIgnored(r)
	if *asJSON {
		if skipped == nil {
			skipped = []fenceSkip{}
		}
		if err := writeIndentedJSON(stdout, map[string]any{
			"fence": out, "files": len(f.Entries), "lines": f.Lines(), "min_hits": f.MinHits,
			"min_line_len": f.MinLineLen, "skipped": skipped, "ignored": ignored,
		}); err != nil {
			return fail("%v", err)
		}
		return 0
	}
	fmt.Fprintf(stdout, "alpha fence: %d line(s) from %d file(s) fenced -> %s\n", f.Lines(), len(f.Entries), out)
	if len(skipped) > 0 {
		fmt.Fprintf(stdout, "%d input file(s) were NOT fenced; see the reasons above\n", len(skipped))
	}
	if !ignored {
		fmt.Fprintln(stdout, "WARNING: could not confirm git ignores the fence; never commit it (the audits refuse a tracked fence)")
	}
	fmt.Fprintln(stdout, "the fence holds salted hashes only and stays local: a holder can test a guessed line against it")
	return 0
}

// fenceSourceFiles expands the arguments to a sorted, de-duplicated file list. Symlinked files
// are followed, since linked skill directories are a common private layout.
func fenceSourceFiles(args []string) ([]string, error) {
	seen := map[string]bool{}
	var files []string
	add := func(path string) {
		if key := fenceKey(path); !seen[key] {
			seen[key] = true
			files = append(files, path)
		}
	}
	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			add(arg)
			continue
		}
		err = filepath.WalkDir(arg, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if d.IsDir() {
				if d.Name() == ".git" {
					return filepath.SkipDir
				}
				return nil
			}
			if target, statErr := os.Stat(path); statErr != nil || target.Mode().IsRegular() {
				add(path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(files)
	return files, nil
}

// fenceKey identifies a file for set membership. Windows paths compare case-insensitively.
func fenceKey(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	abs = filepath.Clean(abs)
	if runtime.GOOS == "windows" {
		abs = strings.ToLower(abs)
	}
	return abs
}

// fenceReadText returns a file's bytes, or the reason it cannot be fenced.
func fenceReadText(path string) ([]byte, string) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, "unreadable"
	}
	if info.Size() > fenceMaxFileBytes {
		return nil, "larger than 4 MiB"
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, "unreadable"
	}
	if bytes.IndexByte(body, 0) >= 0 {
		return nil, "binary or UTF-16 text; save it as UTF-8"
	}
	return body, ""
}

// fenceTrackedState returns the root's tracked files and, of those, the ones that differ from
// the last commit, both keyed by fenceKey. A root that is not a git repository has neither.
func fenceTrackedState(root string) (tracked map[string]string, changed map[string]bool) {
	tracked, changed = map[string]string{}, map[string]bool{}
	tree, err := hooks.ReadTrackedTree(root)
	if err != nil {
		return tracked, changed
	}
	for _, rel := range tree.Paths {
		path := filepath.Join(root, filepath.FromSlash(rel))
		tracked[fenceKey(path)] = path
	}
	out, err := exec.Command("git", "-C", root, "diff", "--name-only", "-z", "HEAD").Output()
	if err != nil {
		// No readable last commit: nothing is known to be published yet.
		for key := range tracked {
			changed[key] = true
		}
		return tracked, changed
	}
	for rel := range strings.SplitSeq(string(out), "\x00") {
		if rel != "" {
			changed[fenceKey(filepath.Join(root, filepath.FromSlash(rel)))] = true
		}
	}
	return tracked, changed
}

// fencePublicBaseline reads the tracked files that match the last commit. Lines found there are
// already published and are left out of the fence. Files with uncommitted edits are not
// trusted as public: a private line pasted in moments ago must still be fenced.
func fencePublicBaseline(tracked map[string]string, changed map[string]bool) [][]byte {
	var public [][]byte
	for key, path := range tracked {
		if changed[key] {
			continue
		}
		if body, reason := fenceReadText(path); reason == "" {
			public = append(public, body)
		}
	}
	return public
}

// fenceEnsureIgnored keeps the fence out of version control: when git does not already ignore
// it, the fence path is added to the repository's local exclude file. It reports whether git
// ignores the fence afterwards.
func fenceEnsureIgnored(root string) bool {
	isIgnored := func() bool {
		return exec.Command("git", "-C", root, "check-ignore", "-q", alphafence.DefaultRel).Run() == nil
	}
	if isIgnored() {
		return true
	}
	out, err := exec.Command("git", "-C", root, "rev-parse", "--git-path", "info/exclude").Output()
	if err != nil {
		return false
	}
	exclude := strings.TrimSpace(string(out))
	if !filepath.IsAbs(exclude) {
		exclude = filepath.Join(root, exclude)
	}
	if err := os.MkdirAll(filepath.Dir(exclude), 0o755); err != nil {
		return false
	}
	file, err := os.OpenFile(exclude, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return false
	}
	_, writeErr := fmt.Fprintf(file, "\n/%s\n", alphafence.DefaultRel)
	if closeErr := file.Close(); writeErr != nil || closeErr != nil {
		return false
	}
	return isIgnored()
}
