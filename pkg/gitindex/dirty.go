package gitindex

import (
	"bytes"
	"context"
	"crypto/sha1"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// Mode selects how ModifiedTrackedWith resolves what the index alone cannot.
type Mode int

const (
	// FallbackAuto (default): decide in-process; shell out to git only for
	// "maybe dirty" paths (eol/filter conversion could apply, exec-bit
	// drift, populated submodules) and for indexes this parser does not
	// support (split index, SHA-256).
	FallbackAuto Mode = iota
	// FallbackNone: never spawn git. "Maybe" paths are reported as modified
	// (a conservative superset); an unsupported index is an error.
	FallbackNone
	// FallbackGit: skip the parser and ask git for everything.
	FallbackGit
)

// Options tunes ModifiedTrackedWith.
type Options struct {
	Mode Mode
	// Workers bounds concurrent lstat/hash work; 0 means 4*GOMAXPROCS.
	Workers int
	// Git is the git executable; empty means "git".
	Git string
}

// Result is a dirty-detection verdict plus cost counters.
type Result struct {
	// Modified is the sorted, de-duplicated list of tracked paths (slash
	// separated, relative to the worktree root) whose working-tree content
	// differs from the index: modified, deleted, type-changed, unmerged.
	Modified []string
	// Maybe lists paths the parser could not decide (before fallback).
	Maybe   []string
	Entries int  // index entries examined
	Hashed  int  // entries whose content was hashed
	UsedGit bool // a git subprocess was consulted
}

// ModifiedTracked returns tracked paths whose working-tree content differs
// from the index of the worktree rooted at repoRoot (FallbackAuto mode).
// Untracked files are out of scope and never reported.
func ModifiedTracked(repoRoot string) ([]string, error) {
	r, err := ModifiedTrackedWith(repoRoot, Options{})
	if err != nil {
		return nil, err
	}
	return r.Modified, nil
}

// ModifiedTrackedWith is ModifiedTracked with explicit options and counters.
func ModifiedTrackedWith(repoRoot string, opt Options) (Result, error) {
	if opt.Git == "" {
		opt.Git = "git"
	}
	if opt.Mode == FallbackGit {
		mod, err := gitModified(opt.Git, repoRoot, nil)
		return Result{Modified: mod, UsedGit: true}, err
	}
	gitDir, commonDir, err := repoDirs(repoRoot)
	if err != nil {
		return Result{}, err
	}
	if sha256Repo(commonDir) {
		return unsupported(repoRoot, opt, fmt.Errorf("%w: sha256 object format", ErrUnsupported))
	}
	idx, err := ReadIndexFile(filepath.Join(gitDir, "index"))
	if err != nil {
		if errors.Is(err, ErrUnsupported) {
			return unsupported(repoRoot, opt, err)
		}
		if errors.Is(err, fs.ErrNotExist) {
			return Result{}, nil // fresh repo: nothing tracked
		}
		return Result{}, err
	}
	res := Result{Entries: len(idx.Entries)}
	risk := newRiskOracle(repoRoot, commonDir, idx)
	verdicts := make([]verdict, len(idx.Entries))
	hashed := make([]bool, len(idx.Entries))

	workers := opt.Workers
	if workers <= 0 {
		workers = 4 * runtime.GOMAXPROCS(0)
	}
	stats := statDirs(repoRoot, idx, workers)
	parallel(len(idx.Entries), workers, func(i int) {
		verdicts[i], hashed[i] = checkEntry(repoRoot, idx, &idx.Entries[i], risk, stats)
	})

	seen := map[string]bool{}
	for i, v := range verdicts {
		if hashed[i] {
			res.Hashed++
		}
		p := idx.Entries[i].Path
		switch v {
		case dirty:
			if !seen[p] {
				seen[p] = true
				res.Modified = append(res.Modified, p)
			}
		case maybe:
			res.Maybe = append(res.Maybe, p)
		}
	}
	if len(res.Maybe) > 0 {
		if opt.Mode == FallbackNone {
			for _, p := range res.Maybe {
				if !seen[p] {
					seen[p] = true
					res.Modified = append(res.Modified, p)
				}
			}
		} else {
			mod, err := gitModified(opt.Git, repoRoot, res.Maybe)
			if err != nil {
				return res, err
			}
			res.UsedGit = true
			for _, p := range mod {
				if !seen[p] {
					seen[p] = true
					res.Modified = append(res.Modified, p)
				}
			}
		}
	}
	sort.Strings(res.Modified)
	return res, nil
}

func unsupported(root string, opt Options, cause error) (Result, error) {
	if opt.Mode == FallbackNone {
		return Result{}, cause
	}
	mod, err := gitModified(opt.Git, root, nil)
	return Result{Modified: mod, UsedGit: true}, err
}

type verdict uint8

const (
	clean verdict = iota
	dirty
	maybe
)

// checkEntry decides one entry; the bool reports whether content was hashed.
func checkEntry(root string, idx *Index, e *Entry, risk *riskOracle, stats dirStats) (verdict, bool) {
	if e.Stage != 0 {
		return dirty, false // unmerged: diff-files lists every conflicted path
	}
	// assume-unchanged (CE_VALID) and skip-worktree entries are, by git's
	// contract, never compared against the working tree.
	if e.AssumeValid || e.SkipWorktree {
		return clean, false
	}
	full := filepath.Join(root, filepath.FromSlash(e.Path))
	st, err := stats.lstat(e.Path, full)
	if err != nil {
		if e.IntentToAdd {
			return clean, false // ita + missing: nothing to compare
		}
		return dirty, false // deleted (or unreadable path component)
	}
	if e.IntentToAdd {
		// git diff-files reports an intent-to-add entry whose file exists as
		// a new file (its index oid is the empty blob placeholder).
		return dirty, false
	}
	typ := e.Mode & modeTypeMask
	if typ == modeGitlink {
		if !st.IsDir() {
			return dirty, false
		}
		if _, err := os.Lstat(filepath.Join(full, ".git")); err != nil {
			return clean, false // unpopulated submodule
		}
		return maybe, false // submodule HEAD/dirtiness: ask git
	}
	isLink := st.Mode()&fs.ModeSymlink != 0
	if st.IsDir() || (!st.Mode().IsRegular() && !isLink) {
		return dirty, false // typechange to directory/special file
	}
	if typ == modeRegular && isLink {
		return dirty, false // regular -> symlink typechange
	}
	execDrift := runtime.GOOS != "windows" && typ == modeRegular &&
		(e.Mode&0100 != 0) != (st.Mode().Perm()&0100 != 0)

	if statMatches(e, st) && !racy(idx, e) {
		if execDrift && risk.fileMode() {
			return maybe, false
		}
		return clean, false
	}
	// git's ie_modified: a size change on an entry whose recorded size is
	// non-zero is DATA_CHANGED without reading content (even if eol
	// conversion would make the bytes equal). Only same-size or smudged
	// (size 0) entries are content-compared.
	if e.Size != 0 && uint32(st.Size()) != e.Size {
		return dirty, false
	}
	// Stat mismatch or racily clean: compare content.
	var data []byte
	if isLink {
		t, err := os.Readlink(full)
		if err != nil {
			return dirty, false
		}
		data = []byte(filepath.ToSlash(t))
	} else {
		data, err = os.ReadFile(full)
		if err != nil {
			return dirty, true
		}
	}
	oid := blobOID(data)
	hasCR := bytes.IndexByte(data, '\r') >= 0
	crRisk, contentRisk := risk.attributes()
	if oid == e.OID {
		if execDrift && risk.fileMode() {
			return maybe, true
		}
		// Raw match is clean unless a text/eol attribute could renormalise
		// CRLF content that the index stored verbatim.
		if hasCR && crRisk {
			return maybe, true
		}
		return clean, true
	}
	if typ == modeSymlink || isLink {
		return dirty, true
	}
	if contentRisk {
		return maybe, true // a clean filter / ident / encoding rule could apply
	}
	if !hasCR {
		return dirty, true // autocrlf/eol conversion only rewrites CR bytes
	}
	norm := bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	if blobOID(norm) == e.OID {
		return maybe, true // clean iff core.autocrlf/eol converts on input
	}
	return dirty, true
}

// statMatches mirrors git's match_stat_data for the fields that matter on
// every platform: mtime and size. ctime/ino/dev/uid/gid are ignored (Git for
// Windows does not populate them meaningfully; ignoring them only makes the
// check stricter, never laxer, because a mismatch merely forces a hash).
func statMatches(e *Entry, st fs.FileInfo) bool {
	if uint32(st.Size()) != e.Size {
		return false
	}
	mt := st.ModTime()
	if uint32(mt.Unix()) != e.MtimeSec {
		return false
	}
	if e.MtimeNsec != 0 && uint32(mt.Nanosecond()) != e.MtimeNsec {
		return false
	}
	return true
}

// racy reports git's "racily clean" condition: the entry was written in the
// same timestamp granule as (or after) the index itself, so a same-size edit
// in that granule would be invisible to stat.
func racy(idx *Index, e *Entry) bool {
	// Seconds-only comparison: a superset of git's nanosecond rule (so we
	// hash at least every entry git would), and exact when git is built
	// without USE_NSEC.
	return idx.Mtime.IsZero() || uint32(idx.Mtime.Unix()) <= e.MtimeSec
}

func blobOID(data []byte) [hashLen]byte {
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(data))
	h.Write(data)
	var o [hashLen]byte
	copy(o[:], h.Sum(nil))
	return o
}

// riskOracle lazily answers "could a conversion apply?" conservatively,
// without spawning git. Two levels: crRisk (text/eol/crlf attributes — they
// only rewrite CR bytes) and contentRisk (filter/ident/working-tree-encoding,
// macro definitions, includes, unreadable sources — anything could change).
type riskOracle struct {
	root, commonDir string
	idx             *Index
	attrOnce        sync.Once
	crRisk          bool
	contentRisk     bool
	modeOnce        sync.Once
	mode            bool
}

func newRiskOracle(root, commonDir string, idx *Index) *riskOracle {
	return &riskOracle{root: root, commonDir: commonDir, idx: idx}
}

// attributes scans every gitattributes source we can locate: tracked
// .gitattributes files, $GIT_COMMON_DIR/info/attributes, the XDG/global
// attributes file, and any config naming core.attributesFile or [include].
func (r *riskOracle) attributes() (crRisk, contentRisk bool) {
	r.attrOnce.Do(func() {
		var paths []string
		for _, e := range r.idx.Entries {
			if e.Path == ".gitattributes" || strings.HasSuffix(e.Path, "/.gitattributes") {
				paths = append(paths, filepath.Join(r.root, filepath.FromSlash(e.Path)))
			}
		}
		paths = append(paths, filepath.Join(r.commonDir, "info", "attributes"))
		if x := os.Getenv("XDG_CONFIG_HOME"); x != "" {
			paths = append(paths, filepath.Join(x, "git", "attributes"))
		}
		if h, err := os.UserHomeDir(); err == nil {
			paths = append(paths, filepath.Join(h, ".config", "git", "attributes"))
		}
		for _, p := range paths {
			b, err := os.ReadFile(p)
			if err != nil {
				if !errors.Is(err, fs.ErrNotExist) {
					r.contentRisk = true
				}
				continue
			}
			cr, content := scanAttributes(b)
			r.crRisk = r.crRisk || cr
			r.contentRisk = r.contentRisk || content
		}
		for _, cfg := range configTexts(r.commonDir) {
			if strings.Contains(cfg, "attributesfile") || strings.Contains(cfg, "[include") {
				r.contentRisk = true
			}
		}
	})
	return r.crRisk, r.contentRisk
}

// scanAttributes classifies one gitattributes file. Negated/unset forms
// (-text, !eol) and diff/merge/whitespace-only attributes carry no risk.
func scanAttributes(b []byte) (crRisk, contentRisk bool) {
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line[0] == '#' {
			continue
		}
		f := strings.Fields(line)
		if strings.HasPrefix(f[0], "[attr]") {
			contentRisk = true // macro could expand to anything
			continue
		}
		for _, a := range f[1:] {
			if a[0] == '-' || a[0] == '!' {
				continue
			}
			name, _, _ := strings.Cut(a, "=")
			switch name {
			case "text", "eol", "crlf":
				crRisk = true
			case "filter", "ident", "working-tree-encoding":
				contentRisk = true
			case "binary", "diff", "merge", "whitespace", "export-ignore", "export-subst",
				"delta", "linguist-generated", "linguist-vendored", "linguist-documentation",
				"linguist-language", "lockable":
			default:
				// Unknown attribute names may be macros defined elsewhere.
				contentRisk = true
			}
		}
	}
	return crRisk, contentRisk
}

// fileMode is true unless the repo config explicitly sets filemode = false.
func (r *riskOracle) fileMode() bool {
	r.modeOnce.Do(func() {
		r.mode = true
		b, err := os.ReadFile(filepath.Join(r.commonDir, "config"))
		if err != nil {
			return
		}
		s := strings.ReplaceAll(strings.ToLower(string(b)), " ", "")
		if strings.Contains(s, "filemode=false") {
			r.mode = false
		}
	})
	return r.mode
}

// configTexts returns the lower-cased text of every config file git would
// read that we can locate (repo, worktree, global, XDG, system candidates).
func configTexts(commonDir string) []string {
	paths := []string{
		filepath.Join(commonDir, "config"),
		filepath.Join(commonDir, "config.worktree"),
	}
	if p := os.Getenv("GIT_CONFIG_GLOBAL"); p != "" {
		paths = append(paths, p)
	}
	if h, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(h, ".gitconfig"), filepath.Join(h, ".config", "git", "config"))
	}
	if p := os.Getenv("GIT_CONFIG_SYSTEM"); p != "" {
		paths = append(paths, p)
	}
	if os.Getenv("GIT_CONFIG_NOSYSTEM") == "" {
		paths = append(paths, "/etc/gitconfig")
	}
	if runtime.GOOS == "windows" && os.Getenv("GIT_CONFIG_NOSYSTEM") == "" {
		for _, env := range []string{"ProgramFiles", "ProgramW6432"} {
			if d := os.Getenv(env); d != "" {
				paths = append(paths, filepath.Join(d, "Git", "etc", "gitconfig"))
			}
		}
		if d := os.Getenv("ProgramData"); d != "" {
			paths = append(paths, filepath.Join(d, "Git", "config"))
		}
	}
	var out []string
	for _, p := range paths {
		if b, err := os.ReadFile(p); err == nil {
			out = append(out, strings.ToLower(string(b)))
		}
	}
	return out
}

func sha256Repo(commonDir string) bool {
	b, err := os.ReadFile(filepath.Join(commonDir, "config"))
	if err != nil {
		return false
	}
	s := strings.ReplaceAll(strings.ToLower(string(b)), " ", "")
	return strings.Contains(s, "objectformat=sha256")
}

// gitModified asks git which of paths (nil = all tracked) differ from the
// index, using `git status --porcelain -z --untracked-files=no` and keeping
// entries with a non-blank worktree column. status has exactly the
// refresh+diff-files semantics (stat refresh, size-change shortcut, eol and
// filter conversion on same-size content compares) but, under
// GIT_OPTIONAL_LOCKS=0, never rewrites the index — unlike
// `git update-index --refresh && git diff-files`, and unlike porcelain
// `git diff`, which drops entries whose converted content is unchanged.
func gitModified(git, root string, paths []string) ([]string, error) {
	base := []string{"--literal-pathspecs", "-C", root, "status", "--porcelain=v1", "-z",
		"--untracked-files=no", "--ignore-submodules=none", "--no-renames", "--"}
	if len(paths) == 0 {
		return runGitStatus(git, base)
	}
	var out []string
	const maxArgBytes = 24 << 10 // stay under Windows' 32K command line
	for len(paths) > 0 {
		args := append([]string(nil), base...)
		n := 0
		for _, p := range paths {
			if n > 0 && n+len(p)+3 > maxArgBytes {
				break
			}
			args = append(args, p)
			n += len(p) + 3
		}
		paths = paths[len(args)-len(base):]
		got, err := runGitStatus(git, args)
		if err != nil {
			return nil, err
		}
		out = append(out, got...)
	}
	return out, nil
}

func runGitStatus(git string, args []string) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, git, args...)
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	b, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("gitindex: git %s: %v: %s", strings.Join(args[:4], " "), err, strings.TrimSpace(stderr.String()))
	}
	var out []string
	for _, rec := range bytes.Split(b, []byte{0}) {
		// "XY path"; Y is the worktree-vs-index column.
		if len(rec) < 4 || rec[1] == ' ' {
			continue
		}
		out = append(out, string(rec[3:]))
	}
	sort.Strings(out)
	return out, nil
}
