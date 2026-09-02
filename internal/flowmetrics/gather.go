package flowmetrics

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// The impure shell. Everything here shells to git or gh; the fold it feeds is
// pure and lives in the other files, so the whole report is testable from
// fixtures without a network or a repository.

// gitFieldSep and gitRecSep are ASCII unit/record separators, chosen because no
// commit subject or body can contain them. Splitting on newlines instead would
// break on every multi-line body, and splitting on a printable sentinel would
// break the day someone quotes it.
const (
	gitFieldSep = "\x1f"
	gitRecSep   = "\x1e"
)

// HeadRev returns the pinned HEAD sha. Every gather should record it: under a
// live fleet the tip moves mid-analysis, and a report that cannot name its rev
// is not reproducible.
func HeadRev(ctx context.Context, root string) (string, error) {
	out, err := runIn(ctx, root, "git", "rev-parse", "HEAD")
	return strings.TrimSpace(out), err
}

// GatherCommits reads the commit facts from git log.
//
// It uses %ct (committer time), not %at: the two differ on ~2% of this history
// and %ct is the instant the serialized trunk land actually stamped, which is
// what a flow measurement is about. Merges are excluded because a merge commit
// carries no authored change and would double-count its side's issue refs.
//
// limit <= 0 reads the whole history.
func GatherCommits(ctx context.Context, root string, limit int) ([]Commit, error) {
	args := []string{"log", "--no-merges",
		"--pretty=format:%H" + gitFieldSep + "%ct" + gitFieldSep + "%s" + gitFieldSep + "%B" + gitRecSep}
	if limit > 0 {
		args = append(args, "-n", strconv.Itoa(limit))
	}
	out, err := runIn(ctx, root, "git", args...)
	if err != nil {
		return nil, fmt.Errorf("git log: %w", err)
	}
	var commits []Commit
	for _, rec := range strings.Split(out, gitRecSep) {
		rec = strings.TrimLeft(rec, "\r\n")
		if strings.TrimSpace(rec) == "" {
			continue
		}
		f := strings.SplitN(rec, gitFieldSep, 4)
		if len(f) < 4 {
			continue
		}
		secs, err := strconv.ParseInt(strings.TrimSpace(f[1]), 10, 64)
		if err != nil {
			continue
		}
		subject := f[2]
		c := Commit{
			SHA:     strings.TrimSpace(f[0]),
			When:    time.Unix(secs, 0).UTC(),
			Subject: subject,
			Leaf:    ParseLeaf(subject),
			Issues:  ParseCommitRefs(subject, f[3]),
		}
		commits = append(commits, c)
	}
	return commits, nil
}

// ghIssue mirrors the `gh issue list --json` wire shape, where labels arrive as
// objects rather than strings.
type ghIssue struct {
	Number    int        `json:"number"`
	Title     string     `json:"title"`
	CreatedAt time.Time  `json:"createdAt"`
	ClosedAt  *time.Time `json:"closedAt"`
	Body      string     `json:"body"`
	Labels    []struct {
		Name string `json:"name"`
	} `json:"labels"`
}

func (g ghIssue) toIssue() Issue {
	iss := Issue{
		Number:    g.Number,
		Title:     g.Title,
		CreatedAt: g.CreatedAt,
		Body:      g.Body,
	}
	// gh emits a zero closedAt for open issues in some versions; treat that
	// as open rather than as a close at the zero time.
	if g.ClosedAt != nil && !g.ClosedAt.IsZero() {
		c := *g.ClosedAt
		iss.ClosedAt = &c
	}
	for _, l := range g.Labels {
		if l.Name != "" {
			iss.Labels = append(iss.Labels, l.Name)
		}
	}
	return iss
}

// DecodeIssues parses a `gh issue list --json ...` array. Split out from the
// fetch so a saved dump can be replayed offline and so tests need no gh.
func DecodeIssues(raw []byte) ([]Issue, error) {
	var wire []ghIssue
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("decode issues: %w", err)
	}
	out := make([]Issue, 0, len(wire))
	for _, w := range wire {
		if w.Number <= 0 {
			continue
		}
		out = append(out, w.toIssue())
	}
	return out, nil
}

// IssueJSONFields is the exact --json field list this package needs.
const IssueJSONFields = "number,title,createdAt,closedAt,labels,body"

// GatherIssues fetches every issue in the given state via gh.
//
// It deliberately does NOT use `gh issue list --search`: the search path routes
// through the GitHub Search API, which silently truncates at 1000 results. On a
// repo with thousands of issues that cap produces a plausible-looking but wrong
// backlog, with no error. The plain list path paginates over GraphQL and is
// exact, so date filtering happens locally in the fold instead.
//
// state is "open", "closed", or "all". limit <= 0 uses a high bound.
func GatherIssues(ctx context.Context, root, state string, limit int) ([]Issue, error) {
	if state == "" {
		state = "all"
	}
	if limit <= 0 {
		limit = 100000
	}
	out, err := runIn(ctx, root, "gh", "issue", "list",
		"--state", state, "--limit", strconv.Itoa(limit), "--json", IssueJSONFields)
	if err != nil {
		return nil, fmt.Errorf("gh issue list: %w", err)
	}
	return DecodeIssues([]byte(out))
}

// LoadIssuesFile reads a saved gh dump, for offline and deterministic runs.
func LoadIssuesFile(path string) ([]Issue, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeIssues(raw)
}

// GatherTree censuses uncommitted work. It reads git status and file mtimes
// only; buildability is a separate opt-in probe because compiling a large module
// costs minutes and most callers want the cheap census.
func GatherTree(ctx context.Context, root string, now time.Time) (TreeWIP, error) {
	t := TreeWIP{}
	rev, err := HeadRev(ctx, root)
	if err != nil {
		return t, err
	}
	t.Rev = rev

	// git status emits paths relative to the repository TOPLEVEL, not to the
	// directory git ran in, so mtimes must be resolved against the toplevel or
	// every stat fails when root is a subdirectory. That failure is silent by
	// nature — the two mtime-derived fields just stay 0, which is exactly what
	// a pristine tree looks like — so resolve it rather than trusting root.
	//
	// --show-cdup, not --show-toplevel: cdup is a RELATIVE hop ("../../"), so
	// the join stays inside whatever path namespace the caller handed us.
	// --show-toplevel returns git's own canonicalized absolute path, which on
	// Windows does not always match the caller's spelling of the same directory
	// (a junctioned %TEMP% is the common case), and a mismatch there reproduces
	// the exact silent-zero bug this resolution exists to prevent.
	base := root
	if cdup, cdErr := runIn(ctx, root, "git", "rev-parse", "--show-cdup"); cdErr == nil {
		base = filepath.Join(root, strings.TrimSpace(cdup))
	}

	// -z keeps paths with spaces or non-ASCII intact; the porcelain v1 format
	// is stable across git versions in a way the human format is not.
	out, err := runIn(ctx, root, "git", "status", "--porcelain", "-z")
	if err != nil {
		return t, fmt.Errorf("git status: %w", err)
	}
	cut := now.Add(-RecentWriterWindowMinutes * time.Minute)
	type recentFile struct {
		path string
		when time.Time
	}
	var (
		dirtyPaths []string
		recent     []recentFile
	)
	for _, ent := range strings.Split(out, "\x00") {
		if len(ent) < 4 {
			continue
		}
		code, path := ent[:2], strings.TrimSpace(ent[3:])
		if path == "" || !isSource(path) {
			continue
		}
		path = cleanRepoPath(path)
		dirtyPaths = append(dirtyPaths, path)
		untracked := code == "??"
		if untracked {
			t.UntrackedGo++
		} else {
			t.ModifiedGo++
		}
		name := filepath.Base(path)
		if isScratchName(name) {
			t.ScratchLitter++
		}
		if strings.HasPrefix(name, ".") {
			t.HiddenGo++
		}
		info, statErr := os.Stat(filepath.Join(base, path))
		if statErr != nil {
			t.StatFailures++
			continue
		}
		if untracked {
			if age := hours(info.ModTime(), now); age > t.OldestUntrackedHours {
				t.OldestUntrackedHours = age
			}
		}
		if info.ModTime().After(cut) {
			t.RecentWriters++
			recent = append(recent, recentFile{path: path, when: info.ModTime()})
		}
	}
	sort.Slice(recent, func(i, j int) bool {
		if recent[i].when.Equal(recent[j].when) {
			return recent[i].path < recent[j].path
		}
		return recent[i].when.After(recent[j].when)
	})
	for i := 0; i < len(recent) && i < RecentWriterPathLimit; i++ {
		t.RecentWriterPaths = append(t.RecentWriterPaths, recent[i].path)
	}
	for _, file := range recent {
		t.recentWriterPaths = append(t.recentWriterPaths, file.path)
	}
	t.DuplicateSymbols = gatherDuplicateSymbols(base, dirtyPaths)

	// Uncommitted churn, whitespace-insensitive so a line-ending rewrite is
	// not counted as work.
	if ns, nsErr := runIn(ctx, root, "git", "diff", "-w", "--numstat"); nsErr == nil {
		for _, line := range strings.Split(ns, "\n") {
			f := strings.Fields(strings.TrimSpace(line))
			if len(f) < 3 {
				continue
			}
			// A binary file reports "-" for both counts; skip it.
			a, aerr := strconv.Atoi(f[0])
			d, derr := strconv.Atoi(f[1])
			if aerr != nil || derr != nil {
				continue
			}
			t.AddedLines += a
			t.DeletedLines += d
		}
	}
	t.Measured = true
	return t, nil
}

// gatherDuplicateSymbols finds package-level declarations that cannot coexist
// in the current build context. Methods, init functions, and files excluded by
// build constraints are skipped because Go permits their names to repeat.
func gatherDuplicateSymbols(base string, paths []string) []DuplicateSymbol {
	type key struct {
		dir, pkg, symbol string
	}
	seen := make(map[key]map[string]struct{})
	for _, path := range paths {
		matched, matchErr := build.Default.MatchFile(filepath.Join(base, filepath.Dir(path)), filepath.Base(path))
		if matchErr != nil || !matched {
			continue
		}
		file, _ := parser.ParseFile(token.NewFileSet(), filepath.Join(base, path), nil, parser.SkipObjectResolution)
		if file == nil || file.Name == nil {
			continue
		}
		for _, symbol := range packageSymbols(file) {
			k := key{dir: filepath.Dir(path), pkg: file.Name.Name, symbol: symbol}
			if seen[k] == nil {
				seen[k] = make(map[string]struct{})
			}
			seen[k][path] = struct{}{}
		}
	}
	var duplicates []DuplicateSymbol
	for k, pathSet := range seen {
		if len(pathSet) < 2 {
			continue
		}
		paths := make([]string, 0, len(pathSet))
		for path := range pathSet {
			paths = append(paths, path)
		}
		sort.Strings(paths)
		duplicates = append(duplicates, DuplicateSymbol{Package: k.pkg, Symbol: k.symbol, Paths: paths})
	}
	sort.Slice(duplicates, func(i, j int) bool {
		if duplicates[i].Package != duplicates[j].Package {
			return duplicates[i].Package < duplicates[j].Package
		}
		if duplicates[i].Symbol != duplicates[j].Symbol {
			return duplicates[i].Symbol < duplicates[j].Symbol
		}
		return strings.Join(duplicates[i].Paths, "\x00") < strings.Join(duplicates[j].Paths, "\x00")
	})
	return duplicates
}

func packageSymbols(file *ast.File) []string {
	var symbols []string
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil && d.Name != nil && d.Name.Name != "init" {
				symbols = append(symbols, d.Name.Name)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Name != nil && s.Name.Name != "_" {
						symbols = append(symbols, s.Name.Name)
					}
				case *ast.ValueSpec:
					for _, name := range s.Names {
						if name.Name != "_" {
							symbols = append(symbols, name.Name)
						}
					}
				}
			}
		}
	}
	return symbols
}

// ProbeBuild runs `go build ./...` and records whether the shared tree compiles.
// Opt-in: it is the expensive part of the census.
//
// A caller must treat a failure under a live fleet with care. When RecentWriters
// is high the tree is being written mid-compile, so a red result may name a peer
// session's half-applied edit rather than anything the caller did; that is
// exactly why the concurrency count is gathered alongside it.
func ProbeBuild(ctx context.Context, root string, t *TreeWIP) {
	t.BuildProbed = true
	out, err := runIn(ctx, root, "go", "build", "./...")
	if err == nil {
		t.Buildable = true
		t.BuildError = ""
		return
	}
	t.Buildable = false
	if msg := strings.TrimSpace(out); msg != "" {
		t.BuildError = msg
	} else {
		t.BuildError = err.Error()
	}
}

// isSource reports whether a path is a Go source file, the population this
// census is about. Dot-prefixed files count: the go tool ignores them, which is
// precisely what makes them invisible WIP.
func isSource(path string) bool { return strings.HasSuffix(path, ".go") }

// isScratchName reports whether a base name looks like a throwaway probe rather
// than intended work: the `zz`-prefixed convention this repo's sessions use for
// refutation probes, or a dot-prefixed source file.
func isScratchName(base string) bool {
	if strings.HasPrefix(base, ".") {
		return true
	}
	if !strings.HasPrefix(base, "zz") {
		return false
	}
	// Require a separator or digit after the prefix so a legitimate file
	// like "zzip.go" is not swept up as litter.
	rest := base[2:]
	return rest != "" && (rest[0] == '_' || rest[0] == '-' || (rest[0] >= '0' && rest[0] <= '9'))
}

// runIn runs a command in a directory and returns its combined output. Combined
// rather than stdout-only because git and go report the diagnostics that matter
// on stderr, and a caller handed only stdout would see an empty failure.
//
// The program is resolved through a switch of string literals (not passed
// straight to exec) so the architest interpreter-free gate can prove every exec
// target is a compiled binary; any other selector is refused, which keeps the
// set fail-closed if a future caller typos a program name.
func runIn(ctx context.Context, dir string, name string, args ...string) (string, error) {
	var cmd *exec.Cmd
	switch name {
	case "git":
		cmd = exec.CommandContext(ctx, "git", args...)
		windowgate.ConfigureBackgroundCommand(cmd)
	case "gh":
		cmd = exec.CommandContext(ctx, "gh", args...)
		windowgate.ConfigureBackgroundCommand(cmd)
	case "go":
		cmd = exec.CommandContext(ctx, "go", args...)
		windowgate.ConfigureBackgroundCommand(cmd)
	default:
		return "", fmt.Errorf("flowmetrics runIn: unsupported program %q (allowed: git, gh, go)", name)
	}
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err
}
