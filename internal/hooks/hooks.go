// Package hooks runs the repo's commit-boundary gates IN ONE PROCESS.
//
// The git hooks (tools/githooks/pre-commit, commit-msg) historically spawned one Python
// interpreter PER gate — 7 for pre-commit + 1 for commit-msg = 8 cold starts, ~2s each on
// a Windows box (process create + Defender scan), so a single `git commit` paid ~12-16s of
// pure interpreter-spawn tax before any checking happened. None of the gates does real work:
// each is regex/substring/os.Stat over `git diff --cached`, sub-millisecond once the
// interpreter is up. This package collapses all 8 gates into one Go process that reads the
// staged diff ONCE and runs every gate over it — the whole measured cost was spawn overhead,
// so a single static-binary start recovers essentially all of it.
//
// Each gate is a byte-faithful port of its tools/check_*.py / scrub_public_copy.py oracle;
// a `parity_test.go` differential harness asserts identical verdicts against the Python
// checkers (kept on disk as the fallback when no `fak` binary resolves, and as the oracle).
// The exit contract every pre-commit gate honors: clean / violation / could-not-run, where
// could-not-run NEVER blocks (fail-open) — a broken check must not wedge every commit.
package hooks

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Runner executes a git subcommand in dir and returns (stdout, exitCode, err). Same contract
// as witness.Runner / safecommit.Runner: err is non-nil ONLY when git could not be EXECUTED;
// a non-zero exit with git present is reported via code. Injectable so the diff reader and
// the gates run over canned evidence in tests with no real git.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// ErrCouldNotRun is the sentinel a gate returns when it cannot reach the evidence it needs
// (git unavailable, a required read failed). The CLI maps it to exit 2 — fail-open, never a
// block — mirroring the Python `run_gate` "status != 1 => skipped" rule.
var ErrCouldNotRun = errors.New("hooks: gate could not run")

// AddedLine is one line added by the staged diff, carrying its new-file line number so a gate
// can cite file:line exactly the way the Python checkers do (parsed from the @@ hunk header).
type AddedLine struct {
	File string
	New  int // 1-based line number in the new file; 0 if the diff gave no hunk header
	Text string
}

// Finding is one gate violation. A gate returns zero findings for a clean staged set.
type Finding struct {
	Gate   string `json:"gate"`   // PUBLIC_LEAK, SECRET_SHAPE, ...
	File   string `json:"file"`   // repo-relative path ("" when not file-scoped)
	Line   int    `json:"line"`   // 0 when not applicable
	Detail string `json:"detail"` // the human message (matches the Python wording where it matters)
}

// Gate is one commit-boundary check. ModeEnv/EscapeEnv name the env vars that soften or skip
// it, exactly as the shell `run_gate` consulted them, so the in-process runner reproduces the
// block/warn/off + one-shot-escape semantics without the shell.
type Gate struct {
	Name      string
	ModeEnv   string // e.g. FLEET_SCRUB_GUARD; default mode is "block"
	EscapeEnv string // e.g. FLEET_ALLOW_LEAK; "1" => skip this gate once
	Check     func(d *StagedDiff) ([]Finding, error)
}

// PreCommitGates returns the seven pre-commit gates in the SAME order tools/githooks/pre-commit
// invoked them, with their mode/escape env vars. Order is preserved so operator output and any
// first-failure behavior match the Python path.
func PreCommitGates() []Gate {
	return []Gate{
		{Name: "PUBLIC_LEAK", ModeEnv: "FLEET_SCRUB_GUARD", EscapeEnv: "FLEET_ALLOW_LEAK", Check: gatePublicLeak},
		{Name: "SECRET_SHAPE", ModeEnv: "FLEET_SHAPE_GUARD", EscapeEnv: "ALLOW_SECRET_SHAPE", Check: gateSecretShape},
		{Name: "DOC_PLACEMENT", ModeEnv: "FLEET_DOC_GUARD", EscapeEnv: "ALLOW_ROOT_DOC", Check: gateDocPlacement},
		{Name: "BROKEN_LINK", ModeEnv: "FLEET_LINK_GUARD", EscapeEnv: "ALLOW_BAD_LINK", Check: gateBrokenLink},
		{Name: "FILE_ADMISSION", ModeEnv: "FLEET_FILE_GUARD", EscapeEnv: "ALLOW_STRAY_FILE", Check: gateFileAdmission},
		{Name: "INDEX_SYNC", ModeEnv: "FLEET_INDEX_GUARD", EscapeEnv: "ALLOW_INDEX_DRIFT", Check: gateIndexSync},
		{Name: "PROVENANCE_LABEL", ModeEnv: "FLEET_PROVENANCE_GUARD", EscapeEnv: "ALLOW_PROVENANCE_DRIFT", Check: gateProvenanceLabel},
	}
}

// realRunner runs the real git binary. Like witness.gitRunner: non-zero exit => code (not err);
// git-unexecutable => err. Stdout decoded as UTF-8 (Go strings are bytes; the Python checkers
// used errors="replace" — Go's string conversion is already lossless over arbitrary bytes).
func realRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = nil
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return out.String(), ee.ExitCode(), nil
	}
	return "", -1, err
}

// StagedDiff is the staged change set read ONCE and shared across every gate. It holds the
// added lines (with new-file line numbers) per file, the staged path lists for two
// diff-filters the gates use (ACMR for "touched", A for "newly added"), and a lazy cache of
// repo files a gate reads (INDEX.md, llms.txt, an arbitrary committed/staged file).
type StagedDiff struct {
	Root              string
	run               Runner
	ctx               context.Context
	AddedByFile       map[string][]AddedLine // file -> its added lines, in order
	StagedPaths       []string               // --diff-filter=ACMR name list (touched)
	AddedPaths        []string               // --diff-filter=A name list (newly added)
	AddedRenamedPaths []string               // --diff-filter=AR name list (file-admission scope)

	fileCache map[string]fileEntry // rel path -> cached read
}

type fileEntry struct {
	data   []byte
	exists bool
}

// ReadStagedDiff runs the one family of `git diff --cached` reads the gates need and folds the
// result into a StagedDiff. A git failure on the core diff returns ErrCouldNotRun so every
// gate fails open (the Python gates each returned exit 2 in that case).
func ReadStagedDiff(root string) (*StagedDiff, error) {
	return readStagedDiffWith(context.Background(), realRunner, root)
}

func readStagedDiffWith(ctx context.Context, run Runner, root string) (*StagedDiff, error) {
	d := &StagedDiff{
		Root:        root,
		run:         run,
		ctx:         ctx,
		AddedByFile: map[string][]AddedLine{},
		fileCache:   map[string]fileEntry{},
	}

	// Core unified diff with zero context — the substring/regex content gates parse this.
	// --diff-filter=ACMR matches the Python checkers' added-line scans.
	out, code, err := run(ctx, root, "diff", "--cached", "--unified=0", "--no-color", "--diff-filter=ACMR")
	if err != nil || code != 0 {
		return nil, ErrCouldNotRun
	}
	d.AddedByFile = parseUnifiedAddedLines(out)

	// The path-class gates each use a specific diff-filter: ACMR (touched, links/secret-shape),
	// A (newly added, index/doc rules), AR (added+renamed, file-admission). Match each gate's
	// Python checker exactly.
	d.StagedPaths = nameList(run, ctx, root, "--diff-filter=ACMR")
	d.AddedPaths = nameList(run, ctx, root, "--diff-filter=A")
	d.AddedRenamedPaths = nameStatusPaths(run, ctx, root, "--diff-filter=AR")

	return d, nil
}

func nameList(run Runner, ctx context.Context, root, filter string) []string {
	out, code, err := run(ctx, root, "diff", "--cached", "--name-only", filter)
	if err != nil || code != 0 {
		return nil
	}
	var paths []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			paths = append(paths, ln)
		}
	}
	return paths
}

// nameStatusPaths runs `git diff --cached --name-status <filter>` and takes the LAST tab-field
// of each line — the Python checkers' `_staged_paths` shape, which for a rename ("R100\told\tnew")
// correctly yields the new path.
func nameStatusPaths(run Runner, ctx context.Context, root, filter string) []string {
	out, code, err := run(ctx, root, "diff", "--cached", "--name-status", filter)
	if err != nil || code != 0 {
		return nil
	}
	var paths []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		fields := strings.Split(ln, "\t")
		paths = append(paths, fields[len(fields)-1])
	}
	return paths
}

// AddedLines returns every added line across all files, in file-then-order, for whole-diff
// scanners (PUBLIC_LEAK, SECRET_SHAPE) that don't care about per-file grouping.
func (d *StagedDiff) AddedLines() []AddedLine {
	var all []AddedLine
	for _, f := range d.sortedFiles() {
		all = append(all, d.AddedByFile[f]...)
	}
	return all
}

func (d *StagedDiff) sortedFiles() []string {
	files := make([]string, 0, len(d.AddedByFile))
	for f := range d.AddedByFile {
		files = append(files, f)
	}
	// stable order for deterministic findings
	sortStrings(files)
	return files
}

// FileBytes reads a repo-relative file once and caches it. Missing file => (nil, false), never
// an error — the gates treat an absent target as "does not resolve", matching os.path.exists.
func (d *StagedDiff) FileBytes(rel string) ([]byte, bool) {
	if e, ok := d.fileCache[rel]; ok {
		return e.data, e.exists
	}
	b, err := os.ReadFile(filepath.Join(d.Root, filepath.FromSlash(rel)))
	e := fileEntry{data: b, exists: err == nil}
	d.fileCache[rel] = e
	return e.data, e.exists
}

// Exists reports whether a repo-relative path exists on disk (file or dir), mirroring
// os.path.exists used by the link/index resolvers.
func (d *StagedDiff) Exists(rel string) bool {
	full := filepath.Join(d.Root, filepath.FromSlash(rel))
	_, err := os.Stat(full)
	return err == nil
}

// Size returns the byte size of a repo-relative file, or (0,false) on error — the size cap
// gate's os.path.getsize twin.
func (d *StagedDiff) Size(rel string) (int64, bool) {
	fi, err := os.Stat(filepath.Join(d.Root, filepath.FromSlash(rel)))
	if err != nil {
		return 0, false
	}
	return fi.Size(), true
}

// IndexMD / LLMsTxt are the two curated index files the placement/sync gates read.
func (d *StagedDiff) IndexMD() (string, bool) { b, ok := d.FileBytes("INDEX.md"); return string(b), ok }
func (d *StagedDiff) LLMsTxt() (string, bool) { b, ok := d.FileBytes("llms.txt"); return string(b), ok }
