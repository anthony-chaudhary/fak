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

	"github.com/anthony-chaudhary/fak/internal/windowgate"
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
	Gate     string `json:"gate"`               // PUBLIC_LEAK, SECRET_SHAPE, ...
	File     string `json:"file"`               // repo-relative path ("" when not file-scoped)
	Line     int    `json:"line"`               // 0 when not applicable
	Detail   string `json:"detail"`             // the human message
	Advisory bool   `json:"advisory,omitempty"` // visible but non-blocking in this push
}

// Gate is one commit-boundary check. ModeEnv/EscapeEnv name the env vars that soften or skip
// it, exactly as the shell `run_gate` consulted them, so the in-process runner reproduces the
// block/warn/off + one-shot-escape semantics without the shell.
type Gate struct {
	Name    string
	ModeEnv string // e.g. FLEET_SCRUB_GUARD; default mode is "block" (unless DefaultMode says otherwise)
	// DefaultMode is the mode used when ModeEnv is UNSET. Empty means the historical default of
	// "block" — every pre-existing gate keeps that. An ADVISORY gate sets this to "warn" so it
	// only warns out of the box (PRIOR_ART), while its ModeEnv can still be set to "block" to
	// hard-enforce it.
	DefaultMode string
	EscapeEnv   string // e.g. FLEET_ALLOW_LEAK; "1" => skip this gate once
	Check       func(d *StagedDiff) ([]Finding, error)
}

// PreCommitGates returns the pre-commit gates in the SAME order tools/githooks/pre-commit
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
		{Name: "CONCEPT_ADMISSION", ModeEnv: "FLEET_CONCEPT_GUARD", EscapeEnv: "ALLOW_CONCEPT_GAP", Check: gateConceptAdmission},
		{Name: "CONCEPT_FRESHNESS", ModeEnv: "FLEET_CONCEPT_FRESHNESS_GUARD", EscapeEnv: "ALLOW_STALE_CONCEPT_DOCS", Check: checkConceptFreshness},
		{Name: "PROVENANCE_LABEL", ModeEnv: "FLEET_PROVENANCE_GUARD", EscapeEnv: "ALLOW_PROVENANCE_DRIFT", Check: gateProvenanceLabel},
		{Name: "HARDWARE_TELL", ModeEnv: "FLEET_HW_GUARD", EscapeEnv: "FLEET_ALLOW_HW", Check: gateHardwareTell},
		// BARE_COMMIT_SWEEP is ADVISORY (issue #3615): DefaultMode "warn" so it never reds a shared
		// trunk out of the box. It closes the raw-git bypass in safecommit's prestaged discipline —
		// a `git commit` / `git add -A && git commit` that did NOT come through `fak commit` (no
		// FAK_SAFECOMMIT_VETTED handshake) would fold the whole staged index, foreign hunks included,
		// into one commit. It fires on any unvetted staged set, naming what would be swept and the
		// pathspec fix. Set FLEET_BARE_COMMIT_GUARD=block to enforce, ALLOW_BARE_COMMIT=1 to skip once;
		// FAK_PRESTAGED_PATH_GUARD=off disables the prestaged family (this gate + safecommit's guard).
		{Name: "BARE_COMMIT_SWEEP", ModeEnv: "FLEET_BARE_COMMIT_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_BARE_COMMIT", Check: gateBareCommitSweep},
		// E2E_OVER_MOCKS is ADVISORY (issue #2901): DefaultMode "warn" so it never blocks a commit
		// out of the box — it names the security-critical floor/quarantine surface a diff touched
		// and asks for a witnessed end-to-end run (the /verify output), not a green mock. Set
		// FLEET_E2E_GUARD=block to hard-enforce it ("failing the merge otherwise"), or ALLOW_NO_E2E=1
		// to skip it once.
		{Name: "E2E_OVER_MOCKS", ModeEnv: "FLEET_E2E_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_NO_E2E", Check: gateE2EOverMocks},
		// PRIOR_ART is ADVISORY: DefaultMode "warn" so it never blocks a commit out of the box —
		// it only prints the SOTA reference + a `Prior-art:` suggestion. Set FLEET_PRIORART_GUARD=block
		// to hard-enforce it, or ALLOW_NO_PRIOR_ART=1 to skip it once. It runs LAST.
		{Name: "TRUST_WIDENING", ModeEnv: "FLEET_TRUST_WIDENING_GUARD", DefaultMode: "warn", EscapeEnv: "FLEET_ALLOW_TRUST_WIDENING", Check: gateTrustWidening},
		{Name: "PRIOR_ART", ModeEnv: "FLEET_PRIORART_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_NO_PRIOR_ART", Check: gatePriorArt},
		// UNTIERED_LEAF is ADVISORY (issue #3614): DefaultMode "warn" so it never reds a shared
		// trunk out of the box. It is the STAGED, commit-boundary twin of the whole-tree
		// TIER_DECLARED hygiene gate — it fires when THIS commit adds a new internal/<leaf>/ non-test
		// .go with no architest tier row, before the untiered leaf reaches the trunk and reds every
		// peer's push (architest's TestEveryPackageDeclaresTier + `fak sync push` refusal). It names
		// the exact one-line edit (or `fak new-leaf`). Set FLEET_TIER_GUARD=block to hard-enforce it,
		// ALLOW_UNTIERED_LEAF=1 to skip it once.
		{Name: "UNTIERED_LEAF", ModeEnv: "FLEET_TIER_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_UNTIERED_LEAF", Check: gateUntieredLeaf},
		// GOFMT is ADVISORY (DefaultMode "warn"): the commit-boundary sibling of make ci's
		// gofmt-check. It fires when a staged .go file is not gofmt-clean, before the drift reds
		// every peer's `make ci` at the trunk — a recurring red the release notes keep clearing
		// ("clear the CI gofmt gate", v0.32.0 x4 / v0.34.0). Set FLEET_GOFMT_GUARD=block to
		// hard-enforce it, ALLOW_GOFMT_DRIFT=1 to skip it once.
		{Name: "GOFMT", ModeEnv: "FLEET_GOFMT_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_GOFMT_DRIFT", Check: gateGofmt},
		// DUPLICATION is ADVISORY (DefaultMode "warn"): the commit-boundary, in-process twin of
		// `fak dup guard --staged`. It brings the clonescan clone engine (the same normalized-token
		// definition the code-slop scorecard grades the whole tree with, a cycle later) to the commit
		// itself, but scopes each added block's comparison to the OTHER tracked .go files in its own
		// directory (Go package) — cheap enough to run every commit, and the most actionable clone to
		// flag ("call the sibling helper"). Cross-package clones stay the whole-tree scorecard's job.
		// Set FLEET_DUP_GUARD=block to hard-enforce it, ALLOW_DUP=1 to skip it once. It runs LAST.
		{Name: "DUPLICATION", ModeEnv: "FLEET_DUP_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_DUP", Check: gateDuplication},
	}
}

// realRunner runs the real git binary. Like witness.gitRunner: non-zero exit => code (not err);
// git-unexecutable => err. Stdout decoded as UTF-8 (Go strings are bytes; the Python checkers
// used errors="replace" — Go's string conversion is already lossless over arbitrary bytes).
func realRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
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
	IndexPaths        []string               // all candidate-index paths (for cross-file semantic gates)
	Treeish           string                 // ":" for index, or a committed tip for CI range checks

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
		Treeish:     ":",
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
	if out, code, err := run(ctx, root, "ls-files"); err == nil && code == 0 {
		for _, line := range strings.Split(out, "\n") {
			if line = strings.TrimSpace(line); line != "" {
				d.IndexPaths = append(d.IndexPaths, filepath.ToSlash(line))
			}
		}
	}

	return d, nil
}

// ReadRangeDiff builds the same admission view from a committed base..tip range.
// CI uses it to repeat the pre-commit decision against immutable objects.
func ReadRangeDiff(root, base, tip string) (*StagedDiff, error) {
	if strings.TrimSpace(base) == "" || strings.TrimSpace(tip) == "" {
		return nil, ErrCouldNotRun
	}
	d := &StagedDiff{Root: root, run: realRunner, ctx: context.Background(), Treeish: tip + ":", AddedByFile: map[string][]AddedLine{}, fileCache: map[string]fileEntry{}}
	out, code, err := realRunner(d.ctx, root, "diff", "--unified=0", "--no-color", "--diff-filter=ACMR", base, tip)
	if err != nil || code != 0 {
		return nil, ErrCouldNotRun
	}
	d.AddedByFile = parseUnifiedAddedLines(out)
	out, code, err = realRunner(d.ctx, root, "diff", "--name-only", "--diff-filter=ACMR", base, tip)
	if err == nil && code == 0 {
		for _, x := range strings.Split(out, "\n") {
			if x = strings.TrimSpace(x); x != "" {
				d.StagedPaths = append(d.StagedPaths, filepath.ToSlash(x))
			}
		}
	}
	out, code, err = realRunner(d.ctx, root, "ls-tree", "-r", "--name-only", tip)
	if err == nil && code == 0 {
		for _, x := range strings.Split(out, "\n") {
			if x = strings.TrimSpace(x); x != "" {
				d.IndexPaths = append(d.IndexPaths, filepath.ToSlash(x))
			}
		}
	}
	return d, nil
}

// stagedPathLines runs a `git diff --cached <listFlag> <filter>` path listing and folds each
// non-blank trimmed output line through pick. nameList / nameStatusPaths differ only in the list
// flag and how a line maps to a path, so the run + split + trim + skip-blank scaffold lives here.
func stagedPathLines(run Runner, ctx context.Context, root, listFlag, filter string, pick func(line string) string) []string {
	out, code, err := run(ctx, root, "diff", "--cached", listFlag, filter)
	if err != nil || code != 0 {
		return nil
	}
	var paths []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		paths = append(paths, pick(ln))
	}
	return paths
}

func nameList(run Runner, ctx context.Context, root, filter string) []string {
	return stagedPathLines(run, ctx, root, "--name-only", filter, func(ln string) string { return ln })
}

// nameStatusPaths runs `git diff --cached --name-status <filter>` and takes the LAST tab-field
// of each line — the Python checkers' `_staged_paths` shape, which for a rename ("R100\told\tnew")
// correctly yields the new path.
func nameStatusPaths(run Runner, ctx context.Context, root, filter string) []string {
	return stagedPathLines(run, ctx, root, "--name-status", filter, func(ln string) string {
		fields := strings.Split(ln, "\t")
		return fields[len(fields)-1]
	})
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
	var b []byte
	exists := false
	if d.run != nil {
		if out, code, err := d.run(d.ctx, d.Root, "show", d.Treeish+filepath.ToSlash(rel)); err == nil && code == 0 {
			b, exists = []byte(out), true
		}
	}
	if !exists {
		var err error
		b, err = os.ReadFile(filepath.Join(d.Root, filepath.FromSlash(rel)))
		exists = err == nil
	}
	e := fileEntry{data: b, exists: exists}
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
