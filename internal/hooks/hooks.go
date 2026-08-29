// Package hooks runs the repo's commit-boundary gates IN ONE PROCESS.
//
// The git hooks (tools/githooks/pre-commit, commit-msg) historically spawned one Python
// interpreter PER gate — 7 for pre-commit + 1 for commit-msg = 8 cold starts, ~2s each on
// a Windows box (process create + Defender scan), so a single `git commit` paid ~12-16s of
// pure interpreter-spawn tax before any checking happened. None of the gates does real work:
// each is regex/substring/os.Stat over `git diff --cached`, sub-millisecond once the
// interpreter is up. This package collapses those gates into one Go process that reads the
// staged diff ONCE and runs every gate over it — the whole measured cost was spawn overhead,
// so a single static-binary start recovers essentially all of it.
//
// The registry has grown well past that Python-era set: PreCommitGates() registers all 24 gates
// today. That number is BOUND, not typed — exhaustiveness_claim_test.go re-derives it from the
// registry and fails when the two disagree, so this sentence cannot quietly decay the way the
// count it replaces did (#5605, epic #5601). Adding a gate is expected to update it.
//
// The pattern, named once here so it is reusable: an exhaustiveness claim in this tree — "all N
// gates", "the only caller", "every package" — carries the witness that would refute it. Either a
// count is bound to the registry it quantifies over (this doc + exhaustiveness_claim_test.go), or
// membership is asserted in both directions (failclosed_ledger_test.go), or the claim names the
// test that enforces it (architest's TestEveryPackageDeclaresTier). A claim carrying none of those
// is prose, and prose decays silently.
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
	"sync"

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
//
// Severity is the one OPTIONAL, gate-specific grade (#4328): a 0-100 magnitude a gate may
// attach when it can measure HOW BAD this finding is, so a caller can act on the degree
// rather than only on the gate's blanket block/warn mode. It is `omitempty` and every gate
// but DUPLICATION leaves it zero, so the JSON report of every other gate is byte-unchanged.
// Zero therefore means UNGRADED, never "graded as harmless" — a consumer must not read an
// absent severity as a low one.
type Finding struct {
	Gate     string `json:"gate"`               // PUBLIC_LEAK, SECRET_SHAPE, ...
	File     string `json:"file"`               // repo-relative path ("" when not file-scoped)
	Line     int    `json:"line"`               // 0 when not applicable
	Detail   string `json:"detail"`             // the human message
	Advisory bool   `json:"advisory,omitempty"` // visible but non-blocking in this push
	Severity int    `json:"severity,omitempty"` // 0 = ungraded; DUPLICATION: copied-coverage percent (0-100)
	View     string `json:"view,omitempty"`     // WORKTREE, LANDS_TREE, or WORKTREE_FALLBACK
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
	return scopeGates([]Gate{
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
		{Name: "NATIVE_FIRST", ModeEnv: "NATIVEFIRST_HOOK_MODE", EscapeEnv: "ALLOW_NATIVE_SUBSTITUTION", Check: checkNativeFirst},
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
		{Name: "DESKTOP_POPUP_REGRESSION", EscapeEnv: "FLEET_ALLOW_POPUP", Check: CheckDesktopPopup},
		// PRIOR_ART is ADVISORY: DefaultMode "warn" so it never blocks a commit out of the box —
		// it only prints the SOTA reference + a `Prior-art:` suggestion. Set FLEET_PRIORART_GUARD=block
		// to hard-enforce it, or ALLOW_NO_PRIOR_ART=1 to skip it once. It runs LAST.
		{Name: "TRUST_WIDENING", ModeEnv: "FLEET_TRUST_WIDENING_GUARD", DefaultMode: "warn", EscapeEnv: "FLEET_ALLOW_TRUST_WIDENING", Check: gateTrustWidening},
		{Name: "PRIOR_ART", ModeEnv: "FLEET_PRIORART_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_NO_PRIOR_ART", Check: gatePriorArt},
		{Name: "MICROHARNESS_WITNESS", ModeEnv: "FLEET_MICROHARNESS_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_NO_MICROHARNESS_WITNESS", Check: gateMicroharnessWitness},
		// UNTIERED_LEAF is ADVISORY (issue #3614): DefaultMode "warn" so it never reds a shared
		// trunk out of the box. It is the STAGED, commit-boundary twin of the whole-tree
		// TIER_DECLARED hygiene gate — it fires when THIS commit adds a new internal/<leaf>/ non-test
		// .go with no architest tier row, before the untiered leaf reaches the trunk and reds every
		// peer's push (architest's TestEveryPackageDeclaresTier + `fak sync push` refusal). It names
		// the exact one-line edit (or `fak new-leaf`). Set FLEET_TIER_GUARD=block to hard-enforce it,
		// ALLOW_UNTIERED_LEAF=1 to skip it once.
		{Name: "UNTIERED_LEAF", ModeEnv: "FLEET_TIER_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_UNTIERED_LEAF", Check: gateUntieredLeaf},
		// CART_BEFORE_HORSE is ADVISORY (#2521): for a newly introduced internal leaf, warn when
		// benchmark/perf/profile/soak/fuzz/proof-matrix/testdata breadth arrives before a named,
		// runnable applied path and its ordinary spine test/witness. The full HEAD-plus-staged view
		// keeps an already-staged spine visible; existing leaves and docs-only commits are out of scope.
		{Name: "CART_BEFORE_HORSE", ModeEnv: "FLEET_CART_BEFORE_HORSE_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_CART_BEFORE_HORSE", Check: gateCartBeforeHorse},
		{Name: "PARALLEL_FABRIC_NUDGE", ModeEnv: "FLEET_PF_NUDGE", DefaultMode: "warn", EscapeEnv: "ALLOW_PF_NUDGE", Check: checkParallelFabricNudge},
		// GIT_HYGIENE_BYPASS is ADVISORY (issue #5588): DefaultMode "warn" so it never reds a shared
		// trunk out of the box. It fires when a staged commit adds hand-rolled git-lock reclamation
		// or object-database maintenance OUTSIDE the packages that own those decisions, and names the
		// evidence-gated route (`fak git-daily`, internal/gitdaily) instead of letting the fifth
		// private copy of "just remove index.lock" reach the trunk. Set FLEET_GIT_HYGIENE_GUARD=block
		// to hard-enforce it, ALLOW_GIT_HYGIENE_BYPASS=1 to skip it once; a `git-hygiene:` note or
		// code already routed through the daily tick silences it in-band.
		{Name: "GIT_HYGIENE_BYPASS", ModeEnv: "FLEET_GIT_HYGIENE_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_GIT_HYGIENE_BYPASS", Check: gateGitHygieneBypass},
		// COMMENT_QUALITY reviews only changed implementation comments and stays advisory because
		// comment value depends on context; it must never turn a prose preference into a commit blockade.
		{Name: "COMMENT_QUALITY", ModeEnv: "FLEET_COMMENT_QUALITY_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_VERBOSE_COMMENTS", Check: gateCommentQuality},
		// GOFMT blocks by default: a staged .go file that is not gofmt-clean would red make ci
		// immediately after landing. Set FLEET_GOFMT_GUARD=warn for an explicit advisory mode,
		// or ALLOW_GOFMT_DRIFT=1 for the existing one-shot escape.
		{Name: "GOFMT", ModeEnv: "FLEET_GOFMT_GUARD", EscapeEnv: "ALLOW_GOFMT_DRIFT", Check: gateGofmt},
		// DUPLICATION is ADVISORY (DefaultMode "warn"): the commit-boundary, in-process twin of
		// `fak dup guard --staged`. It brings the clonescan clone engine (the same normalized-token
		// definition the code-slop scorecard grades the whole tree with, a cycle later) to the commit
		// itself, but scopes each added block's comparison to the OTHER tracked .go files in its own
		// directory (Go package) — cheap enough to run every commit, and the most actionable clone to
		// flag ("call the sibling helper"). Cross-package clones stay the whole-tree scorecard's job.
		// Set FLEET_DUP_GUARD=block to hard-enforce it, ALLOW_DUP=1 to skip it once. It runs LAST.
		{Name: "DUPLICATION", ModeEnv: "FLEET_DUP_GUARD", DefaultMode: "warn", EscapeEnv: "ALLOW_DUP", Check: gateDuplication},
	})
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

	// cacheMu guards fileCache. The pre-commit CLI bounds each gate with a wall-clock budget and
	// ABANDONS a gate that overruns it (#5335) — it cannot cancel one, since Gate.Check takes no
	// context. The abandoned Check keeps running against this same StagedDiff while the loop
	// hands it to the next gate, so two gates can reach fileCache at once. Unsynchronized, that
	// is a concurrent map write: a Go RUNTIME FATAL, unrecoverable, killing the hook — the
	// timeout path would crash the very commit the bound exists to let through.
	cacheMu   sync.Mutex
	fileCache map[string]fileEntry // rel path -> cached read

	// candMu guards candidates, the per-gate CANDIDATE DENOMINATOR ledger (#5602) — how many
	// staged items each gate's own filter admitted for judgement. It is written from inside a
	// gate's Check, so it is reachable by an abandoned over-budget gate concurrently with the
	// next one for exactly the reason cacheMu exists. See candidates.go.
	candMu     sync.Mutex
	candidates map[string]candidateNote // gate name -> what that gate judged over

	// probe replaces working-tree file reads for a scoped view. landsView memoizes the one
	// HEAD-plus-index sibling shared by all LANDS_TREE gates in a run.
	probe      fileReader
	viewMu     sync.Mutex
	landsView  *StagedDiff
	landsTried bool
}

type fileEntry struct {
	data   []byte
	exists bool
}

// ReadStagedDiff runs the one family of `git diff --cached` reads the gates need and folds the
// result into a StagedDiff. A git failure on the core diff returns ErrCouldNotRun so every
// gate fails open (the Python gates each returned exit 2 in that case).
//
// It carries no deadline: the ~5 git reads it spawns all take the index, so on a contended
// shared tree they can block indefinitely. Callers on the commit hot path must use
// ReadStagedDiffWithin instead — see #5335.
func ReadStagedDiff(root string) (*StagedDiff, error) {
	return readStagedDiffWith(context.Background(), realRunner, root)
}

// ReadStagedDiffWithin is ReadStagedDiff bounded by ctx: if the staged-diff reads do not finish
// before ctx expires, it returns ErrCouldNotRun instead of blocking. This is the #5335 fix for
// the hook's PROLOGUE — the reads run `git diff --cached` / `ls-files`, which contend on the
// index, so a peer holding `.git/index.lock` used to wedge every commit in the clone here,
// upstream of (and so uncovered by) the per-gate budget in the pre-commit CLI.
//
// The returned StagedDiff deliberately does NOT retain ctx. Its lazy reads (FileBytes) run under
// a fresh background context, because an EXPIRED ctx would make `git show` fail, and FileBytes
// reports a failed read as "does not resolve" rather than an error — which a gate like
// BROKEN_LINK turns into a finding. Letting the bound reach the lazy reads could therefore
// manufacture a FALSE BLOCK, inverting the fail-open invariant this bound exists to protect. A
// bound may only ever skip work, never add a refusal.
func ReadStagedDiffWithin(ctx context.Context, root string) (*StagedDiff, error) {
	return readStagedDiffWithin(ctx, realRunner, root)
}

func readStagedDiffWithin(ctx context.Context, run Runner, root string) (*StagedDiff, error) {
	type outcome struct {
		d   *StagedDiff
		err error
	}
	// The read runs in a goroutine we are willing to ABANDON, so the bound holds even if the
	// runner ignores ctx entirely — realRunner honors it via exec.CommandContext, but a bound
	// that depends on the thing it is bounding is not a bound. Buffered so an abandoned read
	// still sends and exits rather than leaking. Same shape as the CLI's per-gate budget.
	done := make(chan outcome, 1)
	go func() {
		d, err := readStagedDiffWith(ctx, run, root)
		done <- outcome{d, err}
	}()

	select {
	case o := <-done:
		if o.err != nil {
			return nil, o.err
		}
		if ctx.Err() != nil {
			// The reads RACED the deadline. readStagedDiffWith drops a failed sub-read
			// silently (IndexPaths simply stays empty), so a diff assembled across the
			// expiry may be truncated — and a truncated view is what INDEX_SYNC would read
			// as a real violation. Refuse the partial view instead of gating on it.
			return nil, ErrCouldNotRun
		}
		o.d.ctx = context.Background() // unbind the deadline before any gate reads through it
		return o.d, nil
	case <-ctx.Done():
		return nil, ErrCouldNotRun
	}
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
	if d.probe != nil {
		return d.probe.FileBytes(rel)
	}
	if e, ok := d.cachedFile(rel); ok {
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
	d.storeFile(rel, e)
	return e.data, e.exists
}

// stagedFilesMatching reads text files matching one pathspec from the exact staged index in
// one git process. `git grep -z` prefixes every output line with path+NUL; matching `^` keeps
// blank lines too. Callers fall back to FileBytes unless ok is true.
func (d *StagedDiff) stagedFilesMatching(pathspec string) (map[string][]byte, bool) {
	if d.run == nil || d.Treeish != ":" {
		return nil, false
	}
	out, code, err := d.run(d.ctx, d.Root, "grep", "--cached", "-z", "-I", "-e", "^", "--", pathspec)
	if err != nil || (code != 0 && code != 1) {
		return nil, false
	}
	files := map[string][]byte{}
	if code == 1 {
		return files, true
	}
	for _, line := range strings.SplitAfter(out, "\n") {
		if line == "" {
			continue
		}
		i := strings.IndexByte(line, 0)
		if i <= 0 {
			return nil, false
		}
		rel := filepath.ToSlash(line[:i])
		files[rel] = append(files[rel], line[i+1:]...)
	}
	return files, true
}

// cachedFile / storeFile are the ONLY fileCache accessors, so every read a gate makes is
// serialized against an abandoned gate still running against the same StagedDiff (see cacheMu).
//
// Neither holds cacheMu across the actual file read. A read can be a `git show` that is blocked
// on the very index contention this bound exists for, and holding the lock across it would make
// one wedged gate block every other gate on the mutex — converting a bounded, skippable stall
// into a serialized one. Losing a race just means two gates read the same file twice, which is
// idempotent and cheap; last write wins and both observe identical bytes.
func (d *StagedDiff) cachedFile(rel string) (fileEntry, bool) {
	d.cacheMu.Lock()
	defer d.cacheMu.Unlock()
	e, ok := d.fileCache[rel]
	return e, ok
}

func (d *StagedDiff) storeFile(rel string, e fileEntry) {
	d.cacheMu.Lock()
	defer d.cacheMu.Unlock()
	if d.fileCache == nil {
		return // a hand-built diff with no cache: nothing to warm, and never a nil-map panic
	}
	d.fileCache[rel] = e
}

// Exists reports whether a repo-relative path exists on disk (file or dir), mirroring
// os.path.exists used by the link/index resolvers.
func (d *StagedDiff) Exists(rel string) bool {
	if d.probe != nil {
		return d.probe.Exists(rel)
	}
	full := filepath.Join(d.Root, filepath.FromSlash(rel))
	_, err := os.Stat(full)
	return err == nil
}

// Size returns the byte size of a repo-relative file, or (0,false) on error — the size cap
// gate's os.path.getsize twin.
func (d *StagedDiff) Size(rel string) (int64, bool) {
	if d.probe != nil {
		return d.probe.Size(rel)
	}
	fi, err := os.Stat(filepath.Join(d.Root, filepath.FromSlash(rel)))
	if err != nil {
		return 0, false
	}
	return fi.Size(), true
}

// IndexMD / LLMsTxt are the two curated index files the placement/sync gates read.
func (d *StagedDiff) IndexMD() (string, bool) { b, ok := d.FileBytes("INDEX.md"); return string(b), ok }
func (d *StagedDiff) LLMsTxt() (string, bool) { b, ok := d.FileBytes("llms.txt"); return string(b), ok }
