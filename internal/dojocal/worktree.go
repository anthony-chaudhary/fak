package dojocal

// worktree.go is the dojo-RSI loop's REAL measurement arm — Phase 2 of
// docs/fak/dojo-rsi-loop.md (issue #1024). It is the dojo twin of
// internal/rsiloop/worktree.go: where the rsiloop harness rewrites one
// DefaultCacheSize literal and measures an LRU-hit-rate probe, this harness
// rewrites one claim(...) literal in internal/dojo/claims.go and measures the
// dojo's FoldCalibrable metric by ACTUALLY running `fak dojo run --json` over a
// real corpus, inside a fresh detached git worktree off the pinned baseline SHA.
//
// Every witness field the non-forgeable keep-bit reads is DERIVED here, never
// supplied by the loop author:
//
//   - Before/After Metric = FoldCalibrable over the corpus, baseline measured in
//     an unmodified worktree, candidate measured after the anchored claim swap.
//   - SuiteGreen          = a real `go build`/`go vet` (or WSL `go test`) AND the
//     TWO-DISJOINT-SHARD gate (the candidate must drop FoldCalibrable on BOTH
//     shards — an overfit to the seen shard raises the held-out one and reverts).
//   - TruthClean          = the only TRACKED change in the worktree is claims.go.
//
// Like internal/shipgate and internal/rsiloop/worktree.go, this is the RSI
// harness and uses os/exec; it is not the dispatch hot path, so the os/exec-
// absence proof does not apply here. The repo root and the Go module dir are the
// same for this repo (go.mod lives at the repo root), so a worktree's module dir
// IS its root.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// WorktreeCandidate is the one recalibration the worktree arm measures: the
// (lever, metric) cell to re-point and the corpus-mean claim to swap in. It is
// the Payload a rsiloop.Candidate carries; the command layer picks the worst
// RECALIBRATE from `fak dojo-rsi propose` and builds this from it.
type WorktreeCandidate struct {
	Lever      string
	Metric     string
	NewClaimed float64
}

// WorktreeConfig parameterizes the real dojo worktree harness.
type WorktreeConfig struct {
	// Repo is a path inside the working copy (the module root is fine; git finds
	// the repo from there).
	Repo string
	// BaselineRef is the ref the baseline + the candidate fork from ("main"). The
	// harness resolves it to a SHA ONCE and pins it, so before/after are measured
	// on the identical tree even if main advances mid-run.
	BaselineRef string
	// Corpus is the directory of .jsonl transcripts `fak dojo run --corpus` scores
	// against. It is split into two disjoint shards for the two-shard gate.
	Corpus string
	// Candidate is the single recalibration to measure.
	Candidate WorktreeCandidate
	// SuiteCmds is the suite-green gate; ALL must exit 0. Default: build + vet.
	SuiteCmds [][]string
	// SuitePkgs is the package pattern for the default suite gate ("./...").
	SuitePkgs string
	// DojoArgs are extra args appended to `fak dojo run` (e.g. --ttl 1h, --lever).
	DojoArgs []string
	// DojoRun, when non-nil, overrides the real `fak dojo run --json` invocation.
	// A test injects a fake that returns a fixed report for a corpus dir; the
	// production harness leaves it nil so the real exec runs.
	DojoRun func(moduleDir, corpusDir string) (dojo.Report, error)
	// ScratchDir is the parent for ephemeral worktrees + shard dirs ("" => os.TempDir).
	ScratchDir string
}

// dojoMetricName labels the KPI in the journal.
const dojoMetricName = "dojo_fold_calibrable"

// NewWorktreeHarness wires a rsiloop.Harness to the real dojo worktree/probe/
// suite/truth impls. It is the dojo twin of rsiloop.NewWorktreeHarness: the
// baseline measures FoldCalibrable over a two-shard corpus in a pinned-main
// worktree, the candidate rewrites one claim literal and re-measures, and the
// keep-bit reads a real go-suite plus the two-shard gate plus treeChangedOnly.
func NewWorktreeHarness(cfg WorktreeConfig) rsiloop.Harness {
	if cfg.BaselineRef == "" {
		cfg.BaselineRef = "main"
	}
	if cfg.SuitePkgs == "" {
		cfg.SuitePkgs = "./..."
	}
	if len(cfg.SuiteCmds) == 0 {
		// Windows-safe default: build + vet are a sound native suite-green proxy
		// (`go test` binaries are blocked by OS app-control on this host). A
		// production run overrides this with the WSL test suite (see cmd/dojorsi).
		cfg.SuiteCmds = [][]string{
			{"go", "build", cfg.SuitePkgs},
			{"go", "vet", cfg.SuitePkgs},
		}
	}

	// Pin the baseline SHA + the baseline shard folds exactly ONCE, lazily, on the
	// first resolve (BaselineMetric, which the engine calls before any Measure).
	// Every candidate then forks from this same immutable SHA and competes against
	// the cached baseline shard folds — the two-shard gate needs the baseline
	// per-shard values, so they travel with the pin. Run() is sequential, so the
	// lazy cache needs no lock.
	var pinned string
	var baseFolds shardFolds
	var baseHave bool
	resolveBaseline := func() (string, shardFolds, error) {
		if baseHave {
			return pinned, baseFolds, nil
		}
		sha, err := resolveRef(cfg.Repo, cfg.BaselineRef)
		if err != nil {
			return "", shardFolds{}, err
		}
		var bf shardFolds
		err = withWorktree(cfg, sha, func(p wtPaths) error {
			shardA, shardB, serr := splitCorpus(cfg.Corpus, cfg.ScratchDir)
			if serr != nil {
				return serr
			}
			repA, rerr := runDojo(cfg, p.module, shardA)
			if rerr != nil {
				return rerr
			}
			repB, rerr := runDojo(cfg, p.module, shardB)
			if rerr != nil {
				return rerr
			}
			bf = foldTwoShards(repA, repB)
			return nil
		})
		if err != nil {
			return "", shardFolds{}, err
		}
		pinned, baseFolds, baseHave = sha, bf, true
		return sha, bf, nil
	}

	cand := cfg.Candidate
	return rsiloop.Harness{
		MetricName:      dojoMetricName,
		LowerBetter:     true, // smaller folded calibrable metric wins
		BaselineRefName: cfg.BaselineRef,
		BaselineMetric: func() (float64, string, error) {
			sha, bf, err := resolveBaseline()
			if err != nil {
				return 0, "", err
			}
			return bf.Full, shortSHA(sha), nil
		},
		Candidates: func() []rsiloop.Candidate {
			return []rsiloop.Candidate{{
				Label:   fmt.Sprintf("RECALIBRATE %s/%s -> %g", cand.Lever, cand.Metric, cand.NewClaimed),
				Payload: cand,
			}}
		},
		Measure: func(c rsiloop.Candidate) (rsiloop.Measurement, error) {
			wc, ok := c.Payload.(WorktreeCandidate)
			if !ok {
				return rsiloop.Measurement{}, fmt.Errorf("dojocal: candidate payload is %T, want WorktreeCandidate", c.Payload)
			}
			sha, bf, err := resolveBaseline()
			if err != nil {
				return rsiloop.Measurement{}, err
			}
			var (
				cf        shardFolds
				suiteGreen bool
				truthClean bool
				suiteNote  string
			)
			err = withWorktree(cfg, sha, func(p wtPaths) error {
				if rerr := rewriteClaimInWorktree(p.module, wc); rerr != nil {
					return rerr
				}
				shardA, shardB, serr := splitCorpus(cfg.Corpus, cfg.ScratchDir)
				if serr != nil {
					return serr
				}
				repA, rerr := runDojo(cfg, p.module, shardA)
				if rerr != nil {
					return rerr
				}
				repB, rerr := runDojo(cfg, p.module, shardB)
				if rerr != nil {
					return rerr
				}
				cf = foldTwoShards(repA, repB)
				green, detail := runSuite(p.module, cfg.SuiteCmds)
				suiteGreen = green
				if !green {
					suiteNote = detail
				}
				truthClean = treeChangedOnlyTracked(p.root, ClaimsRelPath)
				return nil
			})
			if err != nil {
				return rsiloop.Measurement{}, err
			}
			return measureCandidate(bf, cf, suiteGreen, truthClean, suiteNote, wc), nil
		},
	}
}

// shardFolds is the two-shard + full-corpus FoldCalibrable view the gate reads.
// Full is the fold over shardA's and shardB's episodes combined; ShardA/ShardB
// are the per-shard folds. Measured is the total measured episode count across
// both shards (the sample behind the metric).
type shardFolds struct {
	Full     float64
	ShardA   float64
	ShardB   float64
	Measured int
}

// foldTwoShards folds two dojo reports into the two-shard + full-corpus view. It
// is pure: the full fold is FoldCalibrable over the concatenation of both
// reports' episodes (NOT the mean of the two shard values — the populations may
// differ in size, so the combined mean must be re-derived over all episodes).
func foldTwoShards(repA, repB dojo.Report) shardFolds {
	combined := make([]dojo.Episode, 0, len(repA.Episodes)+len(repB.Episodes))
	combined = append(combined, repA.Episodes...)
	combined = append(combined, repB.Episodes...)
	full := dojo.FoldCalibrable(combined)
	return shardFolds{
		Full:     full.Value,
		ShardA:   dojo.FoldCalibrable(repA.Episodes).Value,
		ShardB:   dojo.FoldCalibrable(repB.Episodes).Value,
		Measured: full.Measured,
	}
}

// twoShardGate reports whether the candidate dropped FoldCalibrable STRICTLY on
// BOTH disjoint shards versus the baseline — the anti-overfitting defense. A
// recalibration that fits the seen shard but raises the held-out one fails here.
// It is the structural reason a constant-rewrite "gain" cannot sneak through on
// one shard: the held-out shard's calib_err rises and the gate reverts.
func twoShardGate(base, cand shardFolds) bool {
	return cand.ShardA < base.ShardA && cand.ShardB < base.ShardB
}

// measureCandidate is the pure keep/revert decision the git/exec harness wraps:
// given the baseline + candidate shard folds and the suite-green / truth-clean
// witnesses, return the rsiloop.Measurement the engine folds through the
// non-forgeable keep-bit. It is unit-testable without git or a corpus.
//
// The two-shard gate is folded into SuiteGreen (a candidate that overfits the
// seen shard fails its validity suite): shipgate.Evaluate keeps iff strictGain
// (candidate Full < baseline Full, LowerBetter) AND SuiteGreen AND TruthClean,
// so a single-shard "gain" — full drops but one shard does not — must flip
// SuiteGreen to revert. The Measurement.Note names exactly which gate failed so
// a REVERT is diagnosable rather than a silent false.
func measureCandidate(base, cand shardFolds, goSuiteGreen, truthClean bool, suiteNote string, wc WorktreeCandidate) rsiloop.Measurement {
	shardOK := twoShardGate(base, cand)
	effectiveSuite := goSuiteGreen && shardOK
	note := keepNote(base, cand, goSuiteGreen, shardOK, truthClean, suiteNote, wc)
	return rsiloop.Measurement{
		Metric:     cand.Full,
		SuiteGreen: effectiveSuite,
		TruthClean: truthClean,
		Score:      dojoWorktreeScorecard(base, cand),
		Note:       note,
	}
}

// keepNote renders the one-line reason a candidate kept or reverted, naming the
// first gate that failed so the journal row is legible.
func keepNote(base, cand shardFolds, goSuiteGreen, shardOK, truthClean bool, suiteNote string, wc WorktreeCandidate) string {
	fullDelta := base.Full - cand.Full // positive = the full-corpus fold dropped
	kept := fullDelta > 0 && goSuiteGreen && shardOK && truthClean
	if kept {
		return fmt.Sprintf("KEPT: %s/%s %.6g -> %.6g dropped FoldCalibrable %.6g -> %.6g (delta %+.6g) on both shards (%.6g->%.6g, %.6g->%.6g), suite green, truth clean",
			wc.Lever, wc.Metric, 0.0, wc.NewClaimed, base.Full, cand.Full, fullDelta,
			base.ShardA, cand.ShardA, base.ShardB, cand.ShardB)
	}
	switch {
	case !truthClean:
		return fmt.Sprintf("REVERT: %s/%s truth-unclean — the worktree changed more than %s; only the one claim literal may move", wc.Lever, wc.Metric, ClaimsRelPath)
	case !shardOK:
		which := "both shards"
		if !(cand.ShardA < base.ShardA) && !(cand.ShardB < base.ShardB) {
			which = "neither shard"
		} else if !(cand.ShardA < base.ShardA) {
			which = "shard A"
		} else {
			which = "shard B"
		}
		return fmt.Sprintf("REVERT: %s/%s two-shard gate failed — full dropped (%.6g->%.6g) but %s did not (%.6g->%.6g, %.6g->%.6g); an overfit to the seen shard raises the held-out one", wc.Lever, wc.Metric, base.Full, cand.Full, which, base.ShardA, cand.ShardA, base.ShardB, cand.ShardB)
	case !goSuiteGreen:
		detail := suiteNote
		if detail == "" {
			detail = "suite exited non-zero"
		}
		return fmt.Sprintf("REVERT: %s/%s suite red — %s", wc.Lever, wc.Metric, detail)
	case fullDelta <= 0:
		return fmt.Sprintf("REVERT: %s/%s no strict full-corpus gain (%.6g -> %.6g, delta %+.6g); the claim swap did not lower the folded calibrable", wc.Lever, wc.Metric, base.Full, cand.Full, fullDelta)
	default:
		return fmt.Sprintf("REVERT: %s/%s did not keep", wc.Lever, wc.Metric)
	}
}

// dojoWorktreeScorecard is the optional structured readout attached to the
// measurement — telemetry only; the keep-bit reads only Metric/SuiteGreen/
// TruthClean. It carries the per-shard deltas so a reader can see WHY the gate
// kept or reverted without re-running the corpus.
func dojoWorktreeScorecard(base, cand shardFolds) *rsiloop.Scorecard {
	return &rsiloop.Scorecard{
		Name:  dojoMetricName,
		Value: cand.Full,
		Components: []rsiloop.ScoreComponent{
			{Name: "baseline_full", Value: base.Full, Unit: "fold_calibrable"},
			{Name: "candidate_full", Value: cand.Full, Unit: "fold_calibrable"},
			{Name: "full_delta", Value: base.Full - cand.Full, Unit: "fold_calibrable"},
			{Name: "baseline_shard_a", Value: base.ShardA, Unit: "fold_calibrable"},
			{Name: "candidate_shard_a", Value: cand.ShardA, Unit: "fold_calibrable"},
			{Name: "baseline_shard_b", Value: base.ShardB, Unit: "fold_calibrable"},
			{Name: "candidate_shard_b", Value: cand.ShardB, Unit: "fold_calibrable"},
			{Name: "measured", Value: float64(cand.Measured), Unit: "episodes"},
		},
	}
}

// --- git/exec helpers (cloned from internal/rsiloop/worktree.go) -------------

// wtPaths bundles the worktree root (= module dir for this repo).
type wtPaths struct {
	root   string
	module string
}

// resolveRef returns the full SHA a ref points at.
func resolveRef(repo, ref string) (string, error) {
	cmd := exec.Command("git", "-C", repo, "rev-parse", ref)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("rev-parse %s: %w", ref, err)
	}
	return strings.TrimSpace(string(out)), nil
}

func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// repoTopAndRel resolves the git repo root and the module's path relative to it.
func repoTopAndRel(repoArg string) (top, moduleRel string, err error) {
	cmd := exec.Command("git", "-C", repoArg, "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, rerr := cmd.Output()
	if rerr != nil {
		return "", "", fmt.Errorf("rev-parse --show-toplevel: %w", rerr)
	}
	top = filepath.Clean(strings.TrimSpace(string(out)))
	abs, aerr := filepath.Abs(repoArg)
	if aerr != nil {
		return "", "", aerr
	}
	rel, rerr := filepath.Rel(top, abs)
	if rerr != nil {
		return "", "", rerr
	}
	if rel == "" || rel == "." {
		rel = "."
	}
	return top, filepath.ToSlash(rel), nil
}

// withWorktree creates a fresh detached worktree at ref, runs fn against it, and
// always tears it down. main is never modified.
func withWorktree(cfg WorktreeConfig, ref string, fn func(wtPaths) error) error {
	top, moduleRel, err := repoTopAndRel(cfg.Repo)
	if err != nil {
		return err
	}
	parent, err := os.MkdirTemp(cfg.ScratchDir, "dojocal-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(parent)
	wt := filepath.Join(parent, "wt")
	add := exec.Command("git", "-C", top, "worktree", "add", "--detach", wt, ref)
	windowgate.ConfigureBackgroundCommand(add)
	if out, err := add.CombinedOutput(); err != nil {
		return fmt.Errorf("worktree add %s: %v: %s", ref, err, out)
	}
	defer shipgate.RemoveWorktree(top, wt)
	module := wt
	if moduleRel != "." {
		module = filepath.Join(wt, filepath.FromSlash(moduleRel))
	}
	return fn(wtPaths{root: wt, module: module})
}

// runSuite runs every configured suite command in the module dir; green iff all
// exit 0. On failure it returns a short diagnostic so a REVERT is explainable.
func runSuite(moduleDir string, cmds [][]string) (bool, string) {
	for _, c := range cmds {
		if len(c) == 0 {
			continue
		}
		cmd := exec.Command(c[0], c[1:]...)
		windowgate.ConfigureBackgroundCommand(cmd)
		cmd.Dir = moduleDir
		out, err := cmd.CombinedOutput()
		if err == nil {
			continue
		}
		label := "suite cmd " + strings.Join(c, " ")
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			label += " exited non-zero"
		} else {
			label += " could not start (" + err.Error() + ")"
		}
		return false, label + ": " + tail(string(out), 400)
	}
	return true, ""
}

// tail returns the last n bytes of s, for a compact diagnostic.
func tail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n:]
}

// treeChangedOnlyTracked reports whether the only TRACKED change in the worktree
// is `only` (a repo-root-relative, forward-slash path). Untracked entries (the
// `??` porcelain code that a dojo run's scratch files produce) are tolerated;
// a SECOND tracked modification fails closed. A clean tree (no changes) also
// returns false — a candidate that changed nothing is not a real proposal.
//
// This is the dojo arm's truth-clean rung: it differs from rsiloop's
// treeChangedOnly (which requires exactly one path of any kind) because `fak
// dojo run` legitimately drops untracked scratch alongside the one tracked
// claim edit, and those must not mask a real extra edit nor fail the gate.
func treeChangedOnlyTracked(wtRoot, only string) bool {
	cmd := exec.Command("git", "-C", wtRoot, "status", "--porcelain")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	changed := []string{}
	for _, line := range strings.Split(strings.TrimRight(string(out), "\n"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// porcelain v1: "XY <path>"; XY is the 2-char status. Ignore untracked
		// ("??") — dojo-run scratch. Track every other (modified/added/deleted/
		// renamed/typed) entry: only claims.go may appear there.
		if len(line) < 3 {
			continue
		}
		xy := line[:2]
		if xy == "??" {
			continue
		}
		changed = append(changed, filepath.ToSlash(strings.TrimSpace(line[3:])))
	}
	return len(changed) == 1 && changed[0] == only
}

// --- dojo-specific helpers ---------------------------------------------------

// rewriteClaimInWorktree edits the worktree's copy of claims.go, re-pointing the
// (lever, metric) cell's literal to the candidate's new claim. It reuses the
// pure RewriteClaim (anchored, fail-closed), so the swap is byte-exact and the
// worktree's only tracked change is the one recalibrated literal.
func rewriteClaimInWorktree(moduleDir string, wc WorktreeCandidate) error {
	path := filepath.Join(moduleDir, filepath.FromSlash(ClaimsRelPath))
	src, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", ClaimsRelPath, err)
	}
	out, _, err := RewriteClaim(src, wc.Lever, wc.Metric, wc.NewClaimed)
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0o644)
}

// runDojo runs `fak dojo run --json --corpus corpusDir` in the module dir and
// parses the report. A cfg.DojoRun override (test fake) short-circuits the exec.
// The production exec uses `go run ./cmd/fak` so the binary is BUILT inside the
// worktree — that is load-bearing: the claim literal lives in claims.go, which
// is compiled into the binary, so the candidate's rewritten claim only takes
// effect when fak is rebuilt in the candidate worktree.
func runDojo(cfg WorktreeConfig, moduleDir, corpusDir string) (dojo.Report, error) {
	if cfg.DojoRun != nil {
		return cfg.DojoRun(moduleDir, corpusDir)
	}
	args := []string{"run", "./cmd/fak", "dojo", "run", "--json", "--corpus", corpusDir}
	args = append(args, cfg.DojoArgs...)
	cmd := exec.Command("go", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = moduleDir
	// stdout/stderr are captured SEPARATELY (not CombinedOutput): `go run` writes
	// toolchain/module-download progress ("go: downloading ...") to stderr, which
	// would pollute the JSON on stdout and break parsing. stdout holds the report;
	// stderr is folded into the diagnostic only on failure.
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return dojo.Report{}, fmt.Errorf("fak dojo run --corpus %s: %v: %s", filepath.Base(corpusDir), err, tail(stderr.String()+"\n"+stdout.String(), 600))
	}
	return parseDojoReport([]byte(stdout.String()))
}

// parseDojoReport decodes a `fak dojo run --json` envelope into a dojo.Report.
// It accepts both a bare report and a {"report": {...}} gate envelope.
func parseDojoReport(b []byte) (dojo.Report, error) {
	var r dojo.Report
	if err := json.Unmarshal(b, &r); err == nil && r.Schema == dojo.Schema {
		return r, nil
	}
	var env struct {
		Report dojo.Report `json:"report"`
	}
	if err := json.Unmarshal(b, &env); err == nil && env.Report.Schema == dojo.Schema {
		return env.Report, nil
	}
	return dojo.Report{}, fmt.Errorf("dojocal: `fak dojo run --json` did not emit a dojo report: %s", tail(string(b), 300))
}

// splitCorpus walks the corpus dir for .jsonl transcripts, sorts the paths, and
// copies a deterministic disjoint half into each of two temp shard dirs. The
// shards are disjoint by construction (each transcript lands in exactly one) and
// stable across runs (sorted paths), so the baseline and candidate measure the
// SAME two shards. Returns the two shard dir paths.
func splitCorpus(corpusDir, scratchParent string) (shardA, shardB string, err error) {
	files, err := listCorpusFiles(corpusDir)
	if err != nil {
		return "", "", err
	}
	if len(files) == 0 {
		return "", "", fmt.Errorf("dojocal: corpus %s has no .jsonl transcripts to shard", corpusDir)
	}
	parent := scratchParent
	if parent == "" {
		parent = os.TempDir()
	}
	root, err := os.MkdirTemp(parent, "dojocal-shards-")
	if err != nil {
		return "", "", err
	}
	shardA = filepath.Join(root, "a")
	shardB = filepath.Join(root, "b")
	if err := os.MkdirAll(shardA, 0o755); err != nil {
		return "", "", err
	}
	if err := os.MkdirAll(shardB, 0o755); err != nil {
		return "", "", err
	}
	// Even-index files -> shard A, odd-index -> shard B. Both shards non-empty as
	// long as there are >=2 files; a single-file corpus yields one empty shard,
	// which the caller's fold treats as a zero-measurement held-out set (and the
	// two-shard gate reverts — a single transcript cannot support a recalibration).
	mid := len(files) / 2
	if mid < 1 {
		mid = 1
	}
	for i, f := range files {
		dst := shardB
		if i < mid {
			dst = shardA
		}
		if err := copyFile(f, filepath.Join(dst, strconv.Itoa(i)+"_"+filepath.Base(f))); err != nil {
			return "", "", err
		}
	}
	return shardA, shardB, nil
}

// listCorpusFiles recursively collects every *.jsonl path under dir, sorted for
// determinism. `fak dojo run --corpus` scans recursively, so a flat shard dir is
// a valid (degenerate) recursive corpus.
func listCorpusFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, werr error) error {
		if werr != nil {
			return werr
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".jsonl") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk corpus %s: %w", dir, err)
	}
	sort.Strings(files)
	return files, nil
}

// copyFile copies src to dst (creating dst). Used to materialize a shard.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
