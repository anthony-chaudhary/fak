package main

// Commit-time diff-witness binding for the dispatch tick (#1324 proposal #2), the Go
// port of witness_exited_workers in tools/issue_resolve_dispatch.py. For every
// resolve-<N>-<stamp>.log whose pid is provably DEAD and not yet witnessed (no
// .witness sidecar), find the commit it landed for its issue (subject cites #N inside
// the per-worker .basesha..HEAD window) and grade it through `dos commit-audit`: a
// diff-witnessed commit -> CLAIM_WITNESSED, an unwitnessed or wrong-issue commit ->
// CLAIM_UNWITNESSED, no resolving commit -> CLAIM_NO_COMMIT with a structured reason
// classified from the log tail. The verdict is recorded in a .witness sidecar on live
// ticks so a bare `exit 0` / non-empty log never SILENTLY counts as productive, and
// the pick holds the re-blockable guard refusals it surfaces (#1396). Dead-pid gated
// (a still-running worker may not have committed yet -- never mis-blame it) and
// FAIL-OPEN throughout, the same discipline as the live-lane reap.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// Injectable seams mirroring the Python sweep's git= / audit_runner= test params.
var dispatchWitnessResolvingSHA = dispatchWitnessResolvingSHAGit
var dispatchWitnessCommitAudit = dispatchWitnessCommitAuditDos

// dispatchWitnessTestRun is the #3838 test-run seam: given a resolving commit, run its
// changed package's tests and report (ran, passed). Injectable so the sweep test pins
// GREEN/RED/UNRUN with a stubbed runner. It grades the missing rung between "the diff
// looks like the claim" (the diff-shape witness) and "the claim actually holds" — a
// diff-witnessed commit that fails its own tests grades CLAIM_TEST_RED. The default is
// OPT-IN (env-gated), so the hot dispatch loop is never destabilized by an in-tick
// `go test`; the always-on live consumer of this rung is the `verify` skill.
var dispatchWitnessTestRun = dispatchWitnessTestRunGo
var dispatchWitnessCommitPaths = dispatchWitnessCommitPathsGit

// dispatchWitnessTestRunEnv gates the default in-tick runner. Unset/false -> the rung
// records CLAIM_TEST_UNRUN (a valid, surfaced state — #3838 out-of-scope: not every
// commit must carry a test run). Set truthy -> the sweep runs the resolving commit's
// changed-package tests WSL-aware and binds GREEN/RED to the slot.
const dispatchWitnessTestRunEnv = "FAK_WITNESS_TEST_RUN"

// dispatchWitnessLandReap is the seam the #3168 land+reap fires through, injectable
// so the sweep test can assert it runs (before the resolving-SHA scan) and that a
// fault is swallowed. Default lands the worker's worktree diff onto the trunk and
// reaps the worktree.
var dispatchWitnessLandReap = landAndReapWorkerWorktreeDefault

// landAndReapWorkerWorktree lands a dead worker's per-worker worktree (#3168) onto
// the trunk and reaps it, when a .worktree sidecar records one. A no-op (and no
// error surfaced) when the sidecar is absent — a worker that ran in the shared
// trunk. FAIL-OPEN: every step's error is swallowed; the sweep proceeds to audit
// the resolve log regardless.
func landAndReapWorkerWorktree(root, stem, base string) {
	wtPath := ""
	if b, err := os.ReadFile(stem + dispatchWorktreeSidecarSuffix); err == nil {
		wtPath = strings.TrimSpace(string(b))
	}
	if wtPath == "" {
		return
	}
	tree := readResolveLeaseTree(stem + dispatchLeaseTreeSidecarSuffix)
	dispatchWitnessLandReap(root, wtPath, base, tree)
	// Landed once: drop the sidecar so a later sweep never re-lands (the diff is now
	// on the trunk and the worktree is gone).
	_ = os.Remove(stem + dispatchWorktreeSidecarSuffix)
}

// dispatchLandVerify is the #3178 pre-land build witness wired into the live land
// site: `go build ./...` INSIDE the worker's isolated worktree, refusing the land if
// it reds so a broken edit never reaches the trunk. Until now the spine (#3168) landed
// mechanically with verify=nil, leaning only on the downstream dos commit-audit — which
// witnesses the CLAIM, not the build. worktreeWorkerGoBuildVerify FAILS OPEN when the
// go toolchain is absent (a missing `go` never wedges a land). A package var so
// TestDispatchLandVerify can pin a red/green witness without a real toolchain.
var dispatchLandVerify workerworktree.VerifyHook = worktreeWorkerGoBuildVerify

// dispatchLandRefusedAttempts bounds the total Land attempts (first try + retries)
// one sweep makes for one worktree when the land is refused by the readback race
// (#3613). Small on purpose: each refusal means a peer commit swept our paths in
// the gap, and layer 2 (the isolated-index land, with its own #3570 CAS retry)
// already absorbs most contention before this outer bound is ever consulted.
const dispatchLandRefusedAttempts = 3

// Seams for the #3613 refused-land retry, injectable so the retry test pins the
// land/reap sequencing hermetically: the land itself, the reap that destroys the
// worktree, and the between-attempt backoff.
var dispatchLandWorktreeOnce = func(root, wtPath, base string, tree []string) workerworktree.Result {
	return landWorkerWorktreeVerified(root, wtPath, base, tree, nil)
}
var dispatchReapWorktree = func(root, wtPath string) workerworktree.Result {
	return workerworktree.Reap(root, wtPath, nil)
}
var dispatchLandRetrySleep = func(attempt int) {
	time.Sleep(time.Duration(250*attempt) * time.Millisecond)
}

// landAndReapWorkerWorktreeDefault is the production land+reap: apply the worktree's
// diff-since-base onto the trunk as the worker's own stamped commit (scoped to its
// declared lease tree), then force-remove the worktree. Both fail-open.
//
// #3613: a refused land is no longer unconditionally forgotten. A refusal carrying
// the workerworktree.LandReadbackMismatchToken race class is TRANSIENT — a
// concurrent commit on the shared index swept this worker's paths, and the
// worktree still holds the ONLY copy of the diff — so the land is re-attempted
// (bounded, backed off) on the moved HEAD before the reap destroys it.
// Deterministic refusals (red verify, apply conflict) never retry: replaying them
// cannot change the verdict. After the bound is exhausted the reap still runs —
// the pre-#3613 fail-open final resort, so a pathological race can never leak
// worktrees without bound — with every attempt surfaced as a countable line.
func landAndReapWorkerWorktreeDefault(root, wtPath, base string, tree []string) {
	// No commit-message file: Land derives the subject from the worktree tip so the
	// landed commit keeps the worker's own #N-citing, (fak <leaf>)-stamped subject.
	// verify=dispatchLandVerify (#3178): a red `go build ./...` in the worktree refuses
	// the land (nothing applied/committed) so a broken edit never reaches main.
	res := dispatchLandWorktreeOnce(root, wtPath, base, tree)
	attempt := 1
	for !res.OK && workerworktree.LandRefusalRetryable(res.Reason) && attempt < dispatchLandRefusedAttempts {
		fmt.Fprintf(os.Stderr, "fak dispatch: worktree land refused for %s (attempt %d/%d): %s — retrying before reap (#3613)\n",
			filepath.Base(wtPath), attempt, dispatchLandRefusedAttempts, res.Reason)
		dispatchLandRetrySleep(attempt)
		attempt++
		res = dispatchLandWorktreeOnce(root, wtPath, base, tree)
	}
	if !res.OK && strings.TrimSpace(res.Reason) != "" {
		// Surface WHY a worker produced no commit — a refused land is silent otherwise,
		// leaving an operator to guess whether the worker crashed or was refused (#3178).
		// The attempt count makes each race-refusal a countable witness line (#3613).
		fmt.Fprintf(os.Stderr, "fak dispatch: worktree land refused for %s (%d/%d attempts): %s\n",
			filepath.Base(wtPath), attempt, dispatchLandRefusedAttempts, res.Reason)
	}
	_ = dispatchReapWorktree(root, wtPath)
}

// landWorkerWorktreeVerified lands a worker's worktree diff-since-base onto the trunk
// with the #3178 in-worktree build verify (dispatchLandVerify) wired in: a red build
// REFUSES the land (applied:false, committed:false) so a broken edit never reaches
// main. git is nil in production (the default runner); the witness test injects a fake
// git and pins dispatchLandVerify red/green to drive refuse-on-red / pass-on-green.
func landWorkerWorktreeVerified(root, wtPath, base string, tree []string, git workerworktree.GitRunner) workerworktree.Result {
	return workerworktree.Land(root, wtPath, base, "", tree, dispatchLandVerify, git)
}

// dispatchWitnessScanLimit bounds the no-basesha fallback window, mirroring the
// Python worker_resolving_sha scan_limit.
const dispatchWitnessScanLimit = 300

func witnessExitedWorkers(root, runsDir string, live bool) (map[string]any, []dispatchtick.WitnessRecord) {
	audited := []any{}
	buckets := map[string][]any{
		dispatchtick.ClaimWitnessed:   {},
		dispatchtick.ClaimUnwitnessed: {},
		dispatchtick.ClaimNoCommit:    {},
	}
	var records []dispatchtick.WitnessRecord
	for _, log := range resolveLogs(runsDir) {
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if !ok {
			continue
		}
		stem := strings.TrimSuffix(log, filepath.Ext(log))
		if _, err := os.Stat(stem + dispatchtick.WitnessSidecarSuffix); err == nil {
			continue // audited once; a commit's diff (so its verdict) is immutable
		}
		pid, ok := readPID(stem + ".pid")
		if !ok {
			continue // no pid -> cannot prove the worker finished -> not yet auditable
		}
		if dispatchPIDAlive(pid) {
			continue // still running -> it may not have committed yet
		}
		base := ""
		if b, err := os.ReadFile(stem + dispatchtick.BaseSHASidecarSuffix); err == nil {
			base = strings.TrimSpace(string(b))
		}
		// #3515: remember whether this worker ran ISOLATED in a per-worker worktree
		// (the sidecar is consumed by the land below). Such a worker never edited the
		// shared trunk, so any dirty trunk file inside its lane belongs to a live peer
		// — the stranded-poison revert rung below must never fire for it.
		_, wtErr := os.Stat(stem + dispatchWorktreeSidecarSuffix)
		ranInWorktree := wtErr == nil
		// #3168: the pid is provably dead. If this worker ran in a per-worker git
		// worktree, land its diff onto the trunk and reap the worktree BEFORE the
		// resolving-SHA scan, so the just-landed commit is what gets witnessed. All
		// steps fail-open: a land/reap fault is swallowed and the sweep proceeds to
		// audit the resolve log exactly as today (a leaked worktree is reaped later
		// by worktree_doctor.py --sweep-disposable, which knows the marker). Only in a
		// live sweep — a dry-run must never mutate the trunk.
		if live {
			landAndReapWorkerWorktree(root, stem, base)
		}
		sha := dispatchWitnessResolvingSHA(root, issue, base)
		tree := readResolveLeaseTree(stem + dispatchLeaseTreeSidecarSuffix)
		var reverted []string
		var rec dispatchtick.WitnessRecord
		if sha == "" {
			tail, size := dispatchWitnessLogTail(log)
			rec = dispatchtick.WitnessRecord{
				Issue:  issue,
				Log:    filepath.Base(log),
				Claim:  dispatchtick.ClaimNoCommit,
				Reason: dispatchtick.ClassifyNoCommitReason(tail, size),
			}
			// #3515 revert rung: a dead shared-trunk worker that landed NO resolving
			// commit may have stranded uncommitted, non-compiling edits that red
			// `go build` for every peer until a human intervenes. Archive-then-stash
			// exactly its lane-scoped dirty files — live sweeps only (a dry-run never
			// mutates), and never for a worktree-isolated worker (its lane's dirty
			// trunk files can only be a peer's).
			if live && !ranInWorktree {
				reverted = dispatchWitnessRevertStranded(root, runsDir, stem, tree)
			}
		} else {
			verdict, witness := dispatchWitnessCommitAudit(root, sha)
			claim := dispatchtick.ClaimUnwitnessed
			if dispatchtick.CommitWitnessed(verdict, witness) {
				claim = dispatchtick.ClaimWitnessed
			}
			rec = dispatchtick.WitnessRecord{
				Issue:   issue,
				Log:     filepath.Base(log),
				SHA:     sha,
				Claim:   claim,
				Verdict: verdict,
				Witness: witness,
			}
			// #3838 test-run rung: bind the done-claim to a green test of the resolving
			// commit's changed package, ALONGSIDE (never replacing) the diff-shape verdict.
			// The default runner is opt-in, so on a normal tick this grades UNRUN — a valid,
			// surfaced state — while the `verify` skill is the always-on live consumer.
			ran, passed := dispatchWitnessTestRun(root, sha)
			rec.TestClaim = dispatchtick.GradeTestRun(ran, passed)
		}
		if len(tree) > 0 {
			if changed, ok := dispatchWitnessCommitPaths(root, sha); ok {
				rec.OutOfLanePathCount = dispatchCountPathsOutsideTrees(changed, tree)
				if rec.OutOfLanePathCount > 0 {
					rec.FootprintClaim = "CLAIM_OUT_OF_LANE"
				} else {
					rec.FootprintClaim = "CLAIM_SCOPE_CLEAN"
				}
			}
		}
		// Layer 5b: scrape the model the slot was pinned to from its .model sidecar (absent
		// for a floor/seat-default worker -> Model stays ""). It feeds both the .witness
		// record's model key and the Layer-2 downgrade decision (which next chain model to
		// re-dispatch on after a model-switchable wall).
		if b, err := os.ReadFile(stem + dispatchtick.ModelSidecarSuffix); err == nil {
			rec.Model = strings.TrimSpace(string(b))
		}
		// #5416 track E: scrape the rung that actually served the slot from its .zone
		// sidecar, THROUGH the allowlist — a truncated or hand-edited file leaves Zone
		// empty, which the fold counts as unattributed rather than as running on this box.
		// Absent for every slot spawned before the opt-in seam, and that absence is the
		// honest answer: nothing recorded where those ran.
		if b, err := os.ReadFile(stem + dispatchtick.ZoneSidecarSuffix); err == nil {
			if z, ok := dispatchtick.ZoneFromSidecar(string(b)); ok {
				rec.Zone = string(z)
			}
		}
		records = append(records, rec)
		row := rec.Map()
		if len(reverted) > 0 {
			// #3515: surface the revert as first-class evidence on the graded row (and
			// so in the .witness sidecar below) — an operator can see exactly which
			// stranded paths were stashed, and recover them via `git stash pop` or the
			// runs-dir archive copy.
			row["reverted"] = reverted
		}
		// #4324 release-on-exit: this worker is provably finished and the sweep has
		// stopped writing under its lease (the land+reap and the stranded-revert rung
		// both ran above), so a NORMAL exit hands the lane back NOW through the fenced
		// CAS delete instead of stranding it for the ~40-min TTL and refusing peers
		// against a holder that no longer exists. Live sweeps only — a dry-run never
		// mutates a ref. An abnormal exit (an unclassifiable crash, or stranded edits)
		// deliberately keeps its lease: TTL expiry plus the dead-holder reclaim is the
		// correct path when a lane may be mid-write. Fail-open: the outcome is recorded
		// on the graded row and never propagated.
		if live && dispatchWorkerExitReleasesLease(rec, len(reverted)) {
			if id, outcome := dispatchLeaseReleaser(root, stem); id != "" {
				row["lease_released"] = id
			} else {
				row["lease_release_refused"] = outcome
			}
		}
		audited = append(audited, row)
		buckets[rec.Claim] = append(buckets[rec.Claim], row)
		if live {
			if b, err := json.Marshal(row); err == nil {
				_ = os.WriteFile(stem+dispatchtick.WitnessSidecarSuffix, b, 0o644)
			}
		}
	}
	payload := map[string]any{
		"live":        live,
		"audited":     audited,
		"witnessed":   buckets[dispatchtick.ClaimWitnessed],
		"unwitnessed": buckets[dispatchtick.ClaimUnwitnessed],
		"no_commit":   buckets[dispatchtick.ClaimNoCommit],
	}
	return payload, records
}

// dispatchWitnessLogTail reads the last WitnessTailBytes of a worker log (the guard
// summary + final turn live at the end) without loading a possibly multi-MB file.
// Fail-open: a stat error yields ("", -1) so the classifier's size floor disengages.
func dispatchWitnessLogTail(log string) (string, int64) {
	st, err := os.Stat(log)
	if err != nil {
		return "", -1
	}
	size := st.Size()
	f, err := os.Open(log)
	if err != nil {
		return "", size
	}
	defer f.Close()
	if size > dispatchtick.WitnessTailBytes {
		if _, err := f.Seek(-int64(dispatchtick.WitnessTailBytes), io.SeekEnd); err != nil {
			return "", size
		}
	}
	b, err := io.ReadAll(f)
	if err != nil {
		return "", size
	}
	return string(b), size
}

// dispatchWitnessResolvingSHAGit finds the newest commit whose SUBJECT cites #issue,
// scoped to baseSHA..HEAD (the per-worker window recorded at spawn) when the base is
// known, else the most recent dispatchWitnessScanLimit commits. Fail-open: any git
// error yields "" so the slot claims nothing.
func dispatchWitnessResolvingSHAGit(root string, issue int, baseSHA string) string {
	args := []string{"log", "--no-color", "--pretty=format:%H\x1f%s"}
	if baseSHA != "" {
		args = append(args, baseSHA+"..HEAD")
	} else {
		args = append(args, "-n", strconv.Itoa(dispatchWitnessScanLimit))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return dispatchtick.FirstResolvingSHA(string(out), issue)
}

func dispatchWitnessCommitPathsGit(root, sha string) ([]string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "show", "--pretty=format:", "--name-only", sha)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	var paths []string
	for _, line := range strings.Split(string(out), "\n") {
		if path := strings.TrimSpace(line); path != "" {
			paths = append(paths, path)
		}
	}
	return paths, true
}

// dispatchCountPathsOutsideTrees is workerworktree.CountPathsOutsideTrees — the SAME
// out-of-lane rule the land step refuses on, so the witness sweep can never grade a diff
// in-lane that Land would have called out-of-lane. It used to be a byte-identical copy.
func dispatchCountPathsOutsideTrees(changed, trees []string) int {
	return workerworktree.CountPathsOutsideTrees(changed, trees)
}

// dispatchWitnessCommitAuditDos grades sha through `dos commit-audit --json` and
// returns its (verdict, witness) pair. The command emits a JSON array, one row per
// audited sha; a dict is accepted too. Fail-open: an exec/parse failure yields empty
// strings, which grade to the conservative CLAIM_UNWITNESSED.
func dispatchWitnessCommitAuditDos(root, sha string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "commit-audit", sha, "--workspace", root, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", ""
	}
	var parsed any
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(out))), &parsed); err != nil {
		return "", ""
	}
	row := map[string]any{}
	switch doc := parsed.(type) {
	case map[string]any:
		row = doc
	case []any:
		for _, item := range doc {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if rowSHA := dispatchMapString(m, "sha"); rowSHA != "" && strings.HasPrefix(sha, rowSHA) {
				row = m
				break
			}
		}
		if len(row) == 0 && len(doc) > 0 {
			if m, ok := doc[0].(map[string]any); ok {
				row = m
			}
		}
	}
	return dispatchMapString(row, "verdict"), dispatchMapString(row, "witness")
}

// dispatchWitnessTestRunGo is the default #3838 runner: for a resolving commit, run the
// tests of the packages it CHANGED and report (ran, passed). It is OPT-IN via
// dispatchWitnessTestRunEnv so the hot dispatch loop never eats an in-tick `go test`;
// unset -> (false,false) -> the rung grades CLAIM_TEST_UNRUN. FAIL-SAFE: an empty sha,
// no test-bearing changed package, or a launch fault all yield ran=false (UNRUN), so a
// disabled/faulted run can NEVER masquerade as a green pass. WSL-aware: on Windows the
// tests run through test.ps1 (the same seam `fak affected` uses), else `go test`.
func dispatchWitnessTestRunGo(root, sha string) (ran, passed bool) {
	if sha == "" || !dispatchWitnessTestRunEnabled() {
		return false, false // no commit, or the in-tick runner is opt-in and off -> UNRUN
	}
	pkgs := dispatchWitnessChangedTestPkgs(root, sha)
	if len(pkgs) == 0 {
		return false, false // nothing test-bearing changed -> honest UNRUN, not a fake pass
	}
	name, cmdArgs := affectedTestCommand(runtime.GOOS, append([]string{"test", "-count=1"}, pkgs...))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, cmdArgs...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	err := cmd.Run()
	return true, err == nil
}

// dispatchWitnessTestRunEnabled reports whether the opt-in in-tick runner is on. Default
// OFF (unset/false/off/no/0) — the rung then surfaces UNRUN rather than paying an
// in-tick test on every witnessed commit.
func dispatchWitnessTestRunEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(dispatchWitnessTestRunEnv))) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}

// dispatchWitnessChangedTestPkgs maps a resolving commit's changed .go files to the set
// of `./dir/` package patterns that actually carry Go tests. A changed dir with no
// *_test.go is skipped (running it would grade a vacuous pass), so the runner only ever
// goes GREEN/RED on a package whose tests it truly executed. Fail-open: a git error
// yields no packages -> UNRUN.
func dispatchWitnessChangedTestPkgs(root, sha string) []string {
	out, err := gitOut(root, "show", "--name-only", "--pretty=format:", sha)
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var pkgs []string
	for _, line := range strings.Split(out, "\n") {
		f := strings.TrimSpace(line)
		if !strings.HasSuffix(f, ".go") {
			continue
		}
		dir := filepath.ToSlash(filepath.Dir(f))
		if dir == "." || dir == "" || seen[dir] {
			continue
		}
		seen[dir] = true
		matches, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(dir), "*_test.go"))
		if len(matches) == 0 {
			continue // changed package carries no tests -> nothing to bind a pass to
		}
		pkgs = append(pkgs, "./"+dir+"/")
	}
	return pkgs
}

// --- #3515: the crashed-worker stranded-poison revert rung ----------------------
//
// A worker that dies mid-edit in the SHARED trunk (no per-worker worktree) can
// strand an uncommitted, non-compiling file that reds `go build` for every peer
// until a human notices. The witness sweep is the one place that already proves
// "this worker is DEAD and landed NO commit", so it is the safe owner of the
// cleanup. The rung is FAIL-OPEN at every step — it deletes work when it is
// wrong, so any ambiguity (git fault, indeterminate build, green build, no lease
// scoping, un-archived file) stands it down and leaves the tree exactly as found.

// dispatchWitnessDirtyFile is one uncommitted working-tree entry from
// `git status --porcelain -z`: its repo-relative path plus whether git tracks it.
// Untracked strands are archived but never stashed — the sole sanctioned revert
// primitive (`git stash push -- <paths>`) rejects a pathspec naming an untracked
// file outright (handling them needs `-u`, a deliberate follow-on).
type dispatchWitnessDirtyFile struct {
	Path      string
	Untracked bool
}

// Injectable seams for the #3515 rung, mirroring the resolving-SHA / commit-audit /
// test-run seams so the sweep test can drive the rung hermetically (the test stubs
// only the build verdict and exercises real `git status` / `git stash push`).
var dispatchWitnessDirtyPaths = dispatchWitnessDirtyPathsGit
var dispatchWitnessStrandedBuildFails = dispatchWitnessStrandedBuildFailsGo
var dispatchWitnessStashPaths = dispatchWitnessStashPathsGit

// witnessStrandedArchiveSuffix names the per-worker archive dir (under the runs
// dir) the rung copies stranded bytes into BEFORE stashing them — the same
// archive-before-destroy discipline as worktree_doctor.py --sweep-disposable.
const witnessStrandedArchiveSuffix = ".stranded"

// dispatchWitnessRevertStranded reverts a provably-dead, no-commit worker's
// stranded lane-scoped edits, when and only when ALL fire conditions hold:
//
//  1. the worker declared a non-empty lease tree (its .tree sidecar);
//  2. `git status --porcelain` reports dirty files, and root IS the repo toplevel;
//  3. at least one dirty file falls under the lease tree's globs;
//  4. at least one matching dirty file is a .go file (a docs-only strand cannot
//     be "the non-compiling file"), and at least one matching file is TRACKED
//     (else there is nothing the sanctioned stash primitive can revert);
//  5. a SCOPED `go build` of exactly the package dirs containing the matching
//     dirty .go files FAILS — the strand is provably what reds the build.
//
// Then: archive every matching dirty file under the runs dir, and revert the
// tracked ones with the sole gitgate-permitted primitive, exactly
// `git stash push -- <concrete paths>` (recoverable via `git stash pop`; bare
// stash / reset --hard / clean are refused by internal/gitgate for the same
// peer-WIP-sweeping reason this rung matches paths one by one). Every other
// outcome returns nil with the tree untouched. Returns the stashed paths.
func dispatchWitnessRevertStranded(root, runsDir, stem string, tree []string) []string {
	if len(tree) == 0 {
		return nil // no declared lease tree -> no safe attribution -> never touch the tree
	}
	dirty, ok := dispatchWitnessDirtyPaths(root)
	if !ok || len(dirty) == 0 {
		return nil
	}
	var mine []dispatchWitnessDirtyFile
	var stash []string
	goDirs := map[string]bool{}
	for _, f := range dirty {
		if strings.HasSuffix(f.Path, "/") {
			continue // an all-untracked DIR entry: unstashable, and its files are unlisted
		}
		if !dispatchPathInLeaseTree(f.Path, tree) {
			continue // a peer's WIP outside the dead worker's lane: never touched
		}
		mine = append(mine, f)
		if !f.Untracked {
			stash = append(stash, f.Path)
		}
		if strings.HasSuffix(f.Path, ".go") {
			if dir := filepath.ToSlash(filepath.Dir(f.Path)); dir == "." || dir == "" {
				goDirs["."] = true
			} else {
				goDirs["./"+dir] = true
			}
		}
	}
	if len(stash) == 0 || len(goDirs) == 0 {
		return nil // nothing both provably-poisonous AND revertible by the sanctioned primitive
	}
	pkgs := make([]string, 0, len(goDirs))
	for d := range goDirs {
		pkgs = append(pkgs, d)
	}
	sort.Strings(pkgs)
	failed, ok := dispatchWitnessStrandedBuildFails(root, pkgs)
	if !ok || !failed {
		return nil // green or indeterminate -> the strand is not proven poison -> preserve it
	}
	if !dispatchWitnessArchiveStranded(root, runsDir, stem, mine) {
		return nil // an un-archived file is never destroyed
	}
	if err := dispatchWitnessStashPaths(root, stash); err != nil {
		return nil
	}
	return stash
}

// dispatchPathInLeaseTree reports whether a repo-relative path falls under one of
// the lease tree's entries ("cmd/**", "docs/shared.md"). Same directory-ancestry
// semantics as the lease geometry (a trailing /** or /* names the directory's
// subtree), EXCEPT the wildcard-all spellings ("**", "**/*", "*"): a lease naming
// the whole tree provides no per-lane scoping, and this matcher feeds a
// DESTRUCTIVE rung, so wildcard-all conservatively matches nothing here.
func dispatchPathInLeaseTree(path string, tree []string) bool {
	path = strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/")
	if path == "" {
		return false
	}
	for _, t := range tree {
		t = strings.ReplaceAll(strings.TrimSpace(t), "\\", "/")
		t = strings.TrimPrefix(t, "./")
		t = strings.TrimSuffix(t, "/")
		t = strings.TrimSuffix(t, "/**")
		t = strings.TrimSuffix(t, "/*")
		t = strings.TrimSuffix(t, "/")
		if t == "" || t == "**" || t == "**/*" || t == "*" {
			continue
		}
		if path == t || strings.HasPrefix(path, t+"/") {
			return true
		}
	}
	return false
}

// dispatchWitnessDirtyPathsGit lists uncommitted working-tree entries via
// `git status --porcelain -z` (NUL-separated: no quoting ambiguity). It first
// proves root IS the repo toplevel — were root a bare dir under some enclosing
// repo, git would resolve THAT repo and the later stash would mutate a tree the
// sweep does not own. Fail-open: any fault yields (nil, false) and the rung
// stands down.
func dispatchWitnessDirtyPathsGit(root string) ([]dispatchWitnessDirtyFile, bool) {
	top, err := dispatchWitnessGitOut(root, "rev-parse", "--show-toplevel")
	if err != nil || !dispatchWitnessSamePath(strings.TrimSpace(top), root) {
		return nil, false
	}
	out, err := dispatchWitnessGitOut(root, "status", "--porcelain", "-z")
	if err != nil {
		return nil, false
	}
	var files []dispatchWitnessDirtyFile
	entries := strings.Split(out, "\x00")
	for i := 0; i < len(entries); i++ {
		e := entries[i]
		if len(e) < 4 || e[2] != ' ' {
			continue
		}
		code, path := e[:2], e[3:]
		if code[0] == 'R' || code[0] == 'C' {
			i++ // the NEXT NUL token is the rename/copy ORIGIN path
		}
		if code == "!!" {
			continue
		}
		files = append(files, dispatchWitnessDirtyFile{Path: path, Untracked: code == "??"})
	}
	return files, true
}

// dispatchWitnessSamePath compares two on-disk paths for identity, tolerant of
// symlinks (macOS's /var -> /private/var), separators, and Windows casing.
func dispatchWitnessSamePath(a, b string) bool {
	if ra, err := filepath.EvalSymlinks(a); err == nil {
		a = ra
	}
	if rb, err := filepath.EvalSymlinks(b); err == nil {
		b = rb
	}
	return strings.EqualFold(filepath.ToSlash(filepath.Clean(a)), filepath.ToSlash(filepath.Clean(b)))
}

// dispatchWitnessGitOut runs one git command in root and returns its stdout,
// with the same timeout/window discipline as the sweep's other git helpers.
func dispatchWitnessGitOut(root string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	return string(out), err
}

// dispatchWitnessStrandedBuildFailsGo runs a SCOPED `go build` of exactly the
// package dirs containing the dead worker's matching dirty .go files, reporting
// (failed, ok). ok=false means the verdict is INDETERMINATE (no packages, no
// toolchain, or a timeout) and the caller must preserve the work — unlike the
// land-site verify this rung destroys bytes on a positive verdict, so a missing
// toolchain stands it down instead of waving it through. -o points at a
// throwaway dir so a main package's executable never lands in the shared tree.
func dispatchWitnessStrandedBuildFailsGo(root string, pkgs []string) (failed, ok bool) {
	if len(pkgs) == 0 {
		return false, false
	}
	if _, err := exec.LookPath("go"); err != nil {
		return false, false
	}
	tmp, err := os.MkdirTemp("", "fak-stranded-build-")
	if err != nil {
		return false, false
	}
	defer os.RemoveAll(tmp)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	args := append([]string{"build", "-o", tmp + string(os.PathSeparator)}, pkgs...)
	cmd := exec.CommandContext(ctx, "go", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	err = cmd.Run()
	if ctx.Err() != nil {
		return false, false // timeout: indeterminate, never grade it as poison
	}
	return err != nil, true
}

// dispatchWitnessArchiveStranded copies each matching dirty file into the
// per-worker <stem>.stranded/ dir under the runs dir, mirroring the repo-relative
// layout, BEFORE anything is stashed. Returns false on any copy fault so the
// caller never destroys an un-archived byte. A dirty DELETION has no bytes to
// copy and is skipped (the stash itself records it, recoverably).
func dispatchWitnessArchiveStranded(root, runsDir, stem string, files []dispatchWitnessDirtyFile) bool {
	dir := filepath.Join(runsDir, filepath.Base(stem)+witnessStrandedArchiveSuffix)
	for _, f := range files {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(f.Path)))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return false
		}
		dst := filepath.Join(dir, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return false
		}
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			return false
		}
	}
	return true
}

// dispatchWitnessStashPathsGit runs exactly `git stash push -- <paths>` — the
// sole gitgate-permitted revert primitive (internal/gitgate: a pathspec-scoped
// stash create is allowed; bare stash / reset --hard / checkout -- / clean are
// refused). The stash keeps the reverted work recoverable (`git stash pop`) on
// top of the runs-dir archive copy.
func dispatchWitnessStashPathsGit(root string, paths []string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", append([]string{"stash", "push", "--"}, paths...)...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	return cmd.Run()
}
