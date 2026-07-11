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

// landAndReapWorkerWorktreeDefault is the production land+reap: apply the worktree's
// diff-since-base onto the trunk as the worker's own stamped commit (scoped to its
// declared lease tree), then force-remove the worktree. Both fail-open.
func landAndReapWorkerWorktreeDefault(root, wtPath, base string, tree []string) {
	// No commit-message file: Land derives the subject from the worktree tip so the
	// landed commit keeps the worker's own #N-citing, (fak <leaf>)-stamped subject.
	// verify=dispatchLandVerify (#3178): a red `go build ./...` in the worktree refuses
	// the land (nothing applied/committed) so a broken edit never reaches main.
	res := landWorkerWorktreeVerified(root, wtPath, base, tree, nil)
	if !res.OK && strings.TrimSpace(res.Reason) != "" {
		// Surface WHY a worker produced no commit — a refused land is silent otherwise,
		// leaving an operator to guess whether the worker crashed or was refused (#3178).
		fmt.Fprintf(os.Stderr, "fak dispatch: worktree land refused for %s: %s\n",
			filepath.Base(wtPath), res.Reason)
	}
	_ = workerworktree.Reap(root, wtPath, nil)
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
		var rec dispatchtick.WitnessRecord
		if sha == "" {
			tail, size := dispatchWitnessLogTail(log)
			rec = dispatchtick.WitnessRecord{
				Issue:  issue,
				Log:    filepath.Base(log),
				Claim:  dispatchtick.ClaimNoCommit,
				Reason: dispatchtick.ClassifyNoCommitReason(tail, size),
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
		// Layer 5b: scrape the model the slot was pinned to from its .model sidecar (absent
		// for a floor/seat-default worker -> Model stays ""). It feeds both the .witness
		// record's model key and the Layer-2 downgrade decision (which next chain model to
		// re-dispatch on after a model-switchable wall).
		if b, err := os.ReadFile(stem + dispatchtick.ModelSidecarSuffix); err == nil {
			rec.Model = strings.TrimSpace(string(b))
		}
		records = append(records, rec)
		row := rec.Map()
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
