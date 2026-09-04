package workerworktree

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipfence"
)

const (
	// LandResult* are the typed terminal results emitted by worker land (#8813).
	LandResultSuccess   = "success"
	LandResultNoOp      = "no-op"
	LandResultConflict  = "conflict"
	LandResultStaleBase = "stale-base"
)

// LandRefusalRetryable reports whether a refused Land is worth re-attempting on
// the same worktree (#3613). True only for the readback-mismatch race class: a
// concurrent commit on the shared index swept the worker's paths, so the refusal
// is TRANSIENT — the worktree still holds the only copy of the diff, and a
// re-land on the moved HEAD re-applies it cleanly. Every other refusal (red
// verify, apply conflict, disambiguation invariant, git fault) is deterministic
// for the same inputs: replaying it cannot change the verdict.
func LandRefusalRetryable(reason string) bool {
	return strings.Contains(reason, LandReadbackMismatchToken)
}

// casRetrySleep sleeps a short jittered backoff before a lost-CAS retry so N
// colliding landers spread out instead of re-contending the ref in lockstep.
// Package var so unit tests stub it to a no-op (#3570).
var casRetrySleep = func(attempt int) {
	// randomized jitter backoff between CAS attempts (100ms–500ms) (#11235)
	ms := 100 + rand.Intn(401)
	time.Sleep(time.Duration(ms) * time.Millisecond)
}

// mergeTreeRebase performs an in-memory 3-way merge using git merge-tree --write-tree
// to rebase workerCommit (built on baseSHA) onto targetHEAD (#11235).
// Returns the resulting tree SHA or an error if there are conflicts or git failures.
func mergeTreeRebase(root, baseSHA, targetHEAD, workerCommit string, git GitRunner) (string, error) {
	if baseSHA == "" || targetHEAD == "" || workerCommit == "" {
		return "", fmt.Errorf("mergeTreeRebase: baseSHA, targetHEAD, and workerCommit must not be empty")
	}
	rc, out := run(git, root, []string{"merge-tree", "--write-tree", "--merge-base", baseSHA, targetHEAD, workerCommit})
	if rc != 0 {
		return "", fmt.Errorf("mergeTreeRebase: 3-way merge conflict or git error (rc=%d): %s", rc, tail(out, 200))
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 {
		return "", fmt.Errorf("mergeTreeRebase: no output from git merge-tree")
	}
	treeSHA := strings.TrimSpace(lines[0])
	if len(treeSHA) != 40 && len(treeSHA) != 64 {
		return "", fmt.Errorf("mergeTreeRebase: invalid tree SHA %q", treeSHA)
	}
	return treeSHA, nil
}

// MergeTreeRebase is the exported form of mergeTreeRebase.
func MergeTreeRebase(root, baseSHA, targetHEAD, workerCommit string, git GitRunner) (string, error) {
	return mergeTreeRebase(root, baseSHA, targetHEAD, workerCommit, git)
}

// isolatedLandRetryCap reads IsolatedLandRetryEnv: the total bounded CAS attempts
// for one landIsolated call. Unset, unparsable, or <1 falls back to the default —
// fail-open to a sane bound, never to an unbounded loop.
func isolatedLandRetryCap() int {
	v := strings.TrimSpace(os.Getenv(IsolatedLandRetryEnv))
	if v == "" {
		return defaultIsolatedLandRetry
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return defaultIsolatedLandRetry
	}
	return n
}

// Land applies a worker's edits from its isolated worktree onto the TRUNK as one
// stamped, signed-off commit ON main — the serialized commit-to-trunk step that
// keeps OFF_TRUNK from ever tripping (#1334).
//
// It captures the worktree's FULL delta since baseSHA (not HEAD — a guarded
// worker commits IN its detached worktree, so `git diff HEAD` would be empty and
// the change would silently evaporate; diffing against the pinned base captures it
// whether committed, staged, or unstaged) and applies it to the trunk worktree,
// then commits -s by explicit path. The caller holds the lane lease, which
// serializes this so two workers never apply to the same leaf tree at once.
//
// commitMsgFile is the message for `git commit -s -F`; when empty, Land derives
// the message from the worktree tip (`git log -1 --format=%B`) so the landed trunk
// commit keeps the worker's own authored, stampable subject (which cites #N and
// carries the (fak <leaf>) trailer — the worker's acceptance contract). paths,
// when non-empty, scopes the commit to the worker's declared region — never an
// add -A. verify (when non-nil) is a witness run in the worktree before anything
// touches the trunk; a failed witness refuses the land. FAIL-OPEN on git errors.
//
// ONE exception to fail-open: the hard-self core lock (#5392, corelock.go). A diff
// touching a core-locked kernel path is REFUSED unless a maintenance witness
// resolves CONFIRMED — supplied by the WithCoreLockWitness option or by the
// CoreLockWitnessTrailer in the commit message. That gate runs before any apply, so
// a refused land leaves the trunk exactly as it found it.
// CountPathsOutsideTrees counts how many of `changed` fall outside every declared tree in
//
//	rees`. Both sides are normalised to forward slashes and stripped of leading/trailing
//
// separators first, and a tree matches a path exactly or as a "<tree>/" prefix; an empty
// tree entry matches nothing (so a stray "" can never swallow the whole diff).
//
// Exported because the out-of-lane count is a LEASE-SCOPE verdict, and `fak dispatch tick`'s
// witness sweep grades the same worker diffs with it. The two used to carry byte-identical
// copies, which is how a land-time refusal and its post-hoc witness silently disagree.
func CountPathsOutsideTrees(changed, trees []string) int {
	outside := 0
	for _, path := range changed {
		path = strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/")
		inside := false
		for _, tree := range trees {
			tree = strings.Trim(strings.ReplaceAll(tree, "\\", "/"), "/")
			if tree != "" && (path == tree || strings.HasPrefix(path, tree+"/")) {
				inside = true
				break
			}
		}
		if !inside {
			outside++
		}
	}
	return outside
}

func expandLandPaths(wtPath, diffRef string, requested []string, git GitRunner) ([]string, error) {
	args := []string{"ls-files", "--cached", "--others", "--exclude-standard", "--"}
	args = append(args, requested...)
	rc, out := run(git, wtPath, args)
	if rc != 0 {
		return nil, fmt.Errorf("could not expand declared land paths — fail open")
	}
	candidates := strings.Fields(out)
	if len(candidates) > 0 {
		addArgs := []string{"add", "-N", "--"}
		addArgs = append(addArgs, candidates...)
		if rc, _ := run(git, wtPath, addArgs); rc != 0 {
			return nil, fmt.Errorf("could not expose untracked land paths — fail open")
		}
	}
	rc, changedOut := run(git, wtPath, []string{"diff", "--name-only", diffRef})
	if rc != 0 {
		return nil, fmt.Errorf("could not inspect declared land paths — fail open")
	}
	changed := strings.Fields(changedOut)
	expanded := make([]string, 0, len(changed))
	seen := make(map[string]bool)
	for _, req := range requested {
		matched := false
		cleanReq := strings.TrimSuffix(filepath.ToSlash(filepath.Clean(req)), "/")
		for _, name := range changed {
			cleanName := filepath.ToSlash(filepath.Clean(name))
			if cleanName == cleanReq || strings.HasPrefix(cleanName, cleanReq+"/") {
				matched = true
				if !seen[cleanName] {
					seen[cleanName] = true
					expanded = append(expanded, cleanName)
				}
			}
		}
		if !matched {
			return nil, fmt.Errorf("declared land path %q contributes no changed file — refusing partial land", req)
		}
	}
	sort.Strings(expanded)
	return expanded, nil
}
func Land(root, wtPath, baseSHA, commitMsgFile string, paths []string, verify VerifyHook, git GitRunner, opts ...LandOption) (res Result) {
	cfg := newLandConfig(opts)
	tracker := newLandProgressTracker(cfg)
	cfg.tracker = tracker
	finishAdmission := beginLandPhase(tracker, "admission", 0)
	admissionActive := true
	defer func() {
		if admissionActive {
			finishAdmission()
		}
		res.Cost = tracker.receipt()
	}()
	diffRef := baseSHA
	if diffRef == "" {
		diffRef = "HEAD"
	}
	// Make untracked descendants visible to git diff, then collapse directory
	// declarations to the exact changed files they own. The resulting list is the
	// single pathset used by commit, readback, and post-CAS sync.
	if len(paths) > 0 {
		var err error
		paths, err = expandLandPaths(wtPath, diffRef, paths, git)
		if err != nil {
			return Result{OK: false, Reason: err.Error()}
		}
	}
	stripWorktreeWIPFences(wtPath, paths)
	rc, diff := run(git, wtPath, []string{"diff", "--binary", diffRef})
	if rc != 0 {
		return Result{OK: false, Reason: "could not read worktree diff vs " + diffRef + " (git error) — fail open"}
	}
	if strings.TrimSpace(diff) == "" {
		// No net change since the base: the worker landed nothing. The caller's
		// commit-witness (dos commit-audit) decides whether the slot was productive.
		return Result{OK: true, Code: LandResultNoOp, Applied: false, Committed: false,
			Reason: "no net diff in worktree vs " + diffRef + " to land"}
	}
	checkBase := strings.TrimSpace(baseSHA)
	if checkBase == "" {
		if in, err := LoadIntent(wtPath); err == nil {
			checkBase = strings.TrimSpace(in.BaseSHA)
		}
	}
	if checkBase != "" {
		rc, _ := run(git, root, []string{"merge-base", "--is-ancestor", checkBase, "HEAD"})
		if rc != 0 {
			return Result{
				OK:        false,
				Code:      LandResultStaleBase,
				Applied:   false,
				Committed: false,
				Reason:    fmt.Sprintf("base commit %s is not an ancestor of trunk HEAD (stale base)", shortSHA(checkBase)),
			}
		}
	}
	// The landing pathset, read ONCE: it feeds both the out-of-lane ledger and the
	// hard-self core-lock gate below, so the two can never disagree about what this
	// land actually carries. An unreadable name list leaves it empty — the ledger
	// then counts 0 exactly as before, and the gate falls back to the captured
	// diff's own headers rather than treating "unreadable" as "nothing locked".
	namesRC, names := run(git, wtPath, []string{"diff", "--name-only", diffRef})
	if namesRC != 0 {
		names = ""
	}
	tracker.setPatchScope(countPatchScopeFiles(names), int64(len(diff)))
	droppedOutOfLane := 0
	if len(paths) > 0 && names != "" {
		droppedOutOfLane = CountPathsOutsideTrees(strings.Fields(names), paths)
	}
	if verify != nil {
		finishAdmission()
		admissionActive = false
		finishValidation := beginLandPhase(tracker, "prospective-validation", 0)
		if ok, detail := verify(wtPath); !ok {
			finishValidation()
			return Result{OK: false, Applied: false, Committed: false,
				Reason: "worktree verify failed, refusing to land: " + detail}
		}
		finishValidation()
	}
	if admissionActive {
		finishAdmission()
		admissionActive = false
	}
	phaseDone := beginLandPhase(tracker, "policy-admission", 0)
	// Resolve a message file: use the caller's, else materialize the worktree tip's
	// message to a temp file so the landed commit keeps the worker's own subject.
	msgFile, cleanup, err := resolveMsgFile(wtPath, baseSHA, commitMsgFile, git)
	if err != nil {
		phaseDone()
		return Result{OK: false, Applied: false, Committed: false,
			Reason: "could not resolve commit message: " + err.Error()}
	}
	defer cleanup()

	// HARD-SELF CORE LOCK (#5392). The last gate before anything touches the trunk:
	// a diff carrying a core-locked kernel path (internal/adjudicator/**,
	// internal/abi/**, internal/corelocks/**) lands only with a maintenance witness
	// that resolves CONFIRMED — supplied by WithCoreLockWitness (the CLI flag) or by
	// the CoreLockWitnessTrailer in the commit message. This runs HERE, in the
	// lander, because the sanctioned worktree cannot use `fak commit` (OFF_TRUNK on a
	// detached HEAD) and the default isolated path commits through commit-tree, which
	// runs no git hook. Refusing here leaves the trunk index, worktree and HEAD
	// untouched; the worker's diff stays in its worktree.
	if refusal, fired := coreLockLandGate(root, names, diff, msgFile, cfg, git); fired {
		phaseDone()
		refusal.DroppedOutOfLane = droppedOutOfLane
		return refusal
	}
	phaseDone()

	queue := cfg.queue
	if queue == nil {
		queue = DefaultLandingQueue
	}

	landingOp := func() Result {
		// Opt-in race-free layer-2 land (default OFF): stage+commit through a THROWAWAY
		// index so the shared index is never a sweep target. handled=false means it could
		// not isolate safely (detached HEAD, apply conflict, lost CAS, …) and falls through
		// to the baseline shared path below — so enabling it only ever reduces the #3547
		// race window, never regresses it. Path-scoped lands only (a whole-tree land has no
		// safe isolated form here).
		if isolatedLandEnabled() && len(paths) > 0 {
			if r, handled := landIsolated(root, wtPath, diff, msgFile, paths, git, isolatedGitEnv, cfg); handled {
				r.DroppedOutOfLane = droppedOutOfLane
				if r.OK && r.Committed {
					r.Code = LandResultSuccess
				}
				return r
			}
		}

		tracker.setCache("shared-index-fallback", false)
		finishApply := beginLandPhase(tracker, "trunk-apply", 0)
		applied := gitApply(root, diff, git)
		finishApply()
		if !applied.OK {
			return Result{OK: false, Code: LandResultConflict, Applied: false, Committed: false,
				Reason: "git apply to trunk failed", Detail: applied.Detail}
		}
		commitArgs := []string{"commit", "-s", "-F", msgFile}
		if len(paths) > 0 {
			commitArgs = append(commitArgs, "--")
			commitArgs = append(commitArgs, paths...)
		}
		finishCommit := beginLandPhase(tracker, "commit", 0)
		rc, out := run(git, root, commitArgs)
		finishCommit()
		r := Result{OK: rc == 0, Applied: true, Committed: rc == 0, Detail: tail(out, 300), DroppedOutOfLane: droppedOutOfLane}
		if rc == 0 {
			r.Code = LandResultSuccess
		}
		// Opt-in honest-refusal readback (default OFF): confirm the commit we just made
		// actually carries our intended paths. A missing path means our staged change
		// was swept into a concurrent commit on the shared index (#3547); refuse rather
		// than return a false success. FAIL-OPEN — only a positive mismatch flips OK.
		if rc == 0 && len(paths) > 0 && landReadbackEnabled() {
			finishReadback := beginLandPhase(tracker, "land-readback", 0)
			if ok, reason := landReadbackVerify(root, paths, git); !ok {
				r.OK = false
				r.Reason = reason
			}
			finishReadback()
		}
		return r
	}

	res = queue.Coordinate(root, landingOp)
	return res
}

// resolveMsgFile returns a commit-message file path and a cleanup func. A non-empty
// commitMsgFile is used verbatim (no cleanup). Otherwise the worktree tip message
// (git log -1 --format=%B) is written to a temp file so the trunk commit preserves
// the worker's own stamped subject; the temp file is removed by cleanup.
// If the worker has not committed (HEAD == baseSHA), it uses the intent message or
// falls back to a neutral chore subject rather than inheriting an unrelated commit subject (#8813).
func resolveMsgFile(wtPath, baseSHA, commitMsgFile string, git GitRunner) (string, func(), error) {
	if strings.TrimSpace(commitMsgFile) != "" {
		return commitMsgFile, func() {}, nil
	}
	base := strings.TrimSpace(baseSHA)
	if base == "" {
		if in, err := LoadIntent(wtPath); err == nil {
			base = strings.TrimSpace(in.BaseSHA)
		}
	}
	var head string
	if rc, out := run(git, wtPath, []string{"rev-parse", "HEAD"}); rc == 0 {
		head = strings.TrimSpace(out)
	}
	// A worker commit at tip is present when HEAD differs from base.
	hasWorkerCommit := head != "" && (base == "" || head != base)
	var msg string
	if hasWorkerCommit || head == "" {
		rc, out := run(git, wtPath, []string{"log", "-1", "--format=%B"})
		if rc == 0 && strings.TrimSpace(out) != "" {
			msg = strings.TrimRight(out, "\n")
		}
	}
	if strings.TrimSpace(msg) == "" {
		if in, err := LoadIntent(wtPath); err == nil && strings.TrimSpace(in.Message) != "" {
			msg = in.Message
		} else {
			msg = "chore(dispatch): land worker worktree diff"
		}
	}
	f, err := os.CreateTemp("", "fak-wt-msg-*.txt")
	if err != nil {
		return "", func() {}, err
	}
	if _, err := f.WriteString(msg + "\n"); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", func() {}, err
	}
	f.Close()
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// gitApply applies a captured diff to root's working tree via `git apply`, reading
// the patch from a temp file (a long diff exceeds argv limits) and removing it
// after. Kept separate so the apply step is isolated and testable.
func gitApply(root, diff string, git GitRunner) Result {
	patch, cleanup, err := writePatch(diff)
	if err != nil {
		return Result{OK: false, Detail: "could not stage patch temp: " + err.Error()}
	}
	defer cleanup()
	rc, out := run(git, root, []string{"apply", "--whitespace=nowarn", patch})
	return Result{OK: rc == 0, Detail: tail(out, 300)}
}

// writePatch writes a captured diff (newline-terminated) to a temp file and returns
// its path plus a cleanup func. Shared by the baseline working-tree apply and the
// isolated-index `apply --cached`, so both stage byte-identical patch content.
func writePatch(diff string) (string, func(), error) {
	f, err := os.CreateTemp("", "fak-wt-land-*.patch")
	if err != nil {
		return "", func() {}, err
	}
	body := diff
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", func() {}, err
	}
	f.Close()
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// landIsolated is the race-free layer-2 land (#3547). It stages the worker diff into a
// THROWAWAY index seeded from the current trunk HEAD, commits it as a child of that
// exact HEAD via commit-tree, then advances the branch with a compare-and-swap
// update-ref. Two properties fall out:
//   - Staging never touches the SHARED index, so no concurrent `commit -- paths` can
//     sweep our change into its own commit (the false-success layer 1 only detects).
//   - The ref moves ONLY if HEAD has not advanced since we seeded, so a commit built on
//     a stale base can never silently revert a peer's concurrent work — a lost CAS
//     falls back instead.
//
// A LOST CAS is not a fallback trigger anymore (#3570): a peer landing in the gap
// just moved the base, so the loop below re-resolves HEAD, re-seeds the throwaway
// index from the new base, re-applies the same captured diff and re-attempts the
// CAS — optimistic concurrency, bounded by IsolatedLandRetryEnv with a short
// jittered backoff. Under contention the old behavior collapsed nearly every land
// into the racy shared-index fallback precisely when isolation mattered most.
//
// Returns (result, handled). handled=false means "could not isolate safely — use the
// baseline shared path": detached HEAD, unresolved identity, apply conflict (a
// GENUINE same-path overlap, including a conflicting re-apply after a lost CAS),
// exhausted CAS attempts, or any git error. Thus enabling this can only REDUCE the
// race window on the happy path, never regress the baseline. On success the shared
// working tree is synced for `paths` (git checkout <new> -- paths) so trunk builders
// see the landed change, matching the baseline post-state; a sync hiccup is reported
// but does NOT unland.
func landIsolated(root, wtPath, diff, msgFile string, paths []string, git GitRunner, genv GitEnvRunner, configs ...landConfig) (Result, bool) {
	cfg := landConfig{}
	if len(configs) > 0 {
		cfg = configs[0]
	}
	tracker := cfg.tracker
	finishIsolationAdmission := beginLandPhase(tracker, "isolated-admission", 0)
	isolationAdmissionActive := true
	defer func() {
		if isolationAdmissionActive {
			finishIsolationAdmission()
		}
	}()
	// The branch to move. Detached HEAD → no branch ref to CAS safely; fall back.
	rc, ref := run(git, root, []string{"symbolic-ref", "--quiet", "HEAD"})
	branch := strings.TrimSpace(ref)
	if rc != 0 || branch == "" {
		return Result{}, false
	}
	// The exact base our commit parents AND the compare-and-swap old-value.
	rc, head := run(git, root, []string{"rev-parse", "HEAD"})
	oldHEAD := strings.TrimSpace(head)
	if rc != 0 || oldHEAD == "" {
		return Result{}, false
	}
	// commit-tree runs no hook and adds no signoff; compose Signed-off-by ourselves to
	// preserve the baseline `commit -s`. Unresolved identity → fall back (can't honor -s).
	_, name := run(git, root, []string{"config", "user.name"})
	_, email := run(git, root, []string{"config", "user.email"})
	nm, em := strings.TrimSpace(name), strings.TrimSpace(email)
	if nm == "" || em == "" {
		return Result{}, false
	}

	// Throwaway index at a fresh path git creates via read-tree; removed on return.
	idxF, err := os.CreateTemp("", "fak-land-*.index")
	if err != nil {
		return Result{}, false
	}
	idx := idxF.Name()
	idxF.Close()
	os.Remove(idx) // read-tree writes it fresh; a pre-existing empty file would also do
	defer os.Remove(idx)
	env := map[string]string{"GIT_INDEX_FILE": idx}
	if tracker != nil {
		tracker.setCache("fresh-isolated-index", false)
	}

	// The captured diff and the signed message are attempt-invariant: write them once
	// so every CAS attempt stages byte-identical content under the same subject.
	patch, cleanupPatch, err := writePatch(diff)
	if err != nil {
		return Result{}, false
	}
	defer cleanupPatch()
	ctMsg, cleanupMsg, err := composeSignedMsg(msgFile, nm, em)
	if err != nil {
		return Result{}, false
	}
	defer cleanupMsg()
	finishIsolationAdmission()
	isolationAdmissionActive = false

	// Bounded optimistic-concurrency loop (#3570): each attempt seeds the throwaway
	// index from the CURRENT base, builds the commit as a child of that exact base,
	// and CASes the branch forward. Only a lost CAS loops; every other hiccup still
	// falls back immediately, exactly as before.
	attempts := isolatedLandRetryCap()
	var disambiguation *DisambiguationWitnesses
	var lastCommit string
	var lastBase string
	for attempt := 1; attempt <= attempts; attempt++ {
		var treeSHA string
		if attempt > 1 {
			// A peer is actively landing: back off briefly, then re-resolve the base
			// the peer just moved so this attempt re-builds on the NEW HEAD.
			casRetrySleep(attempt)
			finishRebase := beginLandPhase(tracker, "cas-rebase", attempt)
			rc, head := run(git, root, []string{"rev-parse", "HEAD"})
			finishRebase()
			newHEAD := strings.TrimSpace(head)
			if rc != 0 || newHEAD == "" {
				return Result{}, false
			}

			// In-memory 3-way merge tree resolution for CAS landing retry (#11235).
			// If we already constructed an isolated index commit in a previous attempt,
			// rebase it onto moved HEAD in memory when CAS landing encounters divergence.
			rebased := false
			if lastCommit != "" && lastBase != "" && lastBase != newHEAD {
				if rTree, err := mergeTreeRebase(root, lastBase, newHEAD, lastCommit, git); err == nil && rTree != "" {
					treeSHA = rTree
					oldHEAD = newHEAD
					rebased = true
				}
			}

			if !rebased {
				oldHEAD = newHEAD
				// Seed the throwaway index with the current trunk HEAD's tree.
				finishIndex := beginLandPhase(tracker, "index-construction", attempt)
				if rc, _ := runEnv(genv, root, env, []string{"read-tree", oldHEAD}); rc != 0 {
					finishIndex()
					return Result{}, false
				}
				// Stage the worker diff into the throwaway index ONLY (--cached never touches
				// the working tree). A conflict here — first try or re-apply after a lost CAS —
				// means a concurrent change to the SAME paths; let the baseline path adjudicate
				// it exactly as today rather than force it.
				if rc, _ := runEnv(genv, root, env, []string{"apply", "--cached", "--whitespace=nowarn", patch}); rc != 0 {
					finishIndex()
					return Result{}, false
				}
				rc, tree := runEnv(genv, root, env, []string{"write-tree"})
				treeSHA = strings.TrimSpace(tree)
				if rc != 0 || treeSHA == "" {
					finishIndex()
					return Result{}, false
				}
				finishIndex()
			}
		} else {
			// Seed the throwaway index with the current trunk HEAD's tree.
			finishIndex := beginLandPhase(tracker, "index-construction", attempt)
			if rc, _ := runEnv(genv, root, env, []string{"read-tree", oldHEAD}); rc != 0 {
				finishIndex()
				return Result{}, false
			}
			// Stage the worker diff into the throwaway index ONLY (--cached never touches
			// the working tree). A conflict here — first try or re-apply after a lost CAS —
			// means a concurrent change to the SAME paths; let the baseline path adjudicate
			// it exactly as today rather than force it.
			if rc, _ := runEnv(genv, root, env, []string{"apply", "--cached", "--whitespace=nowarn", patch}); rc != 0 {
				finishIndex()
				return Result{}, false
			}
			rc, tree := runEnv(genv, root, env, []string{"write-tree"})
			treeSHA = strings.TrimSpace(tree)
			if rc != 0 || treeSHA == "" {
				finishIndex()
				return Result{}, false
			}
			finishIndex()
		}
		disambiguation = nil
		if disambiguationRelevant(paths) {
			finishAnalysis := beginLandPhase(tracker, "whole-tree-disambiguation", attempt)
			var valid bool
			disambiguation, valid = verifyAppliedDisambiguation(root, wtPath, treeSHA)
			finishAnalysis()
			if !valid {
				return Result{OK: false, Path: root, Reason: "post-apply disambiguation invariant failed", Detail: disambiguation.compactDetail(), Disambiguation: disambiguation}, true
			}
		}
		finishCommit := beginLandPhase(tracker, "commit-construction", attempt)
		rc, commit := runEnv(genv, root, env, []string{"commit-tree", treeSHA, "-p", oldHEAD, "-F", ctMsg})
		finishCommit()
		newCommit := strings.TrimSpace(commit)
		if rc != 0 || newCommit == "" {
			return Result{}, false
		}
		lastCommit = newCommit
		lastBase = oldHEAD
		// Name the off-branch commit before trunk CAS. A process crash from here on
		// leaves an observable, GC-safe recovery candidate instead of a dangling SHA.
		finishRecovery := beginLandPhase(tracker, "recovery-ref-publication", attempt)
		recoveryRef, anchorErr := AnchorRecoveryEntry(root, wtPath, newCommit, func(r string, a []string) (int, string) { return runEnv(genv, r, env, a) })
		if anchorErr != nil {
			finishRecovery()
			return Result{OK: false, Reason: "isolated land recovery anchor failed — trunk unchanged", Detail: anchorErr.Error()}, true
		}
		var remoteReceipt *RemoteReadback
		if cfg.recoveryRemote != "" {
			receipt := PublishRecoveryRef(root, cfg.recoveryRemote, recoveryRef, newCommit, git)
			remoteReceipt = &receipt
			if cfg.requireRemote && !receipt.Witnessed {
				finishRecovery()
				return Result{OK: false, RecoveryRef: recoveryRef, RemoteRecovery: remoteReceipt, Reason: "required remote recovery witness failed — trunk unchanged", Detail: receipt.Reason}, true
			}
		}
		finishRecovery()
		// Compare-and-swap: move the branch ONLY if HEAD is still oldHEAD. A peer commit
		// in the gap fails this → retry on the peer's new HEAD (#3570); the throwaway
		// commit built on the stale base is simply abandoned, unreferenced.
		finishCAS := beginLandPhase(tracker, "trunk-cas", attempt)
		rc, _ = run(git, root, []string{"update-ref", branch, newCommit, oldHEAD})
		finishCAS()
		if rc != 0 {
			continue
		}
		// The ref moved but the shared working tree still holds OLD content for `paths`
		// (we never touched it). Sync just those paths so trunk builders see the landed
		// change, matching the baseline post-state. A sync failure does NOT unland.
		detail := "cas-attempts=" + strconv.Itoa(attempt) + "/" + strconv.Itoa(attempts) + "; recovery-ref=" + recoveryRef
		coArgs := append([]string{"checkout", newCommit, "--"}, paths...)
		finishSync := beginLandPhase(tracker, "working-tree-sync", attempt)
		if rc, out := run(git, root, coArgs); rc != 0 {
			detail += "; landed " + shortSHA(newCommit) + " but working-tree sync failed: " + tail(out, 200)
		}
		finishSync()
		return Result{OK: true, Code: LandResultSuccess, Applied: true, Committed: true,
			Reason: "isolated-index land " + shortSHA(newCommit) + " (race-free, #3547)",
			Detail: detail, Disambiguation: disambiguation, RecoveryRef: recoveryRef, RemoteRecovery: remoteReceipt}, true
	}
	// Every bounded attempt lost its CAS — genuine sustained contention. Fall back to
	// the baseline shared path as the final resort rather than loop unbounded.
	return Result{}, false
}

// composeSignedMsg writes msgFile's content to a new temp file with a Signed-off-by
// trailer appended when absent, so a commit-tree land keeps the baseline `-s` signoff.
// Returns the temp path and a cleanup func.
func composeSignedMsg(msgFile, name, email string) (string, func(), error) {
	raw, err := os.ReadFile(msgFile)
	if err != nil {
		return "", func() {}, err
	}
	body := string(raw)
	signoff := "Signed-off-by: " + name + " <" + email + ">"
	if !strings.Contains(body, signoff) {
		if !strings.HasSuffix(body, "\n") {
			body += "\n"
		}
		body += "\n" + signoff + "\n"
	}
	f, err := os.CreateTemp("", "fak-land-msg-*.txt")
	if err != nil {
		return "", func() {}, err
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", func() {}, err
	}
	f.Close()
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// landReadbackEnabled reports whether the post-commit readback (LandReadbackEnv)
// is on. DEFAULT ON since #3619; only an explicit 0/false/off forces it off.
func landReadbackEnabled() bool { return envDefaultOn(LandReadbackEnv) }

// isolatedLandEnabled reports whether the race-free isolated-index land
// (IsolatedLandEnv) is on. DEFAULT ON since #3619; only an explicit 0/false/off
// forces the shared-index baseline.
func isolatedLandEnabled() bool { return envDefaultOn(IsolatedLandEnv) }

// envDefaultOn reads a boolean gate that DEFAULTS ON: unset/empty counts as on;
// only an explicit 0/false/off (any case) forces it off. The shape a gate takes
// once its soak has proven it safe to default on (#3619), keeping the env as an
// operator escape hatch back to the old behavior.
func envDefaultOn(name string) bool {
	v := strings.TrimSpace(os.Getenv(name))
	return v == "" || (v != "0" && !strings.EqualFold(v, "false") && !strings.EqualFold(v, "off"))
}

// landReadbackVerify confirms trunk HEAD, right after a path-scoped commit, carries
// EVERY intended path. A missing path means the worker's staged change was swept
// into a concurrent commit on the shared index (the #3547 race) — reported as a
// LAND_READBACK_MISMATCH refusal. FAIL-OPEN: any git error yields ok=true, so a
// readback that cannot run never manufactures a refusal. A directory pathspec is
// satisfied by any committed file beneath it.
func landReadbackVerify(root string, paths []string, git GitRunner) (bool, string) {
	rc, head := run(git, root, []string{"rev-parse", "HEAD"})
	if rc != 0 {
		return true, ""
	}
	sha := strings.TrimSpace(head)
	rc, out := run(git, root, []string{"diff-tree", "--no-commit-id", "--name-only", "-r", sha})
	if rc != 0 {
		return true, ""
	}
	committed := map[string]bool{}
	for _, line := range strings.Split(out, "\n") {
		if p := normalizeSlash(strings.TrimSpace(line)); p != "" {
			committed[p] = true
		}
	}
	var missing []string
	for _, p := range paths {
		want := normalizeSlash(strings.TrimSpace(p))
		if want == "" {
			continue
		}
		found := committed[want]
		if !found {
			for c := range committed {
				if strings.HasPrefix(c, want+"/") {
					found = true
					break
				}
			}
		}
		if !found {
			missing = append(missing, p)
		}
	}
	if len(missing) > 0 {
		return false, LandReadbackMismatchToken + ": trunk HEAD " + shortSHA(sha) +
			" does not carry intended path(s) " + strings.Join(missing, ", ") +
			" after commit — shared-index race, land not trusted (#3547)"
	}
	return true, ""
}

// stripWorktreeWIPFences removes ephemeral //go:build wip_<slug> build constraints from
// files in wtPath prior to capturing the landing diff.
func stripWorktreeWIPFences(wtPath string, paths []string) {
	if len(paths) > 0 {
		for _, p := range paths {
			target := filepath.Join(wtPath, filepath.FromSlash(p))
			if fi, err := os.Stat(target); err == nil {
				if fi.IsDir() {
					_ = filepath.Walk(target, func(sub string, info os.FileInfo, werr error) error {
						if werr == nil && !info.IsDir() && strings.HasSuffix(sub, ".go") {
							_ = wipfence.Strip(sub)
						}
						return nil
					})
				} else if strings.HasSuffix(target, ".go") {
					_ = wipfence.Strip(target)
				}
			}
		}
		return
	}
	_ = filepath.Walk(wtPath, func(sub string, info os.FileInfo, werr error) error {
		if werr != nil {
			return nil
		}
		if info.IsDir() && (info.Name() == ".git" || info.Name() == "vendor") {
			return filepath.SkipDir
		}
		if !info.IsDir() && strings.HasSuffix(sub, ".go") {
			_ = wipfence.Strip(sub)
		}
		return nil
	})
}
