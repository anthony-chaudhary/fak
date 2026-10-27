package safesync

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ParkSchema is the schema identifier for path-scoped suspend and reapply receipts (#10076).
const ParkSchema = "fak.sync.park.v1"

// Status constants for ParkReceipt.
const (
	ParkStatusDryRun     = "DRY_RUN"
	ParkStatusParked     = "PARKED"
	ParkStatusIntegrated = "INTEGRATED"
	ParkStatusRestored   = "RESTORED"
	ParkStatusConflict   = "CONFLICT"
)

// Effect classification constants for PathEffect.
const (
	EffectUpstreamIdentical = "upstream_identical"
	EffectSuperseded        = "superseded"
	EffectCleanReapply      = "clean_reapply"
	EffectConflict          = "conflict"
)

// EnvRunner executes a git subcommand in repo with custom environment variables.
type EnvRunner func(ctx context.Context, repo string, env []string, args ...string) RunResult

// ParkOptions configures path-scoped suspend and reapply engine.
type ParkOptions struct {
	Repo           string           `json:"repo"`
	Session        string           `json:"session"`
	Paths          []string         `json:"paths"`
	TargetRef      string           `json:"target_ref,omitempty"`
	Apply          bool             `json:"apply,omitempty"`
	Runner         Runner           `json:"-"`
	EnvRunner      EnvRunner        `json:"-"`
	Now            func() time.Time `json:"-"`
	WriterLeaseTTL time.Duration    `json:"-"`
	LeaseOwner     string           `json:"lease_owner,omitempty"`
	Lease          *WriterLease     `json:"-"`
}

// PathEffect classifies the outcome of reapplying a parked path.
type PathEffect struct {
	Path           string `json:"path"`
	Classification string `json:"classification"`
	Effect         string `json:"effect,omitempty"`
	Detail         string `json:"detail,omitempty"`
}

// ParkReceipt / ParkReport is the typed, evidence-backed result of a park operation.
type ParkReceipt struct {
	Schema                  string       `json:"schema"`
	Session                 string       `json:"session"`
	BaseHEAD                string       `json:"base_head"`
	TargetRef               string       `json:"target_ref"`
	TargetSHA               string       `json:"target_sha"`
	NewHEAD                 string       `json:"new_head,omitempty"`
	SelectedPaths           []string     `json:"selected_paths"`
	UnrelatedPathsPreserved []string     `json:"unrelated_paths_preserved,omitempty"`
	Status                  string       `json:"status"`
	Effects                 []PathEffect `json:"effects,omitempty"`
	CheckpointRef           string       `json:"checkpoint_ref"`
	OK                      bool         `json:"ok"`
	Reason                  string       `json:"reason,omitempty"`
	Detail                  string       `json:"detail,omitempty"`
}

// ParkReport is an alias for ParkReceipt.
type ParkReport = ParkReceipt

// ParkAndReapply executes path suspend, target integration, and reapplying of effects (#10059).
func ParkAndReapply(ctx context.Context, opts ParkOptions) (ParkReceipt, error) {
	opts.Apply = true
	return Park(ctx, opts)
}

// Park suspends owner-authorized dirty paths, integrates target, and reapplies
// unique effects without clobbering unrelated dirty paths (#10076).
func Park(ctx context.Context, opts ParkOptions) (ParkReceipt, error) {
	repo := strings.TrimSpace(opts.Repo)
	if repo == "" {
		repo = "."
	}
	session := strings.TrimSpace(opts.Session)
	if session == "" {
		session = "default"
	}
	checkpointRef := fmt.Sprintf("refs/fak/wip/%s/park", session)

	receipt := ParkReceipt{
		Schema:        ParkSchema,
		Session:       session,
		CheckpointRef: checkpointRef,
	}

	if len(opts.Paths) == 0 {
		receipt.Status = ParkStatusConflict
		receipt.Reason = "no paths specified"
		return receipt, errors.New("no paths specified")
	}

	// Normalize and deduplicate selected paths
	var selectedPaths []string
	selectedMap := make(map[string]bool)
	for _, p := range opts.Paths {
		p = path.Clean(filepath.ToSlash(strings.TrimSpace(p)))
		if p == "" || p == "." {
			continue
		}
		if !selectedMap[p] {
			selectedMap[p] = true
			selectedPaths = append(selectedPaths, p)
		}
	}
	sort.Strings(selectedPaths)
	receipt.SelectedPaths = selectedPaths

	run := opts.Runner
	if run == nil {
		run = RealRunner
	}
	envRun := opts.EnvRunner
	if envRun == nil {
		envRun = func(c context.Context, r string, env []string, args ...string) RunResult {
			if len(env) == 0 && opts.Runner != nil {
				return opts.Runner(c, r, args...)
			}
			cmdArgs := append([]string{"-C", r}, args...)
			cmd := exec.CommandContext(c, "git", cmdArgs...)
			windowgate.ConfigureBackgroundCommand(cmd)
			if len(env) > 0 {
				cmd.Env = append(os.Environ(), env...)
			}
			var stdout, stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr
			err := cmd.Run()
			code := 0
			if err != nil {
				var exit *exec.ExitError
				if errors.As(err, &exit) {
					code = exit.ExitCode()
					err = nil
				} else {
					code = -1
				}
			}
			return RunResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes(), Code: code, Err: err}
		}
	}

	now := opts.Now
	if now == nil {
		now = time.Now
	}
	ttl := opts.WriterLeaseTTL
	if ttl <= 0 {
		ttl = DefaultWriterLeaseTTL
	}
	leaseOwner := strings.TrimSpace(opts.LeaseOwner)
	if leaseOwner == "" {
		leaseOwner = fmt.Sprintf("park-%s-%d", session, os.Getpid())
	}

	// 1. Pin current local HEAD and target SHA
	baseHEAD, err := rev(ctx, run, repo, "HEAD")
	if err != nil {
		receipt.Status = ParkStatusConflict
		receipt.Reason = fmt.Sprintf("resolve HEAD: %v", err)
		return receipt, err
	}
	receipt.BaseHEAD = baseHEAD

	targetRef := strings.TrimSpace(opts.TargetRef)
	if targetRef == "" {
		branch, berr := currentBranch(ctx, run, repo)
		if berr == nil && branch != "" {
			targetRef = "origin/" + branch
		} else {
			devBranch := "main"
			if roles, err := branchrole.Load(repo); err == nil && roles.DevelopmentBranch != "" {
				devBranch = roles.DevelopmentBranch
			}
			targetRef = "origin/" + devBranch
		}
	}
	receipt.TargetRef = targetRef

	targetSHA, err := rev(ctx, run, repo, targetRef)
	if err != nil {
		receipt.Status = ParkStatusConflict
		receipt.Reason = fmt.Sprintf("resolve target ref %q: %v", targetRef, err)
		return receipt, err
	}
	receipt.TargetSHA = targetSHA

	// 2. Verify selected paths & dirty paths
	dirtyPaths, err := workingTreeDirtyPaths(ctx, run, repo)
	if err != nil {
		receipt.Status = ParkStatusConflict
		receipt.Reason = fmt.Sprintf("inspect working tree: %v", err)
		return receipt, err
	}

	// Ensure all selected paths are safe worktree paths
	for _, p := range selectedPaths {
		if _, ok := safeWorktreePath(repo, p); !ok {
			receipt.Status = ParkStatusConflict
			receipt.Reason = fmt.Sprintf("unsafe path %q", p)
			return receipt, fmt.Errorf("unsafe path %q", p)
		}
	}

	// Check merge base and incoming write set
	mbRes := run(ctx, repo, "merge-base", baseHEAD, targetSHA)
	mergeBase := strings.TrimSpace(string(mbRes.Stdout))
	if mergeBase == "" {
		mergeBase = baseHEAD
	}

	diffRes := run(ctx, repo, "diff", "--name-only", mergeBase, targetSHA)
	incomingLines := strings.Fields(string(diffRes.Stdout))
	incomingMap := make(map[string]bool, len(incomingLines))
	for _, line := range incomingLines {
		line = path.Clean(filepath.ToSlash(strings.TrimSpace(line)))
		if line != "" {
			incomingMap[line] = true
		}
	}

	// Check for unowned colliding dirty paths and collect unrelated paths preserved
	var unrelatedPreserved []string
	for _, dp := range dirtyPaths {
		normDP := path.Clean(filepath.ToSlash(dp))
		if selectedMap[normDP] {
			continue
		}
		if incomingMap[normDP] {
			// An unowned / unselected dirty path collides with the incoming write set!
			receipt.Status = ParkStatusConflict
			receipt.Reason = ReasonDirtyWriteOverlap
			receipt.Detail = fmt.Sprintf("unowned dirty path %q collides with incoming integration; ensure working paths are clear or authorized", normDP)
			return receipt, nil
		}
		unrelatedPreserved = append(unrelatedPreserved, normDP)
	}
	sort.Strings(unrelatedPreserved)
	receipt.UnrelatedPathsPreserved = unrelatedPreserved

	// Dry-run mode: preview effects without modifying disk
	if !opts.Apply {
		receipt.Status = ParkStatusDryRun
		var effects []PathEffect
		hasConflict := false

		for _, p := range selectedPaths {
			eff := previewPathEffect(ctx, run, repo, baseHEAD, targetSHA, p)
			effects = append(effects, eff)
			if eff.Classification == EffectConflict {
				hasConflict = true
			}
		}
		receipt.Effects = effects
		if hasConflict {
			receipt.Status = ParkStatusConflict
			receipt.OK = false
			receipt.Reason = "conflict detected during dry-run preview"
		} else {
			receipt.OK = true
			receipt.Reason = "dry-run preview complete; safe to park and integrate"
		}
		return receipt, nil
	}

	for _, p := range selectedPaths {
		for _, ref := range []string{baseHEAD, targetSHA} {
			if _, _, err := readParkGitBlob(ctx, run, repo, ref, p); err != nil {
				receipt.Status = ParkStatusConflict
				receipt.Reason = "cannot inspect selected path before integration"
				receipt.Detail = err.Error()
				return receipt, nil
			}
		}
	}

	// 4. Acquire worktree writer lease
	lease := opts.Lease
	if lease == nil {
		var lerr error
		lease, lerr = AcquireWriterLease(repo, leaseOwner, now, ttl)
		if lerr != nil {
			var held *WriterLeaseHeldError
			if errors.As(lerr, &held) {
				receipt.Status = ParkStatusConflict
				receipt.Reason = fmt.Sprintf("worktree writer lease held by %s; refusing", held.Info.Owner)
				return receipt, nil
			}
			return receipt, lerr
		}
		defer func() { _ = lease.Release() }()
		stopHeartbeat := lease.keepAlive(now)
		defer stopHeartbeat()
	}

	// 3. Mint durable path-scoped recovery checkpoint object
	tmpDir, err := os.MkdirTemp("", "fak-park-idx-")
	if err != nil {
		return receipt, err
	}
	defer os.RemoveAll(tmpDir)

	idxFile := filepath.Join(tmpDir, "index")
	idxEnv := []string{"GIT_INDEX_FILE=" + idxFile}

	rTree := envRun(ctx, repo, idxEnv, "read-tree", baseHEAD)
	if rTree.Err != nil || rTree.Code != 0 {
		return receipt, fmt.Errorf("read-tree %s: %v %s", baseHEAD, rTree.Err, string(rTree.Stderr))
	}

	for _, p := range selectedPaths {
		fullPath := filepath.Join(repo, p)
		if _, statErr := os.Stat(fullPath); statErr == nil {
			addRes := envRun(ctx, repo, idxEnv, "add", "--", p)
			if addRes.Err != nil || addRes.Code != 0 {
				return receipt, fmt.Errorf("git add %s: %v %s", p, addRes.Err, string(addRes.Stderr))
			}
		} else if os.IsNotExist(statErr) {
			rmRes := envRun(ctx, repo, idxEnv, "rm", "--cached", "--ignore-unmatch", "--", p)
			if rmRes.Err != nil || rmRes.Code != 0 {
				return receipt, fmt.Errorf("git rm %s: %v %s", p, rmRes.Err, string(rmRes.Stderr))
			}
		}
	}

	wTree := envRun(ctx, repo, idxEnv, "write-tree")
	if wTree.Err != nil || wTree.Code != 0 {
		return receipt, fmt.Errorf("write-tree: %v %s", wTree.Err, string(wTree.Stderr))
	}
	treeSHA := strings.TrimSpace(string(wTree.Stdout))

	msg := fmt.Sprintf("fak: park %s\n\nPaths: %s", session, strings.Join(selectedPaths, ", "))
	cTree := envRun(ctx, repo, nil, "commit-tree", treeSHA, "-p", baseHEAD, "-m", msg)
	if cTree.Err != nil || cTree.Code != 0 {
		return receipt, fmt.Errorf("commit-tree: %v %s", cTree.Err, string(cTree.Stderr))
	}
	checkpointSHA := strings.TrimSpace(string(cTree.Stdout))

	uRef := envRun(ctx, repo, nil, "update-ref", checkpointRef, checkpointSHA)
	if uRef.Err != nil || uRef.Code != 0 {
		return receipt, fmt.Errorf("update-ref %s: %v %s", checkpointRef, uRef.Err, string(uRef.Stderr))
	}

	// 5. Clear selected paths in the working tree to pinned BaseHEAD
	for _, p := range selectedPaths {
		chk := run(ctx, repo, "cat-file", "-e", baseHEAD+":"+p)
		if chk.Code == 0 {
			coRes := run(ctx, repo, "checkout", "HEAD", "--", p)
			if coRes.Code != 0 {
				return receipt, fmt.Errorf("checkout HEAD -- %s: %s", p, string(coRes.Stderr))
			}
		} else {
			// newly created untracked file: remove from working tree
			fullPath := filepath.Join(repo, p)
			_ = os.Remove(fullPath)
		}
	}

	// 6. Run in-place integration (fast-forward or in-place merge against pinned target SHA)
	newCommit := baseHEAD
	if baseHEAD != targetSHA {
		isAnc, _ := isAncestor(ctx, run, repo, baseHEAD, targetSHA)
		if isAnc {
			newCommit = targetSHA
		} else {
			// Ahead + behind: in-place merge using merge-tree
			mtRes := run(ctx, repo, "merge-tree", "--write-tree", baseHEAD, targetSHA)
			if mtRes.Err != nil || mtRes.Code != 0 {
				// Rollback
				for _, p := range selectedPaths {
					_ = run(ctx, repo, "checkout", checkpointRef, "--", p)
				}
				receipt.Status = ParkStatusConflict
				receipt.OK = false
				receipt.Reason = fmt.Sprintf("in-place merge conflict between %s and %s", baseHEAD, targetSHA)
				return receipt, nil
			}
			mergedTreeSHA := strings.TrimSpace(strings.Split(string(mtRes.Stdout), "\n")[0])
			cTree := envRun(ctx, repo, nil, "commit-tree", mergedTreeSHA, "-p", baseHEAD, "-p", targetSHA, "-m", fmt.Sprintf("fak: in-place integration of %s", targetSHA))
			if cTree.Err != nil || cTree.Code != 0 {
				for _, p := range selectedPaths {
					_ = run(ctx, repo, "checkout", checkpointRef, "--", p)
				}
				receipt.Status = ParkStatusConflict
				receipt.OK = false
				receipt.Reason = fmt.Sprintf("commit-tree failed: %v", cTree.Err)
				return receipt, nil
			}
			newCommit = strings.TrimSpace(string(cTree.Stdout))
		}

		// HEAD follows the actual checkout: its branch when attached, or the
		// worktree-local ref when detached. currentBranch's logical sync-target
		// fallback must never select a shared branch for a detached worker write.
		urRes := run(ctx, repo, "update-ref", "HEAD", newCommit, baseHEAD)
		if urRes.Err != nil || urRes.Code != 0 {
			for _, p := range selectedPaths {
				_ = run(ctx, repo, "checkout", checkpointRef, "--", p)
			}
			receipt.Status = ParkStatusConflict
			receipt.OK = false
			receipt.Reason = fmt.Sprintf("update-ref failed: %v", urRes.Err)
			return receipt, nil
		}

		// Update working tree for paths that changed between baseHEAD and newCommit (excluding selectedPaths)
		diffRes := run(ctx, repo, "diff", "--name-only", baseHEAD, newCommit)
		for _, line := range strings.Fields(string(diffRes.Stdout)) {
			norm := path.Clean(filepath.ToSlash(strings.TrimSpace(line)))
			if norm == "" || selectedMap[norm] {
				continue
			}
			_ = run(ctx, repo, "checkout", newCommit, "--", norm)
		}
	}

	receipt.NewHEAD = newCommit

	// 7. Compare parked content against new HEAD and reapply
	var effects []PathEffect
	hasConflict := false

	for _, p := range selectedPaths {
		eff, mergedBytes, hasMerge := evaluateAndMergePath(ctx, run, repo, baseHEAD, checkpointRef, newCommit, p)
		effects = append(effects, eff)
		if eff.Classification == EffectConflict {
			hasConflict = true
		} else if eff.Classification == EffectCleanReapply && hasMerge {
			// Advance the selected path's index entry to the integrated HEAD
			// before writing the merged working bytes. `git checkout newCommit -- p`
			// writes the new baseline to BOTH the index and the worktree; the
			// subsequent WriteFile replaces only the worktree, so the unique local
			// effect stays unstaged while the index matches upstream. Without this
			// the index keeps the pre-integration base blob (set at the "checkout
			// HEAD" step above) and `git diff --cached` stages an inverse of the
			// upstream hunk (#12690).
			_, exists, err := readParkGitBlob(ctx, run, repo, newCommit, p)
			if err != nil {
				return receipt, fmt.Errorf("inspect selected index baseline %s: %w", p, err)
			}
			indexArgs := []string{"checkout", newCommit, "--", p}
			if !exists {
				indexArgs = []string{"rm", "--cached", "--ignore-unmatch", "--", p}
			}
			if idx := run(ctx, repo, indexArgs...); idx.Err != nil || idx.Code != 0 {
				return receipt, fmt.Errorf("advance selected index %s to %s: %v %s", p, newCommit, idx.Err, string(idx.Stderr))
			}
			fullPath := filepath.Join(repo, p)
			dir := filepath.Dir(fullPath)
			_ = os.MkdirAll(dir, 0755)
			mode := os.FileMode(0644)
			if info, serr := os.Stat(fullPath); serr == nil {
				mode = info.Mode()
			}
			if werr := os.WriteFile(fullPath, mergedBytes, mode); werr != nil {
				return receipt, fmt.Errorf("write reapply %s: %w", p, werr)
			}
		} else if eff.Classification == EffectUpstreamIdentical {
			_ = run(ctx, repo, "checkout", newCommit, "--", p)
		}
	}
	receipt.Effects = effects

	if hasConflict {
		receipt.Status = ParkStatusConflict
		receipt.OK = false
		receipt.Reason = "one or more paths had conflicting changes"
	} else {
		hasCleanReapply := false
		for _, eff := range effects {
			if eff.Classification == EffectCleanReapply {
				hasCleanReapply = true
				break
			}
		}
		if hasCleanReapply {
			receipt.Status = ParkStatusRestored
		} else {
			receipt.Status = ParkStatusIntegrated
		}
		receipt.OK = true
		receipt.Reason = "paths successfully integrated and restored"
	}

	return receipt, nil
}

// readParkGitBlob distinguishes proven absence from a failed content lookup.
func readParkGitBlob(ctx context.Context, run Runner, repo, ref, p string) ([]byte, bool, error) {
	entry := run(ctx, repo, "ls-tree", "-z", "--full-tree", ref, "--", p)
	if entry.Err != nil || entry.Code != 0 {
		return nil, false, fmt.Errorf("inspect Git path %s at %s: %v %s", p, ref, entry.Err, string(entry.Stderr))
	}
	if len(entry.Stdout) == 0 {
		return nil, false, nil
	}
	header, name, ok := strings.Cut(strings.TrimSuffix(string(entry.Stdout), "\x00"), "\t")
	fields := strings.Fields(header)
	if !ok || name != p || len(fields) != 3 || fields[1] != "blob" || (fields[0] != "100644" && fields[0] != "100755") {
		return nil, false, fmt.Errorf("expected one regular Git blob for %s at %s", p, ref)
	}
	content := run(ctx, repo, "show", ref+":"+p)
	if content.Err != nil || content.Code != 0 {
		return nil, true, fmt.Errorf("read Git path %s at %s: %v %s", p, ref, content.Err, string(content.Stderr))
	}
	return content.Stdout, true, nil
}

// previewPathEffect previews the classification of reapplying path p in dry-run mode.
func previewPathEffect(ctx context.Context, run Runner, repo, baseHEAD, targetSHA, p string) PathEffect {
	eff := PathEffect{Path: p}
	fullPath := filepath.Join(repo, p)
	parkedBytes, err := os.ReadFile(fullPath)
	parkedExists := err == nil
	if err != nil && !os.IsNotExist(err) {
		eff.Classification, eff.Effect, eff.Detail = EffectConflict, EffectConflict, err.Error()
		return eff
	}

	baseBytes, baseExists, err := readParkGitBlob(ctx, run, repo, baseHEAD, p)
	if err != nil {
		eff.Classification, eff.Effect, eff.Detail = EffectConflict, EffectConflict, err.Error()
		return eff
	}

	targetBytes, targetExists, err := readParkGitBlob(ctx, run, repo, targetSHA, p)
	if err != nil {
		eff.Classification, eff.Effect, eff.Detail = EffectConflict, EffectConflict, err.Error()
		return eff
	}

	if parkedExists == targetExists && bytes.Equal(parkedBytes, targetBytes) {
		eff.Classification = EffectUpstreamIdentical
		eff.Effect = EffectUpstreamIdentical
		eff.Detail = "upstream already has identical content; suppressed"
		return eff
	}

	if !targetExists && parkedExists && baseExists {
		eff.Classification = EffectSuperseded
		eff.Effect = EffectSuperseded
		eff.Detail = "upstream removed path; local edits superseded"
		return eff
	}

	// 3-way merge check
	mergedBytes, conflict := runThreeWayMerge(ctx, run, repo, targetBytes, baseBytes, parkedBytes)
	if conflict {
		eff.Classification = EffectConflict
		eff.Effect = EffectConflict
		eff.Detail = "conflicting changes between upstream and parked path"
		return eff
	}

	if targetExists && bytes.Equal(mergedBytes, targetBytes) {
		eff.Classification = EffectUpstreamIdentical
		eff.Effect = EffectUpstreamIdentical
		eff.Detail = "upstream already contains all changes from parked path; suppressed"
		return eff
	}

	eff.Classification = EffectCleanReapply
	eff.Effect = EffectCleanReapply
	eff.Detail = "unique effect retained, upstream-identical hunk suppressed"
	return eff
}

// evaluateAndMergePath compares parked content from checkpointRef against newHEAD
// and computes the 3-way merged result.
func evaluateAndMergePath(ctx context.Context, run Runner, repo, baseHEAD, checkpointRef, newHEAD, p string) (PathEffect, []byte, bool) {
	eff := PathEffect{Path: p}

	parkedBytes, parkedExists, err := readParkGitBlob(ctx, run, repo, checkpointRef, p)
	if err != nil {
		eff.Classification, eff.Effect, eff.Detail = EffectConflict, EffectConflict, err.Error()
		return eff, nil, false
	}

	baseBytes, baseExists, err := readParkGitBlob(ctx, run, repo, baseHEAD, p)
	if err != nil {
		eff.Classification, eff.Effect, eff.Detail = EffectConflict, EffectConflict, err.Error()
		return eff, nil, false
	}

	newHeadBytes, newHeadExists, err := readParkGitBlob(ctx, run, repo, newHEAD, p)
	if err != nil {
		eff.Classification, eff.Effect, eff.Detail = EffectConflict, EffectConflict, err.Error()
		return eff, nil, false
	}

	if parkedExists == newHeadExists && bytes.Equal(parkedBytes, newHeadBytes) {
		eff.Classification = EffectUpstreamIdentical
		eff.Effect = EffectUpstreamIdentical
		eff.Detail = "upstream already has identical content; suppressed"
		return eff, nil, false
	}

	if !newHeadExists && parkedExists && baseExists {
		eff.Classification = EffectSuperseded
		eff.Effect = EffectSuperseded
		eff.Detail = "upstream removed path; local edits superseded"
		return eff, nil, false
	}

	// 3-way merge
	mergedBytes, conflict := runThreeWayMerge(ctx, run, repo, newHeadBytes, baseBytes, parkedBytes)
	if conflict {
		eff.Classification = EffectConflict
		eff.Effect = EffectConflict
		eff.Detail = "conflicting changes between upstream and parked path"
		return eff, nil, false
	}

	if newHeadExists && bytes.Equal(mergedBytes, newHeadBytes) {
		eff.Classification = EffectUpstreamIdentical
		eff.Effect = EffectUpstreamIdentical
		eff.Detail = "upstream already contains all changes from parked path; suppressed"
		return eff, nil, false
	}

	eff.Classification = EffectCleanReapply
	eff.Effect = EffectCleanReapply
	eff.Detail = "unique effect retained, upstream-identical hunk suppressed"
	return eff, mergedBytes, true
}

// runThreeWayMerge runs `git merge-file -p -q theirs base ours` using temporary files.
func runThreeWayMerge(ctx context.Context, run Runner, repo string, theirs, base, ours []byte) ([]byte, bool) {
	tmpDir, err := os.MkdirTemp("", "fak-park-merge-")
	if err != nil {
		return nil, true
	}
	defer os.RemoveAll(tmpDir)

	theirsFile := filepath.Join(tmpDir, "theirs")
	baseFile := filepath.Join(tmpDir, "base")
	oursFile := filepath.Join(tmpDir, "ours")

	_ = os.WriteFile(theirsFile, theirs, 0644)
	_ = os.WriteFile(baseFile, base, 0644)
	_ = os.WriteFile(oursFile, ours, 0644)

	res := run(ctx, repo, "merge-file", "-p", "-q", theirsFile, baseFile, oursFile)
	if res.Err != nil || res.Code > 0 || bytes.Contains(res.Stdout, []byte("<<<<<<<")) {
		return nil, true
	}
	return res.Stdout, false
}
