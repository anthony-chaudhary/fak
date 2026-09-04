package safesync

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/mergepreview"
)

// ReconcileSchema is the schema identifier for typed reconciliation assessment.
const ReconcileSchema = "fak.sync.reconcile.v1"

// Route decision constants for trunk reconciliation router.
const (
	RouteNoop               = "ROUTE_NOOP"
	RoutePush               = "ROUTE_PUSH"
	RouteApply              = "ROUTE_APPLY"
	RouteHoldDirtyCollision = "ROUTE_HOLD_DIRTY_COLLISION"
	RouteSupersetMerge      = "ROUTE_SUPERSET_MERGE"
	RouteDisjointIntegrate  = "ROUTE_DISJOINT_INTEGRATE"
	RouteReconcilePacket    = "ROUTE_RECONCILE_PACKET"
	RouteHoldMergeActive    = "ROUTE_HOLD_MERGE_ACTIVE"
	RouteDrain              = "ROUTE_DRAIN"
)

// ReconcileOptions configures ReconcileRouter.
type ReconcileOptions struct {
	Repo           string           `json:"repo"`
	Remote         string           `json:"remote"`
	Branch         string           `json:"branch,omitempty"`
	Goal           string           `json:"goal,omitempty"` // publish (default publish HEAD), publish <sha>, integrate (default integrate origin/main)
	Apply          bool             `json:"apply,omitempty"`
	Fetch          bool             `json:"fetch,omitempty"`
	Runner         Runner           `json:"-"`
	Now            func() time.Time `json:"-"`
	WriterLeaseTTL time.Duration    `json:"-"`
	Contention     bool             `json:"-"` // test or override seam for active contention
}

// GoalInfo represents the parsed reconciliation goal.
type GoalInfo struct {
	Raw    string `json:"raw"`
	Kind   string `json:"kind"`   // "publish" | "integrate"
	Source string `json:"source"` // "HEAD" | "<sha>"
	Target string `json:"target"` // "<remote>/<branch>" | "<ref>"
}

// ReconcileExecution records execution results when --apply is specified.
type ReconcileExecution struct {
	Primitive string `json:"primitive"`
	Success   bool   `json:"success"`
	Pushed    bool   `json:"pushed,omitempty"`
	Applied   bool   `json:"applied,omitempty"`
	NewHead   string `json:"new_head,omitempty"`
	Detail    string `json:"detail,omitempty"`
	Error     string `json:"error,omitempty"`
}

// ReconcileAssessment is the typed, evidence-backed reconciliation verdict.
type ReconcileAssessment struct {
	Schema         string              `json:"schema"`
	Route          string              `json:"route"`
	Primitive      string              `json:"primitive,omitempty"`
	Goal           string              `json:"goal"`
	State          string              `json:"state"`
	OK             bool                `json:"ok"`
	Reason         string              `json:"reason,omitempty"`
	Detail         string              `json:"detail,omitempty"`
	Head           string              `json:"head"`
	Target         string              `json:"target"`
	TargetRef      string              `json:"target_ref"`
	Branch         string              `json:"branch"`
	Remote         string              `json:"remote"`
	Dirty          bool                `json:"dirty"`
	DirtyPaths     []string            `json:"dirty_paths,omitempty"`
	CollidingPaths []string            `json:"colliding_paths,omitempty"`
	MergeActive    bool                `json:"merge_active,omitempty"`
	Contention     bool                `json:"contention,omitempty"`
	Applied        bool                `json:"applied,omitempty"`
	Execution      *ReconcileExecution `json:"execution,omitempty"`
}

// ParseGoal decomposes a goal string into structured GoalInfo.
// Supported shapes:
//   - "" or "publish": kind publish, source HEAD, target <remote>/<branch>
//   - "publish <sha>": kind publish, source <sha>, target <remote>/<branch>
//   - "integrate": kind integrate, source HEAD, target <remote>/<branch> (or origin/main)
//   - "integrate <ref>": kind integrate, source HEAD, target <ref>
func ParseGoal(raw, defaultRemote, defaultBranch string) (GoalInfo, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		raw = "publish"
	}
	fields := strings.Fields(raw)
	kind := strings.ToLower(fields[0])
	switch kind {
	case "publish":
		source := "HEAD"
		if len(fields) > 1 {
			source = fields[1]
		}
		target := defaultRemote + "/main"
		if defaultBranch != "" {
			target = defaultRemote + "/" + defaultBranch
		}
		return GoalInfo{
			Raw:    raw,
			Kind:   "publish",
			Source: source,
			Target: target,
		}, nil
	case "integrate":
		target := defaultRemote + "/main"
		if defaultBranch != "" {
			target = defaultRemote + "/" + defaultBranch
		}
		if len(fields) > 1 {
			target = fields[1]
		}
		return GoalInfo{
			Raw:    raw,
			Kind:   "integrate",
			Source: "HEAD",
			Target: target,
		}, nil
	default:
		return GoalInfo{}, fmt.Errorf("invalid goal %q: must be publish, publish <sha>, or integrate [ref]", raw)
	}
}

// ReconcileRouter orchestrates the shared-trunk reconciliation assessment and routing.
type ReconcileRouter struct {
	opts ReconcileOptions
}

// NewReconcileRouter constructs a ReconcileRouter with normalized defaults.
func NewReconcileRouter(opts ReconcileOptions) *ReconcileRouter {
	if strings.TrimSpace(opts.Repo) == "" {
		opts.Repo = "."
	}
	if strings.TrimSpace(opts.Remote) == "" {
		opts.Remote = "origin"
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.WriterLeaseTTL <= 0 {
		opts.WriterLeaseTTL = DefaultWriterLeaseTTL
	}
	return &ReconcileRouter{opts: opts}
}

// RouteReconciliation inspects repository state and computes the typed reconciliation route.
func RouteReconciliation(ctx context.Context, opts ReconcileOptions) (ReconcileAssessment, error) {
	return NewReconcileRouter(opts).Route(ctx)
}

// Route runs the state inspection pipeline and emits the ReconcileAssessment.
func (r *ReconcileRouter) Route(ctx context.Context) (ReconcileAssessment, error) {
	run := r.opts.Runner
	repo := r.opts.Repo

	branch := strings.TrimSpace(r.opts.Branch)
	if branch == "" {
		b, err := currentBranch(ctx, run, repo)
		if err == nil {
			branch = b
		}
	}
	if branch == "" {
		branch = "main"
	}

	goalInfo, err := ParseGoal(r.opts.Goal, r.opts.Remote, branch)
	if err != nil {
		return ReconcileAssessment{}, err
	}

	if r.opts.Fetch {
		_, _ = checked(ctx, run, repo, "fetch", r.opts.Remote, branch)
	}

	// 1. Observed local HEAD and target remote ref SHA (pinned across all subsequent checks).
	sourceRef := goalInfo.Source
	if sourceRef == "" {
		sourceRef = "HEAD"
	}
	headSHA, err := rev(ctx, run, repo, sourceRef)
	if err != nil {
		return ReconcileAssessment{}, fmt.Errorf("reconcile: resolve source ref %s: %w", sourceRef, err)
	}

	targetRef := goalInfo.Target
	targetSHA, err := rev(ctx, run, repo, targetRef)
	if err != nil {
		var ge *GitError
		if errors.As(err, &ge) {
			return ReconcileAssessment{
				Schema:    ReconcileSchema,
				Goal:      goalInfo.Raw,
				Head:      headSHA,
				TargetRef: targetRef,
				Branch:    branch,
				Remote:    r.opts.Remote,
				State:     StateNoRemoteRef,
				Reason:    fmt.Sprintf("target ref %s not found", targetRef),
				OK:        false,
			}, nil
		}
		return ReconcileAssessment{}, fmt.Errorf("reconcile: resolve target ref %s: %w", targetRef, err)
	}

	// 2. Dirtiness of working tree and dirty path ownership vs incoming changes.
	dirtyPaths, _ := workingTreeDirtyPaths(ctx, run, repo)
	isDirty := len(dirtyPaths) > 0

	// 3. Active MERGE_HEAD check (if present, flags ROUTE_HOLD_MERGE_ACTIVE).
	if isMergeActive(ctx, run, repo) {
		return ReconcileAssessment{
			Schema:      ReconcileSchema,
			Route:       RouteHoldMergeActive,
			Goal:        goalInfo.Raw,
			State:       "merge-active",
			OK:          false,
			Reason:      ReasonMergeActivePeerOwned,
			Detail:      "a merge is currently in progress (MERGE_HEAD present); resolve or abort merge before reconciling",
			Head:        headSHA,
			Target:      targetSHA,
			TargetRef:   targetRef,
			Branch:      branch,
			Remote:      r.opts.Remote,
			Dirty:       isDirty,
			DirtyPaths:  dirtyPaths,
			MergeActive: true,
		}, nil
	}

	// 5. Active lock/quiescence contention check (flags ROUTE_DRAIN).
	contention := r.opts.Contention || isContentionActive(repo, r.opts.Now, r.opts.WriterLeaseTTL)
	if contention {
		return ReconcileAssessment{
			Schema:     ReconcileSchema,
			Route:      RouteDrain,
			Goal:       goalInfo.Raw,
			State:      "contention",
			OK:         false,
			Reason:     ReasonQueuedAwaitingQuiescence,
			Detail:     "trunk is under active lock or quiescence contention; wait for quiescence or drain queue",
			Primitive:  fmt.Sprintf("fak sync drain --remote %s --branch %s", r.opts.Remote, branch),
			Head:       headSHA,
			Target:     targetSHA,
			TargetRef:  targetRef,
			Branch:     branch,
			Remote:     r.opts.Remote,
			Dirty:      isDirty,
			DirtyPaths: dirtyPaths,
			Contention: true,
		}, nil
	}

	// 4. Classify relation.
	// 4a. in-sync: if goal is publish or integrate -> ROUTE_NOOP ("repository is in sync with target")
	if headSHA == targetSHA {
		return ReconcileAssessment{
			Schema:     ReconcileSchema,
			Route:      RouteNoop,
			Goal:       goalInfo.Raw,
			State:      StateInSync,
			OK:         true,
			Reason:     "repository is in sync with target",
			Head:       headSHA,
			Target:     targetSHA,
			TargetRef:  targetRef,
			Branch:     branch,
			Remote:     r.opts.Remote,
			Dirty:      isDirty,
			DirtyPaths: dirtyPaths,
		}, nil
	}

	targetIsAncestor, err := isAncestor(ctx, run, repo, targetSHA, headSHA)
	if err != nil {
		return ReconcileAssessment{}, err
	}

	// 4b. ahead: if goal is publish -> ROUTE_PUSH. If --apply is set, calls SafePush.
	if targetIsAncestor {
		primitive := fmt.Sprintf("fak sync push --remote %s --branch %s", r.opts.Remote, branch)
		route := RoutePush
		reason := fmt.Sprintf("local branch is ahead of remote; ready to publish through `%s`", primitive)
		ok := true
		if goalInfo.Kind == "integrate" {
			route = RouteNoop
			primitive = ""
			reason = "repository already contains target; nothing to integrate"
		}

		assessment := ReconcileAssessment{
			Schema:     ReconcileSchema,
			Route:      route,
			Primitive:  primitive,
			Goal:       goalInfo.Raw,
			State:      StateAhead,
			OK:         ok,
			Reason:     reason,
			Head:       headSHA,
			Target:     targetSHA,
			TargetRef:  targetRef,
			Branch:     branch,
			Remote:     r.opts.Remote,
			Dirty:      isDirty,
			DirtyPaths: dirtyPaths,
		}

		if r.opts.Apply && route == RoutePush {
			pushOpts := PushOptions{
				Repo:           repo,
				Remote:         r.opts.Remote,
				Branch:         branch,
				SourceRef:      headSHA,
				TargetRef:      "refs/heads/" + branch,
				Runner:         run,
				Now:            r.opts.Now,
				VelocityBudget: DefaultPushVelocityBudget,
			}
			pushRes, pushErr := SafePush(ctx, pushOpts)
			exec := &ReconcileExecution{
				Primitive: primitive,
				Pushed:    pushRes.Pushed,
				Success:   pushRes.Pushed,
				Detail:    pushRes.Detail,
			}
			if pushErr != nil {
				exec.Error = pushErr.Error()
			}
			assessment.Execution = exec
			assessment.Applied = pushRes.Pushed
		}
		return assessment, nil
	}

	headIsAncestor, err := isAncestor(ctx, run, repo, headSHA, targetSHA)
	if err != nil {
		return ReconcileAssessment{}, err
	}

	// 4c. behind: checks write-set safety (uncommitted files collision).
	//     - If clean/disjoint: ROUTE_APPLY. If --apply, calls Apply.
	//     - If colliding: ROUTE_HOLD_DIRTY_COLLISION with reason DIRTY_WRITE_OVERLAP.
	if headIsAncestor {
		entries, ffErr := ffWriteSet(ctx, run, repo, headSHA, targetSHA)
		if ffErr != nil {
			return ReconcileAssessment{}, ffErr
		}
		_, divergent := classify(repo, run, ctx, headSHA, targetSHA, entries, false)

		var colliding []string
		for _, d := range divergent {
			colliding = append(colliding, d.Path)
		}
		colliding = uniqueSorted(colliding)

		assessment := ReconcileAssessment{
			Schema:         ReconcileSchema,
			Goal:           goalInfo.Raw,
			State:          StateBehind,
			Head:           headSHA,
			Target:         targetSHA,
			TargetRef:      targetRef,
			Branch:         branch,
			Remote:         r.opts.Remote,
			Dirty:          isDirty,
			DirtyPaths:     dirtyPaths,
			CollidingPaths: colliding,
		}

		if len(divergent) > 0 {
			assessment.Route = RouteHoldDirtyCollision
			assessment.OK = false
			assessment.Reason = ReasonDirtyWriteOverlap
			assessment.Detail = fmt.Sprintf("%d uncommitted file(s) collide with incoming changes; ensure working paths are clear before syncing", len(divergent))
			return assessment, nil
		}

		primitive := fmt.Sprintf("fak sync apply --remote %s --branch %s", r.opts.Remote, branch)
		assessment.Route = RouteApply
		assessment.Primitive = primitive
		assessment.OK = true
		assessment.Reason = "every incoming path is clean or disjoint from working tree; safe to fast-forward"

		if r.opts.Apply {
			applyOpts := Options{
				Repo:                repo,
				Remote:              r.opts.Remote,
				Branch:              branch,
				Runner:              run,
				Now:                 r.opts.Now,
				WriterLeaseTTL:      r.opts.WriterLeaseTTL,
				ApplyVelocityBudget: DefaultPushVelocityBudget,
			}
			applyRes, applyErr := Apply(ctx, applyOpts)
			exec := &ReconcileExecution{
				Primitive: primitive,
				Applied:   applyRes.Applied,
				Success:   applyRes.Applied,
				NewHead:   applyRes.NewHead,
				Detail:    applyRes.Reason,
			}
			if applyErr != nil {
				exec.Error = applyErr.Error()
			}
			assessment.Execution = exec
			assessment.Applied = applyRes.Applied
		}
		return assessment, nil
	}

	// 4d. diverged:
	//     - checks merge preview: if trivial superset (ours or theirs already subsumes, or textless merge): ROUTE_SUPERSET_MERGE.
	//     - if disjoint paths: ROUTE_DISJOINT_INTEGRATE / ROUTE_APPLY.
	//     - if overlapping content conflicts: ROUTE_RECONCILE_PACKET (names reason DIVERGED_OVERLAP).
	assessment := ReconcileAssessment{
		Schema:     ReconcileSchema,
		Goal:       goalInfo.Raw,
		State:      StateDiverged,
		Head:       headSHA,
		Target:     targetSHA,
		TargetRef:  targetRef,
		Branch:     branch,
		Remote:     r.opts.Remote,
		Dirty:      isDirty,
		DirtyPaths: dirtyPaths,
	}

	if checkTrivialSuperset(ctx, run, repo, headSHA, targetSHA) {
		primitive := fmt.Sprintf("fak sync apply --remote %s --branch %s", r.opts.Remote, branch)
		assessment.Route = RouteSupersetMerge
		assessment.Primitive = primitive
		assessment.OK = true
		assessment.Reason = "diverged branch can be textlessly resolved: trivial superset"

		if r.opts.Apply {
			mpRunner := func(ctx context.Context, dir string, args ...string) (mergepreview.RunResult, error) {
				res := run(ctx, dir, args...)
				return mergepreview.RunResult{Stdout: res.Stdout, Stderr: res.Stderr, Code: res.Code}, res.Err
			}
			mpRes, mpErr := mergepreview.Apply(ctx, repo, targetSHA, "", mpRunner)
			exec := &ReconcileExecution{
				Primitive: primitive,
				Applied:   mpRes.ApplyOutcome == mergepreview.ApplyResolvedSuperset,
				Success:   mpRes.ApplyOutcome == mergepreview.ApplyResolvedSuperset,
				Detail:    mpRes.Detail,
			}
			if mpErr != nil {
				exec.Error = mpErr.Error()
			}
			if mpRes.MergeCommit != "" {
				exec.NewHead = mpRes.MergeCommit
			}
			assessment.Execution = exec
			assessment.Applied = exec.Applied
		}
		return assessment, nil
	}

	divergence := classifyDivergedPaths(ctx, run, repo, headSHA, targetSHA)
	if divergence == ReasonDivergedDisjoint {
		primitive := fmt.Sprintf("fak sync apply --remote %s --branch %s", r.opts.Remote, branch)
		assessment.Route = RouteDisjointIntegrate
		assessment.Primitive = primitive
		assessment.OK = true
		assessment.Reason = ReasonDivergedDisjoint
		assessment.Detail = "local and remote changes touch disjoint paths; safe to integrate"

		if r.opts.Apply {
			mergeRes := run(ctx, repo, "merge", "--no-ff", "--no-edit", "--signoff", "-m", fmt.Sprintf("Merge %s (disjoint integrate)", targetRef), targetSHA)
			success := mergeRes.Err == nil && mergeRes.Code == 0
			newHead, _ := rev(ctx, run, repo, "HEAD")
			exec := &ReconcileExecution{
				Primitive: primitive,
				Applied:   success,
				Success:   success,
				NewHead:   newHead,
				Detail:    runDetail(mergeRes),
			}
			if mergeRes.Err != nil {
				exec.Error = mergeRes.Err.Error()
			}
			assessment.Execution = exec
			assessment.Applied = success
		}
		return assessment, nil
	}

	// Overlapping conflicts
	assessment.Route = RouteReconcilePacket
	assessment.OK = false
	assessment.Reason = ReasonDivergedOverlap
	assessment.Detail = "local and remote have conflicting overlapping changes; manual reconciliation packet required"

	// Enumerate conflicting / overlapping paths
	mbRes := run(ctx, repo, "merge-base", headSHA, targetSHA)
	if mbRes.Err == nil && mbRes.Code == 0 {
		mb := strings.TrimSpace(string(mbRes.Stdout))
		if mb != "" {
			localDiff := run(ctx, repo, "diff", "--name-only", mb, headSHA)
			remoteDiff := run(ctx, repo, "diff", "--name-only", mb, targetSHA)
			localFiles := make(map[string]bool)
			for _, f := range strings.Split(strings.TrimSpace(string(localDiff.Stdout)), "\n") {
				f = strings.TrimSpace(f)
				if f != "" {
					localFiles[f] = true
				}
			}
			var overlapping []string
			for _, f := range strings.Split(strings.TrimSpace(string(remoteDiff.Stdout)), "\n") {
				f = strings.TrimSpace(f)
				if f != "" && localFiles[f] {
					overlapping = append(overlapping, f)
				}
			}
			assessment.CollidingPaths = uniqueSorted(overlapping)
		}
	}

	return assessment, nil
}

// checkTrivialSuperset tests if either side already subsumes the other or resolves textlessly.
func checkTrivialSuperset(ctx context.Context, run Runner, repo, head, target string) bool {
	// 1. If trees are identical, trivial superset.
	headTree, err := rev(ctx, run, repo, head+"^{tree}")
	if err == nil {
		targetTree, err := rev(ctx, run, repo, target+"^{tree}")
		if err == nil && headTree != "" && headTree == targetTree {
			return true
		}
	}

	// 2. 3-way merge-tree preview.
	mpRunner := func(ctx context.Context, dir string, args ...string) (mergepreview.RunResult, error) {
		res := run(ctx, dir, args...)
		return mergepreview.RunResult{
			Stdout: res.Stdout,
			Stderr: res.Stderr,
			Code:   res.Code,
		}, res.Err
	}
	pre, err := mergepreview.Preview(ctx, repo, target, mpRunner)
	if err != nil {
		return false
	}
	if pre.Outcome == mergepreview.OutcomeEmptyNetDiff {
		return true
	}
	// Check if target already subsumes HEAD (diff target vs mergeTree is empty).
	if pre.MergeTree != "" {
		diffRes := run(ctx, repo, "diff", "--name-only", "-z", target, pre.MergeTree)
		if diffRes.Err == nil && diffRes.Code == 0 && len(bytes.TrimSpace(diffRes.Stdout)) == 0 {
			return true
		}
	}
	return false
}

// isMergeActive reports whether an unresolved merge is in flight.
func isMergeActive(ctx context.Context, run Runner, repo string) bool {
	res := run(ctx, repo, "rev-parse", "--verify", "-q", "MERGE_HEAD")
	if res.Err == nil && res.Code == 0 && len(bytes.TrimSpace(res.Stdout)) > 0 {
		return true
	}
	gd, err := worktreeGitDir(repo)
	if err == nil {
		if _, statErr := os.Stat(filepath.Join(gd, "MERGE_HEAD")); statErr == nil {
			return true
		}
	}
	return false
}

// isContentionActive checks for active lock or quiescence contention.
func isContentionActive(repo string, now func() time.Time, ttl time.Duration) bool {
	if _, held := ActiveWriterLease(repo, now, ttl); held {
		return true
	}
	gd, err := worktreeGitDir(repo)
	if err == nil {
		lockPath := filepath.Join(gd, "index.lock")
		if fi, statErr := os.Stat(lockPath); statErr == nil {
			if now == nil {
				now = time.Now
			}
			if now().Sub(fi.ModTime()) < 10*time.Minute {
				return true
			}
		}
	}
	qp := filepath.Join(repo, ".fak", "sync-drain-queue.json")
	if b, readErr := os.ReadFile(qp); readErr == nil {
		var q struct {
			Entries []json.RawMessage `json:"entries"`
		}
		if json.Unmarshal(b, &q) == nil && len(q.Entries) > 0 {
			return true
		}
	}
	return false
}

// workingTreeDirtyPaths returns uncommitted dirty paths from git status.
func workingTreeDirtyPaths(ctx context.Context, run Runner, repo string) ([]string, error) {
	res := run(ctx, repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if res.Err != nil || res.Code != 0 {
		return nil, fmt.Errorf("status: code=%d err=%v", res.Code, res.Err)
	}
	fields := splitNUL(res.Stdout)
	var paths []string
	for i := 0; i < len(fields); i++ {
		field := fields[i]
		if len(field) < 3 {
			continue
		}
		code := field[:2]
		path := field[3:]
		if code[0] == 'R' || code[0] == 'C' {
			paths = append(paths, path)
			if i+1 < len(fields) {
				i++
				paths = append(paths, fields[i])
			}
			continue
		}
		paths = append(paths, path)
	}
	return uniqueSorted(paths), nil
}

func uniqueSorted(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	m := make(map[string]bool, len(s))
	for _, v := range s {
		v = strings.TrimSpace(v)
		if v != "" {
			m[v] = true
		}
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func splitNUL(b []byte) []string {
	fields := strings.Split(string(b), "\x00")
	var out []string
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}
