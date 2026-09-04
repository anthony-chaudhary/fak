package safesync

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

// PacketSchema is the schema identifier for typed reconciliation packets.
const PacketSchema = "fak.sync.packet.v1"

// Disposition represents the typed reconciliation outcome.
type Disposition string

const (
	DispositionSafeDisjoint               Disposition = "safe-disjoint"
	DispositionTrivialSuperset            Disposition = "trivial-superset"
	DispositionWaitForOwner               Disposition = "wait-for-owner"
	DispositionOwnerHandoffRequired       Disposition = "owner-handoff-required"
	DispositionSemanticConflictReview     Disposition = "semantic-conflict-review"
	DispositionOwnerAuthorizedPathSuspend Disposition = "owner-authorized-path-suspend"
)

// CommitInfo captures metadata for a commit participating in divergence.
type CommitInfo struct {
	SHA     string   `json:"sha"`
	Subject string   `json:"subject"`
	Paths   []string `json:"paths"`
}

// PathOwnerInfo captures lane and lease ownership for a path.
type PathOwnerInfo struct {
	Lane      string `json:"lane"`
	Owner     string `json:"owner,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	Active    bool   `json:"active"`
}

// MergePreviewInfo captures 3-way merge-tree preview results.
type MergePreviewInfo struct {
	Clean     bool     `json:"clean"`
	Superset  bool     `json:"superset"`
	Conflicts []string `json:"conflicts"`
}

// ReconciliationPacket is the owner-aware reconciliation packet for genuine divergence.
type ReconciliationPacket struct {
	Schema            string                   `json:"schema"`
	GeneratedAt       string                   `json:"generated_at"`
	Repo              string                   `json:"repo"`
	LocalHead         string                   `json:"local_head"`
	MergeBase         string                   `json:"merge_base"`
	TargetRef         string                   `json:"target_ref"`
	TargetSHA         string                   `json:"target_sha"`
	LocalCommits      []CommitInfo             `json:"local_commits"`
	RemoteCommits     []CommitInfo             `json:"remote_commits"`
	DirtyPaths        []string                 `json:"dirty_paths"`
	PathOwnership     map[string]PathOwnerInfo `json:"path_ownership"`
	MergePreview      MergePreviewInfo         `json:"merge_preview"`
	Disposition       Disposition              `json:"disposition"`
	Dispatchable      bool                     `json:"dispatchable"`
	RequiredWitnesses []string                 `json:"required_witnesses"`
}

// PacketOptions configures BuildReconciliationPacket.
type PacketOptions struct {
	Repo         string           `json:"repo"`
	Remote       string           `json:"remote"`
	Branch       string           `json:"branch,omitempty"`
	TargetRef    string           `json:"target_ref,omitempty"`
	TargetSHA    string           `json:"target_sha,omitempty"`
	LocalHead    string           `json:"local_head,omitempty"`
	Session      string           `json:"session,omitempty"`
	SuspendPaths []string         `json:"suspend_paths,omitempty"`
	Fetch        bool             `json:"fetch,omitempty"`
	Runner       Runner           `json:"-"`
	Now          func() time.Time `json:"-"`
}

// BuildReconciliationPacket inspects repository state and builds an owner-aware reconciliation packet.
func BuildReconciliationPacket(ctx context.Context, opts PacketOptions) (*ReconciliationPacket, error) {
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

	repo := opts.Repo
	run := opts.Runner

	branch := strings.TrimSpace(opts.Branch)
	if branch == "" {
		b, err := currentBranch(ctx, run, repo)
		if err == nil && b != "" {
			branch = b
		} else {
			branch = "main"
		}
	}

	targetRef := strings.TrimSpace(opts.TargetRef)
	if targetRef == "" {
		targetRef = opts.Remote + "/" + branch
	}

	if opts.Fetch {
		_, _ = checked(ctx, run, repo, "fetch", opts.Remote, branch)
	}

	localHead := strings.TrimSpace(opts.LocalHead)
	if localHead == "" {
		var err error
		localHead, err = rev(ctx, run, repo, "HEAD")
		if err != nil {
			return nil, fmt.Errorf("resolve HEAD: %w", err)
		}
	}

	targetSHA := strings.TrimSpace(opts.TargetSHA)
	if targetSHA == "" {
		var err error
		targetSHA, err = rev(ctx, run, repo, targetRef)
		if err != nil {
			return nil, fmt.Errorf("resolve target ref %q: %w", targetRef, err)
		}
	}

	mergeBase := ""
	if localHead == targetSHA {
		mergeBase = localHead
	} else {
		mbRes := run(ctx, repo, "merge-base", localHead, targetSHA)
		if mbRes.Err == nil && mbRes.Code == 0 {
			mergeBase = strings.TrimSpace(string(mbRes.Stdout))
		}
	}

	localCommits, err := readCommitsBetween(ctx, run, repo, mergeBase, localHead)
	if err != nil {
		return nil, fmt.Errorf("read local commits: %w", err)
	}
	remoteCommits, err := readCommitsBetween(ctx, run, repo, mergeBase, targetSHA)
	if err != nil {
		return nil, fmt.Errorf("read remote commits: %w", err)
	}

	dirtyPaths, err := workingTreeDirtyPaths(ctx, run, repo)
	if err != nil || dirtyPaths == nil {
		dirtyPaths = []string{}
	}

	preview := MergePreviewInfo{
		Conflicts: []string{},
	}
	if localHead == targetSHA {
		preview.Clean = true
		preview.Superset = true
	} else {
		merged := run(ctx, repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", localHead, targetSHA)
		if merged.Err == nil && (merged.Code == 0 || merged.Code == 1) {
			fields := splitNUL(merged.Stdout)
			if merged.Code == 0 {
				preview.Clean = true
				preview.Superset = checkTrivialSuperset(ctx, run, repo, localHead, targetSHA) ||
					(mergeBase != "" && (mergeBase == targetSHA || mergeBase == localHead))
			} else {
				preview.Clean = false
				preview.Superset = false
				if len(fields) > 1 {
					var conflicts []string
					for _, f := range fields[1:] {
						f = strings.TrimSpace(f)
						if f != "" {
							conflicts = append(conflicts, filepath.Clean(filepath.ToSlash(f)))
						}
					}
					preview.Conflicts = uniqueSorted(conflicts)
				}
			}
		} else {
			preview.Clean = false
			preview.Superset = false
		}
	}

	allPathsMap := make(map[string]bool)
	for _, c := range localCommits {
		for _, p := range c.Paths {
			allPathsMap[p] = true
		}
	}
	for _, c := range remoteCommits {
		for _, p := range c.Paths {
			allPathsMap[p] = true
		}
	}
	for _, p := range dirtyPaths {
		allPathsMap[p] = true
	}
	for _, p := range preview.Conflicts {
		allPathsMap[p] = true
	}

	allPaths := make([]string, 0, len(allPathsMap))
	for p := range allPathsMap {
		allPaths = append(allPaths, p)
	}
	sort.Strings(allPaths)

	tax, _ := regionadmit.LoadTaxonomy(repo)

	lrRunner := func(c context.Context, dir string, args ...string) (string, int, error) {
		r := run(c, dir, args...)
		return string(r.Stdout), r.Code, r.Err
	}
	store := leaseref.NewWithRunner(lrRunner, repo)
	nowTime := opts.Now()
	liveLeases, _, _ := store.Live(ctx, nowTime)
	writerLease, hasWriterLease := ActiveWriterLease(repo, opts.Now, DefaultWriterLeaseTTL)

	pathOwnership := make(map[string]PathOwnerInfo)
	for _, p := range allPaths {
		lane := resolveLaneForPath(p, tax)
		info := PathOwnerInfo{
			Lane:   lane,
			Active: false,
		}

		for _, rec := range liveLeases {
			matches := false
			if rec.ID == lane && lane != "" {
				matches = true
			} else if len(rec.TreeGlobs) > 0 && (dispatchorder.TreesOverlap([]string{p}, rec.TreeGlobs) || matchesAnyGlob(p, rec.TreeGlobs)) {
				matches = true
			}
			if matches {
				info.Owner = rec.Holder
				info.SessionID = rec.SessionID
				info.Active = !rec.Expired(nowTime)
				break
			}
		}

		if !info.Active && hasWriterLease && writerLease != nil {
			info.Owner = writerLease.Owner
			info.Active = true
		}

		if info.SessionID == "" && opts.Session != "" {
			for _, dp := range dirtyPaths {
				if dp == p {
					info.SessionID = opts.Session
					if info.Owner == "" {
						info.Owner = opts.Session
					}
					break
				}
			}
		}

		pathOwnership[p] = info
	}

	localWriteSet := make(map[string]bool)
	for _, c := range localCommits {
		for _, p := range c.Paths {
			localWriteSet[p] = true
		}
	}
	remoteWriteSet := make(map[string]bool)
	for _, c := range remoteCommits {
		for _, p := range c.Paths {
			remoteWriteSet[p] = true
		}
	}
	dirtySet := make(map[string]bool)
	for _, p := range dirtyPaths {
		dirtySet[p] = true
	}

	suspendMap := make(map[string]bool)
	for _, sp := range opts.SuspendPaths {
		suspendMap[filepath.Clean(filepath.ToSlash(sp))] = true
	}

	var overlappingCommits []string
	for p := range remoteWriteSet {
		if localWriteSet[p] {
			overlappingCommits = append(overlappingCommits, p)
		}
	}

	var collidingDirty []string
	for p := range remoteWriteSet {
		if dirtySet[p] {
			collidingDirty = append(collidingDirty, p)
		}
	}

	allDirtySuspended := len(collidingDirty) > 0
	for _, p := range collidingDirty {
		if !suspendMap[p] {
			allDirtySuspended = false
			break
		}
	}

	conflictingPathsMap := make(map[string]bool)
	for _, p := range preview.Conflicts {
		conflictingPathsMap[p] = true
	}
	for _, p := range overlappingCommits {
		conflictingPathsMap[p] = true
	}
	for _, p := range collidingDirty {
		if !suspendMap[p] {
			conflictingPathsMap[p] = true
		}
	}

	activePeerLeaseHeld := false
	for p := range conflictingPathsMap {
		if info, ok := pathOwnership[p]; ok && info.Active {
			if opts.Session == "" || (info.SessionID != opts.Session && info.Owner != opts.Session) {
				activePeerLeaseHeld = true
				break
			}
		}
	}

	var disposition Disposition
	var dispatchable bool

	switch {
	case activePeerLeaseHeld:
		disposition = DispositionWaitForOwner
		dispatchable = false
	case !preview.Clean || len(preview.Conflicts) > 0:
		disposition = DispositionSemanticConflictReview
		dispatchable = false
	case len(conflictingPathsMap) > 0:
		disposition = DispositionOwnerHandoffRequired
		dispatchable = false
	case allDirtySuspended && len(overlappingCommits) == 0 && preview.Clean:
		disposition = DispositionOwnerAuthorizedPathSuspend
		dispatchable = true
	case preview.Superset:
		disposition = DispositionTrivialSuperset
		dispatchable = true
	default:
		disposition = DispositionSafeDisjoint
		dispatchable = true
	}

	return &ReconciliationPacket{
		Schema:            PacketSchema,
		GeneratedAt:       nowTime.UTC().Format(time.RFC3339),
		Repo:              repo,
		LocalHead:         localHead,
		MergeBase:         mergeBase,
		TargetRef:         targetRef,
		TargetSHA:         targetSHA,
		LocalCommits:      localCommits,
		RemoteCommits:     remoteCommits,
		DirtyPaths:        dirtyPaths,
		PathOwnership:     pathOwnership,
		MergePreview:      preview,
		Disposition:       disposition,
		Dispatchable:      dispatchable,
		RequiredWitnesses: []string{"tests", "build check", "remote containment"},
	}, nil
}

func readCommitsBetween(ctx context.Context, run Runner, repo, base, tip string) ([]CommitInfo, error) {
	if tip == "" {
		return []CommitInfo{}, nil
	}
	if base == tip {
		return []CommitInfo{}, nil
	}
	var args []string
	if base == "" {
		args = []string{"log", "--name-only", "--format=COMMIT:%H%x00%s", tip}
	} else {
		args = []string{"log", "--name-only", "--format=COMMIT:%H%x00%s", fmt.Sprintf("%s..%s", base, tip)}
	}
	res := run(ctx, repo, args...)
	if res.Err != nil || res.Code != 0 {
		return nil, fmt.Errorf("git %s: code=%d err=%v", strings.Join(args, " "), res.Code, res.Err)
	}
	return parseGitLogOutput(res.Stdout), nil
}

func parseGitLogOutput(b []byte) []CommitInfo {
	commits := make([]CommitInfo, 0)
	var cur *CommitInfo
	lines := strings.Split(string(b), "\n")
	for _, line := range lines {
		line = strings.TrimRight(line, "\r")
		if strings.HasPrefix(line, "COMMIT:") {
			if cur != nil {
				cur.Paths = uniqueSorted(cur.Paths)
				commits = append(commits, *cur)
			}
			rest := strings.TrimPrefix(line, "COMMIT:")
			parts := strings.SplitN(rest, "\x00", 2)
			sha := parts[0]
			subject := ""
			if len(parts) > 1 {
				subject = parts[1]
			}
			cur = &CommitInfo{
				SHA:     sha,
				Subject: subject,
				Paths:   []string{},
			}
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && cur != nil {
			cleanPath := filepath.Clean(filepath.ToSlash(trimmed))
			cur.Paths = append(cur.Paths, cleanPath)
		}
	}
	if cur != nil {
		cur.Paths = uniqueSorted(cur.Paths)
		commits = append(commits, *cur)
	}
	return commits
}

func resolveLaneForPath(path string, tax regionadmit.Taxonomy) string {
	if len(tax.Trees) > 0 {
		if l := laneadmit.LaneForPath(path, laneadmit.Taxonomy{Trees: tax.Trees, Exclusive: tax.Exclusive, Loaded: true}, laneadmit.GranLeaf); l != "" {
			return l
		}
	}
	norm := filepath.ToSlash(path)
	segs := strings.Split(norm, "/")
	if len(segs) >= 2 {
		switch segs[0] {
		case "internal":
			return segs[1]
		case "cmd":
			return "cmd"
		case "docs", "tools", "examples", "visuals", "experiments":
			return segs[0]
		}
	}
	return ""
}

func matchesAnyGlob(path string, globs []string) bool {
	p := filepath.ToSlash(path)
	for _, g := range globs {
		g = filepath.ToSlash(g)
		if g == p {
			return true
		}
		if strings.HasSuffix(g, "/**") {
			prefix := strings.TrimSuffix(g, "/**")
			if strings.HasPrefix(p, prefix+"/") || p == prefix {
				return true
			}
		}
		if strings.HasSuffix(g, "/*") {
			dir := strings.TrimSuffix(g, "/*")
			if filepath.Dir(p) == dir {
				return true
			}
		}
		if m, _ := filepath.Match(g, p); m {
			return true
		}
	}
	return false
}
