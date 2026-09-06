package safecommit

import (
	"context"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// ReasonPeerWIPCollision reports that a path-scoped git op (especially a directory pathspec)
// would sweep WIP owned by a peer session or untracked conflicting peer work (#11232).
const ReasonPeerWIPCollision = "PEER_WIP_COLLISION"

const peerWIPGuardEnvVar = "FAK_PEER_WIP_GUARD"

// One budget covers the entire attribution scan, not each path or subprocess.
const peerWIPAttributionTimeout = 30 * time.Second

// peerWIPGuardMode reads FAK_PEER_WIP_GUARD (block|warn|off, default block).
func peerWIPGuardMode() staleBaseMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(peerWIPGuardEnvVar))) {
	case "off", "0", "false":
		return staleBaseOff
	case "warn", "advisory":
		return staleBaseWarn
	default:
		return staleBaseBlock
	}
}

// PathAttributionOptions configures path attribution validation.
type PathAttributionOptions struct {
	SessionID              string
	SessionScope           []string
	PeerWIP                map[string]string // path -> peer session id
	PeerWIPChecker         func(path string) (peerSession string, isPeer bool)
	RestrictToSessionScope bool
}

// PathAttributionResult is the outcome of validating paths against peer WIP and session scope.
type PathAttributionResult struct {
	OK             bool     `json:"ok"`
	Reason         string   `json:"reason,omitempty"`
	Detail         string   `json:"detail,omitempty"`
	CollidingPaths []string `json:"colliding_paths,omitempty"`
	PeerSessions   []string `json:"peer_sessions,omitempty"`
	ExpandedPaths  []string `json:"expanded_paths,omitempty"`
	EffectivePaths []string `json:"effective_paths,omitempty"`
}

// ValidatePathAttribution validates that requested paths (including directory pathspecs)
// do not sweep peer WIP or untracked conflicting peer work.
func ValidatePathAttribution(ctx context.Context, run Runner, dir string, requestedPaths []string, opts PathAttributionOptions) (PathAttributionResult, error) {
	if d, ok := ctx.Deadline(); !ok || time.Until(d) > peerWIPAttributionTimeout {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, peerWIPAttributionTimeout)
		defer cancel()
	}
	if err := ctx.Err(); err != nil {
		return PathAttributionResult{}, err
	}
	norm, ok := normalizePaths(requestedPaths)
	if !ok || len(norm) == 0 {
		return PathAttributionResult{OK: false, Reason: ReasonNoPath, Detail: "no valid repo-relative pathspec given"}, nil
	}
	statusArgs := append([]string{"status", "--porcelain", "--"}, norm...)
	statusOut, err := runPeerWIPLookup(ctx, run, dir, statusArgs...)
	if err != nil {
		return PathAttributionResult{}, err
	}
	return checkPathAttributionFromStatus(ctx, run, dir, norm, statusOut, opts)
}

// checkPathAttributionFromStatus checks whether dirty/staged/untracked paths under requested
// directory pathspecs collide with peer WIP or untracked peer work.
func checkPathAttributionFromStatus(ctx context.Context, run Runner, dir string, requestedPaths []string, statusOut string, opts PathAttributionOptions) (PathAttributionResult, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, peerWIPAttributionTimeout)
		defer cancel()
	}
	if err := ctx.Err(); err != nil {
		return PathAttributionResult{}, err
	}
	changedPaths := statusChangedPaths(statusOut)
	if err := ctx.Err(); err != nil {
		return PathAttributionResult{}, err
	}
	if len(changedPaths) == 0 {
		return PathAttributionResult{
			OK:             true,
			EffectivePaths: requestedPaths,
		}, nil
	}

	sessionID := strings.TrimSpace(opts.SessionID)
	if sessionID == "" {
		sessionID = strings.TrimSpace(os.Getenv("FAK_SESSION_ID"))
	}
	if sessionID == "" {
		sessionID = strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID"))
	}

	type pathStatusInfo struct {
		untracked bool
	}
	statusMap := make(map[string]pathStatusInfo)
	for _, line := range strings.Split(strings.ReplaceAll(statusOut, "\r\n", "\n"), "\n") {
		p := statusLinePath(line)
		if p == "" {
			continue
		}
		xy := ""
		if len(line) >= 2 {
			xy = line[:2]
		}
		statusMap[p] = pathStatusInfo{
			untracked: xy == "??",
		}
	}

	var gitOwners map[string]string
	if len(opts.PeerWIP) == 0 && opts.PeerWIPChecker == nil && run != nil {
		var err error
		gitOwners, err = resolveGitPeerOwners(ctx, run, dir, changedPaths, sessionID)
		if err != nil {
			return PathAttributionResult{}, err
		}
	}

	var collidingPaths []string
	var peerSessions []string
	seenPeers := make(map[string]bool)
	var descriptions []string

	for _, cp := range changedPaths {
		if err := ctx.Err(); err != nil {
			return PathAttributionResult{}, err
		}
		isDirSweep := false
		for _, req := range requestedPaths {
			if gitgate.TreeContains(req, cp) && req != cp {
				isDirSweep = true
				break
			}
		}

		var peerOwner string
		var isPeer bool

		// 1. Explicit checker
		if opts.PeerWIPChecker != nil {
			if peer, isP := opts.PeerWIPChecker(cp); isP {
				peerOwner = peer
				isPeer = true
			}
		}

		if err := ctx.Err(); err != nil {
			return PathAttributionResult{}, err
		}

		// 2. PeerWIP map
		if !isPeer && opts.PeerWIP != nil {
			if peer, exists := opts.PeerWIP[cp]; exists && peer != "" {
				if sessionID == "" || peer != sessionID {
					peerOwner = peer
					isPeer = true
				}
			}
		}

		// 3. Git peer checkpoints
		if !isPeer && len(opts.PeerWIP) == 0 && opts.PeerWIPChecker == nil && run != nil {
			peerOwner = gitOwners[cp]
			if peerOwner != "" {
				isPeer = true
			}
		}

		// 4. SessionScope check: if session scope is declared and this file is outside
		// it under a directory sweep, treat as peer WIP or untracked scratch.
		if !isPeer && len(opts.SessionScope) > 0 {
			inScope := gitgate.CoveredByAnyTree(cp, opts.SessionScope)
			if !inScope && isDirSweep {
				info := statusMap[cp]
				if info.untracked {
					peerOwner = "untracked-peer-work"
				} else {
					peerOwner = "peer"
				}
				isPeer = true
			}
		}

		if isPeer {
			collidingPaths = append(collidingPaths, cp)
			if peerOwner != "" && !seenPeers[peerOwner] {
				seenPeers[peerOwner] = true
				peerSessions = append(peerSessions, peerOwner)
			}
			desc := fmt.Sprintf("%s belongs to session %s", cp, peerOwner)
			if statusMap[cp].untracked {
				desc = fmt.Sprintf("%s (untracked peer work owned by %s)", cp, peerOwner)
			}
			descriptions = append(descriptions, desc)
		}
	}

	if err := ctx.Err(); err != nil {
		return PathAttributionResult{}, err
	}
	if len(collidingPaths) == 0 {
		return PathAttributionResult{
			OK:             true,
			ExpandedPaths:  changedPaths,
			EffectivePaths: requestedPaths,
		}, nil
	}

	sort.Strings(collidingPaths)
	sort.Strings(peerSessions)

	// If restrict requested and session scope is declared:
	if opts.RestrictToSessionScope && len(opts.SessionScope) > 0 {
		collidingSet := make(map[string]bool, len(collidingPaths))
		for _, c := range collidingPaths {
			collidingSet[c] = true
		}
		var kept []string
		for _, cp := range changedPaths {
			if err := ctx.Err(); err != nil {
				return PathAttributionResult{}, err
			}
			if !collidingSet[cp] && gitgate.CoveredByAnyTree(cp, opts.SessionScope) {
				kept = append(kept, cp)
			}
		}
		if len(kept) > 0 {
			sort.Strings(kept)
			kept = dedupeStrings(kept)
			if err := ctx.Err(); err != nil {
				return PathAttributionResult{}, err
			}
			return PathAttributionResult{
				OK:             true,
				CollidingPaths: collidingPaths,
				PeerSessions:   peerSessions,
				ExpandedPaths:  changedPaths,
				EffectivePaths: kept,
				Detail: fmt.Sprintf("restricted directory pathspec to %d session-scoped file(s); excluded %d peer path(s) (%s)",
					len(kept), len(collidingPaths), strings.Join(collidingPaths, ", ")),
			}, nil
		}
	}

	return PathAttributionResult{
		OK:             false,
		Reason:         ReasonPeerWIPCollision,
		CollidingPaths: collidingPaths,
		PeerSessions:   peerSessions,
		ExpandedPaths:  changedPaths,
		Detail: fmt.Sprintf("directory pathspec would sweep peer WIP: %s — reconcile rather than sweep, or narrow pathspec to explicit files",
			strings.Join(descriptions, "; ")),
	}, nil
}

// runPeerWIPLookup never interprets an incomplete lookup as an unowned path.
// Runner can report a killed process as an exit code without a Go error.
func runPeerWIPLookup(ctx context.Context, run Runner, dir string, args ...string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	out, code, err := run(ctx, dir, args...)
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	if err != nil {
		return "", fmt.Errorf("safecommit: peer WIP %s: %w", args[0], err)
	}
	if code != 0 {
		return "", fmt.Errorf("safecommit: peer WIP %s exited %d", args[0], code)
	}
	return out, nil
}

// Git object IDs bound argv size on Windows as well as subprocess amplification.
const peerWIPBatchSize = 128
const peerWIPRefFormat = "--format=%(refname)%00%(objectname)%00%(objecttype)%00%(contents:size)%00%(contents)%00"

// resolveGitPeerOwners snapshots ordered refs and their messages in one process.
// Messages are byte-length framed; embedded newlines/NULs cannot forge records.
// Deltas are fetched by immutable object ID in bounded, deduplicated chunks.
// Evaluate refs in snapshot order, so an earlier delta still beats a later scope.
func resolveGitPeerOwners(ctx context.Context, run Runner, dir string, targetPaths []string, selfSession string) (map[string]string, error) {
	// Cross-check the length-framed metadata against a compact ordered manifest.
	// This also detects successful output truncated at a whole-record boundary;
	// concurrent ref updates refuse rather than combining different snapshots.
	manifest, err := runPeerWIPLookup(ctx, run, dir, "for-each-ref", "--sort=refname", "--format=%(refname) %(objectname)", "refs/fak/wip")
	if err != nil {
		return nil, err
	}
	expected := strings.Split(strings.TrimSuffix(manifest, "\n"), "\n")
	if manifest == "" {
		expected = nil
	}
	out, err := runPeerWIPLookup(ctx, run, dir, "for-each-ref", "--sort=refname", peerWIPRefFormat, "refs/fak/wip")
	if err != nil {
		return nil, err
	}
	type checkpoint struct {
		peer, oid string
		scope     []string
	}
	var checkpoints []checkpoint
	var objects []string
	seen := make(map[string]bool)
	previous := ""
	record := 0
	for out != "" {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var fields [4]string
		for i := range fields {
			var ok bool
			fields[i], out, ok = strings.Cut(out, "\x00")
			if !ok {
				return nil, fmt.Errorf("safecommit: incomplete peer WIP metadata")
			}
		}
		ref, oid, kind := fields[0], fields[1], fields[2]
		size, err := strconv.Atoi(fields[3])
		if err != nil || size < 0 || size > len(out) || !strings.HasPrefix(out[size:], "\x00\n") {
			return nil, fmt.Errorf("safecommit: invalid peer WIP message frame")
		}
		msg := out[:size]
		out = out[size+2:]
		if !strings.HasPrefix(ref, "refs/fak/wip/") || ref <= previous || !peerWIPObjectID(oid) {
			return nil, fmt.Errorf("safecommit: invalid peer WIP ref snapshot")
		}
		if record >= len(expected) || expected[record] != ref+" "+oid {
			return nil, fmt.Errorf("safecommit: incomplete or changed peer WIP snapshot")
		}
		record++
		previous = ref
		peer := strings.TrimPrefix(ref, "refs/fak/wip/")
		if peer == "" {
			return nil, fmt.Errorf("safecommit: empty peer WIP owner")
		}
		// Exclude the ref, not its object: peers can point at the same commit as self.
		if peer == selfSession {
			continue
		}
		if kind != "commit" {
			return nil, fmt.Errorf("safecommit: peer WIP %s is not a commit", ref)
		}
		stamp, _ := wipref.DecodeStamp(msg)
		checkpoints = append(checkpoints, checkpoint{peer, oid, stamp.Scope})
		if !seen[oid] {
			seen[oid] = true
			objects = append(objects, oid)
		}
	}
	if record != len(expected) {
		return nil, fmt.Errorf("safecommit: incomplete peer WIP snapshot")
	}
	deltas := make(map[string]map[string]bool)
	if len(objects) == 0 {
		return make(map[string]string), ctx.Err()
	}
	// A root commit is a no-delta terminator under log.showRoot=false. Appending
	// it to every batch proves that even the final delta was received in full.
	rootOut, err := runPeerWIPLookup(ctx, run, dir, "rev-list", "--max-parents=0", "--max-count=1", objects[0], "--")
	if err != nil {
		return nil, err
	}
	root := strings.TrimSpace(rootOut)
	if !peerWIPObjectID(root) {
		return nil, fmt.Errorf("safecommit: invalid peer WIP batch terminator")
	}
	deltas[root] = make(map[string]bool)
	filtered := objects[:0]
	for _, oid := range objects {
		if oid != root {
			filtered = append(filtered, oid)
		}
	}
	objects = filtered
	for start := 0; start < len(objects); start += peerWIPBatchSize {
		end := min(start+peerWIPBatchSize, len(objects))
		batch := objects[start:end]
		args := []string{"-c", "log.showRoot=false", "log", "--no-walk=unsorted", "--format=%H", "--raw", "-z", "--no-abbrev", "--no-renames", "--no-ext-diff", "--no-textconv", "--diff-merges=off", "-r"}
		args = append(args, batch...)
		args = append(args, root)
		raw, err := runPeerWIPLookup(ctx, run, dir, args...)
		if err != nil {
			return nil, err
		}
		want := make(map[string]bool, len(batch))
		for _, oid := range batch {
			want[oid] = true
		}
		current := ""
		terminated := false
		for raw != "" {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			token, rest, ok := strings.Cut(raw, "\x00")
			if !ok {
				return nil, fmt.Errorf("safecommit: truncated peer WIP delta")
			}
			raw = rest
			token = strings.TrimPrefix(token, "\n")
			if token == root {
				if raw != "" {
					return nil, fmt.Errorf("safecommit: invalid peer WIP batch terminator")
				}
				terminated = true
				break
			}
			if want[token] {
				if deltas[token] != nil {
					return nil, fmt.Errorf("safecommit: duplicate peer WIP delta")
				}
				current = token
				deltas[current] = make(map[string]bool)
				continue
			}
			// With rename detection disabled, every raw record has exactly one path.
			fields := strings.Fields(token)
			if current == "" || len(fields) != 5 || len(fields[0]) != 7 || fields[0][0] != ':' || len(fields[1]) != 6 || !peerWIPObjectID(fields[2]) || !peerWIPObjectID(fields[3]) || len(fields[4]) != 1 || !strings.Contains("ACDMTUXB", fields[4]) {
				return nil, fmt.Errorf("safecommit: invalid peer WIP delta record")
			}
			path, rest, ok := strings.Cut(raw, "\x00")
			if !ok || path == "" {
				return nil, fmt.Errorf("safecommit: truncated peer WIP delta path")
			}
			raw = rest
			deltas[current][path] = true
		}
		if !terminated {
			return nil, fmt.Errorf("safecommit: truncated peer WIP batch")
		}
		for _, oid := range batch {
			if deltas[oid] == nil {
				return nil, fmt.Errorf("safecommit: missing peer WIP delta %s", oid)
			}
		}
	}
	owners := make(map[string]string)
	for _, checkpoint := range checkpoints {
		for _, path := range targetPaths {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if owners[path] == "" && (gitgate.CoveredByAnyTree(path, checkpoint.scope) || deltas[checkpoint.oid][path]) {
				owners[path] = checkpoint.peer
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return owners, nil
}

func peerWIPObjectID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
