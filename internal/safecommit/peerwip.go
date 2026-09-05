package safecommit

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// ReasonPeerWIPCollision reports that a path-scoped git op (especially a directory pathspec)
// would sweep WIP owned by a peer session or untracked conflicting peer work (#11232).
const ReasonPeerWIPCollision = "PEER_WIP_COLLISION"

const peerWIPGuardEnvVar = "FAK_PEER_WIP_GUARD"

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
	norm, ok := normalizePaths(requestedPaths)
	if !ok || len(norm) == 0 {
		return PathAttributionResult{OK: false, Reason: ReasonNoPath, Detail: "no valid repo-relative pathspec given"}, nil
	}
	statusArgs := append([]string{"status", "--porcelain", "--"}, norm...)
	statusOut, code, err := run(ctx, dir, statusArgs...)
	if err != nil {
		return PathAttributionResult{}, fmt.Errorf("safecommit: git not executable: %w", err)
	}
	if code != 0 {
		return PathAttributionResult{OK: true, EffectivePaths: norm}, nil
	}
	return checkPathAttributionFromStatus(ctx, run, dir, norm, statusOut, opts)
}

// checkPathAttributionFromStatus checks whether dirty/staged/untracked paths under requested
// directory pathspecs collide with peer WIP or untracked peer work.
func checkPathAttributionFromStatus(ctx context.Context, run Runner, dir string, requestedPaths []string, statusOut string, opts PathAttributionOptions) (PathAttributionResult, error) {
	changedPaths := statusChangedPaths(statusOut)
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

	var collidingPaths []string
	var peerSessions []string
	seenPeers := make(map[string]bool)
	var descriptions []string

	for _, cp := range changedPaths {
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
			peerOwner = resolveGitPeerOwner(ctx, run, dir, cp, sessionID)
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
			if !collidingSet[cp] && gitgate.CoveredByAnyTree(cp, opts.SessionScope) {
				kept = append(kept, cp)
			}
		}
		if len(kept) > 0 {
			sort.Strings(kept)
			kept = dedupeStrings(kept)
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

// resolveGitPeerOwner queries refs/fak/wip/* to determine if targetPath is owned/checkpointed by a peer session.
func resolveGitPeerOwner(ctx context.Context, run Runner, dir, targetPath, selfSession string) string {
	out, code, err := run(ctx, dir, "for-each-ref", "--format=%(refname)", "refs/fak/wip")
	if err != nil || code != 0 || strings.TrimSpace(out) == "" {
		return ""
	}
	for _, ref := range strings.Split(out, "\n") {
		ref = strings.TrimSpace(ref)
		if ref == "" || !strings.HasPrefix(ref, "refs/fak/wip/") {
			continue
		}
		peer := strings.TrimPrefix(ref, "refs/fak/wip/")
		if peer == selfSession || peer == "" {
			continue
		}
		// 1. Check commit message for Stamp
		msg, logCode, _ := run(ctx, dir, "log", "-1", "--format=%B", ref)
		if logCode == 0 {
			if stamp, ok := wipref.DecodeStamp(msg); ok && len(stamp.Scope) > 0 {
				if gitgate.CoveredByAnyTree(targetPath, stamp.Scope) {
					return peer
				}
			}
		}
		// 2. Check diff-tree for files touched in this checkpoint
		diffOut, diffCode, _ := run(ctx, dir, "diff-tree", "--no-commit-id", "--name-only", "-r", ref)
		if diffCode == 0 {
			for _, line := range strings.Split(diffOut, "\n") {
				if strings.TrimSpace(line) == targetPath {
					return peer
				}
			}
		}
	}
	return ""
}
