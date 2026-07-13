package toolprocgate

import (
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// The subagent FS lane floor (#2891): the wire that decides whether a child's
// tool call touching a sibling lane's files is refused. Hermes isolates
// workers by pinning an env var in the spawned env — isolation by convention,
// which the child could read around. Here the scope is part of the spawn
// GRANT (CapabilityEnvelope.LaneTree, validated at admission), so the floor
// adjudicates what the broker granted, never what the child's environment
// claims, and every refusal is a structured leak-event row (action fs_denied)
// citing a closed reason token.
const (
	// ReasonSiblingLaneTouch — the touched path resolves outside the lane
	// tree the spawn grant carried: a sibling lane's files.
	ReasonSiblingLaneTouch = "SIBLING_LANE_TOUCH"
	// ReasonMissingLaneFloor — the grant carried no lane tree at all. The
	// floor fails closed: an absent scope (Hermes' missing env var) is a
	// refusal, not a wildcard.
	ReasonMissingLaneFloor = "MISSING_LANE_FLOOR"
	// ReasonLanePathEscape — the touched path is absolute or traverses out
	// of the workspace root, so no granted lane tree can contain it.
	ReasonLanePathEscape = "LANE_PATH_ESCAPE"
)

// FSTouch is one child tool call's filesystem intent, presented to the floor
// BEFORE the tool runs. Path is workspace-relative (the lane-tree coordinate
// system of dos.toml [lanes.trees]); an absolute path is an escape by
// definition here, not something the floor resolves.
type FSTouch struct {
	ToolCallID string
	TraceID    string
	Path       string
	AtMS       int64
}

// FSDecision is the floor's verdict on one touch. On refusal Reason carries a
// closed token from the vocabulary above and LeakEvent is the witnessed
// fs_denied row (byte-free: the path travels only as a digest). An allow is
// silent — no event, matching the output-admission precedent.
type FSDecision struct {
	Verdict   string
	Reason    string
	LeakEvent *LeakEvent
}

// Allowed reports the touch was admitted by the floor.
func (d FSDecision) Allowed() bool { return d.Verdict == SpawnVerdictAllow }

// LaneFloor is the inherited FS capability floor: the lane tree the spawn
// grant carried, plus the grant identity the refusal witness cites. Build it
// with SpawnGrant.LaneFloor — a floor never comes from the child's env.
type LaneFloor struct {
	Grant SpawnGrant
}

// LaneFloor returns the FS floor this grant carries.
func (g SpawnGrant) LaneFloor() LaneFloor { return LaneFloor{Grant: g} }

// AdmitTouch adjudicates one filesystem touch against the granted lane tree.
// Deny-by-structure, in order: a path that escapes the workspace can never be
// in-lane (LANE_PATH_ESCAPE); a grant with no lane tree admits nothing
// (MISSING_LANE_FLOOR); a path outside every granted pattern is a sibling
// lane's file (SIBLING_LANE_TOUCH).
func (f LaneFloor) AdmitTouch(t FSTouch) FSDecision {
	deny := func(reason string) FSDecision {
		ev := f.refusalEvent(t, reason)
		return FSDecision{Verdict: SpawnVerdictDeny, Reason: reason, LeakEvent: &ev}
	}
	rel, ok := laneRelPath(t.Path)
	if !ok {
		return deny(ReasonLanePathEscape)
	}
	if len(f.Grant.Envelope.LaneTree) == 0 {
		return deny(ReasonMissingLaneFloor)
	}
	for _, pattern := range f.Grant.Envelope.LaneTree {
		if laneTreeMatch(pattern, rel) {
			return FSDecision{Verdict: SpawnVerdictAllow}
		}
	}
	return deny(ReasonSiblingLaneTouch)
}

// AdmitFSTouch adjudicates one child touch against the grant's lane floor
// and, on refusal, appends the witnessed fs_denied row to the broker's
// leak-event stream — the same append-only stream the spawn verdicts ride,
// so one operator report covers both boundaries.
func (b *SpawnBroker) AdmitFSTouch(g SpawnGrant, t FSTouch) FSDecision {
	d := g.LaneFloor().AdmitTouch(t)
	if d.LeakEvent != nil && b != nil {
		b.mu.Lock()
		b.leaks = append(b.leaks, *d.LeakEvent)
		b.mu.Unlock()
	}
	return d
}

// refusalEvent builds the byte-free witness row for a refused touch. Grant
// identity comes from the grant (never the child's env); the touched path
// travels only as a digest, per the leak-event no-payload contract.
func (f LaneFloor) refusalEvent(t FSTouch, reason string) LeakEvent {
	g := f.Grant
	return LeakEvent{
		Schema:          LeakEventSchema,
		Action:          LeakFSDenied,
		AtMS:            positiveLeakAtMS(t.AtMS),
		AgentRunID:      safeLeakToken(g.AgentRunID, 256, "unknown"),
		ParentRunID:     safeLeakToken(g.ParentRunID, 256, "unknown"),
		ToolCallID:      safeLeakToken(strmatch.FirstTrimmed(t.ToolCallID, g.ToolCallID), 256, "unknown"),
		TraceID:         safeLeakToken(strmatch.FirstTrimmed(t.TraceID, t.ToolCallID, g.ToolCallID), 256, "unknown"),
		PolicyDigest:    safeLeakToken(g.PolicyDigest, 256, "unknown"),
		Backend:         safeLeakToken(g.Backend, 256, "unknown"),
		Reason:          safeLeakReason(reason, "FS_DENIED"),
		BoundedRef:      BoundedRef{Kind: "path", Digest: digest([]string{t.Path}), Len: int64(len(t.Path))},
		SourceChannel:   "fs",
		DescendantState: DescendantRunning,
	}
}

// normalizeLaneTree validates and canonicalizes the granted lane tree at
// spawn admission: slash paths relative to the workspace root, dos.toml
// [lanes.trees] syntax (e.g. "internal/toolprocgate/**"). A pattern that is
// empty, absolute, NUL-carrying, or traverses upward denies the spawn
// outright — a floor that cannot be trusted is never granted.
func normalizeLaneTree(in []string) ([]string, error) {
	if len(in) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(in))
	seen := map[string]bool{}
	for _, pattern := range in {
		pattern = strings.ReplaceAll(strings.TrimSpace(pattern), "\\", "/")
		if pattern == "" || strings.ContainsRune(pattern, '\x00') || !laneRelForm(pattern) {
			return nil, fmt.Errorf("INVALID_LANE_TREE")
		}
		if seen[pattern] {
			continue
		}
		seen[pattern] = true
		out = append(out, pattern)
	}
	sort.Strings(out)
	return out, nil
}

// laneRelForm reports s is in workspace-relative form: not absolute (POSIX or
// drive-letter) and with no upward-traversal segment.
func laneRelForm(s string) bool {
	if strings.HasPrefix(s, "/") || (len(s) >= 2 && s[1] == ':') {
		return false
	}
	for _, seg := range strings.Split(s, "/") {
		if seg == ".." {
			return false
		}
	}
	return true
}

// laneRelPath canonicalizes a touched path into the lane-tree coordinate
// system. It refuses (escape) anything empty, absolute, NUL-carrying, that
// carries ANY ".." segment — even one that would lexically clean back
// in-lane, since lexical cleaning does not match filesystem resolution under
// symlinks — or that names the workspace root itself.
func laneRelPath(p string) (string, bool) {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	if p == "" || strings.ContainsRune(p, '\x00') || !laneRelForm(p) {
		return "", false
	}
	if p = path.Clean(p); p == "." {
		return "", false
	}
	return p, true
}

// laneTreeMatch matches one canonical relative path against one granted lane
// pattern: "**" grants the whole workspace, "dir/**" grants the subtree (and
// the directory itself), anything else matches path.Match semantics.
func laneTreeMatch(pattern, rel string) bool {
	if pattern == "**" {
		return true
	}
	if dir, ok := strings.CutSuffix(pattern, "/**"); ok {
		return rel == dir || strings.HasPrefix(rel, dir+"/")
	}
	ok, err := path.Match(pattern, rel)
	return err == nil && ok
}
