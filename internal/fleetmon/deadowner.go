package fleetmon

// deadowner.go — the OWNER-LIVENESS reaper dimension (#3596). The janitor
// (janitor.go) reaps stale CHILD commands under a LIVE worker root; the loop
// reaper (internal/looporphan, `fak loop reap`) KEEPS any supervisor that still
// parents live work; procguard reaps by CPU/thread level or idle-shell name+age.
// None of them reaps a whole fak-owned loop/worker TREE whose OWNER — its run-id
// lease / run-registry row — is DEAD while its children are BUSY. That is exactly
// the tree a stopped `fak c` / `dos loop` drainer leaves on Windows (TaskStop does
// not kill the subtree) or that a crashed owning session strands: a live subtree
// with no live owner, invisible to every existing heuristic because it is neither
// idle nor cheap nor reparented-to-init. Those trees are what accumulates at 100x.
//
// This evaluator adds the missing OWNERSHIP-liveness key. It tags each fak-owned
// tree root by the run-id/lane already present on its command line, consults an
// INJECTED owner-liveness verdict (the cmd shell resolves it from the lease store
// — internal/leaseref, where a Record within TTL is live and an expired/absent one
// is dead — and/or the run registry; this core does NO I/O so the whole
// classification is table-testable with no git and no live fleet), and flags a
// tree whose owner is provably DEAD or ABSENT as a reap candidate.
//
// It keeps the existing no-false-reap contract, root-first:
//   - a PROTECTED OS name (procguard.ProtectedNames) is reported, never a candidate;
//   - an ATTENDED-terminal parent (a LIVE interactive shell — a human may be
//     watching) is reported, never a candidate; a DETACHED tree whose launcher
//     already exited is NOT attended and stays reapable;
//   - a candidate whose subtree still holds a protected/persistent process (an MCP
//     server, a `fak guard`/serve) is demoted — a tree-kill would take it down too;
//   - an owner whose liveness cannot be proven (no run-id tag on the root, or a key
//     absent from the injected verdict map) fails CLOSED to spared.
//
// Like the janitor it is report-first: it never kills; the cmd shell tree-kills
// only the flagged candidates behind an explicit --enact, exactly as
// `fak fleet janitor --apply` does over JanitorResult.Stale.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// DeadOwnerSchema tags the dead-owner reaper's machine-readable payload so the
// control pane folds it alongside the janitor and loop-reap contracts.
const DeadOwnerSchema = "fak-fleet-deadowner/1"

// OwnerState is the liveness verdict for a fak-owned tree's owning run. The cmd
// shell resolves it (a leaseref.Record within TTL => Live; an expired record or a
// registry row absent-or-terminal => Dead/Absent); this pure evaluator only reads
// it. Only Dead and Absent make a tree reapable — Live and Unknown always spare.
type OwnerState string

const (
	OwnerLive    OwnerState = "live"    // lease present and within TTL / registry row active — SPARE
	OwnerDead    OwnerState = "dead"    // lease present but expired / registry row terminal — REAP candidate
	OwnerAbsent  OwnerState = "absent"  // owner looked up, NO lease/registry row (crashed owner) — REAP candidate
	OwnerUnknown OwnerState = "unknown" // owner key not resolvable / not in the verdict map — SPARE (fail closed)
)

// reapable reports whether an owner verdict makes its tree a reap candidate. Only
// an AFFIRMATIVE dead-or-absent verdict does; Unknown (the default for any root
// whose key the caller did not resolve) fails closed to spared.
func (s OwnerState) reapable() bool { return s == OwnerDead || s == OwnerAbsent }

// DefaultDeadOwnerRootMarkers are the cmdline substrings that identify a fak-owned
// loop/worker tree ROOT — the process whose owning run-lease this reaper keys on.
// It is the union of the loop-reap supervisor+worker markers and the detached
// `fak guard` session engine. A match here only makes a tree a CANDIDATE for the
// liveness check; the owner verdict, not the marker, decides reap vs spare.
var DefaultDeadOwnerRootMarkers = []string{
	"fak c ", "dos loop", "loop drive", "superloop drive", "fak guard", "claude -p",
}

// DefaultOwnerKeyFlags are the cmdline flags a fak-owned root carries its owning
// run identity in, tried in order — the key looked up in the injected verdict map.
// The run-id is preferred; lane/loop/goal/region are the coarser fallbacks the
// loop reaper already groups on (parseLoopLane).
var DefaultOwnerKeyFlags = []string{"--run-id", "--run", "--lane", "--loop", "--goal", "--region"}

// DeadOwnerInput is the injected snapshot the pure evaluator folds. Procs is a
// relations scan (PID/PPID/Name/Cmdline). OwnerStates maps an owner key to its
// resolved liveness; a root whose key is absent from the map classifies Unknown
// (fail closed). The marker / key-flag / protected-name / attended-parent sets
// default to the fak-owned vocabulary when nil.
type DeadOwnerInput struct {
	Procs           []procguard.Proc
	OwnerStates     map[string]OwnerState
	RootMarkers     []string
	OwnerKeyFlags   []string
	ProtectedNames  map[string]bool
	AttendedParents map[string]bool
	ProtectedCmd    []string // extra cmdline substrings marking a protected subtree process (default: MCP/guard)
}

// DeadOwnerTree is one fak-owned tree root the reaper classified.
type DeadOwnerTree struct {
	RootPID     int        `json:"root_pid"`
	PPID        int        `json:"ppid"`
	Name        string     `json:"name"`
	Command     string     `json:"command"`
	OwnerKey    string     `json:"owner_key,omitempty"`
	Owner       OwnerState `json:"owner_state"`
	TreePIDs    []int      `json:"tree_pids"` // root + every live descendant (the tree-kill set)
	Descendants int        `json:"descendants"`
	Candidate   bool       `json:"candidate"` // a dead-owner reap candidate
	Protected   bool       `json:"protected"` // OS-critical name / protected subtree — reported, never reaped
	Attended    bool       `json:"attended"`  // live interactive-terminal parent — reported, never reaped
	Reason      string     `json:"reason"`
}

// DeadOwnerResult is the machine-readable payload the control pane folds. OK is
// false exactly when a dead-owner tree is live (a candidate exists) — the
// report-first ACTION bit, matching the janitor/procguard contracts.
type DeadOwnerResult struct {
	Schema         string          `json:"schema"`
	OK             bool            `json:"ok"`
	Scanned        int             `json:"scanned"`
	Trees          []DeadOwnerTree `json:"trees"`
	Candidates     []DeadOwnerTree `json:"candidates"`
	CandidateCount int             `json:"candidate_count"`
	NextAction     string          `json:"next_action"`
}

// EvaluateDeadOwnerReaper classifies every fak-owned tree root against its owning
// run's liveness. Pure: no I/O, no kills — the cmd shell resolves OwnerStates and
// tree-kills the candidates behind --enact.
func EvaluateDeadOwnerReaper(in DeadOwnerInput) DeadOwnerResult {
	markers := in.RootMarkers
	if markers == nil {
		markers = DefaultDeadOwnerRootMarkers
	}
	keyFlags := in.OwnerKeyFlags
	if keyFlags == nil {
		keyFlags = DefaultOwnerKeyFlags
	}
	protectedNames := in.ProtectedNames
	if protectedNames == nil {
		protectedNames = procguard.ProtectedNames
	}
	attended := in.AttendedParents
	if attended == nil {
		attended = procguard.DefaultInteractiveParentNames
	}
	protectedCmd := in.ProtectedCmd
	if protectedCmd == nil {
		protectedCmd = defaultProtectedCmd
	}

	byPID := map[int]procguard.Proc{}
	children := map[int][]int{}
	for _, p := range in.Procs {
		if p.PID <= 0 {
			continue
		}
		byPID[p.PID] = p
		if p.PPID != nil {
			children[*p.PPID] = append(children[*p.PPID], p.PID)
		}
	}

	var trees []DeadOwnerTree
	for _, p := range in.Procs {
		if p.PID <= 0 || !matchesDeadOwnerMarker(p, markers) {
			continue
		}
		key := parseOwnerKey(p.Cmdline, keyFlags)
		state := OwnerUnknown
		if key != "" {
			if s, ok := in.OwnerStates[key]; ok {
				state = s
			}
		}
		ppid := derefInt(p.PPID)
		tree := subtree(p.PID, children)

		row := DeadOwnerTree{
			RootPID:     p.PID,
			PPID:        ppid,
			Name:        p.Name,
			Command:     trimCmd(strings.TrimSpace(p.Cmdline)),
			OwnerKey:    key,
			Owner:       state,
			TreePIDs:    tree,
			Descendants: len(tree) - 1,
		}

		// The protection ladder — the first rung that applies wins, and every rung
		// EXCEPT the final owner check can only SPARE a tree, never reap one.
		parentAlive, parentName := parentNameIfAlive(ppid, byPID)
		switch {
		case protectedNames[stemLower(p.Name)]:
			row.Protected = true
			row.Reason = fmt.Sprintf("protected OS name %q — reported, never reaped", p.Name)
		case parentAlive && attended[stemLower(parentName)]:
			row.Attended = true
			row.Reason = fmt.Sprintf("attended-terminal parent (%s, pid %d live) — reported, never reaped", stemLower(parentName), ppid)
		case state.reapable() && subtreeHoldsProtected(tree, p.PID, byPID, protectedNames, protectedCmd):
			row.Protected = true
			row.Reason = "subtree holds a protected/persistent process — a tree-kill would take it down; spared"
		case state.reapable():
			row.Candidate = true
			row.Reason = fmt.Sprintf("owner %q is %s: %d live descendant(s) with no live owner — busy orphan tree", key, state, row.Descendants)
		case state == OwnerLive:
			row.Reason = fmt.Sprintf("owner %q live — spared", key)
		default:
			row.Reason = "owner liveness unknown (no run-id tag or unresolved) — spared (fail closed)"
		}
		trees = append(trees, row)
	}

	sort.SliceStable(trees, func(i, j int) bool { return trees[i].RootPID < trees[j].RootPID })

	var candidates []DeadOwnerTree
	for _, t := range trees {
		if t.Candidate {
			candidates = append(candidates, t)
		}
	}
	return DeadOwnerResult{
		Schema:         DeadOwnerSchema,
		OK:             len(candidates) == 0,
		Scanned:        len(in.Procs),
		Trees:          trees,
		Candidates:     candidates,
		CandidateCount: len(candidates),
		NextAction:     deadOwnerNextAction(candidates),
	}
}

// matchesDeadOwnerMarker reports whether a process's name+cmdline contains any
// root marker (case-insensitive substring) — the same shape as the loop reaper's
// procMatchesLoopMarker.
func matchesDeadOwnerMarker(p procguard.Proc, markers []string) bool {
	hay := strings.ToLower(p.Name + " " + p.Cmdline)
	for _, m := range markers {
		if m = strings.ToLower(strings.TrimSpace(m)); m != "" && strings.Contains(hay, m) {
			return true
		}
	}
	return false
}

// parseOwnerKey extracts the owning run identity from a root's cmdline: the first
// of keyFlags that resolves (space- or =-separated). "" when none is present — the
// tree then classifies Unknown and is spared. The space form guards against a
// missing value ("--run-id --json" must not yield "--json").
func parseOwnerKey(cmdline string, keyFlags []string) string {
	fields := strings.Fields(cmdline)
	for i, f := range fields {
		for _, name := range keyFlags {
			if f == name && i+1 < len(fields) {
				if v := fields[i+1]; !strings.HasPrefix(v, "-") {
					return strings.TrimSpace(v)
				}
			}
			if strings.HasPrefix(f, name+"=") {
				return strings.TrimSpace(strings.TrimPrefix(f, name+"="))
			}
		}
	}
	return ""
}

// parentNameIfAlive returns (true, name) only when ppid points at a process still
// present in the snapshot — i.e. the parent is ALIVE. A parent absent from the
// snapshot (the detached-orphan case: the launcher already exited) returns
// (false, ""), so a dead launcher never counts as an attended terminal.
func parentNameIfAlive(ppid int, byPID map[int]procguard.Proc) (bool, string) {
	if ppid <= 0 {
		return false, ""
	}
	if p, ok := byPID[ppid]; ok {
		return true, p.Name
	}
	return false, ""
}

// subtreeHoldsProtected reports whether any DESCENDANT of root (excluding root
// itself) is a protected OS process or a persistent MCP/guard server. A tree-kill
// of the root would take such a process down, so its dead-owner root is demoted to
// spared — the same no-false-reap rule the janitor applies to a protected subtree.
func subtreeHoldsProtected(tree []int, root int, byPID map[int]procguard.Proc, protectedNames map[string]bool, protectedCmd []string) bool {
	for _, pid := range tree {
		if pid == root {
			continue
		}
		p := byPID[pid]
		if protectedNames[stemLower(p.Name)] || matchesAny(p.Cmdline, p.Name, protectedCmd) {
			return true
		}
	}
	return false
}

func deadOwnerNextAction(candidates []DeadOwnerTree) string {
	if len(candidates) == 0 {
		return "no dead-owner orphan tree; no action"
	}
	names := make([]string, 0, len(candidates))
	for _, c := range candidates {
		names = append(names, fmt.Sprintf("%s(pid %d, owner %s)", c.Name, c.RootPID, c.Owner))
	}
	return fmt.Sprintf("re-run with --enact to tree-kill %d dead-owner orphan tree(s): %s", len(candidates), strings.Join(names, ", "))
}
