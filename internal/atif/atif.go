package atif

import (
	"encoding/json"
	"io"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// SchemaVersion is fak's ATIF profile tag, stamped on every emitted bundle and
// trajectory. It is a fak profile of the Agent Trajectory Interchange Format: the
// step/subagent shape is faithful to the interchange format, the field set is fak's
// (redacted Turn metadata). Bump the minor only additively so an older reader keeps
// parsing.
const SchemaVersion = "atif/1"

// Bundle is the top-level ATIF artifact a `fak traj export --format atif` writes: the
// schema tag plus the root trajectories (each of which may carry nested subagent
// trajectories). A single-session export has one root trajectory; a corpus that mixes
// several traces has one per root trace.
type Bundle struct {
	SchemaVersion string       `json:"schema_version"`
	Trajectories  []Trajectory `json:"trajectories"`
}

// Trajectory is one agent's ordered run: identity, the step list, and any subagent
// trajectories spawned from within it. StepCount and CompactionEvents mirror the
// counters agent-lens's ATIF adapter tracks (_step_counter, compaction_events) so a
// consumer sees the same top-line shape.
type Trajectory struct {
	SchemaVersion    string            `json:"schema_version"`
	TrajectoryID     string            `json:"trajectory_id"` // the source trace id
	Agent            string            `json:"agent,omitempty"`
	CreatedUnixNano  int64             `json:"created_unix_nano,omitempty"` // first step's wall-clock anchor
	StepCount        int               `json:"step_count"`
	CompactionEvents int               `json:"compaction_events"`
	Metadata         map[string]string `json:"metadata,omitempty"` // trace-level labels (full-fidelity only)
	Steps            []Step            `json:"steps"`
	Subagents        []Trajectory      `json:"subagents,omitempty"`
}

// Step is one adjudicated agent action in ATIF shape. A fak Turn is a single
// tool call the kernel adjudicated, so it folds to one step carrying both the call
// identity (tool, args digest) and the result identity (result digest, taint) — the
// redacted metadata, never the bodies. ToolUseID is the stable per-step handle a
// subagent's ParentToolUseID points back at.
type Step struct {
	Index           int               `json:"index"` // 0-based, monotonic within the trajectory
	Seq             int               `json:"seq"`   // source Turn.Seq (1-based)
	Type            string            `json:"type"`  // tool_call | decision | cache_hit
	Role            string            `json:"role"`  // assistant | tool | system
	Tool            string            `json:"tool,omitempty"`
	ToolUseID       string            `json:"tool_use_id"`                      // stable id: "<trace>:<seq>"
	ParentToolUseID string            `json:"parent_tool_use_id,omitempty"`     // set on a spawned subagent's steps
	SubagentRef     string            `json:"subagent_trajectory_id,omitempty"` // set on the step that spawned a subagent
	Query           string            `json:"query,omitempty"`                  // full-fidelity only
	Verdict         string            `json:"verdict,omitempty"`
	Reason          string            `json:"reason,omitempty"`
	Taint           string            `json:"taint,omitempty"`
	Materialized    string            `json:"materialized,omitempty"`
	ArgsDigest      string            `json:"args_digest,omitempty"`
	ResultDigest    string            `json:"result_digest,omitempty"`
	Tokens          int               `json:"tokens,omitempty"`
	Bytes           int64             `json:"bytes,omitempty"`
	CacheHit        bool              `json:"cache_hit,omitempty"`
	TSUnixNano      int64             `json:"ts_unix_nano,omitempty"`
	Labels          map[string]string `json:"labels,omitempty"` // full-fidelity only
}

// Options controls the mapping. The zero value is the safe default: redacted
// (no query text, no labels) and the default parent-linkage label keys.
type Options struct {
	// FullFidelity includes the human query text and the producer label map. OFF (the
	// default) emits only structural identity — a conscious operator choice, since a
	// query can carry sensitive prose.
	FullFidelity bool
	// ParentLabelKeys are the Turn label keys consulted, in order, to find a trace's
	// parent for subagent nesting. Empty => DefaultParentLabelKeys.
	ParentLabelKeys []string
	// Agent labels the emitting agent (e.g. a model id) on each trajectory. Optional.
	Agent string
}

// DefaultParentLabelKeys are the label keys FromTurns consults to link a child trace
// to the parent step that spawned it — the fak analogue of agent-lens routing
// subagent events by parent_tool_use_id. The value is either a parent tool_use_id
// ("<trace>:<seq>") or a bare parent trace id; both are resolved.
var DefaultParentLabelKeys = []string{"parent_tool_use_id", "parent_trace_id", "parent_trace", "subagent_of"}

// compactionLabelKeys mark a turn as a compaction event (counted, not emitted as a
// distinct step type — the count is the top-line signal a consumer wants).
var compactionLabelKeys = []string{"compaction", "compacted", "compaction_event"}

// FromTurns folds a Turn corpus into an ATIF Bundle. Traces become trajectories;
// a trace whose turns carry a parent-linkage label nests as a subagent under the
// parent trajectory, and the parent step it was spawned from is back-referenced
// (SubagentRef). Ordering is deterministic: root trajectories in first-seen trace
// order, steps in seq order, subagents in first-seen order under their parent.
func FromTurns(turns []trajectory.Turn, opts Options) Bundle {
	keys := opts.ParentLabelKeys
	if len(keys) == 0 {
		keys = DefaultParentLabelKeys
	}

	// Group turns by trace, preserving first-seen trace order.
	byTrace := map[string][]trajectory.Turn{}
	var traceOrder []string
	for _, t := range turns {
		if _, seen := byTrace[t.TraceID]; !seen {
			traceOrder = append(traceOrder, t.TraceID)
		}
		byTrace[t.TraceID] = append(byTrace[t.TraceID], t)
	}

	type built struct {
		traj        Trajectory
		parentTrace string
		parentStep  string // parent tool_use_id, "" if only a trace was named
	}
	built0 := map[string]*built{}
	for _, id := range traceOrder {
		tr := byTrace[id]
		sort.SliceStable(tr, func(i, j int) bool { return tr[i].Seq < tr[j].Seq })
		traj := buildTrajectory(id, tr, opts)
		pTrace, pStep := parentOf(tr, keys)
		built0[id] = &built{traj: traj, parentTrace: pTrace, parentStep: pStep}
	}

	// Attach children to parents; collect roots. A parent that is not itself in the
	// corpus (dangling link) leaves the child as a root, so nothing is dropped.
	var roots []string
	for _, id := range traceOrder {
		b := built0[id]
		if b.parentTrace == "" {
			roots = append(roots, id)
			continue
		}
		p, ok := built0[b.parentTrace]
		if !ok {
			roots = append(roots, id)
			continue
		}
		p.traj.Subagents = append(p.traj.Subagents, b.traj)
		if b.parentStep != "" {
			// Back-reference the spawning step (attach_subagent_refs analogue).
			for i := range p.traj.Steps {
				if p.traj.Steps[i].ToolUseID == b.parentStep {
					p.traj.Steps[i].SubagentRef = id
					break
				}
			}
		}
	}

	out := Bundle{SchemaVersion: SchemaVersion}
	for _, id := range roots {
		out.Trajectories = append(out.Trajectories, built0[id].traj)
	}
	if out.Trajectories == nil {
		out.Trajectories = []Trajectory{}
	}
	return out
}

// buildTrajectory maps one trace's (seq-sorted) turns to a Trajectory.
func buildTrajectory(id string, turns []trajectory.Turn, opts Options) Trajectory {
	traj := Trajectory{
		SchemaVersion: SchemaVersion,
		TrajectoryID:  id,
		Agent:         opts.Agent,
		Steps:         make([]Step, 0, len(turns)),
	}
	for i, t := range turns {
		if i == 0 {
			traj.CreatedUnixNano = t.TSUnixNano
		}
		if isCompaction(t) {
			traj.CompactionEvents++
		}
		traj.Steps = append(traj.Steps, buildStep(i, t, opts))
	}
	traj.StepCount = len(traj.Steps)
	if opts.FullFidelity {
		if md := traceMetadata(turns); len(md) > 0 {
			traj.Metadata = md
		}
	}
	return traj
}

// buildStep maps one Turn to an ATIF step.
func buildStep(index int, t trajectory.Turn, opts Options) Step {
	s := Step{
		Index:        index,
		Seq:          t.Seq,
		Type:         stepType(t),
		Role:         "assistant",
		Tool:         t.Tool,
		ToolUseID:    toolUseID(t.TraceID, t.Seq),
		Verdict:      t.Verdict,
		Reason:       t.Reason,
		Taint:        t.Taint,
		Materialized: t.Materialized,
		ArgsDigest:   t.ArgsDigest,
		ResultDigest: t.ResultDigest,
		Tokens:       t.TokenEstimate,
		Bytes:        t.Bytes,
		CacheHit:     t.CacheHit,
		TSUnixNano:   t.TSUnixNano,
	}
	if opts.FullFidelity {
		s.Query = t.Query
		if len(t.Labels) > 0 {
			s.Labels = copyLabels(t.Labels)
		}
	}
	return s
}

// parentOf resolves a trace's parent from its turns' labels, consulting keys in
// order and returning the first hit as (parentTraceID, parentToolUseID). A value of
// the form "<trace>:<seq>" names a specific parent step (its tool_use_id); a bare id
// names only the parent trace.
func parentOf(turns []trajectory.Turn, keys []string) (parentTrace, parentStep string) {
	for _, t := range turns {
		for _, k := range keys {
			v, ok := t.Labels[k]
			if !ok || v == "" {
				continue
			}
			if tr, _, isStep := splitToolUseID(v); isStep {
				return tr, v
			}
			return v, ""
		}
	}
	return "", ""
}

// stepType classifies a Turn's ATIF step type from its shape.
func stepType(t trajectory.Turn) string {
	if t.CacheHit || t.Verdict == "VDSO_HIT" {
		return "cache_hit"
	}
	if t.Verdict == "DENY" || t.Verdict == "QUARANTINE" {
		return "decision"
	}
	return "tool_call"
}

func isCompaction(t trajectory.Turn) bool {
	for _, k := range compactionLabelKeys {
		if v, ok := t.Labels[k]; ok && v != "" && v != "false" && v != "0" {
			return true
		}
	}
	return false
}

// traceMetadata collects labels that are constant across a trace (session-level
// facts like model or account), excluding the per-step parent-linkage keys.
func traceMetadata(turns []trajectory.Turn) map[string]string {
	if len(turns) == 0 {
		return nil
	}
	common := map[string]string{}
	for k, v := range turns[0].Labels {
		common[k] = v
	}
	for _, t := range turns[1:] {
		for k, v := range common {
			if t.Labels[k] != v {
				delete(common, k)
			}
		}
	}
	for _, k := range DefaultParentLabelKeys {
		delete(common, k)
	}
	if len(common) == 0 {
		return nil
	}
	return common
}

func toolUseID(trace string, seq int) string { return trace + ":" + itoa(seq) }

// splitToolUseID parses "<trace>:<seq>" back into its parts; isStep is false when the
// value is not that shape (a bare trace id, no ":<int>" suffix).
func splitToolUseID(v string) (trace string, seq int, isStep bool) {
	i := lastColon(v)
	if i < 0 || i == len(v)-1 {
		return "", 0, false
	}
	n, ok := atoi(v[i+1:])
	if !ok {
		return "", 0, false
	}
	return v[:i], n, true
}

func copyLabels(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// WriteBundle marshals a Bundle as indented JSON (the trajectory.json artifact).
func WriteBundle(w io.Writer, b Bundle) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(b)
}

// ---------------------------------------------------------------------------
// small dependency-free helpers (mirrors internal/trajhook's local ints)
// ---------------------------------------------------------------------------

func lastColon(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == ':' {
			return i
		}
	}
	return -1
}

func atoi(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	n := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
