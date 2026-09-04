// Package timeaware provides deterministic accounting and health signals for agent work.
// It records monotonic, half-open spans; attributes them to execution dimensions; and
// deliberately keeps elapsed time, aggregate effort, active-union time, and critical
// path separate.
package timeaware

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	// SpanSchema identifies the versioned time span schema.
	SpanSchema = "fak-time-span/1"
	// MetadataSchema identifies the versioned metadata schema.
	MetadataSchema = "fak-time-metadata/1"
	// SnapshotSchema identifies the versioned time snapshot schema.
	SnapshotSchema = "fak-time-snapshot/1"
)

// Phase classifies an execution interval into active, waiting, or recovery states.
type Phase string

const (
	// PhaseActive represents productive execution on the main work path.
	PhaseActive Phase = "active"
	// PhaseQueue represents time spent waiting in a queue before admission.
	PhaseQueue Phase = "queue"
	// PhaseWait represents external blocking waits (e.g. downstream responses).
	PhaseWait Phase = "wait"
	// PhaseStall represents churn or lack of forward progress during active execution.
	PhaseStall Phase = "stall"
	// PhaseIdle represents deliberate inactivity or pause.
	PhaseIdle Phase = "idle"
	// PhaseUnknown represents unclassified or missing phase telemetry.
	PhaseUnknown Phase = "unknown"
	// PhaseSpeculative represents speculative work that may be discarded.
	PhaseSpeculative Phase = "speculative"
	// PhasePostCancel represents work performed after a cancellation signal.
	PhasePostCancel Phase = "post-cancel"
)

// Dimensions records identity and provenance attributes for a span.
type Dimensions struct {
	SessionID       string `json:"session_id,omitempty"`
	RunID           string `json:"run_id,omitempty"`
	GoalID          string `json:"goal_id,omitempty"`
	IntentID        string `json:"intent_id,omitempty"`
	LineItemID      string `json:"line_item_id,omitempty"`
	AgentID         string `json:"agent_id,omitempty"`
	SubagentID      string `json:"subagent_id,omitempty"`
	Skill           string `json:"skill,omitempty"`
	Plugin          string `json:"plugin,omitempty"`
	Hook            string `json:"hook,omitempty"`
	Tool            string `json:"tool,omitempty"`
	OperationFamily string `json:"operation_family,omitempty"`
	Retry           int    `json:"retry,omitempty"`
	Poll            int    `json:"poll,omitempty"`
}

// Span represents a monotonic, half-open execution interval [StartNS, EndNS).
type Span struct {
	Schema     string     `json:"schema"`
	ID         string     `json:"id"`
	ParentID   string     `json:"parent_id,omitempty"`
	StartNS    int64      `json:"start_ns"`
	EndNS      int64      `json:"end_ns"`
	Phase      Phase      `json:"phase"`
	Dimensions Dimensions `json:"dimensions"`
}

// DurationNS returns the duration of the span in nanoseconds.
func (s Span) DurationNS() int64 { return s.EndNS - s.StartNS }

// Validate checks whether the span satisfies schema and monotonic bounds.
func (s Span) Validate() error {
	if s.Schema != SpanSchema {
		return fmt.Errorf("schema %q: want %q", s.Schema, SpanSchema)
	}
	if s.ID == "" {
		return errors.New("span id is required")
	}
	if s.StartNS < 0 || s.EndNS < 0 {
		return errors.New("monotonic timestamps must be non-negative")
	}
	if s.EndNS < s.StartNS {
		return errors.New("half-open span end precedes start")
	}
	switch s.Phase {
	case PhaseActive, PhaseQueue, PhaseWait, PhaseStall, PhaseIdle, PhaseUnknown, PhaseSpeculative, PhasePostCancel:
	default:
		return fmt.Errorf("unknown phase %q", s.Phase)
	}
	return nil
}

// Metadata holds environment and version attributes for cohort comparison.
type Metadata struct {
	Schema        string `json:"schema"`
	BuildVersion  string `json:"build_version,omitempty"`
	ModuleVersion string `json:"module_version,omitempty"`
	SchemaVersion string `json:"schema_version,omitempty"`
	ConfigDigest  string `json:"config_digest,omitempty"`
	PolicyDigest  string `json:"policy_digest,omitempty"`
	Platform      string `json:"platform,omitempty"`
	Architecture  string `json:"architecture,omitempty"`
	Model         string `json:"model,omitempty"`
	Engine        string `json:"engine,omitempty"`
	Component     string `json:"component,omitempty"`
	Runtime       string `json:"runtime,omitempty"`
}

// Validate checks whether metadata satisfies required schema.
func (m Metadata) Validate() error {
	if m.Schema != MetadataSchema {
		return fmt.Errorf("schema %q: want %q", m.Schema, MetadataSchema)
	}
	return nil
}

// CohortKey returns a stable, deliberately strict comparability identity.
func (m Metadata) CohortKey() string {
	return fmt.Sprintf("build=%s|module=%s|schema=%s|config=%s|policy=%s|platform=%s/%s|model=%s|engine=%s|component=%s|runtime=%s",
		m.BuildVersion, m.ModuleVersion, m.SchemaVersion, m.ConfigDigest, m.PolicyDigest,
		m.Platform, m.Architecture, m.Model, m.Engine, m.Component, m.Runtime)
}

// EdgeKind denotes the semantic relationship represented by an edge.
type EdgeKind string

const (
	// EdgeContains represents hierarchical containment.
	EdgeContains EdgeKind = "contains"
	// EdgeDependsOn represents execution ordering dependencies.
	EdgeDependsOn EdgeKind = "depends-on"
	// EdgeRetryOf represents a retry relationship to an earlier attempt.
	EdgeRetryOf EdgeKind = "retry-of"
	// EdgePollOf represents a polling check associated with an operation.
	EdgePollOf EdgeKind = "poll-of"
)

// Edge represents an attribution relationship between two spans or entities.
type Edge struct {
	From string   `json:"from"`
	To   string   `json:"to"`
	Kind EdgeKind `json:"kind"`
}

// Measures records decomposed duration metrics across all execution phases.
type Measures struct {
	WallNS         int64 `json:"wall_ns"`
	EffortNS       int64 `json:"effort_ns"`
	UnionActiveNS  int64 `json:"union_active_ns"`
	CriticalPathNS int64 `json:"critical_path_ns"`
	QueueNS        int64 `json:"queue_ns"`
	WaitNS         int64 `json:"wait_ns"`
	StallNS        int64 `json:"stall_ns"`
	IdleNS         int64 `json:"idle_ns"`
	UnknownNS      int64 `json:"unknown_ns"`
	SpeculativeNS  int64 `json:"speculative_ns"`
	PostCancelNS   int64 `json:"post_cancel_ns"`
}

// Rollup contains aggregated duration measures and counts.
type Rollup struct {
	Measures           Measures `json:"measures"`
	SpanCount          int      `json:"span_count"`
	InvalidSpanCount   int      `json:"invalid_span_count"`
	DuplicateSpanCount int      `json:"duplicate_span_count"`
	RetryCount         int      `json:"retry_count"`
	PollCount          int      `json:"poll_count"`
}

type interval struct{ start, end int64 }

// Invariant: time-aware span aggregation is fail-closed and deterministic.
// Invalid, unparseable, or duplicate spans are excluded from duration rollups and
// tracked in error counters. Elapsed time, active effort, and waiting durations remain
// strictly segregated.
//
// Aggregate calculates order-independent rollups from spans and attribution edges.
// Invalid and duplicate spans are counted and excluded. Effort is the sum of active,
// speculative, and post-cancel worker spans; waiting classes remain separate and are
// never presented as elapsed or worker effort.
func Aggregate(spans []Span, edges []Edge) Rollup {
	var r Rollup
	byID := make(map[string]Span, len(spans))
	valid := make([]Span, 0, len(spans))
	active := make([]interval, 0, len(spans))
	first, last, have := int64(0), int64(0), false
	for _, s := range spans {
		if s.Validate() != nil {
			r.InvalidSpanCount++
			continue
		}
		if _, ok := byID[s.ID]; ok {
			r.DuplicateSpanCount++
			continue
		}
		byID[s.ID] = s
		valid = append(valid, s)
		d := s.DurationNS()
		if !have || s.StartNS < first {
			first = s.StartNS
		}
		if !have || s.EndNS > last {
			last = s.EndNS
		}
		have = true
		if s.Dimensions.Retry > 0 {
			r.RetryCount++
		}
		if s.Dimensions.Poll > 0 {
			r.PollCount++
		}
		switch s.Phase {
		case PhaseActive:
			r.Measures.EffortNS += d
			active = append(active, interval{s.StartNS, s.EndNS})
		case PhaseQueue:
			r.Measures.QueueNS += d
		case PhaseWait:
			r.Measures.WaitNS += d
		case PhaseStall:
			r.Measures.StallNS += d
		case PhaseIdle:
			r.Measures.IdleNS += d
		case PhaseUnknown:
			r.Measures.UnknownNS += d
		case PhaseSpeculative:
			r.Measures.SpeculativeNS += d
			r.Measures.EffortNS += d
		case PhasePostCancel:
			r.Measures.PostCancelNS += d
			r.Measures.EffortNS += d
		}
	}
	r.SpanCount = len(valid)
	if have {
		r.Measures.WallNS = last - first
	}
	r.Measures.UnionActiveNS = unionDuration(active)
	r.Measures.CriticalPathNS = criticalPath(byID, edges)
	return r
}

func unionDuration(xs []interval) int64 {
	if len(xs) == 0 {
		return 0
	}
	sort.Slice(xs, func(i, j int) bool {
		if xs[i].start == xs[j].start {
			return xs[i].end < xs[j].end
		}
		return xs[i].start < xs[j].start
	})
	cur, total := xs[0], int64(0)
	for _, x := range xs[1:] {
		if x.start <= cur.end {
			if x.end > cur.end {
				cur.end = x.end
			}
			continue
		}
		total += cur.end - cur.start
		cur = x
	}
	return total + cur.end - cur.start
}

// criticalPath computes the longest duration path over depends-on edges. With no
// dependency relation, independent spans are parallel candidates and the longest wins.
func criticalPath(spans map[string]Span, edges []Edge) int64 {
	deps := make(map[string][]string)
	for _, e := range edges {
		if e.Kind == EdgeDependsOn {
			if _, a := spans[e.From]; a {
				if _, b := spans[e.To]; b {
					deps[e.From] = append(deps[e.From], e.To)
				}
			}
		}
	}
	memo := map[string]int64{}
	visiting := map[string]bool{}
	var visit func(string) int64
	visit = func(id string) int64 {
		if v, ok := memo[id]; ok {
			return v
		}
		if visiting[id] {
			return 0
		}
		visiting[id] = true
		best := int64(0)
		for _, d := range deps[id] {
			if v := visit(d); v > best {
				best = v
			}
		}
		visiting[id] = false
		v := spans[id].DurationNS() + best
		memo[id] = v
		return v
	}
	best := int64(0)
	for id, s := range spans {
		if s.Phase != PhaseActive && s.Phase != PhaseSpeculative && s.Phase != PhasePostCancel {
			continue
		}
		if v := visit(id); v > best {
			best = v
		}
	}
	return best
}

// Provenance states who can vouch for an activity or time value. Rendered
// commentary never upgrades the provenance supplied by the producer.
type Provenance string

const (
	// ProvenanceFact represents witnessed, direct telemetry or records.
	ProvenanceFact Provenance = "FACT"
	// ProvenanceInference represents logically deduced values based on facts.
	ProvenanceInference Provenance = "INFERENCE"
	// ProvenanceForecast represents projected future values or estimates.
	ProvenanceForecast Provenance = "FORECAST"
	// ProvenanceClaim represents self-reported or unverified statements.
	ProvenanceClaim Provenance = "CLAIM"
)

// ActivityProvenance is an alias for Provenance.
type ActivityProvenance = Provenance

const (
	// ActivityFact is an alias for ProvenanceFact.
	ActivityFact = ProvenanceFact
	// ActivityInference is an alias for ProvenanceInference.
	ActivityInference = ProvenanceInference
	// ActivityForecast is an alias for ProvenanceForecast.
	ActivityForecast = ProvenanceForecast
	// ActivityClaim is an alias for ProvenanceClaim.
	ActivityClaim = ProvenanceClaim
)

// State classifies an agent's current lifecycle activity.
type State string

const (
	// StateWorking indicates active execution toward a task.
	StateWorking State = "working"
	// StateWaiting indicates waiting on an external event, resource, or human.
	StateWaiting State = "waiting"
	// StateRecovering indicates handling errors or attempting recovery.
	StateRecovering State = "recovering"
	// StateBlocked indicates progress is halted by an unresolved blocker.
	StateBlocked State = "blocked"
	// StateDone indicates task completion.
	StateDone State = "done"
	// StateUncertain indicates activity state cannot be confidently determined.
	StateUncertain State = "uncertain"
)

// ActivityState is an alias for State.
type ActivityState = State

const (
	// ActivityWorking is an alias for StateWorking.
	ActivityWorking = StateWorking
	// ActivityWaiting is an alias for StateWaiting.
	ActivityWaiting = StateWaiting
	// ActivityRecovering is an alias for StateRecovering.
	ActivityRecovering = StateRecovering
	// ActivityBlocked is an alias for StateBlocked.
	ActivityBlocked = StateBlocked
	// ActivityDone is an alias for StateDone.
	ActivityDone = StateDone
	// ActivityUncertain is an alias for StateUncertain.
	ActivityUncertain = StateUncertain
)

// Motion describes whether forward progress is occurring.
type Motion string

const (
	// MotionAdvancing indicates measurable forward progress.
	MotionAdvancing Motion = "advancing"
	// MotionFlat indicates execution is ongoing with no forward progress.
	MotionFlat Motion = "flat"
	// MotionRegressing indicates state or quality is regressing.
	MotionRegressing Motion = "regressing"
	// MotionOscillating indicates churning between repeating states.
	MotionOscillating Motion = "oscillating"
	// MotionUnknown indicates motion trend cannot be determined.
	MotionUnknown Motion = "unknown"
)

// ActivityMotion is an alias for Motion.
type ActivityMotion = Motion

const (
	// ActivityAdvancing is an alias for MotionAdvancing.
	ActivityAdvancing = MotionAdvancing
	// ActivityFlat is an alias for MotionFlat.
	ActivityFlat = MotionFlat
	// ActivityRegressing is an alias for MotionRegressing.
	ActivityRegressing = MotionRegressing
	// ActivityOscillating is an alias for MotionOscillating.
	ActivityOscillating = MotionOscillating
	// ActivityMotionUnknown is an alias for MotionUnknown.
	ActivityMotionUnknown = MotionUnknown
)

// DenominatorClass identifies what a scope denominator counts. Keeping
// this explicit prevents an attempt count from being presented as work scope.
type DenominatorClass string

const (
	// DenominatorBudget counts allocated budget units.
	DenominatorBudget DenominatorClass = "budget"
	// DenominatorDeclaredWork counts upfront declared units of work.
	DenominatorDeclaredWork DenominatorClass = "declared_work"
	// DenominatorDiscoveredWork counts dynamically discovered units of work.
	DenominatorDiscoveredWork DenominatorClass = "discovered_work"
	// DenominatorAttempt counts trial or invocation attempts.
	DenominatorAttempt DenominatorClass = "attempt"
	// DenominatorOpportunity counts eligible candidates or targets.
	DenominatorOpportunity DenominatorClass = "opportunity"
	// DenominatorCriticalPath counts critical path units.
	DenominatorCriticalPath DenominatorClass = "critical_path"
	// DenominatorEvidence counts verification or evidence artifacts.
	DenominatorEvidence DenominatorClass = "evidence"
)

// ActivityDenominatorClass is an alias for DenominatorClass.
type ActivityDenominatorClass = DenominatorClass

const (
	// ActivityBudget is an alias for DenominatorBudget.
	ActivityBudget = DenominatorBudget
	// ActivityDeclaredWork is an alias for DenominatorDeclaredWork.
	ActivityDeclaredWork = DenominatorDeclaredWork
	// ActivityDiscoveredWork is an alias for DenominatorDiscoveredWork.
	ActivityDiscoveredWork = DenominatorDiscoveredWork
	// ActivityAttempt is an alias for DenominatorAttempt.
	ActivityAttempt = DenominatorAttempt
	// ActivityOpportunity is an alias for DenominatorOpportunity.
	ActivityOpportunity = DenominatorOpportunity
	// ActivityCriticalPath is an alias for DenominatorCriticalPath.
	ActivityCriticalPath = DenominatorCriticalPath
	// ActivityEvidence is an alias for DenominatorEvidence.
	ActivityEvidence = DenominatorEvidence
)

// Count distinguishes a witnessed zero from unavailable telemetry.
type Count struct {
	Value     int  `json:"value"`
	Available bool `json:"available"`
}

// ActivityCount is an alias for Count.
type ActivityCount = Count

// KnownCount creates a Count with Available set to true.
func KnownCount(value int) Count {
	return Count{Value: value, Available: true}
}

// KnownActivityCount is an alias for KnownCount.
func KnownActivityCount(value int) ActivityCount {
	return KnownCount(value)
}

// UnavailableCount creates an unavailable Count.
func UnavailableCount() Count { return Count{} }

// UnavailableActivityCount is an alias for UnavailableCount.
func UnavailableActivityCount() ActivityCount { return UnavailableCount() }

// Render formats the count as a number or "?" if unavailable.
func (c Count) Render() string {
	if !c.Available {
		return "?"
	}
	return fmt.Sprintf("%d", c.Value)
}

// Scope is revisioned because discovery can honestly change the
// denominator without implying that execution moved backwards.
type Scope struct {
	Completed        int              `json:"completed"`
	Total            Count            `json:"total"`
	DenominatorClass DenominatorClass `json:"denominator_class"`
	Revision         int              `json:"revision"`
}

// ActivityScope is an alias for Scope.
type ActivityScope = Scope

// Render returns the formatted scope string.
func (s Scope) Render() string {
	completed := s.Completed
	if completed < 0 {
		completed = 0
	}
	class := s.DenominatorClass
	if class == "" {
		class = DenominatorDeclaredWork
	}
	revision := s.Revision
	if revision < 0 {
		revision = 0
	}
	return fmt.Sprintf("%d/%s %s @r%d", completed, s.Total.Render(), class, revision)
}

// Summary pairs short commentary text with explicit provenance.
type Summary struct {
	Text       string     `json:"text"`
	Provenance Provenance `json:"provenance"`
}

// ActivitySummary is an alias for Summary.
type ActivitySummary = Summary

// Render returns the summary text with provenance tag.
func (s Summary) Render() string {
	text := strings.TrimSpace(s.Text)
	if text == "" {
		text = "?"
	}
	provenance := s.Provenance
	if provenance == "" {
		provenance = ProvenanceClaim
	}
	return fmt.Sprintf("%s [%s]", text, provenance)
}

// Transition records a semantic state change, not an individual tool
// invocation. MaterialEvidence should contain only the latest evidence that
// changed what an observer knows about the episode.
type Transition struct {
	Operation        string     `json:"operation"`
	MaterialEvidence string     `json:"material_evidence"`
	Provenance       Provenance `json:"provenance"`
}

// ActivityTransition is an alias for Transition.
type ActivityTransition = Transition

// Episode groups calls that pursue one intent. Calls remain available
// in lower-level provenance logs; this model deliberately does not count them.
type Episode struct {
	Intent     string     `json:"intent"`
	Transition Transition `json:"transition"`
	Age        string     `json:"age"`
}

// ActivityEpisode is an alias for Episode.
type ActivityEpisode = Episode

// ActivitySnapshot is the deterministic input to the native info surface.
// Lifecycle values are independently available: queue telemetry may be absent
// while an in-flight gauge is known (including a known zero).
type ActivitySnapshot struct {
	State    State     `json:"state"`
	Motion   Motion    `json:"motion"`
	Scope    Scope     `json:"scope"`
	Queued   Count     `json:"queued"`
	InFlight Count     `json:"in_flight"`
	Current  Summary   `json:"current"`
	Next     Summary   `json:"next"`
	Episodes []Episode `json:"episodes,omitempty"`
}

// Snapshot is an alias for ActivitySnapshot.
type Snapshot = ActivitySnapshot

// FormatActivitySnapshot returns a single, deterministic line no wider than
// width runes. Width <= 0 means unbounded. Truncation uses one ellipsis rune.
func FormatActivitySnapshot(snapshot ActivitySnapshot, width int) string {
	state := snapshot.State
	if state == "" {
		state = StateUncertain
	}
	motion := snapshot.Motion
	if motion == "" {
		motion = MotionUnknown
	}
	parts := []string{
		string(state),
		string(motion),
		"scope " + snapshot.Scope.Render(),
		"queued " + snapshot.Queued.Render(),
		"in-flight " + snapshot.InFlight.Render(),
		"current " + snapshot.Current.Render(),
		"next " + snapshot.Next.Render(),
	}
	return BoundString(strings.Join(parts, " · "), width)
}

// Render returns a formatted single-line activity string bounded to width runes.
func (s ActivitySnapshot) Render(width int) string {
	return FormatActivitySnapshot(s, width)
}

// Format returns a formatted single-line activity string bounded to width runes.
func (s ActivitySnapshot) Format(width int) string {
	return FormatActivitySnapshot(s, width)
}

// BoundString truncates text to width runes using a trailing ellipsis rune.
// Width <= 0 means unbounded.
func BoundString(text string, width int) string {
	if width <= 0 || utf8.RuneCountInString(text) <= width {
		return text
	}
	if width == 1 {
		return "…"
	}
	runes := []rune(text)
	return string(runes[:width-1]) + "…"
}
