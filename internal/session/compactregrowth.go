package session

// compactregrowth.go — decompose the post-compaction REGROWTH of resident context
// (#4768, extending the #4763 compaction-health miner in compactaudit.go).
//
// The #4763 audit proved the cut works at the event boundary (median ~242k -> ~25k).
// The open question this file answers is why ~175k comes BACK so quickly: the native
// Codex telemetry pass behind the issue found 699 of 1,044 observable fires returned
// to >=200k resident tokens before the rollout ended, median 1,322 s. A large one-time
// shed does not establish sustained value if the window refills in 22 minutes — but
// fast regrowth is only waste if it is not useful new work and not cheap cache reads.
//
// So each fire gets a REGROWTH TRAJECTORY (floor, velocity, threshold crossings,
// censoring) and a CONTENT-CLASS ATTRIBUTION of the transcript bytes appended between
// the fire and its rebound/censor point. Attribution is body-blind: rows are classified
// from the leading fields of their payload (type/role/tool name), measured by length,
// and deduplicated by a content hash computed from the streamed bytes — lengths and
// hashes are retained, bodies never are.
//
// Two honesty rules carry over from #4763 and are load-bearing here:
//
//   - Impossible wall clocks are typed, not asserted. The issue's own audit found two
//     "zero-second" rebounds spanning 132 token samples. Such windows are flagged
//     TIMESTAMP_SUSPECT and excluded from every wall-clock aggregate while still
//     counting as observed rebounds.
//   - Regrowth is priced net of reuse. Each window joins the provider's
//     cached_input_tokens telemetry, so a fast rebound served mostly from cache reads
//     is not reported as gross waste.
//
// Regrowth anomalies live on the trajectory (CompactFire.Regrowth), NOT on the fire or
// session anomaly lists, so #4763's verdict vocabulary and its corpus witness stay
// stable.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
	"time"
)

// Regrowth thresholds and classification bars. Named so an operator reading a flag
// knows the bar it tripped.
const (
	// RegrowthReboundTokens — the resident count that counts as "the window came back".
	// Matches the issue's 200k rebound bar.
	RegrowthReboundTokens = 200000
	// RegrowthFastReboundSeconds — the fast/slow cohort split (30 minutes; the issue's
	// audit saw 508 of 699 rebounds inside it).
	RegrowthFastReboundSeconds = 1800
	// RegrowthWithin15MinSeconds — the issue's tighter headline bucket.
	RegrowthWithin15MinSeconds = 900
	// RegrowthOversizedRowBytes — one transcript row at/above this is an oversized
	// single event (a paste, image, or giant tool result the shedder will fight).
	RegrowthOversizedRowBytes = 256 << 10
	// RegrowthDupMinBytes — a row must be at least this large for a repeat of it to
	// count as duplication; identical tiny rows ("ok") are noise, not reinjection.
	RegrowthDupMinBytes = 2048
	// RegrowthDupToolMinRows — repeated tool output needs at least this many duplicate
	// result rows in one window; a single re-read is routine.
	RegrowthDupToolMinRows = 2
	// RegrowthSuffixWindowSamples / RegrowthSuffixBurstTokens — regaining this many
	// tokens within the first few post-fire samples is the #3071 suffix-recreation
	// signature: the window jumps right back before any real work happened.
	RegrowthSuffixWindowSamples = 2
	RegrowthSuffixBurstTokens   = 40000
	// regrowthDupMapCap bounds the session-wide dedup table so a pathological rollout
	// cannot grow it without limit; overflow rows simply stop registering as new.
	regrowthDupMapCap = 1 << 17
)

// RegrowthThresholds are the resident-token milestones each trajectory times.
var RegrowthThresholds = []int{50000, 100000, 150000, RegrowthReboundTokens}

// Regrowth anomaly tokens — a closed vocabulary, same contract as the #4763 set.
const (
	// AnomalyDuplicateSetup counts ROLLOUT-level restatement, not resident waste. The
	// dedup table is session-scoped across fires, so a post-fire reinjection of setup
	// the cut already discarded reads as a duplicate even though it is the window's
	// only resident copy. The #5255 attribution refuted the dedupe reading: within a
	// post-fire window every instruction payload occurs exactly once, the fires that
	// append setup are precisely those whose replacement_history omits it, and the
	// emitter is the upstream CLI's post-compaction rebuild, not this repo — so
	// removing the reinjection would strip the agent's instructions, not save context.
	// Kept deliberately as an honest restatement counter; decision witness:
	// testdata/compactaudit/setup-reinjection-decision-2026-08-09.md.
	AnomalyDuplicateSetup     = "DUPLICATE_SETUP_REINJECTION"
	AnomalyRepeatedToolResult = "REPEATED_TOOL_RESULT"
	AnomalyOversizedEvent     = "OVERSIZED_EVENT"
	AnomalySuffixRecreation   = "SUFFIX_RECREATION"
	AnomalyTimestampSuspect   = "TIMESTAMP_SUSPECT"
)

// Censor tokens: why a trajectory ended without reaching the rebound bar. A censored
// window is an observation limit, not a verdict.
const (
	RegrowthCensorNextFire   = "NEXT_FIRE"
	RegrowthCensorRolloutEnd = "ROLLOUT_END"
)

// Content classes. Tool traffic is keyed per tool ("tool_call/shell",
// "tool_result/shell") so the attribution table can name the dominant tool.
const (
	RegrowClassInstructions   = "instructions"
	RegrowClassUserMessage    = "message/user"
	RegrowClassSystemMessage  = "message/system"
	RegrowClassDeveloperMsg   = "message/developer"
	RegrowClassAssistantMsg   = "message/assistant"
	RegrowClassReasoning      = "reasoning"
	RegrowClassCompactSummary = "compaction_summary"
	RegrowClassToolCallPrefix = "tool_call/"
	RegrowClassToolResPrefix  = "tool_result/"
	RegrowClassUnknown        = "unknown"
)

// regrowthInstructionMarkers mark a message as reinjected instruction/skill payload
// rather than conversation. They appear in the leading bytes of the content.
var regrowthInstructionMarkers = []string{
	"<user_instructions>",
	"<environment_context>",
	"<ENVIRONMENT_CONTEXT>",
	"<skill",
	"AGENTS.md",
}

// RegrowthClassStat is one content class's share of a window (or of the corpus).
// Lengths and duplicate counts only — never bodies.
type RegrowthClassStat struct {
	Rows     int   `json:"rows"`
	Bytes    int64 `json:"bytes"`
	DupRows  int   `json:"dup_rows,omitempty"`
	DupBytes int64 `json:"dup_bytes,omitempty"`
}

// RegrowthCrossing times one resident-token milestone after a fire.
type RegrowthCrossing struct {
	Threshold int     `json:"threshold"`
	Seconds   float64 `json:"seconds"`
	Samples   int     `json:"samples"`
	Turns     int     `json:"turns"`
	ToolCalls int     `json:"tool_calls"`
}

// CompactRegrowth is one fire's regrowth trajectory: how fast the window refilled,
// out of what, and how the observation ended.
type CompactRegrowth struct {
	// PostFloorTokens is the lowest resident sample observed after the fire — the
	// point regrowth is measured from.
	PostFloorTokens    int `json:"post_floor_tokens"`
	LastResidentTokens int `json:"last_resident_tokens"`
	GrowthTokens       int `json:"growth_tokens"`

	Samples   int     `json:"samples"`
	Turns     int     `json:"turns"`
	ToolCalls int     `json:"tool_calls"`
	Seconds   float64 `json:"seconds"`

	TokensPerSample float64 `json:"tokens_per_sample,omitempty"`
	TokensPerTurn   float64 `json:"tokens_per_turn,omitempty"`
	// TokensPerSecond is only computed on a clean clock; a TIMESTAMP_SUSPECT window
	// never reports a wall-clock velocity.
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`

	Crossings []RegrowthCrossing `json:"crossings,omitempty"`

	// Rebounded — resident reached RegrowthReboundTokens within this window.
	// Censored names why observation stopped short when it did not.
	Rebounded       bool    `json:"rebounded"`
	Censored        string  `json:"censored,omitempty"`
	NextFireSeconds float64 `json:"next_fire_seconds,omitempty"`

	// Cache join: summed provider input/cache-read tokens across the window's
	// samples. CacheReadFraction = cache reads / total input, i.e. how much of the
	// regrowth pricing was reuse. -1 when the window carried no samples.
	WindowInputTokens int     `json:"window_input_tokens"`
	CacheReadTokens   int     `json:"cache_read_tokens"`
	CacheReadFraction float64 `json:"cache_read_fraction"`

	// Classes attribute the transcript bytes appended during the window.
	Classes map[string]*RegrowthClassStat `json:"classes,omitempty"`

	Anomalies  []string `json:"anomalies,omitempty"`
	Confidence string   `json:"confidence"`
	Reason     string   `json:"reason"`
}

// RegrowthCohort summarizes one side of the fast/slow comparison, so "optimize the
// rebound away" can be checked against what the fast sessions were actually doing.
type RegrowthCohort struct {
	Windows                 int     `json:"windows"`
	MedianToolCalls         int     `json:"median_tool_calls"`
	MedianTurns             int     `json:"median_turns"`
	MedianGrowthTokens      int     `json:"median_growth_tokens"`
	MedianCacheReadFraction float64 `json:"median_cache_read_fraction"`
}

// CompactRegrowthRollup is the corpus roll-up: the issue's headline counts plus the
// ranked attribution table.
type CompactRegrowthRollup struct {
	// FiresWithTelemetry — fires with at least one subsequent resident sample (the
	// issue's 1,044 denominator).
	FiresWithTelemetry int `json:"fires_with_telemetry"`
	// Rebounds — windows that reached RegrowthReboundTokens (the issue's 699).
	Rebounds int `json:"rebounds"`
	Censored int `json:"censored"`
	// TimestampSuspect windows still count as rebounds but are excluded from every
	// wall-clock statistic below.
	TimestampSuspect int `json:"timestamp_suspect"`

	MedianSecondsToRebound float64 `json:"median_seconds_to_rebound"`
	P90SecondsToRebound    float64 `json:"p90_seconds_to_rebound"`
	MedianSamplesToRebound int     `json:"median_samples_to_rebound"`
	ReboundsWithin15Min    int     `json:"rebounds_within_15min"`
	ReboundsWithin30Min    int     `json:"rebounds_within_30min"`
	MedianNextFireSeconds  float64 `json:"median_next_fire_seconds"`

	MedianCacheReadFraction float64 `json:"median_cache_read_fraction"`

	ClassTotals   map[string]*RegrowthClassStat `json:"class_totals,omitempty"`
	AnomalyCounts map[string]int                `json:"anomaly_counts,omitempty"`

	Fast RegrowthCohort `json:"fast_cohort"`
	Slow RegrowthCohort `json:"slow_cohort"`
}

// regrowKey identifies row content for dedup: the payload-content hash plus the full
// row length. Content hashes deliberately exclude per-call ids (call_id/id), so the
// same tool output returned under two call ids still reads as the same span.
type regrowKey struct {
	hash uint64
	size int64
}

// regrowthWindow is the open trajectory being accumulated for the most recent fire.
type regrowthWindow struct {
	fireIdx  int
	at       time.Time
	turnBase int
	toolBase int
	reg      *CompactRegrowth
	crossed  map[int]bool
	haveTS   bool
}

// regrowthTracker rides ScanCompactRollout's single pass. It sees every transcript
// row (for the session-wide dedup table) and every token sample (for trajectories),
// and owns nothing about fire admission — compactaudit.go stays the referee there.
type regrowthTracker struct {
	seen       map[regrowKey]struct{}
	toolByCall map[string]string
	windows    []*regrowthWindow // closed and open, in fire order
	cur        *regrowthWindow
	// replay is nil unless a caller armed the #5254 counterfactual dedup replay
	// (compactregrowth_replay.go). Nil is the production path and costs nothing.
	replay *regrowthReplay
}

func newRegrowthTracker() *regrowthTracker {
	return &regrowthTracker{
		seen:       make(map[regrowKey]struct{}),
		toolByCall: make(map[string]string),
	}
}

// regrowthItemProbe is the head-parse of a response_item payload: only the leading
// identity fields plus raw content slots for hashing. Bodies pass through the hash and
// are never retained.
type regrowthItemProbe struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Name      string          `json:"name"`
	CallID    string          `json:"call_id"`
	Content   json.RawMessage `json:"content"`
	Output    json.RawMessage `json:"output"`
	Arguments json.RawMessage `json:"arguments"`
}

// onFire closes the current window (censoring it if the rebound bar was not reached)
// and opens the next one. Called only when compactaudit.go ADMITS a new fire, so the
// compacted/context_compacted twin never splits a window.
func (tr *regrowthTracker) onFire(fireIdx int, ts time.Time, turn, toolCalls int) {
	tr.closeCurrent(RegrowthCensorNextFire, ts)
	w := &regrowthWindow{
		fireIdx:  fireIdx,
		at:       ts,
		turnBase: turn,
		toolBase: toolCalls,
		crossed:  make(map[int]bool),
		reg: &CompactRegrowth{
			CacheReadFraction: -1,
			Classes:           map[string]*RegrowthClassStat{},
		},
		haveTS: !ts.IsZero(),
	}
	tr.windows = append(tr.windows, w)
	tr.cur = w
}

// observeSample advances the open trajectory with one non-zero resident sample.
func (tr *regrowthTracker) observeSample(resident, inputTokens, reuseTokens int, ts time.Time, turn, toolCalls int) {
	w := tr.cur
	if w == nil {
		return
	}
	r := w.reg
	r.Samples++
	if r.PostFloorTokens == 0 || resident < r.PostFloorTokens {
		r.PostFloorTokens = resident
	}
	r.LastResidentTokens = resident
	r.Turns = turn - w.turnBase
	r.ToolCalls = toolCalls - w.toolBase
	if w.haveTS && !ts.IsZero() {
		if d := ts.Sub(w.at).Seconds(); d > r.Seconds {
			r.Seconds = round4(d)
		}
	}
	r.WindowInputTokens += inputTokens
	r.CacheReadTokens += reuseTokens

	for _, th := range RegrowthThresholds {
		if resident < th || w.crossed[th] {
			continue
		}
		w.crossed[th] = true
		c := RegrowthCrossing{
			Threshold: th,
			Samples:   r.Samples,
			Turns:     r.Turns,
			ToolCalls: r.ToolCalls,
		}
		if w.haveTS && !ts.IsZero() {
			c.Seconds = round4(ts.Sub(w.at).Seconds())
		}
		r.Crossings = append(r.Crossings, c)
		if th == RegrowthReboundTokens {
			r.Rebounded = true
		}
	}

	// Suffix recreation (#3071's signature): the window jumps a burst's worth above
	// its floor within the first couple of samples, before any real work could have
	// produced it.
	if r.Samples <= RegrowthSuffixWindowSamples &&
		resident-r.PostFloorTokens >= RegrowthSuffixBurstTokens {
		r.Anomalies = appendUnique(r.Anomalies, AnomalySuffixRecreation)
	}
}

// observeResponseItem classifies one transcript row, measures it, and registers its
// content hash. Rows before the first fire feed only the dedup table — that is what
// lets a post-fire reinjection of pre-fire setup read as a duplicate.
func (tr *regrowthTracker) observeResponseItem(parsed bool, payload, head []byte, rowLen int64) {
	class, key := tr.classifyItem(parsed, payload, head, rowLen)
	dup := tr.observeClassed(class, key, rowLen)
	// The counterfactual replay needs the tool-result BODY, which the audit itself is careful
	// never to retain. Re-extract it here rather than threading it through classifyItem, so the
	// production path (replay == nil) keeps its body-blind hot loop untouched.
	if tr.replay != nil && tr.cur != nil && strings.HasPrefix(class, RegrowClassToolResPrefix) {
		tr.replay.observe(toolResultSlot(parsed, payload, head), rowLen, dup, !parsed)
	}
}

// toolResultSlot re-reads the `output` slot of a tool-result row for the replay only. It mirrors
// exactly what classifyItem hashed, so the replay scores the same bytes the anomaly was raised on.
func toolResultSlot(parsed bool, payload, head []byte) []byte {
	if parsed && len(payload) > 0 {
		var p struct {
			Output json.RawMessage `json:"output"`
		}
		if json.Unmarshal(payload, &p) == nil && p.Output != nil {
			return p.Output
		}
	}
	return headContentSlice(head, `"output":`)
}

// observeCompacted attributes the compaction summary row itself (its
// replacement_history is what the compactor injects into the fresh window).
func (tr *regrowthTracker) observeCompacted(rowLen int64) {
	tr.observeClassed(RegrowClassCompactSummary, regrowKey{}, rowLen)
}

// observeClassed attributes one row to its class and reports whether the row was a session-wide
// content duplicate, so the counterfactual replay scores rows under the audit's own verdict rather
// than a second, possibly divergent, rule.
func (tr *regrowthTracker) observeClassed(class string, key regrowKey, rowLen int64) bool {
	dup := false
	if key.hash != 0 && rowLen >= RegrowthDupMinBytes {
		if _, ok := tr.seen[key]; ok {
			dup = true
		} else if len(tr.seen) < regrowthDupMapCap {
			tr.seen[key] = struct{}{}
		}
	}
	w := tr.cur
	if w == nil {
		return dup
	}
	st := w.reg.Classes[class]
	if st == nil {
		st = &RegrowthClassStat{}
		w.reg.Classes[class] = st
	}
	st.Rows++
	st.Bytes += rowLen
	if dup {
		st.DupRows++
		st.DupBytes += rowLen
	}
	if rowLen >= RegrowthOversizedRowBytes {
		w.reg.Anomalies = appendUnique(w.reg.Anomalies, AnomalyOversizedEvent)
	}
	return dup
}

// classifyItem maps a response_item row to its content class and content-hash key.
// The hash covers the content slot only (output/content/name+arguments), never the
// per-call ids, and for over-long rows it covers the streamed head after the content
// key plus the total length — enough that two identical giant outputs still collide.
func (tr *regrowthTracker) classifyItem(parsed bool, payload, head []byte, rowLen int64) (string, regrowKey) {
	var p regrowthItemProbe
	if parsed && len(payload) > 0 {
		if json.Unmarshal(payload, &p) != nil {
			return RegrowClassUnknown, regrowKey{}
		}
	} else {
		p.Type = typeValueAt(head, 2)
		p.Role = firstStringField(head, "role")
		p.Name = firstStringField(head, "name")
		p.CallID = firstStringField(head, "call_id")
	}

	switch p.Type {
	case "message":
		content := p.Content
		if content == nil {
			content = headContentSlice(head, `"content":`)
		}
		class := RegrowClassUnknown
		switch p.Role {
		case "user":
			class = RegrowClassUserMessage
		case "system":
			class = RegrowClassSystemMessage
		case "developer":
			class = RegrowClassDeveloperMsg
		case "assistant":
			class = RegrowClassAssistantMsg
		}
		for _, m := range regrowthInstructionMarkers {
			if bytes.Contains(content, []byte(m)) {
				class = RegrowClassInstructions
				break
			}
		}
		return class, regrowKey{hash: hashRegrowContent([]byte(p.Role), content), size: rowLen}
	case "function_call", "custom_tool_call", "local_shell_call":
		name := p.Name
		if p.Type == "local_shell_call" {
			name = "local_shell"
		}
		if name == "" {
			name = "unknown"
		}
		if p.CallID != "" {
			tr.rememberTool(p.CallID, name)
		}
		args := p.Arguments
		if args == nil {
			args = headContentSlice(head, `"arguments":`)
		}
		return RegrowClassToolCallPrefix + name,
			regrowKey{hash: hashRegrowContent([]byte(name), args), size: rowLen}
	case "function_call_output", "custom_tool_call_output", "local_shell_call_output":
		name := tr.toolByCall[p.CallID]
		if p.Type == "local_shell_call_output" {
			name = "local_shell"
		}
		if name == "" {
			name = "unknown"
		}
		out := p.Output
		if out == nil {
			out = headContentSlice(head, `"output":`)
		}
		return RegrowClassToolResPrefix + name,
			regrowKey{hash: hashRegrowContent(nil, out), size: rowLen}
	case "reasoning":
		return RegrowClassReasoning, regrowKey{}
	}
	return RegrowClassUnknown, regrowKey{}
}

func (tr *regrowthTracker) rememberTool(callID, name string) {
	if len(tr.toolByCall) < regrowthDupMapCap {
		tr.toolByCall[callID] = name
	}
}

// headContentSlice returns the head bytes after the first occurrence of key — the
// hashable content slot of a row too long to unmarshal. The tail past the head bound
// was discarded, which is why the dedup key also carries the row length.
func headContentSlice(head []byte, key string) []byte {
	i := bytes.Index(head, []byte(key))
	if i < 0 {
		return nil
	}
	return head[i+len(key):]
}

func hashRegrowContent(prefix, content []byte) uint64 {
	if len(content) == 0 && len(prefix) == 0 {
		return 0
	}
	h := fnv.New64a()
	h.Write(prefix)
	h.Write(content)
	v := h.Sum64()
	if v == 0 {
		v = 1 // 0 is the "no content hash" sentinel
	}
	return v
}

// closeCurrent finishes the open window: censor typing, velocities, duplicate-span
// anomalies, cache fraction, and the timestamp-suspect exclusion.
func (tr *regrowthTracker) closeCurrent(censor string, nextFireAt time.Time) {
	w := tr.cur
	if w == nil {
		return
	}
	tr.cur = nil
	// Score the counterfactual replay on exactly the audit's window boundaries, before any
	// telemetry-driven early return below can skip it.
	if tr.replay != nil {
		tr.replay.closeWindow()
	}
	r := w.reg

	if censor == RegrowthCensorNextFire && w.haveTS && !nextFireAt.IsZero() {
		if d := nextFireAt.Sub(w.at).Seconds(); d >= 0 {
			r.NextFireSeconds = round4(d)
		}
	}
	if !r.Rebounded {
		r.Censored = censor
	}

	if r.Samples == 0 {
		r.Confidence = CompactConfidenceNone
		r.Reason = CompactReasonTelemetryMissing
		sort.Strings(r.Anomalies)
		return
	}

	r.GrowthTokens = r.LastResidentTokens - r.PostFloorTokens
	if r.Samples > 0 {
		r.TokensPerSample = round4(float64(r.GrowthTokens) / float64(r.Samples))
	}
	if r.Turns > 0 {
		r.TokensPerTurn = round4(float64(r.GrowthTokens) / float64(r.Turns))
	}

	// A rebound whose 200k CROSSING shows no wall-clock lapse across 2+ samples is
	// the issue's "zero-second" telemetry artifact — an ordering/timestamp anomaly,
	// not a real velocity. Typed suspect, excluded from clock math. Keyed on the
	// crossing, not the whole window: samples after the rebound can carry sane
	// clocks and must not launder the impossible crossing.
	if c := reboundCrossingOf(r); c != nil && c.Seconds <= 0 && c.Samples >= 2 {
		r.Anomalies = appendUnique(r.Anomalies, AnomalyTimestampSuspect)
	}
	if r.Seconds > 0 && !hasRegrowAnomaly(r, AnomalyTimestampSuspect) {
		r.TokensPerSecond = round4(float64(r.GrowthTokens) / r.Seconds)
	}

	if r.WindowInputTokens > 0 {
		r.CacheReadFraction = round4(float64(r.CacheReadTokens) / float64(r.WindowInputTokens))
	}

	var setupDupBytes int64
	var toolDupRows int
	var toolDupBytes int64
	for class, st := range r.Classes {
		switch {
		case class == RegrowClassInstructions ||
			strings.HasPrefix(class, "message/"):
			setupDupBytes += st.DupBytes
		case strings.HasPrefix(class, RegrowClassToolResPrefix):
			toolDupRows += st.DupRows
			toolDupBytes += st.DupBytes
		}
	}
	if setupDupBytes >= RegrowthDupMinBytes {
		r.Anomalies = appendUnique(r.Anomalies, AnomalyDuplicateSetup)
	}
	if toolDupRows >= RegrowthDupToolMinRows && toolDupBytes >= RegrowthDupMinBytes {
		r.Anomalies = appendUnique(r.Anomalies, AnomalyRepeatedToolResult)
	}

	switch {
	case hasRegrowAnomaly(r, AnomalyTimestampSuspect):
		r.Confidence = CompactConfidenceLow
		r.Reason = AnomalyTimestampSuspect
	default:
		r.Confidence = CompactConfidenceHigh
		r.Reason = CompactReasonOK
	}
	sort.Strings(r.Anomalies)
}

// reboundCrossingOf returns the RegrowthReboundTokens crossing, or nil before the
// window rebounded.
func reboundCrossingOf(r *CompactRegrowth) *RegrowthCrossing {
	for i := range r.Crossings {
		if r.Crossings[i].Threshold == RegrowthReboundTokens {
			return &r.Crossings[i]
		}
	}
	return nil
}

func hasRegrowAnomaly(r *CompactRegrowth, want string) bool {
	for _, a := range r.Anomalies {
		if a == want {
			return true
		}
	}
	return false
}

// finalize closes the last window as end-of-session censored and attaches every
// trajectory to its fire.
func (tr *regrowthTracker) finalize(rep *CompactSessionReport) {
	tr.closeCurrent(RegrowthCensorRolloutEnd, time.Time{})
	for _, w := range tr.windows {
		if w.fireIdx >= 0 && w.fireIdx < len(rep.Fires) {
			rep.Fires[w.fireIdx].Regrowth = w.reg
		}
	}
}

// regrowthCohortAccum collects the per-window samples behind ONE regrowth cohort. The
// rollup splits windows into a fast cohort and a slow cohort, and the two are the same
// cohort measured over different windows, not two different measurements — so which
// per-window facts a cohort collects, and how those samples become the reported
// medians, are each stated once here rather than mirrored per cohort. Mirroring is
// what makes the fast and slow rows quietly incomparable: a fact added to one arm and
// missed in the other, or a cohort summarised with a different rounding, breaks the
// only thing the two rows exist to support — reading them against each other.
type regrowthCohortAccum struct {
	tools, turns, growth []int
	reuse                []float64
}

// add records one window. A negative CacheReadFraction means the window carried no
// reuse telemetry at all, so it is omitted from the reuse samples instead of being
// folded in as a zero — a missing measurement must not drag the median down.
func (c *regrowthCohortAccum) add(r *CompactRegrowth) {
	c.tools = append(c.tools, r.ToolCalls)
	c.turns = append(c.turns, r.Turns)
	c.growth = append(c.growth, r.GrowthTokens)
	if r.CacheReadFraction >= 0 {
		c.reuse = append(c.reuse, r.CacheReadFraction)
	}
}

// cohort reports the accumulated windows as the published cohort row. Windows counts
// every added window (the reuse samples can be fewer); the reuse median keeps the
// rollup's round4 so a cohort figure never renders at a different precision than the
// corpus figures beside it.
func (c *regrowthCohortAccum) cohort() RegrowthCohort {
	return RegrowthCohort{
		Windows:                 len(c.tools),
		MedianToolCalls:         medianInt(c.tools),
		MedianTurns:             medianInt(c.turns),
		MedianGrowthTokens:      medianInt(c.growth),
		MedianCacheReadFraction: round4(medianFloat(c.reuse)),
	}
}

// rollupCompactRegrowth rolls every fire's trajectory into the corpus answer.
// Wall-clock statistics use clean-clock rebounds only; suspect windows are counted,
// never timed.
func rollupCompactRegrowth(reports []CompactSessionReport) *CompactRegrowthRollup {
	agg := &CompactRegrowthRollup{
		ClassTotals:   map[string]*RegrowthClassStat{},
		AnomalyCounts: map[string]int{},
	}
	var cleanSeconds []float64
	var reboundSamples []int
	var nextFire []float64
	var reuseFrac []float64
	var fast, slow regrowthCohortAccum

	for _, rep := range reports {
		for _, f := range rep.Fires {
			r := f.Regrowth
			if r == nil || r.Samples == 0 {
				continue
			}
			agg.FiresWithTelemetry++
			for _, a := range r.Anomalies {
				agg.AnomalyCounts[a]++
			}
			for class, st := range r.Classes {
				tot := agg.ClassTotals[class]
				if tot == nil {
					tot = &RegrowthClassStat{}
					agg.ClassTotals[class] = tot
				}
				tot.Rows += st.Rows
				tot.Bytes += st.Bytes
				tot.DupRows += st.DupRows
				tot.DupBytes += st.DupBytes
			}
			if r.NextFireSeconds > 0 {
				nextFire = append(nextFire, r.NextFireSeconds)
			}
			if r.CacheReadFraction >= 0 {
				reuseFrac = append(reuseFrac, r.CacheReadFraction)
			}

			suspect := hasRegrowAnomaly(r, AnomalyTimestampSuspect)
			if suspect {
				agg.TimestampSuspect++
			}
			if !r.Rebounded {
				agg.Censored++
				slow.add(r)
				continue
			}
			agg.Rebounds++
			var reboundSec float64
			for _, c := range r.Crossings {
				if c.Threshold == RegrowthReboundTokens {
					reboundSec = c.Seconds
					reboundSamples = append(reboundSamples, c.Samples)
					break
				}
			}
			isFast := false
			if !suspect && reboundSec > 0 {
				cleanSeconds = append(cleanSeconds, reboundSec)
				if reboundSec <= RegrowthWithin15MinSeconds {
					agg.ReboundsWithin15Min++
				}
				if reboundSec <= RegrowthFastReboundSeconds {
					agg.ReboundsWithin30Min++
					isFast = true
				}
			}
			if isFast {
				fast.add(r)
			} else {
				slow.add(r)
			}
		}
	}

	agg.MedianSecondsToRebound = round4(medianFloat(cleanSeconds))
	agg.P90SecondsToRebound = round4(percentileFloat(cleanSeconds, 0.90))
	agg.MedianSamplesToRebound = medianInt(reboundSamples)
	agg.MedianNextFireSeconds = round4(medianFloat(nextFire))
	agg.MedianCacheReadFraction = round4(medianFloat(reuseFrac))
	agg.Fast = fast.cohort()
	agg.Slow = slow.cohort()
	if agg.FiresWithTelemetry == 0 {
		return nil
	}
	return agg
}

// percentileFloat is the nearest-rank percentile (the value at ceil(p*n), 1-based).
func percentileFloat(v []float64, p float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	idx := int(p*float64(len(s))+0.9999999) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

// writeCompactRegrowthSection writes the regrowth section of the human report: the rebound
// headline, the ranked attribution table, and the fast/slow comparison.
func writeCompactRegrowthSection(w interface{ Write([]byte) (int, error) }, agg *CompactRegrowthRollup, topN int) {
	if agg == nil || agg.FiresWithTelemetry == 0 {
		return
	}
	fmt.Fprintf(w, "  regrowth after compaction — %d fires with post-fire telemetry\n", agg.FiresWithTelemetry)
	fmt.Fprintf(w, "    rebounded to >=%dk resident: %d (%d censored; %d timestamp-suspect excluded from wall-clock stats)\n",
		RegrowthReboundTokens/1000, agg.Rebounds, agg.Censored, agg.TimestampSuspect)
	if agg.Rebounds > 0 {
		fmt.Fprintf(w, "    time back to %dk: median %.0f s, p90 %.0f s, median %d samples; %d within 15 min, %d within 30 min\n",
			RegrowthReboundTokens/1000, agg.MedianSecondsToRebound, agg.P90SecondsToRebound,
			agg.MedianSamplesToRebound, agg.ReboundsWithin15Min, agg.ReboundsWithin30Min)
	}
	if agg.MedianNextFireSeconds > 0 {
		fmt.Fprintf(w, "    median time to the NEXT fire: %.0f s\n", agg.MedianNextFireSeconds)
	}
	if agg.MedianCacheReadFraction > 0 {
		fmt.Fprintf(w, "    cache join: median %.0f%% of window input tokens were cache reads (regrowth priced net of reuse)\n",
			agg.MedianCacheReadFraction*100)
	}

	if len(agg.ClassTotals) > 0 && topN > 0 {
		type rankedClass struct {
			class string
			st    *RegrowthClassStat
		}
		ranked := make([]rankedClass, 0, len(agg.ClassTotals))
		for c, st := range agg.ClassTotals {
			ranked = append(ranked, rankedClass{c, st})
		}
		sort.Slice(ranked, func(i, j int) bool {
			if ranked[i].st.Bytes != ranked[j].st.Bytes {
				return ranked[i].st.Bytes > ranked[j].st.Bytes
			}
			return ranked[i].class < ranked[j].class
		})
		if len(ranked) > topN {
			ranked = ranked[:topN]
		}
		fmt.Fprintln(w, "    regrowth attribution (transcript bytes appended post-fire, ranked):")
		for _, rc := range ranked {
			line := fmt.Sprintf("      %-28s %9s in %d rows", rc.class, humanBytes(rc.st.Bytes), rc.st.Rows)
			if rc.st.DupBytes > 0 {
				line += fmt.Sprintf("  (duplicated: %s in %d rows)", humanBytes(rc.st.DupBytes), rc.st.DupRows)
			}
			fmt.Fprintln(w, line)
		}
	}
	if len(agg.AnomalyCounts) > 0 {
		fmt.Fprintln(w, "    regrowth anomalies:")
		for _, k := range sortedKeys(agg.AnomalyCounts) {
			fmt.Fprintf(w, "      %-28s %d windows\n", k, agg.AnomalyCounts[k])
		}
	}
	if agg.Fast.Windows > 0 || agg.Slow.Windows > 0 {
		fmt.Fprintf(w, "    fast (<=30 min rebound) vs slower/censored windows: tool calls %d vs %d, turns %d vs %d, growth %d vs %d tokens, cache-read %.0f%% vs %.0f%%\n",
			agg.Fast.MedianToolCalls, agg.Slow.MedianToolCalls,
			agg.Fast.MedianTurns, agg.Slow.MedianTurns,
			agg.Fast.MedianGrowthTokens, agg.Slow.MedianGrowthTokens,
			agg.Fast.MedianCacheReadFraction*100, agg.Slow.MedianCacheReadFraction*100)
	}
	fmt.Fprintln(w)
}
