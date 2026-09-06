package trajctl

// detour.go — issue #2546, the detour-objective rung of the trajectory-control
// epic (#2533): error side-quests become visible as budgeted child objectives.
// An error burst followed by hours of repair looks identical to progress in
// every prior signal (epic use cases 3 and 4); the doctrine's answer is
// "detours are objectives too" — a side-quest is opened as a child with a
// budget, scored on repair, and closed when the work *returns*.
//
// This file is that lifecycle. A W2 tool-stream detector folds the ordered
// (tool, target, is_error) stream a recorded transcript carries: an ERROR
// BURST (detourErrorBurstMin errors inside a detourBurstWindow-event window)
// plus a SUSTAINED TOPIC SHIFT (detourShiftRun consecutive calls off the
// pre-burst parent shape, starting within detourShiftHorizon of the burst)
// opens a detour span; the stream RETURNING to the parent shape
// (detourReturnRun consecutive non-error calls back on pre-burst topics)
// closes it. DetourRows turns each span into ledger rows: a child objective
// with the default detour budget plus the parent flipped to PAUSED on open,
// and the child MET plus the parent restored to ACTIVE on close — exactly the
// shapes curve.go's DETOUR_OVERRUN gate and rollup.go's tree fold already
// read. Each open/close also ledgers a W2 score row on the child (0 at open,
// 1 at return) carrying the transcript-span evidence, so the detour's repair
// curve has witnessed endpoints.
//
// The fold stays pure and tier-1: DetectDetourSpans and DetourRows read only
// the injected event stream and objective; the transcript walk is a bounded
// tool_use/tool_result pairing (ReadToolStream), heuristics only — model-based
// topic detection is out of scope; budget ENFORCEMENT (the return-to-main loop, #2552)
// fires a return-to-main nudge via the spine steer channel when DETOUR_OVERRUN trips,
// escalating to warn on repeat.

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// DetourDetectorMethod / DetourDetectorVersion identify the detector's W2
// score rows. Bump the version when the burst/shift/return constants or the
// topic key change, so folded curves distinguish detector generations.
const (
	DetourDetectorMethod  = "detour-detector"
	DetourDetectorVersion = "1"
)

// The pinned stream-shape constants. Heuristics first (#2546 out-of-scope
// fences model-based detection); each is small enough to reason about on a
// fixture and pinned so the fixture tests are stable.
const (
	// detourBurstWindow is the sliding window (in tool events) the burst
	// counter scans.
	detourBurstWindow = 6
	// detourErrorBurstMin errors inside one window is an error burst.
	detourErrorBurstMin = 3
	// detourShiftRun consecutive off-parent-topic calls is a sustained shift —
	// one stray off-topic probe is not a side-quest.
	detourShiftRun = 3
	// detourShiftHorizon bounds how far after the burst's first error the
	// shift run may START and still be attributed to the burst: a topic move
	// long after a self-recovered burst is legitimate new work, not a detour.
	detourShiftHorizon = 12
	// detourReturnRun consecutive non-error on-parent-topic calls closes the
	// detour — a single mid-detour check against parent files must not.
	detourReturnRun = 2
	// defaultDetourBudgetTurns is the turn budget a detector-opened detour
	// starts with: wide enough for a small repair, small enough that
	// DETOUR_OVERRUN (curve.go) fires before a lost afternoon.
	defaultDetourBudgetTurns = 5
)

// DefaultDetourBudget is the budget a detector-opened detour child objective
// is declared with ("the default budget", #2546 in-scope clause).
func DefaultDetourBudget() Budget {
	return Budget{Turns: defaultDetourBudgetTurns}
}

// ToolEvent is one tool call in stream order: the minimal slice of a recorded
// transcript the detector folds. Target is the call's primary operand (file
// path, search path, or command) — the topic proxy; IsError is whether the
// paired tool_result carried is_error.
type ToolEvent struct {
	Tool    string `json:"tool"`
	Target  string `json:"target,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

// DetourSpan is one detected side-quest, in stream-index coordinates.
type DetourSpan struct {
	// BurstIndex is the first errored event of the triggering burst.
	BurstIndex int `json:"burst_index"`
	// OpenIndex is where the sustained off-parent shift begins — the detour's
	// open point.
	OpenIndex int `json:"open_index"`
	// CloseIndex is the return point (first event of the proven return run),
	// or -1 while the detour is still open at stream end.
	CloseIndex int `json:"close_index"`
	// Errors is the errored-event count in the burst window starting at
	// BurstIndex.
	Errors int `json:"errors"`
	// Topics are the distinct off-parent topic keys of the shift run, sorted.
	Topics []string `json:"topics,omitempty"`
}

// Closed reports whether the stream returned to the parent shape.
func (sp DetourSpan) Closed() bool { return sp.CloseIndex >= 0 }

// topicKey reduces an event to its topic: a path-like target folds to its
// directory (normalized to forward slashes, lowercased) so sibling files share
// a topic; anything else folds to its first token (a command's program); an
// empty target falls back to the tool name. A "" key (tool and target both
// empty) is neutral — it never joins the parent baseline and resets both the
// shift and return runs.
func topicKey(ev ToolEvent) string {
	t := strings.TrimSpace(ev.Target)
	if t == "" {
		return strings.ToLower(strings.TrimSpace(ev.Tool))
	}
	if strings.ContainsAny(t, `/\`) {
		norm := strings.ToLower(strings.ReplaceAll(t, `\`, "/"))
		if i := strings.LastIndex(norm, "/"); i > 0 {
			return norm[:i]
		}
		return norm
	}
	if f := strings.Fields(t); len(f) > 0 {
		return strings.ToLower(f[0])
	}
	return strings.ToLower(strings.TrimSpace(ev.Tool))
}

// DetectDetourSpans folds the ordered tool stream into detour spans. Pure and
// deterministic: same stream, same spans. The parent shape is the topic set of
// non-error events before each burst, excluding events inside already-detected
// spans so a first detour's topics never pollute a later baseline. A burst
// with no established parent shape, or no sustained shift within the horizon,
// opens nothing (an in-place retry loop is a stall, not a side-quest — the
// stall scorer's rung). Scanning resumes after a proven return, so repair
// errors inside an open detour never open a second one.
func DetectDetourSpans(events []ToolEvent) []DetourSpan {
	spans := make([]DetourSpan, 0)
	i := 0
	for i < len(events) {
		burst, trigger := findBurst(events, i)
		if burst < 0 {
			break
		}
		base := baselineTopics(events, burst, spans)
		if len(base) == 0 {
			i = trigger + 1
			continue
		}
		open, topics := findShift(events, burst, base)
		if open < 0 {
			i = trigger + 1
			continue
		}
		span := DetourSpan{
			BurstIndex: burst,
			OpenIndex:  open,
			CloseIndex: -1,
			Errors:     countErrors(events, burst, burst+detourBurstWindow),
			Topics:     topics,
		}
		if ret := findReturn(events, open+detourShiftRun, base); ret >= 0 {
			span.CloseIndex = ret
			spans = append(spans, span)
			i = ret + detourReturnRun
			continue
		}
		spans = append(spans, span) // still open at stream end
		break
	}
	return spans
}

// findBurst scans triggers t >= from for a window [max(from,t-W+1), t] holding
// >= detourErrorBurstMin errors; it returns (first error index in that window,
// t), or (-1, -1). Clipping at from keeps a previous span's errors from
// seeding a new burst.
func findBurst(events []ToolEvent, from int) (int, int) {
	for t := from; t < len(events); t++ {
		lo := t - detourBurstWindow + 1
		if lo < from {
			lo = from
		}
		n, first := 0, -1
		for j := lo; j <= t; j++ {
			if events[j].IsError {
				n++
				if first < 0 {
					first = j
				}
			}
		}
		if n >= detourErrorBurstMin {
			return first, t
		}
	}
	return -1, -1
}

// baselineTopics is the parent shape: distinct topic keys of non-error events
// before upto, skipping events inside already-detected spans.
func baselineTopics(events []ToolEvent, upto int, spans []DetourSpan) map[string]bool {
	base := map[string]bool{}
	for j := 0; j < upto && j < len(events); j++ {
		if events[j].IsError || inSpan(j, spans) {
			continue
		}
		if k := topicKey(events[j]); k != "" {
			base[k] = true
		}
	}
	return base
}

// inSpan reports whether stream index j falls inside a detected span's
// [OpenIndex, CloseIndex) range (an unclosed span covers through stream end).
func inSpan(j int, spans []DetourSpan) bool {
	for _, sp := range spans {
		if j >= sp.OpenIndex && (!sp.Closed() || j < sp.CloseIndex) {
			return true
		}
	}
	return false
}

// findShift scans from the burst for detourShiftRun consecutive off-baseline
// topics whose run starts within detourShiftHorizon of the burst. It returns
// the run's start index plus its distinct topics (sorted), or (-1, nil).
func findShift(events []ToolEvent, burst int, base map[string]bool) (int, []string) {
	run, start := 0, -1
	for j := burst; j < len(events); j++ {
		k := topicKey(events[j])
		if k == "" || base[k] {
			run, start = 0, -1
			continue
		}
		if run == 0 {
			start = j
			if start > burst+detourShiftHorizon {
				return -1, nil // a shift this late is new work, not this burst's detour
			}
		}
		run++
		if run >= detourShiftRun {
			seen := map[string]bool{}
			for _, ev := range events[start : start+detourShiftRun] {
				seen[topicKey(ev)] = true
			}
			topics := make([]string, 0, len(seen))
			for k := range seen {
				topics = append(topics, k)
			}
			sort.Strings(topics)
			return start, topics
		}
	}
	return -1, nil
}

// findReturn scans from `from` for detourReturnRun consecutive non-error
// on-baseline events and returns the run's start — the return point — or -1.
func findReturn(events []ToolEvent, from int, base map[string]bool) int {
	run, start := 0, -1
	for j := from; j < len(events); j++ {
		if !events[j].IsError && base[topicKey(events[j])] {
			if run == 0 {
				start = j
			}
			run++
			if run >= detourReturnRun {
				return start
			}
			continue
		}
		run, start = 0, -1
	}
	return -1
}

// countErrors counts errored events in [from, to) clipped to the stream.
func countErrors(events []ToolEvent, from, to int) int {
	n := 0
	for j := from; j < to && j < len(events); j++ {
		if events[j].IsError {
			n++
		}
	}
	return n
}

// DetourRows turns detected spans into the ledger's detour open/close trail
// for one parent objective. Per span k (1-based in the child id): OPEN appends
// the child objective ("<parent>-detour-<k>", ParentID set, the default detour
// budget, ACTIVE) plus the parent flipped PAUSED plus a W2 open row (value 0)
// on the child; CLOSE — only for a span the stream returned from — appends the
// child MET plus the parent restored ACTIVE plus a W2 close row (value 1).
// streamRef names the transcript behind the evidence refs ("tool-stream" when
// empty). Pure: the caller owns the append (AppendDetourRows); replaying the
// same spans re-derives the same rows, and the latest-wins objective fold
// converges. A parent that cannot validate as an objective row (empty id or
// statement) yields nil.
func DetourRows(parent Objective, spans []DetourSpan, streamRef string, unixMillis int64, stamp Stamp) []Row {
	if parent.ID == "" || parent.Statement == "" {
		return nil
	}
	if streamRef == "" {
		streamRef = "tool-stream"
	}
	rows := make([]Row, 0, 6*len(spans))
	for k, sp := range spans {
		open, closed := detourSpanRows(parent, sp, k, streamRef, unixMillis, stamp)
		rows = append(rows, open...)
		rows = append(rows, closed...)
	}
	return rows
}

// detourSpanRows builds the ledger rows for one detected span k (0-based; child id
// "<parent>-detour-<k+1>"): the OPEN trio always (the budgeted child objective
// ACTIVE, the parent flipped PAUSED, and a W2 value-0 open marker on the child),
// plus the CLOSE trio (child MET, parent restored ACTIVE, W2 value-1 close marker)
// only when the span has returned. It is split out so the batch [DetourRows] fold
// and the live turn-end [LiveDetourRows] fold share one source of truth for the row
// shapes; callers pass an already-defaulted streamRef.
func detourSpanRows(parent Objective, sp DetourSpan, k int, streamRef string, unixMillis int64, stamp Stamp) (open, closed []Row) {
	child := Objective{
		ID:       fmt.Sprintf("%s-detour-%d", parent.ID, k+1),
		ParentID: parent.ID,
		Statement: fmt.Sprintf("detour: repair the %d-error burst at tool call %d (topics: %s)",
			sp.Errors, sp.BurstIndex, strings.Join(sp.Topics, ", ")),
		Budget: DefaultDetourBudget(),
		Status: StatusActive,
	}
	paused := parent
	paused.Status = StatusPaused
	open = []Row{
		ObjectiveRecord(child),
		ObjectiveRecord(paused),
		ScoreRecord(detourMarker(child.ID, 0, streamRef, fmt.Sprintf(
			"opened at events[%d]: %d-error burst at events[%d] + sustained shift to %s",
			sp.OpenIndex, sp.Errors, sp.BurstIndex, strings.Join(sp.Topics, ", ")), unixMillis, stamp)),
	}
	if !sp.Closed() {
		return open, nil // parent stays paused; budget enforcement is handled by the return-to-main loop (#2552)
	}
	met := child
	met.Status = StatusMet
	resumed := parent
	resumed.Status = StatusActive
	closed = []Row{
		ObjectiveRecord(met),
		ObjectiveRecord(resumed),
		ScoreRecord(detourMarker(child.ID, 1, streamRef, fmt.Sprintf(
			"closed at events[%d]: stream returned to the parent shape", sp.CloseIndex), unixMillis, stamp)),
	}
	return open, closed
}

// LiveDetourRows is the turn-end fold of [DetourRows]: given the ledger STATE
// already recorded on earlier turns, it returns only the rows that ADVANCE each
// span's detour trail, so replaying the same (growing) transcript at successive
// turn ends never double-opens a child and never re-appends a closed detour. For
// span k (child id "<parent>-detour-<k+1>"):
//
//   - child absent           -> the span's open trio, plus its close trio when the
//     span has already returned;
//   - child still ACTIVE and the span has since returned -> only the close trio —
//     the live open-on-one-turn / close-on-a-later-turn transition (child MET,
//     parent resumed ACTIVE);
//   - child already MET, or child ACTIVE while the span is still open -> nothing.
//
// Pure and deterministic like DetourRows: same (parent, spans, state) -> same rows;
// the caller owns the append. An undeclarable parent yields nil. Span numbering is
// stable across turns because DetectDetourSpans appends spans left-to-right, so an
// earlier span keeps its k as the stream grows.
func LiveDetourRows(parent Objective, spans []DetourSpan, state State, streamRef string, unixMillis int64, stamp Stamp) []Row {
	if parent.ID == "" || parent.Statement == "" {
		return nil
	}
	if streamRef == "" {
		streamRef = "tool-stream"
	}
	rows := make([]Row, 0)
	for k, sp := range spans {
		childID := fmt.Sprintf("%s-detour-%d", parent.ID, k+1)
		open, closed := detourSpanRows(parent, sp, k, streamRef, unixMillis, stamp)
		switch existing, seen := state.Objectives[childID]; {
		case !seen:
			rows = append(rows, open...)
			rows = append(rows, closed...)
		case existing.Status == StatusActive && sp.Closed():
			rows = append(rows, closed...)
		}
	}
	return rows
}

// TurnEndDetourRows is the pure turn-end producer: it detects detour spans in the
// live tool stream and, for every OPEN ROOT objective (a top-level objective with
// no parent, active or paused) in state, returns the ledger rows that advance that
// root's detour trail without double-opening on replay. It is the detour twin of
// [Sample] — pure, deterministic, and clock-free — so the cmd/fak Stop-hook wiring
// is a bounded, fail-open pass over the transcript path and the folded ledger. No
// spans, or no open root, yields nil. A detour opens under the ROOT because the
// detector reads the session-global tool stream, not a single sub-objective;
// budget enforcement and the return-to-main nudge (#2552) enforce budgets via the
// steer channel when a detour overruns.
func TurnEndDetourRows(state State, events []ToolEvent, streamRef string, unixMillis int64, stamp Stamp) []Row {
	spans := DetectDetourSpans(events)
	if len(spans) == 0 {
		return nil
	}
	rows := make([]Row, 0)
	for _, id := range openRootObjectiveIDs(state.Objectives) {
		rows = append(rows, LiveDetourRows(state.Objectives[id], spans, state, streamRef, unixMillis, stamp)...)
	}
	return rows
}

// openRootObjectiveIDs returns the ids of the open (active or paused) ROOT
// objectives — those with no parent id — in lexical order, so a detour pass is
// deterministic regardless of map iteration order. A detour child carries a parent
// id, so it is never itself a root: detours never nest.
func openRootObjectiveIDs(objectives map[string]Objective) []string {
	ids := make([]string, 0, len(objectives))
	for id, obj := range objectives {
		if obj.ParentID == "" && objectiveOpen(obj.Status) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

// detourMarker is one W2 endpoint row of a detour's repair curve.
func detourMarker(childID string, value float64, streamRef, detail string, unixMillis int64, stamp Stamp) ScoreRow {
	return ScoreRow{
		ObjectiveID: childID,
		Value:       value,
		Method:      DetourDetectorMethod,
		Version:     DetourDetectorVersion,
		Witness:     W2,
		Evidence:    []EvidenceRef{{Kind: "transcript-span", Ref: streamRef, Detail: detail}},
		UnixMillis:  unixMillis,
		SessionID:   stamp.SessionID,
		RunID:       stamp.RunID,
	}
}

// AppendDetourRows writes the detour trail to the ledger at path, in order,
// returning the count appended — the thin I/O wrapper mirroring AppendSample
// and AppendSteerDecisions. A row that fails validation stops the append and
// returns the error with the count written so far.
func AppendDetourRows(path string, rows []Row) (int, error) {
	n := 0
	for _, row := range rows {
		if err := Append(path, row); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// ReadToolStream extracts the ordered tool stream from a recorded transcript
// at path (Claude Code JSONL). Unlike the fail-open ledger reader, a missing
// or unreadable transcript is an explicit error — a detector fed nothing must
// not report "no detours" as if it had looked.
func ReadToolStream(path string) ([]ToolEvent, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseToolStream(string(b)), nil
}

// streamRecord / streamBlock are the minimal transcript shapes the extractor
// reads: tool_use blocks on assistant records, tool_result blocks on user
// records. Everything else in a record is ignored.
type streamRecord struct {
	Type    string `json:"type"`
	Message struct {
		Content json.RawMessage `json:"content"`
	} `json:"message"`
}

type streamBlock struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	IsError   bool            `json:"is_error"`
}

// ParseToolStream folds transcript JSONL into ToolEvents, one per tool_use in
// call order, with each tool_result's is_error attributed back to its call by
// tool_use_id. Tolerant like ParseLedger: malformed lines, string-content
// records, id-less calls, and unmatched results are skipped, never fatal. A
// call whose result never arrives stays IsError=false (an unanswered call is
// a stall signal, not an error one).
func ParseToolStream(content string) []ToolEvent {
	events := make([]ToolEvent, 0)
	byID := map[string]int{}
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec streamRecord
		if json.Unmarshal([]byte(line), &rec) != nil || len(rec.Message.Content) == 0 {
			continue
		}
		var blocks []streamBlock
		if json.Unmarshal(rec.Message.Content, &blocks) != nil {
			continue
		}
		for _, b := range blocks {
			switch b.Type {
			case "tool_use":
				if b.ID != "" {
					byID[b.ID] = len(events)
				}
				events = append(events, ToolEvent{Tool: b.Name, Target: streamTarget(b.Input)})
			case "tool_result":
				if idx, ok := byID[b.ToolUseID]; ok && b.IsError {
					events[idx].IsError = true
				}
			}
		}
	}
	return events
}

// streamTarget is a tool_use input's primary operand: the first non-empty of
// file_path, notebook_path, path, pattern, command. Path-like fields lead so a
// Grep's locality (path) beats its pattern as the topic proxy.
func streamTarget(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(input, &m) != nil {
		return ""
	}
	for _, key := range []string{"file_path", "notebook_path", "path", "pattern", "command"} {
		var s string
		if raw, ok := m[key]; ok && json.Unmarshal(raw, &s) == nil && s != "" {
			return s
		}
	}
	return ""
}

// ComposeReturnToMain serializes the return-to-main nudge or warn packet: detour statement,
// budget overrun details, and an explicit reference to the paused parent objective and plan.
func ComposeReturnToMain(detour, parent Objective, oc ObjectiveCurve, warn bool) string {
	var b strings.Builder
	prefix := "[fak trajctl return-to-main]"
	if warn {
		prefix = "[fak trajctl return-to-main WARN]"
	}
	fmt.Fprintf(&b, "%s Detour %q has overrun its budget: %s\n", prefix, detour.ID, oc.Detail)
	fmt.Fprintf(&b, "Detour objective: %s\n", detour.Statement)
	fmt.Fprintf(&b, "Paused parent objective %q: %s\n", parent.ID, parent.Statement)
	if len(parent.Plan) > 0 {
		b.WriteString("Parent plan state:")
		for _, p := range parent.Plan {
			if p.Title != "" {
				fmt.Fprintf(&b, " %s (%s);", p.ID, p.Title)
			} else {
				fmt.Fprintf(&b, " %s;", p.ID)
			}
		}
		b.WriteString("\n")
	}
	if pts := progressPoints(oc.Methods); len(pts) > 0 {
		if len(pts) > reAnchorExcerptPoints {
			pts = pts[len(pts)-reAnchorExcerptPoints:]
		}
		vals := make([]string, 0, len(pts))
		for _, p := range pts {
			vals = append(vals, fmt.Sprintf("%.2f", p.Value))
		}
		fmt.Fprintf(&b, "Curve excerpt (%s, last %d): %s (latest %.2f, delta %+.2f)\n",
			CommitScorerMethod, len(pts), strings.Join(vals, " -> "), oc.Latest, oc.Delta)
	}
	if warn {
		fmt.Fprintf(&b, "WARNING: Detour %q is repeatedly over budget while parent %q remains paused. Conclude or abandon the detour immediately and return to the parent objective.", detour.ID, parent.ID)
	} else {
		fmt.Fprintf(&b, "The detour has exceeded its allocated budget while parent %q remains paused. Wrap up the side-quest and return to the parent objective before your next action.", parent.ID)
	}
	return b.String()
}

// DecideReturnToMain enforces detour budgets when a detour child overruns its
// budget while its parent is paused (the return-to-main loop, #2552). Pure:
// the first overrun composes a return-to-main nudge referencing the paused parent;
// repeated overrun escalates one rung to warn; outstanding delivered warns hold.
func (s State) DecideReturnToMain(oc ObjectiveCurve) SteerDecision {
	d := SteerDecision{ObjectiveID: oc.ObjectiveID, Action: ActionNone, Signal: oc.Signal}
	if oc.Signal != SignalDetourOverrun {
		d.Reason = "regime gate: return-to-main only enforces DETOUR_OVERRUN"
		return d
	}
	obj, ok := s.Objectives[oc.ObjectiveID]
	if !ok {
		d.Reason = "regime gate: objective was never declared — nothing to re-anchor on"
		return d
	}
	parentID := oc.ParentID
	if parentID == "" {
		parentID = obj.ParentID
	}
	parent, okParent := s.Objectives[parentID]
	if parentID == "" || !okParent || parent.Status != StatusPaused {
		d.Reason = "regime gate: DETOUR_OVERRUN belongs to the return-to-main rung, not the re-anchor nudge"
		return d
	}
	steers := s.SteersFor(oc.ObjectiveID)
	deliveredNudge, deliveredWarn := detourOverrunHistory(steers)
	if deliveredWarn {
		d.Reason = "regime gate: a delivered warn is outstanding for this episode — re-arms when the curve returns to HEALTHY"
		return d
	}
	if deliveredNudge {
		d.Action = ActionWarn
		d.Reason = fmt.Sprintf("regime gate: repeated %s — %s (escalated to warn)", oc.Signal, oc.Detail)
		d.Packet = ComposeReturnToMain(obj, parent, oc, true)
		return d
	}
	d.Action = ActionNudge
	d.Reason = fmt.Sprintf("regime gate: %s — %s", oc.Signal, oc.Detail)
	d.Packet = ComposeReturnToMain(obj, parent, oc, false)
	return d
}

// detourOverrunHistory scans steer decisions for an objective in DETOUR_OVERRUN:
// a delivered nudge marks the first overrun response; a delivered warn marks
// the escalation. A decision on a HEALTHY curve re-arms the episode.
func detourOverrunHistory(steers []SteerDecision) (deliveredNudge bool, deliveredWarn bool) {
	for _, d := range steers {
		switch {
		case d.Signal == SignalHealthy:
			deliveredNudge = false
			deliveredWarn = false
		case d.Signal == SignalDetourOverrun && d.Action == ActionNudge && d.Delivered:
			deliveredNudge = true
		case d.Signal == SignalDetourOverrun && d.Action == ActionWarn && d.Delivered:
			deliveredWarn = true
		}
	}
	return deliveredNudge, deliveredWarn
}
