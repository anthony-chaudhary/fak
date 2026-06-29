package sessionobs

// ingest.go -- the pure per-transcript fold: turn ONE Claude Code session
// transcript (JSONL) into a scrubbed Record. This is the "collect more data"
// engine that climbs the corpus from rung 1 (capture: bytes on disk) to rung 3
// (link: a record tied to its outcome).
//
// The split of duties: FoldTranscript reads one transcript stream and emits the
// scrubbed Record plus the Evidence the outcome is derived from. It is PURE over an
// io.Reader, so it tests against a fixture and never touches the filesystem or git.
// The impure shell (a `fak sessions` command) owns discovery across the host's
// .claude-* homes and the one fact the transcript cannot self-certify -- whether a
// committed SHA is still in git history (witnessed vs reverted) -- and passes that
// back through Classify.
//
// THE PRECISE LINK. The hard part of session->outcome linking is usually fuzzy
// time-window correlation. We avoid it: `git commit` prints "[<branch> <sha>]
// <subject>" to stdout, which the harness captures verbatim into the Bash
// tool_result. So a session NAMES the commits it landed in its own transcript. The
// fold extracts those SHAs; the outcome link is evidence, not a guess.

import (
	"bufio"
	"encoding/json"
	"io"
	"regexp"
	"strings"
)

// FoldMeta is the out-of-band context the shell knows about a transcript that the
// bytes do not carry in a single convenient place (the session id from the filename,
// which account home produced it). It is stamped onto the Record; none of it is prose.
type FoldMeta struct {
	SessionID string
	Namespace string
	Account   string
}

// Evidence is the in-transcript proof the fold extracts -- the precise signals an
// outcome is derived from, kept separate from the Record so a caller can audit WHY a
// session was classified the way it was. CommitSHAs are the non-fuzzy outcome link.
type Evidence struct {
	CommitSHAs    []string // distinct SHAs from "[<branch> <sha>] <subject>" git-commit success markers
	Mutated       bool     // an Edit / Write / NotebookEdit ran (the session changed the tree)
	Interrupts    int      // assistant turns the user/harness interrupted
	GoalEvents    int      // /goal directives the session ran under
	StopMarks     int      // goal-block / stop-hook system-reminders (a blocked Stop)
	ToolErrors    int      // is_error tool_result blocks (a tool call that failed)
	GuardRefusals int      // [fak] guard DENY/QUARANTINE banners the gateway injected into a turn
}

// committed reports whether the session landed at least one commit.
func (e Evidence) committed() bool { return len(e.CommitSHAs) > 0 }

// stopped reports whether the session shows a waste-side terminal signal: a blocked
// Stop or an interrupt. (committed takes precedence in Classify, so a session that
// both committed and was later interrupted still reads as a ship.)
func (e Evidence) stopped() bool { return e.StopMarks > 0 || e.Interrupts > 0 }

// Classify maps the extracted Evidence to an Outcome via the shared ClassifyOutcome
// policy. witnessed is the one bit the fold cannot know -- the shell sets it true
// when a committed SHA is still reachable in git history (survived, not reverted).
func Classify(e Evidence, witnessed bool) Outcome {
	return ClassifyOutcome(e.committed(), witnessed, e.stopped(), e.Mutated)
}

// readOnlyTools mirrors tools/session_audit.py: tools that observe but never mutate
// or spawn. The read-only fraction is a cheap behavior feature (an all-Read session
// that ships nothing looks very different from a tight edit/commit loop).
var readOnlyTools = map[string]bool{
	"Read": true, "Glob": true, "Grep": true, "LS": true, "NotebookRead": true,
	"WebFetch": true, "WebSearch": true, "TodoRead": true, "ToolSearch": true,
	"Monitor": true, "TaskGet": true, "TaskList": true, "TaskOutput": true,
	"ReadMcpResourceTool": true, "ListMcpResourcesTool": true, "ReadMcpResourceDirTool": true,
}

// mutatingTools change the working tree directly (Bash is handled separately -- only
// a git-commit Bash counts toward the commit link, but any Bash is non-read-only).
var mutatingTools = map[string]bool{
	"Edit": true, "Write": true, "NotebookEdit": true,
}

// commitMarker matches the line `git commit` prints on success: "[main 809c5339] ..."
// or "[main (root-commit) abcdef0] ...". The branch token is non-greedy up to the
// 7-40 hex SHA; we capture the SHA. Anchored to a line start after optional space.
var commitMarker = regexp.MustCompile(`\[[^\]]*?\b([0-9a-f]{7,40})\]`)

// goalDirective matches a /goal slash-command wrapper or a typed "/goal" prompt.
var goalDirective = regexp.MustCompile(`(?i)(<command-name>\s*/?goal|^\s*/goal\b)`)

// guardRefusalMarkers are the literal banners the fak gateway injects into an ASSISTANT
// turn when the kernel DENIES a proposed tool call or QUARANTINES an inbound tool result
// -- the guard-friction signal the contrast ranks first (see contrast.go). They are the
// emission strings of internal/gateway/{http.go:adjudicationNote/denySummary,
// messages.go:resultAdmissionNote}. We scan them ONLY in assistant text, where the
// gateway actually emits them; a tool_result that merely QUOTES one (a Read of this file
// or a gateway test) lands in a user block and so cannot be mistaken for a real refusal.
// Bare "[fak]" is deliberately NOT a marker -- it also prefixes "[fak] compacted" and
// "[fak] ctxview-elided" notes, which are not refusals.
//
// NON-CIRCULARITY: GuardRefusals is a behavior FEATURE the value-vs-waste contrast ranks,
// so it must never feed the cohort-defining stopped() evidence -- if a refusal defined the
// waste cohort, "refusals separate waste" would be tautology, not a finding. Count it as a
// signal; never classify an outcome on it. (A refusal that escalates to a blocked Stop is
// already counted on the StopMarks pathway.)
var guardRefusalMarkers = []string{
	"[fak] refused",                              // adjudicationNote: proposed call denied (Anthropic wire)
	"refused by the fak kernel",                  // denySummary: all-denied turn (fak-unaware wires)
	"held out of context as a safety precaution", // resultAdmissionNote: quarantine page-out
}

// hasGuardRefusal reports whether an assistant text block carries a guard-refusal banner.
func hasGuardRefusal(s string) bool {
	for _, m := range guardRefusalMarkers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

// transcriptLine is the minimal projection of a transcript JSONL record the fold
// needs. Everything else (thinking text, attachments, file-history snapshots) is
// ignored. Decoding is defensive: a wrong-shaped field is skipped, never panics.
type transcriptLine struct {
	Type                 string          `json:"type"`
	Message              json.RawMessage `json:"message"`
	InterruptedMessageID string          `json:"interruptedMessageId"`
}

type transcriptMessage struct {
	ID         string          `json:"id"`
	Role       string          `json:"role"`
	Model      string          `json:"model"`
	StopReason string          `json:"stop_reason"`
	Usage      transcriptUsage `json:"usage"`
	Content    json.RawMessage `json:"content"` // string OR []block
}

type transcriptUsage struct {
	OutputTokens int64 `json:"output_tokens"`
}

type contentBlock struct {
	Type    string          `json:"type"`
	Name    string          `json:"name"`    // tool_use
	Input   json.RawMessage `json:"input"`   // tool_use
	Content json.RawMessage `json:"content"` // tool_result: string OR []block
	Text    string          `json:"text"`    // assistant text block (carries the gateway's [fak] notes)
	IsError bool            `json:"is_error"`
}

// FoldTranscript streams one transcript and folds it into a scrubbed Record plus the
// Evidence the outcome rests on. Outcome is left Unknown here; the caller sets it with
// Classify once it knows the witnessed bit. A malformed line is skipped (a torn final
// line never aborts the fold). It never retains prose: only counts and the commit SHAs.
func FoldTranscript(r io.Reader, meta FoldMeta) (Record, Evidence) {
	rec := Record{SessionID: meta.SessionID, Namespace: meta.Namespace, Account: meta.Account}
	var ev Evidence
	seenTurns := map[string]bool{} // de-dup billed assistant turns by message.id (see session_audit)
	shaSet := map[string]bool{}

	// A transcript line (a browser snapshot, a huge Read) can be many MB, past any
	// fixed Scanner cap; ReadString grows unbounded so no session is silently truncated.
	br := bufio.NewReader(r)
	for {
		raw, err := br.ReadString('\n')
		line := strings.TrimSpace(raw)
		if line == "" {
			if err != nil {
				break
			}
			continue
		}
		var tl transcriptLine
		if json.Unmarshal([]byte(line), &tl) != nil {
			continue
		}
		if tl.InterruptedMessageID != "" {
			ev.Interrupts++
		}
		if len(tl.Message) == 0 {
			continue
		}
		var msg transcriptMessage
		if json.Unmarshal(tl.Message, &msg) != nil {
			continue
		}

		switch tl.Type {
		case "assistant":
			if msg.ID != "" {
				if seenTurns[msg.ID] {
					continue // already folded this billed turn
				}
				seenTurns[msg.ID] = true
			}
			rec.AssistantTurns++
			rec.OutputTokens += msg.Usage.OutputTokens
			if msg.StopReason == "interrupted" {
				ev.Interrupts++
			}
			foldAssistantContent(msg.Content, &rec, &ev)
		case "user":
			foldUserContent(msg.Content, &ev, shaSet)
		}
	}
	ev.CommitSHAs = sortedKeys(shaSet)
	// Promote the commit count into the committable Signals (the loop's strongest feature).
	rec.Signals.Commits = len(ev.CommitSHAs)
	rec.Signals.Interrupts = ev.Interrupts
	rec.Signals.GoalEvents = ev.GoalEvents
	rec.Signals.StopEvents = ev.StopMarks
	rec.Signals.ToolErrors = ev.ToolErrors
	rec.Signals.GuardRefusals = ev.GuardRefusals
	return rec, ev
}

// foldAssistantContent walks an assistant message's content blocks, counting tool
// calls (read-only vs mutating) and the guard-refusal banners the gateway prepends as
// text. One refusal-bearing text block counts as one guard-friction event (the gateway
// emits at most one note per turn), so the count tracks refusal turns, not refused calls.
func foldAssistantContent(raw json.RawMessage, rec *Record, ev *Evidence) {
	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return
	}
	for _, b := range blocks {
		switch b.Type {
		case "tool_use":
			rec.ToolCalls++
			if readOnlyTools[b.Name] {
				rec.ReadOnlyCalls++
			}
			if mutatingTools[b.Name] {
				ev.Mutated = true
			}
		case "text":
			if hasGuardRefusal(b.Text) {
				ev.GuardRefusals++
			}
		}
	}
}

// foldUserContent walks a user message. A string body is a typed prompt / hook /
// reminder -- scanned for a /goal directive and a stop-hook block. A list body holds
// tool_result blocks -- scanned for the git-commit SHA marker and tool errors.
func foldUserContent(raw json.RawMessage, ev *Evidence, shaSet map[string]bool) {
	// Try the string form first (a typed prompt or a hook/reminder).
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if goalDirective.MatchString(s) {
			ev.GoalEvents++
		}
		if strings.Contains(s, "Stop hook") || strings.Contains(s, "blocked stopping") ||
			strings.Contains(s, "block stopping until") {
			ev.StopMarks++
		}
		return
	}
	// Otherwise the list form: tool_result blocks.
	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return
	}
	for _, b := range blocks {
		if b.Type != "tool_result" {
			continue
		}
		if b.IsError {
			ev.ToolErrors++ // a tool call that failed -- the friction signal the contrast ranks
		}
		text := blockText(b.Content)
		for _, m := range commitMarker.FindAllStringSubmatch(text, -1) {
			if len(m) > 1 {
				shaSet[m[1]] = true
			}
		}
	}
}

// blockText flattens a tool_result content field (string OR []{text|content}) to a
// plain string for marker scanning. It keeps no reference to the bytes after return.
func blockText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	var sb strings.Builder
	for _, b := range blocks {
		var t string
		if json.Unmarshal(b.Content, &t) == nil {
			sb.WriteString(t)
			sb.WriteByte('\n')
		}
	}
	return sb.String()
}

// sortedKeys returns the map keys in lexical order (deterministic CommitSHAs).
func sortedKeys(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	// insertion sort: SHA sets are tiny (a session lands a handful of commits).
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
