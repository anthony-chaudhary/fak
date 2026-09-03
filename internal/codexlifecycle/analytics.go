// analytics.go — the #4767 half of this leaf: native Codex critical-path and TYPED
// tool-outcome analytics over the same rollout store the #4785 lifecycle fold reads.
//
// WHY TYPED, NOT REGEX-ONLY. A naive output parser over the observed corpus counted
// 18,408 "non-zero/error-shaped" tool results — but 675 of them are expected
// `git rev-parse -q --verify MERGE_HEAD` negatives and 283 are `wait` exit-1 control
// outcomes. Counting those as failures poisons every failure ranking. This file
// decodes the STRUCTURED tool envelope first, maps the outcome into a CLOSED
// vocabulary (the non-guard tool-failure classes #2129 defined), and classifies
// registered expected-negative probes and control outcomes SEPARATELY from genuine
// failures. Missing results are typed off the #4785 reconciled task boundary —
// an unmatched call at an interruption boundary stays a typed interrupted outcome
// with inferred confidence, never a success and never a generic error.
//
// PRIVACY CONTRACT (bodies are never retained). Ingestion streams the JSONL and
// keeps only ids, timestamps, tool names, decoded envelope numbers, apply_patch
// target paths, and a BOUNDED command head used exclusively for probe/loop
// classification. Prompts, result bodies, and agent messages are dropped at read
// time; corpus reports export classes, reasons, and hashed signatures — no raw
// commands.
package codexlifecycle

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"regexp"
	"strings"
	"time"
)

// ToolClass is the CLOSED tool-outcome vocabulary: every joined call/result pair
// maps to exactly one member. Expected negatives and control exits are first-class
// members precisely so they can never be lumped into failure counts again.
type ToolClass string

const (
	ToolOK                ToolClass = "ok"
	ToolExpectedNegative  ToolClass = "expected_negative"  // a registered probe whose non-zero exit IS the answer
	ToolControlExit       ToolClass = "control_exit"       // flow-control exit (e.g. `wait` propagating a child's status)
	ToolFailure           ToolClass = "failure"            // genuine command failure
	ToolTimeout           ToolClass = "timeout"            // killed at the harness deadline, partial state possible
	ToolMissingResult     ToolClass = "missing_result"     // call with no output despite later task evidence
	ToolMalformedEnvelope ToolClass = "malformed_envelope" // output present but its envelope is undecodable
	ToolInterrupted       ToolClass = "interrupted"        // call open at an aborted/superseded/dead task boundary
	ToolLiveTail          ToolClass = "live_tail"          // call open at the fresh tail — may genuinely still run
)

// CountsAsFailure reports whether the class belongs in a failure ranking. Expected
// negatives, control exits, and open-tail unknowns do NOT.
func (c ToolClass) CountsAsFailure() bool { return c == ToolFailure || c == ToolTimeout }

// Confidence says how the classifier knows. Observed = read from a decoded
// envelope; Assumed = no envelope semantics exist for the tool, absence of an error
// signal is being trusted; Inferred = synthesized from a reconciled task boundary.
type Confidence string

const (
	ConfidenceObserved Confidence = "observed"
	ConfidenceAssumed  Confidence = "assumed"
	ConfidenceInferred Confidence = "inferred"
)

// EnvelopeKind is the decode path that produced an Envelope.
type EnvelopeKind string

const (
	EnvelopeStructured EnvelopeKind = "structured" // JSON object output (metadata.exit_code form or tool-native JSON)
	EnvelopeText       EnvelopeKind = "text"       // "Exit code: N\nWall time: X seconds\nOutput:\n…" harness form
	EnvelopeOpaque     EnvelopeKind = "opaque"     // free text with no envelope semantics (e.g. update_plan acks)
	EnvelopeMalformed  EnvelopeKind = "malformed"  // looks like an envelope but cannot be decoded
)

// Envelope is the decoded, body-free summary of one function_call_output.
type Envelope struct {
	Kind     EnvelopeKind `json:"kind"`
	HasExit  bool         `json:"has_exit,omitempty"`
	ExitCode int          `json:"exit_code,omitempty"`
	WallS    float64      `json:"wall_s,omitempty"`
	TimedOut bool         `json:"timed_out,omitempty"`
}

var (
	textExitRE  = regexp.MustCompile(`^Exit code:\s*(-?\d+)`)
	textWallRE  = regexp.MustCompile(`Wall time:\s*([0-9.]+) seconds`)
	timeoutHead = regexp.MustCompile(`(?i)^command timed out after|^timed out after`)
)

// DecodeEnvelope decodes one output body into envelope numbers, structured form
// first, and drops the body. Only the FIRST body line is consulted for the harness
// timeout marker so result content mentioning "timed out" cannot fake a timeout.
func DecodeEnvelope(output string) Envelope {
	s := strings.TrimSpace(output)
	if s == "" {
		return Envelope{Kind: EnvelopeOpaque}
	}
	if strings.HasPrefix(s, "{") {
		var obj map[string]any
		if json.Unmarshal([]byte(s), &obj) != nil {
			return Envelope{Kind: EnvelopeMalformed}
		}
		env := Envelope{Kind: EnvelopeStructured}
		if md, ok := obj["metadata"].(map[string]any); ok {
			if code, ok := md["exit_code"].(float64); ok {
				env.HasExit = true
				env.ExitCode = int(code)
			}
			if dur, ok := md["duration_seconds"].(float64); ok {
				env.WallS = dur
			}
		}
		if body, ok := obj["output"].(string); ok {
			env.TimedOut = timeoutHead.MatchString(firstLine(body))
		}
		if env.HasExit && (env.ExitCode == 124 || env.ExitCode == 143) {
			env.TimedOut = true
		}
		return env
	}
	if strings.HasPrefix(s, "Exit code:") {
		m := textExitRE.FindStringSubmatch(s)
		if m == nil {
			return Envelope{Kind: EnvelopeMalformed}
		}
		env := Envelope{Kind: EnvelopeText, HasExit: true}
		fmt.Sscanf(m[1], "%d", &env.ExitCode)
		if w := textWallRE.FindStringSubmatch(s); w != nil {
			fmt.Sscanf(w[1], "%f", &env.WallS)
		}
		if _, body, found := strings.Cut(s, "Output:\n"); found {
			env.TimedOut = timeoutHead.MatchString(firstLine(body))
		}
		if env.ExitCode == 124 || env.ExitCode == 143 {
			env.TimedOut = true
		}
		return env
	}
	return Envelope{Kind: EnvelopeOpaque}
}

func firstLine(s string) string {
	s = strings.TrimLeft(s, "\r\n \t")
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// outcomeProbe is one REGISTERED command shape with a stable reason token. The
// registry is deliberately closed and short: a probe earns its row by being an
// intentional negative (the caller wants the non-zero answer) with DOCUMENTED
// exit semantics, never by frequency. OnlyExit, when non-zero, restricts the
// probe to that exact exit code (grep's 1-vs-2 distinction).
type outcomeProbe struct {
	Reason   string
	Re       *regexp.Regexp
	OnlyExit int
}

var expectedNegativeProbes = []outcomeProbe{
	// `git rev-parse -q --verify MERGE_HEAD` exits 1 when no merge is in progress —
	// that IS the answer the caller wanted (675 in the observed corpus).
	{"merge_head_probe", regexp.MustCompile(`(?i)\bgit\b[^|;&]*\brev-parse\b[^|;&]*\bMERGE_HEAD\b`), 0},
	// rg / grep / git grep DOCUMENT exit 1 as "no matches" — the answer a search
	// probe exists to produce. Exit 2 (a real grep error) stays a failure, so the
	// probe is exit-gated. Registered at segment start only: a command merely
	// piping through a matcher does not qualify.
	{"grep_no_match", regexp.MustCompile(`(?im)(?:^|;|&&|\|\|)[ \t]*(?:rg|grep|egrep|fgrep|git[ \t]+grep)[ \t]`), 1},
}

var controlExitProbes = []outcomeProbe{
	// The shell `wait` builtin propagates the awaited child's exit status; a
	// non-zero exit reports the CHILD's outcome, it does not mean `wait` failed.
	// Only a wait at a command-segment start counts: commands merely CONTAINING
	// "wait" (rg patterns, `--max-wait-s`, `-probe-wait`, Wait-Process pipelines)
	// stay in their own class — sweeping those in is the naive-parser defect
	// this vocabulary exists to remove.
	{"wait_control_exit", regexp.MustCompile(`(?im)(?:^|;|&&|\|\|)[ \t]*wait(?:[\s;]|$)`), 0},
}

// waitCommandRE marks calls whose WALL TIME is deliberate blocking/waiting rather
// than useful tool work — the wait/blocked bucket of the critical path.
var waitCommandRE = regexp.MustCompile(`(?i)^\s*(wait|sleep|start-sleep)\b`)

// sleepPollRE is the #2365 foreground-polling detector, ported to command heads.
var sleepPollRE = regexp.MustCompile(`(?i)^\s*(?:sleep|start-sleep)\b`)

// patchTargetRE extracts apply_patch FILE TARGETS (paths only; the patch body is
// dropped) so the #2365 edit-churn detector works on Codex event shape.
var patchTargetRE = regexp.MustCompile(`\*\*\* (?:Update|Add|Delete) File: ([^\r\n]+)`)

// ClassifyOutcome maps one decoded call/result pair into the closed vocabulary.
// commandHead is the bounded head retained at ingestion (empty for non-shell tools).
func ClassifyOutcome(commandHead string, env Envelope) (ToolClass, string, Confidence) {
	switch env.Kind {
	case EnvelopeMalformed:
		return ToolMalformedEnvelope, "undecodable_envelope", ConfidenceObserved
	case EnvelopeOpaque:
		return ToolOK, "no_envelope", ConfidenceAssumed
	}
	if env.TimedOut {
		return ToolTimeout, "timeout", ConfidenceObserved
	}
	if !env.HasExit {
		return ToolOK, "structured_no_exit", ConfidenceAssumed
	}
	if env.ExitCode == 0 {
		return ToolOK, "exit_0", ConfidenceObserved
	}
	for _, p := range expectedNegativeProbes {
		if (p.OnlyExit == 0 || p.OnlyExit == env.ExitCode) && p.Re.MatchString(commandHead) {
			return ToolExpectedNegative, p.Reason, ConfidenceObserved
		}
	}
	for _, p := range controlExitProbes {
		if (p.OnlyExit == 0 || p.OnlyExit == env.ExitCode) && p.Re.MatchString(commandHead) {
			return ToolControlExit, p.Reason, ConfidenceObserved
		}
	}
	return ToolFailure, fmt.Sprintf("exit_%d", env.ExitCode), ConfidenceObserved
}

// headBytes bounds the retained command head: enough for probe classification and
// loop signatures, never the whole command.
const headBytes = 256

// ARecord is one retained, body-free analytics record from a rollout.
type ARecord struct {
	Kind             string // task_started | task_complete | turn_aborted | function_call | function_call_output | token_count | compacted
	PayloadKind      string // the record's raw payload type ("function_call" vs "custom_tool_call", …) when one exists
	TS               time.Time
	TurnID           string
	CallID           string
	Tool             string
	Head             string   // bounded command head (classification only; corpus reports never export it)
	Targets          []string // apply_patch file targets, paths only
	Env              Envelope
	Reason           string
	DurationMS       int64
	InputTokens      int  // token_count only: payload.info.last_token_usage.input_tokens (#10662)
	GoalContinuation bool // structured harness envelope; no prompt body retained

	// ErrClasses is the #10668(c) closed status-class vocabulary reduced from a
	// TERMINAL task_complete's free text at read time (5xx / 429 / 400 only,
	// code-anchored — see errorclass.go). The text itself is dropped here, so
	// no consumer ever sees it; non-terminal records carry nil.
	ErrClasses []string
}

// ReadAnalyticsRollout streams one rollout and returns its Meta plus body-free
// analytics records in file order. Torn/non-JSON lines are skipped — the truncated
// tail is process-death evidence, not a read error.
func ReadAnalyticsRollout(r io.Reader) (Meta, []ARecord, error) {
	return readAnalyticsRollout(r, nil)
}

// ReadAnalyticsRolloutCensus is ReadAnalyticsRollout plus the #10668 sidecar:
// every event_msg payload type counted by name (including the ones this reader
// does not interpret, so upstream's future typed error/retry/terminal events
// are observed, not dropped) and the torn-tail shape. The returned Meta and
// records are identical to ReadAnalyticsRollout's; counting is never an error.
func ReadAnalyticsRolloutCensus(r io.Reader) (Meta, []ARecord, RolloutCensus, error) {
	var census RolloutCensus
	meta, records, err := readAnalyticsRollout(r, &census)
	return meta, records, census, err
}

func readAnalyticsRollout(r io.Reader, census *RolloutCensus) (Meta, []ARecord, error) {
	var meta Meta
	var out []ARecord
	haveMeta := false
	lastParsed := true

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 256*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Timestamp string `json:"timestamp"`
			Type      string `json:"type"`
			Payload   struct {
				Type             string `json:"type"`
				TurnID           string `json:"turn_id"`
				Reason           string `json:"reason"`
				DurationMS       int64  `json:"duration_ms"`
				ID               string `json:"id"`
				AltID            string `json:"session_id"`
				ModelProvider    string `json:"model_provider"`
				CLIVersion       string `json:"cli_version"`
				CWD              string `json:"cwd"`
				Originator       string `json:"originator"`
				Source           string `json:"source"`
				ThreadSource     string `json:"thread_source"`
				Role             string `json:"role"`
				LastAgentMessage string `json:"last_agent_message"`
				Content          []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
				CallID    string          `json:"call_id"`
				Name      string          `json:"name"`
				Arguments string          `json:"arguments"`
				Input     string          `json:"input"`
				Output    json.RawMessage `json:"output"`
				Info      struct {
					LastTokenUsage struct {
						InputTokens int `json:"input_tokens"`
					} `json:"last_token_usage"`
				} `json:"info"`
			} `json:"payload"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil {
			lastParsed = false
			continue
		}
		lastParsed = true
		ts, _ := time.Parse(time.RFC3339, rec.Timestamp)
		switch rec.Type {
		case "session_meta":
			if haveMeta {
				continue
			}
			haveMeta = true
			meta = Meta{
				RolloutID:    firstNonEmpty(rec.Payload.ID, rec.Payload.AltID),
				Provider:     strings.TrimSpace(rec.Payload.ModelProvider),
				CLIVersion:   strings.TrimSpace(rec.Payload.CLIVersion),
				CWD:          strings.TrimSpace(rec.Payload.CWD),
				Originator:   strings.TrimSpace(rec.Payload.Originator),
				Source:       strings.TrimSpace(rec.Payload.Source),
				ThreadSource: strings.TrimSpace(rec.Payload.ThreadSource),
			}
		case kindCompacted:
			out = append(out, ARecord{Kind: kindCompacted, TS: ts})
		case "response_item":
			if rec.Payload.Type == "message" && rec.Payload.Role == "user" {
				for _, item := range rec.Payload.Content {
					if item.Type == "input_text" && strings.Contains(item.Text, "<codex_"+"internal_"+`context source="goal">`) {
						out = append(out, ARecord{Kind: "goal_continuation", TS: ts, GoalContinuation: true})
						break
					}
				}
			}
			switch rec.Payload.Type {
			case kindToolCall, "custom_tool_call":
				head, targets := callHead(rec.Payload.Name, firstNonEmpty(rec.Payload.Arguments, rec.Payload.Input))
				out = append(out, ARecord{
					Kind: kindToolCall, PayloadKind: rec.Payload.Type, TS: ts,
					CallID: firstNonEmpty(rec.Payload.CallID, rec.Payload.ID),
					Tool:   strings.TrimSpace(rec.Payload.Name),
					Head:   head, Targets: targets,
				})
			case "function_call_output", "custom_tool_call_output":
				out = append(out, ARecord{
					Kind: "function_call_output", PayloadKind: rec.Payload.Type, TS: ts,
					CallID: firstNonEmpty(rec.Payload.CallID, rec.Payload.ID),
					Env:    DecodeEnvelope(outputText(rec.Payload.Output)),
				})
			}
		case "event_msg":
			// THE #10668 CENSUS: an unrecognized payload type is counted by
			// name and never an error — unknown events stay invisible to the
			// analytics fold, but no longer invisible to the operator. The
			// interpreted kinds are counted too, so the census is the full
			// event-type inventory, not just its unknown tail.
			census.addPayload(rec.Payload.Type)
			switch rec.Payload.Type {
			case KindStarted, KindComplete, KindAborted:
				out = append(out, ARecord{
					Kind: rec.Payload.Type, PayloadKind: rec.Payload.Type, TS: ts,
					TurnID: strings.TrimSpace(rec.Payload.TurnID),
					Reason: rec.Payload.Reason, DurationMS: rec.Payload.DurationMS,
					ErrClasses: errClassesFor(rec.Payload.Type, rec.Payload.LastAgentMessage),
				})
			case kindTokens:
				out = append(out, ARecord{Kind: kindTokens, PayloadKind: rec.Payload.Type, TS: ts,
					InputTokens: rec.Payload.Info.LastTokenUsage.InputTokens})
			}
		}
	}
	if census != nil {
		census.TornTail = !lastParsed
	}
	if err := sc.Err(); err != nil {
		return meta, out, err
	}
	return meta, out, nil
}

// errClassesFor reduces a terminal task_complete's free text to the closed
// #10668 status-class tokens, dropping the text at read time. Anything that is
// not a terminal task_complete carries no classes.
func errClassesFor(kind, lastAgentMessage string) []string {
	if kind != KindComplete {
		return nil
	}
	return ClassifyStatusClasses(lastAgentMessage)
}

// outputText unwraps a function_call_output payload's output field, which is a JSON
// string in the observed corpus but may be a structured object in other producers.
func outputText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}

// callHead extracts the bounded command head and any patch targets from a call's
// arguments, then drops the arguments.
func callHead(tool, args string) (string, []string) {
	var parsed struct {
		Command string `json:"command"`
		Input   string `json:"input"`
		Patch   string `json:"patch"`
	}
	_ = json.Unmarshal([]byte(args), &parsed)
	body := firstNonEmpty(parsed.Command, parsed.Input, parsed.Patch)
	var targets []string
	if strings.Contains(strings.ToLower(tool), "patch") || strings.Contains(body, "*** Begin Patch") {
		// The regex runs on the DECODED body (real newlines), never the raw JSON.
		for _, m := range patchTargetRE.FindAllStringSubmatch(body, -1) {
			targets = append(targets, strings.TrimSpace(m[1]))
		}
	}
	head := body
	if len(head) > headBytes {
		head = head[:headBytes]
	}
	return head, targets
}

// sigHash is the scrubbed loop signature: tool + bounded head, hashed so reports
// carry a stable identifier and never the command text.
func sigHash(tool, head string) string {
	h := fnv.New32a()
	_, _ = h.Write([]byte(tool))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(head))
	return fmt.Sprintf("%08x", h.Sum32())
}
