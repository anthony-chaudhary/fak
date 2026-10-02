package toolproc

// repeatingest.go — the INGESTION front for #4764: turn a native Codex rollout
// JSONL stream into the []CallRecord the classifier and the reuse store consume.
// This is the "stream native Codex rollout logs and normalize tool calls by tool +
// canonical arguments without retaining secrets or raw output bodies" bullet, as a
// pure, hermetic library: bytes in, normalized records out, no I/O of its own.
//
// WHAT A ROLLOUT LINE IS. Each line is {timestamp, type, payload}. The payload's
// own `type` is the record kind. A tool CALL is a `function_call` /
// `local_shell_call` / `custom_tool_call`; its RESULT is the matching
// `*_output` record, joined by `call_id`. We keep only the output's SIZE (len),
// never its body — the analytics contract — and defer secret redaction to Normalize.
//
// COMMAND EXTRACTION. A Codex shell call arrives as a `command` array, usually
// wrapped `["bash","-lc","<script>"]` (or pwsh `-c`); the SCRIPT is the real command
// the classifier reasons about, so the wrapper is unwrapped to its inner line. A
// non-shell tool (apply_patch, update_plan) keeps its tool name as the record Tool
// and falls through to CmdUnknown in the classifier — fail-closed, never reused.
//
// TOLERANT BY DESIGN. A malformed line, an unknown payload type, or a call with no
// output is skipped or scored zero-bytes, never a parse error — a rollout is an
// append-only log fak did not write, so ingestion must never crash on one bad row.

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
	"time"
)

// rolloutLine is the outer JSONL envelope: an ISO-8601 timestamp and the record
// payload. Unknown outer fields are ignored (additive evolution).
type rolloutLine struct {
	Timestamp string          `json:"timestamp"`
	Payload   json.RawMessage `json:"payload"`
}

// rolloutPayload is the union of the tool-call and tool-output record shapes we
// read. Absent fields stay zero — a payload is matched by Type, then only the
// fields that type carries are consulted.
type rolloutPayload struct {
	Type      string          `json:"type"`
	Name      string          `json:"name"`
	CallID    string          `json:"call_id"`
	Arguments string          `json:"arguments"` // function_call: a JSON string
	Input     string          `json:"input"`     // custom_tool_call: raw tool input
	Output    json.RawMessage `json:"output"`    // *_output: string or object
	PayloadN  json.RawMessage `json:"payload"`   // response_item: nested payload
	Action    *struct {
		Command []string `json:"command"`
	} `json:"action"` // local_shell_call
}

// shellCallTypes are the payload types that denote a shell command. Others are
// non-shell tools, keyed by their tool name.
var shellCallTypes = map[string]bool{
	"local_shell_call": true,
}

// shellFunctionNames are function_call names that wrap a shell command line.
// `shell_command` is the name the captured rollouts overwhelmingly use (#5120's
// replay measured it at ~95% of all calls, matching #4764's own audit); omitting it
// sent every shell call down the non-shell branch, where the command is never
// extracted and the whole inventory folds into one UNKNOWN bucket.
var shellFunctionNames = map[string]bool{
	"shell": true, "bash": true, "container.exec": true, "local_shell": true,
	"exec_command": true, "shell_command": true,
}

// outputTypes map an output record to the call it completes.
var outputTypes = map[string]bool{
	"function_call_output": true, "custom_tool_call_output": true, "local_shell_call_output": true,
}

// pendingCall is a tool call awaiting its output, kept in stream order.
type pendingCall struct {
	callID string
	tool   string
	raw    string
	atMS   int64
}

// IngestRollout streams a native Codex rollout JSONL log into normalized
// CallRecords — one per tool call, its OutputBytes joined from the matching output
// record by call_id. It retains no output body (only the size) and no secret
// (Normalize redacts downstream). Records are returned in call order; a call whose
// output never arrives is emitted with OutputBytes 0.
func IngestRollout(r io.Reader) []CallRecord {
	br := bufio.NewReader(r)
	var calls []pendingCall
	outBytes := map[string]int64{}

	for {
		line, err := br.ReadString('\n')
		if s := strings.TrimSpace(line); s != "" {
			ingestLine(s, &calls, outBytes)
		}
		if err != nil {
			break // io.EOF or a read error: stop after consuming the last partial line
		}
	}

	recs := make([]CallRecord, 0, len(calls))
	for _, c := range calls {
		recs = append(recs, CallRecord{
			Tool:        c.tool,
			Raw:         c.raw,
			AtMS:        c.atMS,
			OutputBytes: outBytes[c.callID], // 0 if the output never arrived
		})
	}
	return recs
}

// ingestLine folds one JSONL row into the call list / output map. Unknown or
// malformed rows are silently skipped.
func ingestLine(s string, calls *[]pendingCall, outBytes map[string]int64) {
	var outer rolloutLine
	if err := json.Unmarshal([]byte(s), &outer); err != nil || len(outer.Payload) == 0 {
		return
	}
	var p rolloutPayload
	if err := json.Unmarshal(outer.Payload, &p); err != nil {
		return
	}
	switch {
	case outputTypes[p.Type]:
		if p.CallID != "" {
			outBytes[p.CallID] += outputLen(p.Output)
		}
	case p.Type == "function_call", p.Type == "custom_tool_call", shellCallTypes[p.Type]:
		tool, raw := callToolAndRaw(p)
		if raw == "" && tool == "" {
			return
		}
		*calls = append(*calls, pendingCall{
			callID: p.CallID,
			tool:   tool,
			raw:    raw,
			atMS:   parseRolloutTS(outer.Timestamp),
		})
	}
}

// callToolAndRaw derives the (Tool, Raw) pair the classifier consumes from a call
// payload: a shell call becomes Tool "shell_command" + the unwrapped command line; a
// non-shell tool keeps its name as Tool and a best-effort argument line as Raw.
func callToolAndRaw(p rolloutPayload) (tool, raw string) {
	// A local_shell_call carries the command array directly.
	if p.Action != nil && len(p.Action.Command) > 0 {
		return "shell_command", unwrapShell(p.Action.Command)
	}
	// A function_call whose name is a shell verb carries {"command": ...} in args.
	if p.Type == "function_call" && shellFunctionNames[strings.ToLower(p.Name)] {
		if cmd := commandFromArgs(p.Arguments); cmd != "" {
			return "shell_command", cmd
		}
	}
	// A non-shell tool: key by its name; the raw is the tool name plus a compact
	// argument tail so repeats of the same tool-call still fold.
	name := strings.TrimSpace(p.Name)
	if name == "" {
		name = p.Type
	}
	arg := strings.TrimSpace(p.Arguments)
	if arg == "" {
		arg = strings.TrimSpace(p.Input)
	}
	raw = name
	if arg != "" {
		raw = name + " " + collapseWS(arg)
	}
	return name, raw
}

// commandFromArgs pulls the command line out of a shell function_call's JSON
// arguments. It accepts {"command": ["bash","-lc","git status"]} and
// {"command": "git status"}.
func commandFromArgs(argsJSON string) string {
	if strings.TrimSpace(argsJSON) == "" {
		return ""
	}
	var a struct {
		Command json.RawMessage `json:"command"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &a); err != nil || len(a.Command) == 0 {
		return ""
	}
	var arr []string
	if err := json.Unmarshal(a.Command, &arr); err == nil {
		return unwrapShell(arr)
	}
	var one string
	if err := json.Unmarshal(a.Command, &one); err == nil {
		return strings.TrimSpace(one)
	}
	return ""
}

// unwrapShell reduces a command array to the command the classifier reasons about:
// the inner script of a `bash -lc <script>` / `sh -c` / `pwsh -Command` wrapper, else
// the whole array joined. This is what lets `git status` inside a bash wrapper
// classify as a mutable query rather than as the program `bash`.
func unwrapShell(cmd []string) string {
	if len(cmd) == 0 {
		return ""
	}
	shell := strings.ToLower(baseName(cmd[0]))
	switch shell {
	case "bash", "sh", "zsh", "dash":
		if i := indexFlag(cmd, "-lc", "-c", "-lic"); i >= 0 && i+1 < len(cmd) {
			return strings.TrimSpace(cmd[i+1])
		}
	case "pwsh", "powershell", "powershell.exe", "pwsh.exe":
		if i := indexFlag(cmd, "-command", "-c"); i >= 0 && i+1 < len(cmd) {
			return strings.TrimSpace(cmd[i+1])
		}
	}
	return strings.TrimSpace(strings.Join(cmd, " "))
}

// indexFlag returns the index of the first token matching any of flags
// (case-insensitive), or -1.
func indexFlag(cmd []string, flags ...string) int {
	want := map[string]bool{}
	for _, f := range flags {
		want[f] = true
	}
	for i, t := range cmd {
		if want[strings.ToLower(t)] {
			return i
		}
	}
	return -1
}

// outputLen returns the size of an output record's payload without retaining it: the
// length of the string when the output is a JSON string, else the raw byte length.
func outputLen(raw json.RawMessage) int64 {
	if len(raw) == 0 {
		return 0
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return int64(len(s))
	}
	return int64(len(raw))
}

// collapseWS squeezes runs of whitespace to single spaces so an argument line folds
// across formatting differences.
func collapseWS(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// parseRolloutTS parses an ISO-8601 rollout timestamp to unix milliseconds; 0 on
// failure (a record with no usable clock still classifies, just without spacing).
func parseRolloutTS(s string) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.UnixMilli()
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UnixMilli()
	}
	return 0
}

// ---------------------------------------------------------------------------
// Evidence view (#13416): a LOSSLESS, coverage-bearing companion to
// IngestRollout. IngestRollout is deliberately lossy and tolerant — it
// normalizes arguments to a Raw line and skips unknown/malformed rows — which is
// right for analytics but cannot distinguish "a completely understood zero-call
// transcript" from "we did not understand this input". IngestRolloutEvidence
// answers that question without changing the legacy API: it preserves original
// call inputs and reports explicit coverage counters, and it NEVER promotes an
// unsupported or failed input to an observed zero.
// ---------------------------------------------------------------------------

// RolloutCallKind is the wire shape a call arrived as.
type RolloutCallKind string

const (
	RolloutKindFunctionCall RolloutCallKind = "function_call"    // arguments live in Arguments
	RolloutKindCustomTool   RolloutCallKind = "custom_tool_call" // input lives in Input
	RolloutKindLocalShell   RolloutCallKind = "local_shell_call" // command lives in Command
)

// RolloutCall is one tool call preserved at its original wire granularity.
// Unlike CallRecord.Raw it is never normalized: Arguments/Input are the exact
// bytes the rollout carried, and Command is a copied array (not a synthesized
// string), so a consumer can reason about losslessness. No output body is ever
// retained.
type RolloutCall struct {
	Kind      RolloutCallKind `json:"kind"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"` // function_call: unchanged
	Input     string          `json:"input,omitempty"`     // custom_tool_call: unchanged
	Command   []string        `json:"command,omitempty"`   // local_shell_call: copied
	CallID    string          `json:"call_id,omitempty"`
	AtMS      int64           `json:"at_ms,omitempty"`
}

// RolloutEvidence is the typed coverage verdict over a Codex rollout stream.
// Complete is true only for a recognized stream read through EOF with every
// nonblank record classified under an explicit supported schema, with at least
// one record read — including a legitimately empty call list. Any unknown
// top-level kind or response_item payload type, any malformed row, empty input,
// or a non-EOF read error makes Complete false and is counted so a consumer can
// tell "zero" from "unknown".
type RolloutEvidence struct {
	Calls       []RolloutCall `json:"calls"`
	Complete    bool          `json:"complete"`
	Records     int           `json:"records"`     // nonblank rows read
	Unsupported int           `json:"unsupported"` // unknown top-level/response_item kinds
	Malformed   int           `json:"malformed"`   // rows that were not valid JSON or lacked a payload
	ReadError   error         `json:"-"`
}

// rolloutEvidenceRecord is the per-row classification. Any record that reaches a
// supported branch is recognized; call records additionally emit a RolloutCall.
type rolloutEvidenceRecord struct {
	call       *RolloutCall
	recognized bool
	malformed  bool
}

// evidenceEnvelopeTypes are known non-tool top-level envelopes. They contribute
// coverage but never calls or output bodies.
var evidenceEnvelopeTypes = map[string]bool{
	"session_meta": true, "turn_context": true, "event_msg": true, "compacted": true,
}

// evidenceResponseItemTypes are known nested response_item payload types that
// carry no call of their own (messages, results, reasoning). A call-kind nested
// payload is still ingested as a call.
var evidenceResponseItemTypes = map[string]bool{
	"message": true, "function_call_output": true, "custom_tool_call_output": true,
	"local_shell_call_output": true, "reasoning": true,
}

// IngestRolloutEvidence streams a native Codex rollout JSONL log and returns a
// lossless call view plus an explicit coverage verdict. It shares the scanner and
// decoder with IngestRollout but never normalizes inputs and never silently
// ignores a row it could not classify (see RolloutEvidence.Complete).
func IngestRolloutEvidence(r io.Reader) RolloutEvidence {
	br := bufio.NewReader(r)
	var ev RolloutEvidence

	for {
		line, err := br.ReadString('\n')
		if s := strings.TrimSpace(line); s != "" {
			ev.Records++
			rec := classifyEvidenceRow(s)
			switch {
			case rec.malformed:
				ev.Malformed++
			case rec.call != nil:
				ev.Calls = append(ev.Calls, *rec.call)
			case rec.recognized:
				// coverage-only row (envelope, output, message)
			default:
				ev.Unsupported++
			}
		}
		if err != nil {
			if err != io.EOF {
				ev.ReadError = err
			}
			break
		}
	}

	// Complete only when the stream was fully understood: at least one record
	// read, no unsupported rows, no malformed rows, and no read error. A
	// malformed row and an unknown kind are tracked separately so the consumer
	// can tell them apart, but either makes the coverage incomplete.
	ev.Complete = ev.ReadError == nil && ev.Unsupported == 0 && ev.Malformed == 0 && ev.Records > 0
	return ev
}

// classifyEvidenceRow parses one JSONL row and classifies it. It mirrors
// ingestLine's tolerance (never panics) but reports rather than discards: a row
// that cannot be parsed is left unrecognized so the caller counts it Malformed.
func classifyEvidenceRow(s string) rolloutEvidenceRecord {
	var outer rolloutLine
	if err := json.Unmarshal([]byte(s), &outer); err != nil || len(outer.Payload) == 0 {
		return rolloutEvidenceRecord{malformed: true} // not JSON, or no payload object
	}
	var p rolloutPayload
	if err := json.Unmarshal(outer.Payload, &p); err != nil {
		return rolloutEvidenceRecord{malformed: true}
	}
	// A response_item wraps its own payload; unwrap one level so a nested
	// function_call is still a call and a nested message still contributes
	// coverage. An unknown nested type stays unsupported.
	if p.Type == "response_item" {
		if len(p.PayloadN) == 0 {
			return rolloutEvidenceRecord{recognized: true} // bare envelope
		}
		var nested rolloutPayload
		if err := json.Unmarshal(p.PayloadN, &nested); err != nil {
			return rolloutEvidenceRecord{malformed: true} // a malformed nested payload
		}
		if nested.Type == "" {
			return rolloutEvidenceRecord{recognized: true}
		}
		return classifyEvidencePayload(nested, outer.Timestamp)
	}
	return classifyEvidencePayload(p, outer.Timestamp)
}

// classifyEvidencePayload classifies a single (possibly unwrapped) payload.
func classifyEvidencePayload(p rolloutPayload, ts string) rolloutEvidenceRecord {
	switch {
	case evidenceEnvelopeTypes[p.Type]:
		return rolloutEvidenceRecord{recognized: true}
	case outputTypes[p.Type]:
		return rolloutEvidenceRecord{recognized: true} // coverage only; body never retained
	case p.Type == "function_call":
		return rolloutEvidenceRecord{recognized: true, call: &RolloutCall{
			Kind: RolloutKindFunctionCall, Name: p.Name, Arguments: p.Arguments,
			CallID: p.CallID, AtMS: parseRolloutTS(ts),
		}}
	case p.Type == "custom_tool_call":
		return rolloutEvidenceRecord{recognized: true, call: &RolloutCall{
			Kind: RolloutKindCustomTool, Name: p.Name, Input: p.Input,
			CallID: p.CallID, AtMS: parseRolloutTS(ts),
		}}
	case shellCallTypes[p.Type]:
		var cmd []string
		if p.Action != nil && len(p.Action.Command) > 0 {
			cmd = append([]string(nil), p.Action.Command...) // copy: never alias the decoder buffer
		}
		return rolloutEvidenceRecord{recognized: true, call: &RolloutCall{
			Kind: RolloutKindLocalShell, Name: p.Name, Command: cmd,
			CallID: p.CallID, AtMS: parseRolloutTS(ts),
		}}
	case evidenceResponseItemTypes[p.Type]:
		return rolloutEvidenceRecord{recognized: true}
	}
	return rolloutEvidenceRecord{} // unknown top-level kind
}
