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
