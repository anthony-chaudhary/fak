package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// FSMState represents the state of a StreamingToolFSM as it processes token chunks.
type FSMState int

// State is an alias for FSMState.
type State = FSMState

const (
	// StateScanning: initially scanning tokens/text, waiting to detect a tool call start or pattern.
	StateScanning FSMState = iota
	// StateToolIdentified: tool name has been extracted and speculatability evaluated.
	StateToolIdentified
	// StateArgumentsStreaming: tool arguments are currently streaming in.
	StateArgumentsStreaming
	// StateCompleted: tool call syntax and arguments completed and parsed.
	StateCompleted
	// StateFailed: parsing encountered an unrecoverable syntax error or malformed structure.
	StateFailed
	// StateSquashed: speculative execution was squashed (cancelled/discarded).
	StateSquashed
)

// String returns the string representation of the FSMState.
func (s FSMState) String() string {
	switch s {
	case StateScanning:
		return "StateScanning"
	case StateToolIdentified:
		return "StateToolIdentified"
	case StateArgumentsStreaming:
		return "StateArgumentsStreaming"
	case StateCompleted:
		return "StateCompleted"
	case StateFailed:
		return "StateFailed"
	case StateSquashed:
		return "StateSquashed"
	default:
		return "StateUnknown"
	}
}

var (
	// ErrSpeculationMismatch is returned when authoritative arguments diverge from speculative arguments.
	ErrSpeculationMismatch = errors.New("streaming tool fsm: speculation arguments mismatch")
	// ErrSpeculationSquashed is returned when speculative execution was squashed.
	ErrSpeculationSquashed = errors.New("streaming tool fsm: speculation was squashed")
	// ErrNotSpeculatable is returned when speculative dispatch is attempted on an effectful/non-read-only tool.
	ErrNotSpeculatable = errors.New("streaming tool fsm: tool is not speculatable")
	// ErrNoSpeculativeDispatch is returned when Commit is called without a speculative dispatch in flight.
	ErrNoSpeculativeDispatch = errors.New("streaming tool fsm: no speculative dispatch in flight")
	// ErrToolCallMalformed is returned when tool call syntax is malformed.
	ErrToolCallMalformed = errors.New("streaming tool fsm: malformed tool call")
)

var (
	reJSONName     = regexp.MustCompile(`"(?:name|tool)"\s*:\s*"([^"]+)"`)
	reFuncCallName = regexp.MustCompile(`"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"`)
	reXMLFunc      = regexp.MustCompile(`<function=([^>\s]+)>`)
	reXMLParam     = regexp.MustCompile(`<parameter=([^>]+)>([\s\S]*?)(?:</parameter>|$)`)
	rePythonCall   = regexp.MustCompile(`<tool_call>\s*([a-zA-Z0-9_.-]+)\s*\(`)
)

// SpeculativeRunner executes a tool call in the background during speculative dispatch.
type SpeculativeRunner interface {
	Execute(ctx context.Context, tool string, args string) (string, error)
}

// SpeculativeRunnerFunc adapts a plain function to the SpeculativeRunner interface.
type SpeculativeRunnerFunc func(ctx context.Context, tool string, args string) (string, error)

func (f SpeculativeRunnerFunc) Execute(ctx context.Context, tool string, args string) (string, error) {
	return f(ctx, tool, args)
}

// FSMOption configures a StreamingToolFSM instance.
type FSMOption func(*StreamingToolFSM)

// WithSpeculatableFunc supplies a custom predicate for evaluating tool speculatability.
func WithSpeculatableFunc(fn func(tool, args string) (vdso.SpecReason, bool)) FSMOption {
	return func(fsm *StreamingToolFSM) {
		fsm.speculatableFn = fn
	}
}

// WithToolDefinitions seeds tool declarations that can inform speculatability hints.
func WithToolDefinitions(defs []ToolDef) FSMOption {
	return func(fsm *StreamingToolFSM) {
		fsm.toolDefs = append(fsm.toolDefs, defs...)
	}
}

// StreamingToolFSM is an incremental streaming lexer/parser and speculative dispatch
// state machine. It consumes token chunks as they arrive from the model, identifies
// tool calls within early tokens (typically 15-25 tokens), checks effect-free speculatability
// via vdso.Speculatable, launches background speculative execution, and commits in 0ms on match
// or squashes on mismatch. Thread-safe for concurrent operations.
type StreamingToolFSM struct {
	mu sync.Mutex

	state FSMState
	// Config options
	speculatableFn func(tool, args string) (vdso.SpecReason, bool)
	toolDefs       []ToolDef

	// Parser state
	buffer              strings.Builder
	tokenCount          int
	toolCallStartToken  int
	toolIdentifiedToken int
	toolName            string
	rawArgs             string
	earlyArgs           string
	isSpeculatable      bool
	specReason          vdso.SpecReason

	// Speculative execution lifecycle
	dispatched   bool
	specTool     string
	specArgs     string
	specResult   string
	specErr      error
	squashReason string
	cancel       context.CancelFunc
	doneCh       chan struct{}
}

// NewStreamingToolFSM creates an initialized StreamingToolFSM in StateScanning.
func NewStreamingToolFSM(opts ...FSMOption) *StreamingToolFSM {
	fsm := &StreamingToolFSM{
		state: StateScanning,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(fsm)
		}
	}
	return fsm
}

// State returns the current state machine state.
func (fsm *StreamingToolFSM) State() FSMState {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.state
}

// ToolName returns the identified tool name, or empty if not yet identified.
func (fsm *StreamingToolFSM) ToolName() string {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.toolName
}

// EarlyArgs returns early parsed arguments extracted during streaming.
func (fsm *StreamingToolFSM) EarlyArgs() string {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.earlyArgs
}

// RawArgs returns the completed argument string, or empty if not yet complete.
func (fsm *StreamingToolFSM) RawArgs() string {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.rawArgs
}

// FinalArgs returns the final parsed arguments (alias for RawArgs).
func (fsm *StreamingToolFSM) FinalArgs() string {
	return fsm.RawArgs()
}

// IsSpeculatable returns whether the identified tool call is safe for effect-free speculative dispatch.
func (fsm *StreamingToolFSM) IsSpeculatable() bool {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.isSpeculatable
}

// SpecReason returns the closed refusal or approval verdict from the speculatable gate.
func (fsm *StreamingToolFSM) SpecReason() vdso.SpecReason {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.specReason
}

// TokenCount returns the total number of token chunks fed to the FSM.
func (fsm *StreamingToolFSM) TokenCount() int {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.tokenCount
}

// ToolIdentifiedToken returns the token count at which the tool name was identified.
func (fsm *StreamingToolFSM) ToolIdentifiedToken() int {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.toolIdentifiedToken
}

// IsDispatched returns whether speculative dispatch has been launched.
func (fsm *StreamingToolFSM) IsDispatched() bool {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.dispatched
}

// SquashReason returns the reason speculative execution was squashed, if any.
func (fsm *StreamingToolFSM) SquashReason() string {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.squashReason
}

// Buffer returns the accumulated raw stream buffer.
func (fsm *StreamingToolFSM) Buffer() string {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()
	return fsm.buffer.String()
}

// Reset clears the FSM back to StateScanning, cancelling any background speculative dispatch.
func (fsm *StreamingToolFSM) Reset() {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if fsm.cancel != nil {
		fsm.cancel()
		fsm.cancel = nil
	}
	fsm.state = StateScanning
	fsm.buffer.Reset()
	fsm.tokenCount = 0
	fsm.toolCallStartToken = 0
	fsm.toolIdentifiedToken = 0
	fsm.toolName = ""
	fsm.rawArgs = ""
	fsm.earlyArgs = ""
	fsm.isSpeculatable = false
	fsm.specReason = vdso.SpecRefusedNilCall
	fsm.dispatched = false
	fsm.specTool = ""
	fsm.specArgs = ""
	fsm.specResult = ""
	fsm.specErr = nil
	fsm.squashReason = ""
	fsm.doneCh = nil
}

// Feed consumes an incremental chunk of tokens from the model stream.
func (fsm *StreamingToolFSM) Feed(chunk string) error {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if fsm.state == StateSquashed || fsm.state == StateFailed {
		return nil
	}

	fsm.tokenCount++
	fsm.buffer.WriteString(chunk)
	buf := fsm.buffer.String()

	// 1. If scanning, look for tool call indicators and extract tool name
	if fsm.state == StateScanning {
		hasTrigger := strings.Contains(buf, "<tool_call>") ||
			strings.Contains(buf, "<function_call>") ||
			strings.Contains(buf, "<function=") ||
			strings.Contains(buf, `"name"`) ||
			strings.Contains(buf, `"tool"`) ||
			strings.Contains(buf, `"function"`)

		if hasTrigger && fsm.toolCallStartToken == 0 {
			fsm.toolCallStartToken = fsm.tokenCount
		}

		var toolName string
		if m := reXMLFunc.FindStringSubmatch(buf); len(m) > 1 {
			toolName = m[1]
		} else if m := reFuncCallName.FindStringSubmatch(buf); len(m) > 1 {
			toolName = m[1]
		} else if m := reJSONName.FindStringSubmatch(buf); len(m) > 1 {
			toolName = m[1]
		} else if m := rePythonCall.FindStringSubmatch(buf); len(m) > 1 {
			toolName = m[1]
		}

		if toolName != "" {
			fsm.toolName = toolName
			fsm.toolIdentifiedToken = fsm.tokenCount
			fsm.state = StateToolIdentified
			fsm.checkSpeculatable()
		}
	}

	// 2. Check if arguments start streaming
	if fsm.state == StateToolIdentified {
		if strings.Contains(buf, "<parameter") || strings.Contains(buf, `"arguments"`) {
			fsm.state = StateArgumentsStreaming
		}
	}

	// 3. If streaming arguments or tool identified, extract early or complete arguments
	if fsm.state == StateArgumentsStreaming || fsm.state == StateToolIdentified {
		fsm.extractArguments(buf)
	}

	// 4. If end tag detected without a valid tool call, mark failed
	if fsm.state == StateScanning && (strings.Contains(buf, "</tool_call>") || strings.Contains(buf, "</function_call>")) {
		fsm.state = StateFailed
	}

	return nil
}

// FeedToken is an alias for Feed.
func (fsm *StreamingToolFSM) FeedToken(token string) error {
	return fsm.Feed(token)
}

// ProcessChunk is an alias for Feed.
func (fsm *StreamingToolFSM) ProcessChunk(chunk string) error {
	return fsm.Feed(chunk)
}

// FeedToolCallDelta handles structured tool-call deltas directly (e.g. from OpenAI streaming chunks).
func (fsm *StreamingToolFSM) FeedToolCallDelta(name, argsDelta string) error {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if fsm.state == StateSquashed || fsm.state == StateFailed {
		return nil
	}

	fsm.tokenCount++
	if name != "" && fsm.toolName == "" {
		fsm.toolName = name
		fsm.toolIdentifiedToken = fsm.tokenCount
		fsm.state = StateToolIdentified
		fsm.checkSpeculatable()
	}

	if argsDelta != "" {
		fsm.state = StateArgumentsStreaming
		if fsm.rawArgs != "" {
			fsm.rawArgs += argsDelta
		} else if fsm.earlyArgs != "" {
			fsm.earlyArgs += argsDelta
		} else {
			fsm.earlyArgs = argsDelta
		}

		candidate := fsm.earlyArgs
		var v any
		if err := json.Unmarshal([]byte(candidate), &v); err == nil {
			fsm.rawArgs = candidate
			fsm.earlyArgs = candidate
			fsm.state = StateCompleted
		} else if closed := autoCloseJSON(candidate); closed != "" {
			fsm.earlyArgs = closed
		}
		fsm.checkSpeculatable()
	}

	return nil
}

// extractArguments parses parameter tags or JSON arguments from the accumulated buffer.
func (fsm *StreamingToolFSM) extractArguments(buf string) {
	// Pattern 1: XML parameter tags (<parameter=key>val</parameter>)
	if strings.Contains(buf, "<parameter") {
		params := reXMLParam.FindAllStringSubmatch(buf, -1)
		if len(params) > 0 {
			paramMap := make(map[string]string)
			for _, p := range params {
				k := strings.TrimSpace(p[1])
				v := strings.TrimSpace(p[2])
				paramMap[k] = v
			}
			jsonBytes, _ := json.Marshal(paramMap)
			fsm.earlyArgs = string(jsonBytes)
			if strings.Contains(buf, "</function>") || strings.Contains(buf, "</tool_call>") {
				fsm.rawArgs = string(jsonBytes)
				fsm.state = StateCompleted
			}
			fsm.checkSpeculatable()
			return
		}
	}

	// Pattern 2: JSON "arguments"
	argIdx := strings.Index(buf, `"arguments"`)
	if argIdx == -1 {
		// If outer structure completed with empty arguments
		if fsm.rawArgs == "" && (strings.Contains(buf, "</tool_call>") || strings.Contains(buf, "</function_call>")) {
			fsm.rawArgs = "{}"
			fsm.earlyArgs = "{}"
			fsm.state = StateCompleted
			fsm.checkSpeculatable()
		}
		return
	}

	afterArg := buf[argIdx+len(`"arguments"`):]
	colonIdx := strings.IndexByte(afterArg, ':')
	if colonIdx == -1 {
		return
	}
	valStr := strings.TrimLeft(afterArg[colonIdx+1:], " \t\r\n")
	if len(valStr) == 0 {
		return
	}

	switch valStr[0] {
	case '{':
		depth := 0
		inStr := false
		escape := false
		end := -1
		for i := 0; i < len(valStr); i++ {
			c := valStr[i]
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '"' {
				inStr = !inStr
				continue
			}
			if !inStr {
				if c == '{' {
					depth++
				} else if c == '}' {
					depth--
					if depth == 0 {
						end = i + 1
						break
					}
				}
			}
		}
		if end != -1 {
			fsm.rawArgs = strings.TrimSpace(valStr[:end])
			fsm.earlyArgs = fsm.rawArgs
			fsm.state = StateCompleted
		} else {
			if closed := autoCloseJSON(valStr); closed != "" {
				fsm.earlyArgs = closed
			}
		}
		fsm.checkSpeculatable()

	case '"':
		_ = true
		escape := false
		end := -1
		for i := 1; i < len(valStr); i++ {
			c := valStr[i]
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '"' {
				end = i + 1
				break
			}
		}
		if end != -1 {
			var unquoted string
			if err := json.Unmarshal([]byte(valStr[:end]), &unquoted); err == nil {
				fsm.rawArgs = strings.TrimSpace(unquoted)
				fsm.earlyArgs = fsm.rawArgs
				fsm.state = StateCompleted
			}
		} else {
			quoted := valStr + `"`
			var unquoted string
			if err := json.Unmarshal([]byte(quoted), &unquoted); err == nil {
				if closed := autoCloseJSON(unquoted); closed != "" {
					fsm.earlyArgs = closed
				}
			}
		}
		fsm.checkSpeculatable()

	case '[':
		depth := 0
		inStr := false
		escape := false
		end := -1
		for i := 0; i < len(valStr); i++ {
			c := valStr[i]
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '"' {
				inStr = !inStr
				continue
			}
			if !inStr {
				if c == '[' {
					depth++
				} else if c == ']' {
					depth--
					if depth == 0 {
						end = i + 1
						break
					}
				}
			}
		}
		if end != -1 {
			fsm.rawArgs = strings.TrimSpace(valStr[:end])
			fsm.earlyArgs = fsm.rawArgs
			fsm.state = StateCompleted
		}
		fsm.checkSpeculatable()
	}
}

// checkSpeculatable evaluates whether the current tool and arguments are effect-free and speculatable.
func (fsm *StreamingToolFSM) checkSpeculatable() {
	if fsm.toolName == "" {
		fsm.isSpeculatable = false
		fsm.specReason = vdso.SpecRefusedNilCall
		return
	}

	args := fsm.rawArgs
	if args == "" {
		args = fsm.earlyArgs
	}

	if fsm.speculatableFn != nil {
		fsm.specReason, fsm.isSpeculatable = fsm.speculatableFn(fsm.toolName, args)
		return
	}

	call := &abi.ToolCall{
		Tool: fsm.toolName,
		Meta: metaFor(fsm.toolName),
	}
	if args != "" {
		call.Args = abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(args),
			Len:    int64(len(args)),
		}
	}
	fsm.specReason, fsm.isSpeculatable = vdso.Speculatable(call)
}

// SpeculativeDispatch launches background execution of the speculatable tool call
// while model decode finishes.
func (fsm *StreamingToolFSM) SpeculativeDispatch(ctx context.Context, runner any) error {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if fsm.state == StateSquashed {
		return ErrSpeculationSquashed
	}
	if !fsm.isSpeculatable {
		return ErrNotSpeculatable
	}
	if fsm.toolName == "" {
		return errors.New("streaming tool fsm: tool name not identified yet")
	}

	args := fsm.rawArgs
	if args == "" {
		args = fsm.earlyArgs
	}
	if args == "" {
		return errors.New("streaming tool fsm: arguments not available yet")
	}

	runFn, err := resolveRunner(runner)
	if err != nil {
		return err
	}

	if fsm.dispatched {
		return nil
	}

	fsm.dispatched = true
	fsm.specTool = fsm.toolName
	fsm.specArgs = args

	dispatchCtx, cancel := context.WithCancel(ctx)
	fsm.cancel = cancel
	fsm.doneCh = make(chan struct{})

	specTool := fsm.specTool
	specArgs := fsm.specArgs

	go func() {
		defer close(fsm.doneCh)
		res, runErr := runFn(dispatchCtx, specTool, specArgs)
		fsm.mu.Lock()
		defer fsm.mu.Unlock()
		fsm.specResult = res
		fsm.specErr = runErr
	}()

	return nil
}

// Commit compares authoritative tool name and arguments against speculative execution.
// If exact match, returns the pre-computed speculative result (0ms wait).
// If mismatched, squashes speculative execution and returns ErrSpeculationMismatch.
func (fsm *StreamingToolFSM) Commit(finalToolName, finalArgs string) (string, error) {
	fsm.mu.Lock()
	if fsm.state == StateSquashed {
		reason := fsm.squashReason
		if reason == "" {
			reason = "speculation squashed"
		}
		fsm.mu.Unlock()
		return "", fmt.Errorf("%w: %s", ErrSpeculationSquashed, reason)
	}
	if !fsm.dispatched {
		fsm.mu.Unlock()
		return "", ErrNoSpeculativeDispatch
	}

	if fsm.specTool != finalToolName || !argsMatch(fsm.specArgs, finalArgs) {
		fsm.state = StateSquashed
		fsm.squashReason = fmt.Sprintf("argument mismatch: speculated (%s %s) vs final (%s %s)",
			fsm.specTool, fsm.specArgs, finalToolName, finalArgs)
		if fsm.cancel != nil {
			fsm.cancel()
		}
		fsm.mu.Unlock()
		return "", ErrSpeculationMismatch
	}

	doneCh := fsm.doneCh
	fsm.mu.Unlock()

	// Wait for background execution to complete
	<-doneCh

	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if fsm.state == StateSquashed {
		return "", fmt.Errorf("%w: %s", ErrSpeculationSquashed, fsm.squashReason)
	}
	if fsm.specErr != nil {
		return "", fsm.specErr
	}
	fsm.state = StateCompleted
	return fsm.specResult, nil
}

// Squash cancels background speculative execution and marks the FSM as StateSquashed.
func (fsm *StreamingToolFSM) Squash(reason string) {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if fsm.state == StateSquashed {
		return
	}
	fsm.state = StateSquashed
	fsm.squashReason = reason
	if fsm.cancel != nil {
		fsm.cancel()
	}
}

// resolveRunner extracts an executable run function from the supplied runner argument.
func resolveRunner(runner any) (func(context.Context, string, string) (string, error), error) {
	if runner == nil {
		return nil, errors.New("streaming tool fsm: runner cannot be nil")
	}
	switch r := runner.(type) {
	case SpeculativeRunner:
		return r.Execute, nil
	case SpeculativeRunnerFunc:
		return r.Execute, nil
	case func(context.Context, string, string) (string, error):
		return r, nil
	case func(string, string) (string, error):
		return func(_ context.Context, t, a string) (string, error) {
			return r(t, a)
		}, nil
	case interface {
		RunTool(context.Context, string, string) (string, error)
	}:
		return r.RunTool, nil
	case interface {
		Execute(context.Context, string, string) (string, error)
	}:
		return r.Execute, nil
	case interface {
		Run(context.Context, string, string) (string, error)
	}:
		return r.Run, nil
	default:
		return nil, fmt.Errorf("streaming tool fsm: unsupported runner type %T", runner)
	}
}

// autoCloseJSON inspects partial JSON text and attempts to close open quotes and braces
// to synthesize a valid JSON object. Returns valid JSON or empty string.
func autoCloseJSON(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	var openBraces int
	inString := false
	var escape bool

	for i := 0; i < len(s); i++ {
		c := s[i]
		if escape {
			escape = false
			continue
		}
		if c == '\\' {
			escape = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if !inString {
			if c == '{' {
				openBraces++
			} else if c == '}' {
				openBraces--
			}
		}
	}
	if openBraces < 0 {
		return ""
	}
	candidate := s
	if inString {
		candidate += `"`
	}
	for i := 0; i < openBraces; i++ {
		candidate += "}"
	}
	var v any
	if err := json.Unmarshal([]byte(candidate), &v); err == nil {
		return candidate
	}
	return ""
}

// normalizeJSON canonizes JSON representations by unmarshaling and re-marshaling with sorted keys.
func normalizeJSON(raw string) ([]byte, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, errors.New("empty")
	}
	// If unquoted JSON string, unquote first
	var unquoted string
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		if err := json.Unmarshal([]byte(raw), &unquoted); err == nil {
			raw = strings.TrimSpace(unquoted)
		}
	}
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return nil, err
	}
	return json.Marshal(v)
}

// argsMatch reports whether two argument representations match exactly or canonically.
func argsMatch(a, b string) bool {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == b {
		return true
	}
	normA, errA := normalizeJSON(a)
	normB, errB := normalizeJSON(b)
	if errA == nil && errB == nil {
		return bytes.Equal(normA, normB)
	}
	return false
}
