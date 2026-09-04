package vdso

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TurnState captures the conversational, planning, and contextual state of an agent turn
// prior to model reasoning generation and dispatch.
type TurnState struct {
	TurnIndex      int               `json:"turn_index"`
	PreviousOutput string            `json:"previous_output,omitempty"`
	PlanStep       string            `json:"plan_step,omitempty"`
	UserPrompt     string            `json:"user_prompt,omitempty"`
	WorkDir        string            `json:"work_dir,omitempty"`
	TargetTool     string            `json:"target_tool,omitempty"`
	TargetPath     string            `json:"target_path,omitempty"`
	TargetPattern  string            `json:"target_pattern,omitempty"`
	SessionID      string            `json:"session_id,omitempty"`
	Principal      string            `json:"principal,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// InlineToolResult represents a synthesized tool execution served inline from the vDSO
// fast path without model reasoning generation or remote latency/tokens.
type InlineToolResult struct {
	Call         *abi.ToolCall     `json:"call"`
	Result       *abi.Result       `json:"result"`
	Tool         string            `json:"tool"`
	Path         string            `json:"path,omitempty"`
	Pattern      string            `json:"pattern,omitempty"`
	Content      []byte            `json:"content,omitempty"`
	ModelLatency time.Duration     `json:"model_latency"`
	RemoteTokens int               `json:"remote_tokens"`
	ServedInline bool              `json:"served_inline"`
	TurnState    TurnState         `json:"turn_state"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// ProactiveInterceptor probes the vDSO cache and filesystem state before dispatching
// to a reasoning model. When a turn's plan step or previous output unambiguously identifies
// a deterministic read target whose result is fresh in vDSO (or zero-choice deterministically
// readable), it serves the result inline with 0ms model latency and 0 remote tokens.
type ProactiveInterceptor struct {
	mu                      sync.RWMutex
	vdso                    *VDSO
	allowZeroChoiceDiskRead bool
	workDir                 string

	evaluations   int64
	interceptions int64
	fallthroughs  int64
}

// InterceptorOption configures a ProactiveInterceptor.
type InterceptorOption func(*ProactiveInterceptor)

// WithVDSO sets the vDSO instance used for probing and cache evaluation.
func WithVDSO(v *VDSO) InterceptorOption {
	return func(pi *ProactiveInterceptor) {
		pi.vdso = v
	}
}

// WithZeroChoiceDiskRead allows the interceptor to read unambiguous deterministic files
// directly from disk and populate the vDSO cache if the entry is not yet cached and not stale.
func WithZeroChoiceDiskRead(enabled bool) InterceptorOption {
	return func(pi *ProactiveInterceptor) {
		pi.allowZeroChoiceDiskRead = enabled
	}
}

// WithWorkDir sets the default working directory for path resolution.
func WithWorkDir(dir string) InterceptorOption {
	return func(pi *ProactiveInterceptor) {
		pi.workDir = dir
	}
}

// NewProactiveInterceptor constructs a new ProactiveInterceptor.
func NewProactiveInterceptor(opts ...InterceptorOption) *ProactiveInterceptor {
	pi := &ProactiveInterceptor{
		vdso: Default,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(pi)
		}
	}
	if pi.vdso == nil {
		pi.vdso = Default
	}
	return pi
}

// VDSO returns the configured vDSO instance.
func (pi *ProactiveInterceptor) VDSO() *VDSO {
	if pi == nil || pi.vdso == nil {
		return Default
	}
	return pi.vdso
}

// Stats returns cumulative evaluation, interception, and fallthrough counts.
func (pi *ProactiveInterceptor) Stats() (evaluations, interceptions, fallthroughs int64) {
	if pi == nil {
		return 0, 0, 0
	}
	return atomic.LoadInt64(&pi.evaluations), atomic.LoadInt64(&pi.interceptions), atomic.LoadInt64(&pi.fallthroughs)
}

// SetZeroChoiceDiskRead updates the zero-choice disk read configuration safely.
func (pi *ProactiveInterceptor) SetZeroChoiceDiskRead(enabled bool) {
	if pi == nil {
		return
	}
	pi.mu.Lock()
	pi.allowZeroChoiceDiskRead = enabled
	pi.mu.Unlock()
}

type readTarget struct {
	tool    string
	path    string
	pattern string
}

var (
	mutationVerbsRe = regexp.MustCompile(`(?i)\b(edit|write|delete|remove|create|overwrite|modify|update|truncate|rm|mv|git\s+commit|git\s+push)\b`)
	grepWordRe      = regexp.MustCompile(`(?i)\b(grep|search)\b`)
	globWordRe      = regexp.MustCompile(`(?i)\b(glob|find\s+files)\b`)
	readWordRe      = regexp.MustCompile(`(?i)\b(read|cat|inspect|examine|view|open|load)\b`)
	backtickRe      = regexp.MustCompile("`([^`]+)`")
	doubleQuoteRe   = regexp.MustCompile(`"([^"]+)"`)
	singleQuoteRe   = regexp.MustCompile(`'([^']+)'`)
	pathWithSlashRe = regexp.MustCompile(`(?:^|[\s(])([a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)+)(?:$|[\s),;:?])`)
	fileWithExtRe   = regexp.MustCompile(`(?:^|[\s(])([a-zA-Z0-9_\-\./\\]+\.(?:go|txt|md|json|toml|yaml|yml|c|h|cpp|py|ts|js|sh|rs|proto|sql|html|css))(?:$|[\s),;:?])`)
)

func cleanPathString(p string) string {
	p = strings.TrimSpace(p)
	p = strings.Trim(p, "`\"'()[]{}<>,;:")
	if p == "" {
		return ""
	}
	cleaned := filepath.Clean(p)
	return filepath.ToSlash(cleaned)
}

func resolveDiskPath(workDir, p string) string {
	cleaned := cleanPathString(p)
	if filepath.IsAbs(cleaned) {
		return cleaned
	}
	if workDir != "" {
		cand := filepath.Join(workDir, cleaned)
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	return cleaned
}

// extractReadTarget analyzes turn state to determine if an unambiguous deterministic read target exists.
func (pi *ProactiveInterceptor) extractReadTarget(turn TurnState) (*readTarget, bool) {
	// 1. Explicit target fields in turn
	if turn.TargetTool != "" {
		tool := strings.TrimSpace(turn.TargetTool)
		switch strings.ToLower(tool) {
		case "read", "readfile", "read_file":
			if turn.TargetPath == "" {
				return nil, false
			}
			return &readTarget{tool: ToolClaudeRead, path: cleanPathString(turn.TargetPath)}, true
		case "glob":
			pat := strings.TrimSpace(turn.TargetPattern)
			path := strings.TrimSpace(turn.TargetPath)
			if pat == "" && path == "" {
				return nil, false
			}
			if pat == "" && (strings.ContainsAny(path, "*?[]{}") || strings.HasPrefix(path, "*")) {
				pat = path
				path = "."
			}
			if path == "" {
				path = "."
			}
			return &readTarget{tool: ToolClaudeGlob, path: cleanPathString(path), pattern: pat}, true
		case "grep":
			pat := strings.TrimSpace(turn.TargetPattern)
			if pat == "" {
				return nil, false
			}
			path := strings.TrimSpace(turn.TargetPath)
			if path == "" {
				path = "."
			}
			return &readTarget{tool: ToolClaudeGrep, path: cleanPathString(path), pattern: pat}, true
		default:
			// Non-read tool explicitly specified -> not a deterministic read target.
			return nil, false
		}
	}

	// 2. PlanStep analysis (highest intent priority)
	if strings.TrimSpace(turn.PlanStep) != "" {
		if target, ok := parseReadTargetFromText(turn.PlanStep); ok {
			return target, true
		}
		// If PlanStep was provided but could not be parsed or was rejected, do not fall back.
		return nil, false
	}

	// 3. PreviousOutput analysis
	if strings.TrimSpace(turn.PreviousOutput) != "" {
		return parseReadTargetFromText(turn.PreviousOutput)
	}

	return nil, false
}

func stripQuoted(s string) string {
	s = backtickRe.ReplaceAllString(s, " ")
	s = doubleQuoteRe.ReplaceAllString(s, " ")
	s = singleQuoteRe.ReplaceAllString(s, " ")
	return s
}

func parseReadTargetFromText(text string) (*readTarget, bool) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, false
	}

	// Structured JSON tool call check
	if strings.Contains(text, "{") && strings.Contains(text, "}") {
		if target, ok := parseJSONToolTarget(text); ok {
			return target, true
		}
	}

	// Structured markdown tool call check
	if idx := strings.Index(text, "[tool_call:"); idx != -1 {
		endIdx := strings.Index(text[idx:], "]")
		if endIdx != -1 {
			snippet := text[idx : idx+endIdx+1]
			if target, ok := parseToolCallSnippet(snippet); ok {
				return target, true
			}
		}
	}

	// Reject if text outside quoted strings states mutation intent
	outside := stripQuoted(text)
	if mutationVerbsRe.MatchString(outside) {
		return nil, false
	}

	// Tool-specific detection for Glob
	if target, ok := parseGlobIntent(text); ok {
		return target, true
	}

	// Tool-specific detection for Grep
	if target, ok := parseGrepIntent(text); ok {
		return target, true
	}

	// Tool-specific detection for Read
	return parseReadIntent(text)
}

func parseJSONToolTarget(text string) (*readTarget, bool) {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start == -1 || end == -1 || end <= start {
		return nil, false
	}
	raw := text[start : end+1]
	var m map[string]any
	if json.Unmarshal([]byte(raw), &m) != nil {
		return nil, false
	}

	var toolName string
	for _, k := range []string{"tool", "name", "tool_name"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok {
				toolName = strings.TrimSpace(s)
				break
			}
		}
	}

	// If a mutation tool is named, explicitly reject
	if IsClaudeNativeWriteTool(toolName) {
		return nil, false
	}

	var filePath string
	for _, k := range filePathArgKeys {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok {
				filePath = cleanPathString(s)
				break
			}
		}
	}

	var pattern string
	for _, k := range []string{"pattern", "query", "regex", "search"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok {
				pattern = strings.TrimSpace(s)
				break
			}
		}
	}

	switch strings.ToLower(toolName) {
	case "read", "readfile", "read_file":
		if filePath != "" {
			return &readTarget{tool: ToolClaudeRead, path: filePath}, true
		}
	case "glob":
		if pattern != "" || filePath != "" {
			if pattern == "" && strings.ContainsAny(filePath, "*?[]{}") {
				pattern = filePath
				filePath = "."
			}
			if filePath == "" {
				filePath = "."
			}
			return &readTarget{tool: ToolClaudeGlob, path: filePath, pattern: pattern}, true
		}
	case "grep":
		if pattern != "" {
			if filePath == "" {
				filePath = "."
			}
			return &readTarget{tool: ToolClaudeGrep, path: filePath, pattern: pattern}, true
		}
	}

	// If no tool was explicitly named, but filePath was present and no mutation was stated
	if filePath != "" && pattern == "" {
		return &readTarget{tool: ToolClaudeRead, path: filePath}, true
	}

	return nil, false
}

func parseToolCallSnippet(snippet string) (*readTarget, bool) {
	// Format: [tool_call: read for absolute_path '...'] or [tool_call: Read filePath="..."]
	lower := strings.ToLower(snippet)
	if strings.Contains(lower, "write") || strings.Contains(lower, "edit") {
		return nil, false
	}
	if strings.Contains(lower, "read") {
		// Extract path from quotes or backticks
		for _, re := range []*regexp.Regexp{backtickRe, singleQuoteRe, doubleQuoteRe} {
			if m := re.FindStringSubmatch(snippet); len(m) > 1 {
				p := cleanPathString(m[1])
				if p != "" {
					return &readTarget{tool: ToolClaudeRead, path: p}, true
				}
			}
		}
	}
	return nil, false
}

func parseGrepIntent(text string) (*readTarget, bool) {
	outside := stripQuoted(text)
	if !grepWordRe.MatchString(outside) {
		return nil, false
	}

	var pattern string
	var path string = "."

	matches := backtickRe.FindAllStringSubmatch(text, -1)
	if len(matches) == 1 {
		pattern = strings.TrimSpace(matches[0][1])
		path = "."
	} else if len(matches) >= 2 {
		pattern = strings.TrimSpace(matches[0][1])
		path = cleanPathString(matches[1][1])
	} else {
		// Check single or double quotes
		qMatches := doubleQuoteRe.FindAllStringSubmatch(text, -1)
		if len(qMatches) == 0 {
			qMatches = singleQuoteRe.FindAllStringSubmatch(text, -1)
		}
		if len(qMatches) == 1 {
			pattern = strings.TrimSpace(qMatches[0][1])
			path = "."
		} else if len(qMatches) >= 2 {
			pattern = strings.TrimSpace(qMatches[0][1])
			path = cleanPathString(qMatches[1][1])
		}
	}

	if path == "." {
		if inIdx := strings.Index(strings.ToLower(text), " in "); inIdx != -1 {
			remainder := strings.TrimSpace(text[inIdx+4:])
			cand := cleanPathString(remainder)
			if cand != "" && cand != "." {
				path = cand
			}
		}
	}

	if pattern != "" {
		if path == "" {
			path = "."
		}
		return &readTarget{tool: ToolClaudeGrep, path: path, pattern: pattern}, true
	}
	return nil, false
}

func parseGlobIntent(text string) (*readTarget, bool) {
	outside := stripQuoted(text)
	if !globWordRe.MatchString(outside) {
		return nil, false
	}

	var pattern string
	var path string = "."

	matches := backtickRe.FindAllStringSubmatch(text, -1)
	if len(matches) == 1 {
		pattern = strings.TrimSpace(matches[0][1])
		path = "."
	} else if len(matches) >= 2 {
		pattern = strings.TrimSpace(matches[0][1])
		path = cleanPathString(matches[1][1])
	} else {
		tokens := strings.Fields(text)
		for _, tok := range tokens {
			cleanTok := strings.Trim(tok, "`\"',;:()")
			if strings.ContainsAny(cleanTok, "*?[]{}") {
				pattern = cleanTok
				break
			}
		}
		path = "."
	}

	if path == "." {
		if inIdx := strings.Index(strings.ToLower(text), " in "); inIdx != -1 {
			remainder := strings.TrimSpace(text[inIdx+4:])
			cand := cleanPathString(remainder)
			if cand != "" && cand != "." {
				path = cand
			}
		}
	}

	if pattern != "" {
		if path == "" {
			path = "."
		}
		return &readTarget{tool: ToolClaudeGlob, path: path, pattern: pattern}, true
	}
	return nil, false
}

func parseReadIntent(text string) (*readTarget, bool) {
	lower := strings.ToLower(text)
	hasReadKeyword := false
	for _, kw := range []string{"read", "cat", "inspect", "examine", "view", "open", "check", "load", "file:", "path:", "target:"} {
		if strings.Contains(lower, kw) {
			hasReadKeyword = true
			break
		}
	}

	candidates := extractFileCandidates(text)

	// If there are multiple distinct files mentioned, it's ambiguous -> do not guess.
	if len(candidates) > 1 {
		return nil, false
	}

	if len(candidates) == 1 {
		// If a read keyword was found, or the text is solely/primarily the file path
		if hasReadKeyword || isSolelyFilePath(text, candidates[0]) {
			return &readTarget{tool: ToolClaudeRead, path: candidates[0]}, true
		}
	}

	return nil, false
}

func extractFileCandidates(text string) []string {
	seen := make(map[string]struct{})
	var result []string

	add := func(p string) {
		cp := cleanPathString(p)
		if cp == "" || cp == "." || cp == "/" {
			return
		}
		// Ignore pure keywords or non-file tokens
		if strings.EqualFold(cp, "read") || strings.EqualFold(cp, "file") || strings.EqualFold(cp, "path") {
			return
		}
		if _, exists := seen[cp]; !exists {
			seen[cp] = struct{}{}
			result = append(result, cp)
		}
	}

	// 1. Backticked strings
	for _, m := range backtickRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	// 2. Quoted strings
	for _, m := range doubleQuoteRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 && looksLikePath(m[1]) {
			add(m[1])
		}
	}
	for _, m := range singleQuoteRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 && looksLikePath(m[1]) {
			add(m[1])
		}
	}

	// 3. Unquoted paths with slashes
	for _, m := range pathWithSlashRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	// 4. Unquoted files with standard extensions
	for _, m := range fileWithExtRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 {
			add(m[1])
		}
	}

	return result
}

func looksLikePath(s string) bool {
	s = strings.TrimSpace(s)
	if strings.Contains(s, "/") || strings.Contains(s, "\\") {
		return true
	}
	return fileWithExtRe.MatchString(s)
}

func isSolelyFilePath(text string, path string) bool {
	trimmed := strings.TrimSpace(text)
	trimmed = strings.Trim(trimmed, "`\"'()[]{}<>,;:")
	return cleanPathString(trimmed) == path
}

// Evaluate checks whether a turn's next action is an unambiguous deterministic read target
// that can be served inline directly from vDSO cache (or zero-choice deterministic disk read).
// Returns the synthesized inline result and true on hit, or nil and false to fall through to the remote model.
func (pi *ProactiveInterceptor) Evaluate(ctx context.Context, turn TurnState) (*InlineToolResult, bool) {
	if pi == nil {
		return nil, false
	}
	atomic.AddInt64(&pi.evaluations, 1)

	target, ok := pi.extractReadTarget(turn)
	if !ok || target == nil {
		atomic.AddInt64(&pi.fallthroughs, 1)
		return nil, false
	}

	v := pi.VDSO()
	workDir := turn.WorkDir
	if workDir == "" {
		pi.mu.RLock()
		workDir = pi.workDir
		pi.mu.RUnlock()
	}

	diskPath := resolveDiskPath(workDir, target.path)

	// For file read tools, verify disk existence before probing cache.
	if target.tool == ToolClaudeRead {
		st, err := os.Stat(diskPath)
		if err != nil || st.IsDir() {
			// Missing file or directory: falls through to remote model.
			atomic.AddInt64(&pi.fallthroughs, 1)
			return nil, false
		}
	}

	// Probe vDSO cache and disk freshness across candidate argument shapes.
	call, res, hit := pi.probeVDSO(ctx, v, target, diskPath, turn)
	if hit && res != nil {
		return pi.synthesizeInlineResult(ctx, v, call, res, target, turn)
	}

	// Check if zero-choice deterministic disk read is permitted.
	pi.mu.RLock()
	allowZeroChoice := pi.allowZeroChoiceDiskRead
	pi.mu.RUnlock()

	if allowZeroChoice && target.tool == ToolClaudeRead {
		if inlineRes, served := pi.tryZeroChoiceDiskRead(ctx, v, target, diskPath, turn); served {
			return inlineRes, true
		}
	}

	atomic.AddInt64(&pi.fallthroughs, 1)
	return nil, false
}

func (pi *ProactiveInterceptor) probeVDSO(
	ctx context.Context,
	v *VDSO,
	target *readTarget,
	diskPath string,
	turn TurnState,
) (*abi.ToolCall, *abi.Result, bool) {
	var candidateArgs []string

	switch target.tool {
	case ToolClaudeRead:
		candidateArgs = append(candidateArgs, fmt.Sprintf(`{"filePath":%q}`, target.path))
		if diskPath != target.path && diskPath != "" {
			candidateArgs = append(candidateArgs, fmt.Sprintf(`{"filePath":%q}`, diskPath))
		}
		candidateArgs = append(candidateArgs, fmt.Sprintf(`{"file_path":%q}`, target.path))
		candidateArgs = append(candidateArgs, fmt.Sprintf(`{"path":%q}`, target.path))
	case ToolClaudeGlob:
		candidateArgs = append(candidateArgs, fmt.Sprintf(`{"pattern":%q,"path":%q}`, target.pattern, target.path))
		if target.path == "." || target.path == "" {
			candidateArgs = append(candidateArgs, fmt.Sprintf(`{"pattern":%q}`, target.pattern))
		}
	case ToolClaudeGrep:
		candidateArgs = append(candidateArgs, fmt.Sprintf(`{"pattern":%q,"path":%q}`, target.pattern, target.path))
		if target.path == "." || target.path == "" {
			candidateArgs = append(candidateArgs, fmt.Sprintf(`{"pattern":%q}`, target.pattern))
		}
	default:
		return nil, nil, false
	}

	for _, argsStr := range candidateArgs {
		call := &abi.ToolCall{
			Tool: target.tool,
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(argsStr), Len: int64(len(argsStr))},
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		}
		if turn.Principal != "" {
			call.Meta["principal"] = turn.Principal
		}
		if turn.SessionID != "" {
			call.Meta["session_id"] = turn.SessionID
		}

		res, hit := v.Lookup(ctx, call)
		if hit && res != nil {
			return call, res, true
		}
	}

	return nil, nil, false
}

func (pi *ProactiveInterceptor) synthesizeInlineResult(
	ctx context.Context,
	v *VDSO,
	call *abi.ToolCall,
	res *abi.Result,
	target *readTarget,
	turn TurnState,
) (*InlineToolResult, bool) {
	payloadBytes := v.bytes(ctx, res.Payload)
	if len(payloadBytes) == 0 && len(res.Payload.Inline) > 0 {
		payloadBytes = res.Payload.Inline
	}

	if res.Meta == nil {
		res.Meta = make(map[string]string)
	}
	res.Meta["served_by"] = "vdso"
	res.Meta["proactive_intercepted"] = "true"
	res.Meta["model_latency_ms"] = "0"
	res.Meta["remote_tokens"] = "0"

	// Advance turn state directly: increment turn index, record output, clear consumed plan step.
	nextTurn := turn
	nextTurn.TurnIndex = turn.TurnIndex + 1
	nextTurn.PreviousOutput = string(payloadBytes)
	nextTurn.PlanStep = ""
	if nextTurn.Metadata == nil {
		nextTurn.Metadata = make(map[string]string)
	} else {
		m := make(map[string]string, len(nextTurn.Metadata)+5)
		for k, val := range nextTurn.Metadata {
			m[k] = val
		}
		nextTurn.Metadata = m
	}
	nextTurn.Metadata["proactive_interception"] = "true"
	nextTurn.Metadata["intercepted_tool"] = target.tool
	nextTurn.Metadata["intercepted_target"] = target.path
	nextTurn.Metadata["model_latency_ms"] = "0"
	nextTurn.Metadata["remote_tokens"] = "0"

	inlineRes := &InlineToolResult{
		Call:         call,
		Result:       res,
		Tool:         target.tool,
		Path:         target.path,
		Pattern:      target.pattern,
		Content:      payloadBytes,
		ModelLatency: 0,
		RemoteTokens: 0,
		ServedInline: true,
		TurnState:    nextTurn,
		Metadata: map[string]string{
			"served_by":             "vdso",
			"proactive_intercepted": "true",
			"model_latency_ms":      "0",
			"remote_tokens":         "0",
		},
	}

	atomic.AddInt64(&pi.interceptions, 1)
	return inlineRes, true
}

func (pi *ProactiveInterceptor) tryZeroChoiceDiskRead(
	ctx context.Context,
	v *VDSO,
	target *readTarget,
	diskPath string,
	turn TurnState,
) (*InlineToolResult, bool) {
	st, err := os.Stat(diskPath)
	if err != nil || st.IsDir() {
		return nil, false
	}

	// Check if a witness for this file is revoked (stale cache eviction)
	fileWit := deriveFileWitness(target.tool, []byte(fmt.Sprintf(`{"filePath":%q}`, target.path)))
	if fileWit != "" {
		v.mu.Lock()
		isRevoked := v.revokedLocked(fileWit)
		v.mu.Unlock()
		if isRevoked {
			return nil, false
		}
	}

	content, err := os.ReadFile(diskPath)
	if err != nil {
		return nil, false
	}

	argsJSON := fmt.Sprintf(`{"filePath":%q}`, target.path)
	call := &abi.ToolCall{
		Tool: target.tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(argsJSON), Len: int64(len(argsJSON))},
		Meta: map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "true",
		},
	}
	if turn.Principal != "" {
		call.Meta["principal"] = turn.Principal
	}
	if turn.SessionID != "" {
		call.Meta["session_id"] = turn.SessionID
	}

	emitRes := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: content, Len: int64(len(content)), Taint: abi.TaintTrusted},
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"tier":                  "2",
			"proactive_intercepted": "true",
			"model_latency_ms":      "0",
			"remote_tokens":         "0",
		},
	}

	// Populate vDSO cache via Emit so subsequent lookups hit tier-2.
	// Note: served_by must NOT be "vdso" at Emit time, or StoreResult skips caching.
	v.Emit(abi.Event{
		Kind:   abi.EvComplete,
		Call:   call,
		Result: emitRes,
	})

	return pi.synthesizeInlineResult(ctx, v, call, emitRes, target, turn)
}
