// Package gym coordinates agent evaluation environments, sub-10ms CoW
// snapshot lifecycles, and isolated execution trajectories.
package gym

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// OpenCodeEvalSchema defines the canonical schema identifier for OpenCode evaluation receipts.
const OpenCodeEvalSchema = "fak.gym.opencode.eval.v1"

// Scenario categories for OpenCode tool evaluation.
const (
	CategoryToolExecution       = "tool_execution"
	CategoryMalformedRepair     = "malformed_repair"
	CategoryRefusalHandling     = "refusal_handling"
	CategoryAdversarialBoundary = "adversarial_boundary"
)

// OpenCodeToolKind identifies the client-side execution interface.
type OpenCodeToolKind string

const (
	OpenCodeToolKindDesktop OpenCodeToolKind = "desktop" // Visual desktop OpenCode tool calling (camelCase args)
	OpenCodeToolKindCLI     OpenCodeToolKind = "cli"     // Headless CLI execution (bash/exec_command)
	OpenCodeToolKindMCP     OpenCodeToolKind = "mcp"     // MCP protocol-mediated OpenCode dialect
)

// OpenCodeToolCall represents an OpenCode tool proposal.
type OpenCodeToolCall struct {
	ID        string           `json:"id"`
	Name      string           `json:"name"`
	Arguments string           `json:"arguments"`
	Kind      OpenCodeToolKind `json:"kind,omitempty"`
	SessionID string           `json:"session_id,omitempty"`
}

// OpenCodeEvalScenario defines an isolated evaluation scenario for OpenCode tool behaviors.
type OpenCodeEvalScenario struct {
	ID                    string             `json:"id"`
	Name                  string             `json:"name"`
	Category              string             `json:"category"`
	InitialFiles          map[string]string  `json:"initial_files,omitempty"`
	ToolCalls             []OpenCodeToolCall `json:"tool_calls"`
	ExpectedVerdict       string             `json:"expected_verdict,omitempty"`        // "ALLOW", "DENY", "REPAIRED"
	ExpectedRefusalReason string             `json:"expected_refusal_reason,omitempty"` // "DEFAULT_DENY", "POLICY_BLOCK", "SELF_MODIFY", "MALFORMED"
	ExpectedRepairs       int                `json:"expected_repairs,omitempty"`
	ExpectANSIStripped    bool               `json:"expect_ansi_stripped,omitempty"`
	ExpectedOutputDigest  string             `json:"expected_output_digest,omitempty"`
}

// OpenCodeAdjudicationRecord captures the adjudication and execution outcome of one proposed tool call.
type OpenCodeAdjudicationRecord struct {
	ToolCallID         string         `json:"tool_call_id"`
	Tool               string         `json:"tool"`
	CanonicalTool      string         `json:"canonical_tool"`
	Verdict            string         `json:"verdict"` // "ALLOW", "DENY", "REPAIRED"
	Reason             string         `json:"reason,omitempty"`
	ReasonCode         abi.ReasonCode `json:"reason_code,omitempty"`
	WasRepaired        bool           `json:"was_repaired"`
	RepairedArguments  string         `json:"repaired_arguments,omitempty"`
	RawOutput          string         `json:"raw_output,omitempty"`
	NormalizedOutput   string         `json:"normalized_output,omitempty"`
	OutputDigest       string         `json:"output_digest,omitempty"`
	ExecutionError     string         `json:"execution_error,omitempty"`
	ANSISequencesCount int            `json:"ansi_sequences_count,omitempty"`
}

// OpenCodeEvalReceipt records the evaluation outcome, deterministic hashes, and execution trace.
type OpenCodeEvalReceipt struct {
	Schema           string                       `json:"schema"`
	ScenarioID       string                       `json:"scenario_id"`
	Timestamp        time.Time                    `json:"timestamp"`
	Outcome          string                       `json:"outcome"` // "PASS", "FAIL"
	FailureReason    string                       `json:"failure_reason,omitempty"`
	EvaluatedCalls   int                          `json:"evaluated_calls"`
	AllowedCount     int                          `json:"allowed_count"`
	RepairedCount    int                          `json:"repaired_count"`
	DeniedCount      int                          `json:"denied_count"`
	Adjudications    []OpenCodeAdjudicationRecord `json:"adjudications"`
	EvaluationHash   string                       `json:"evaluation_hash"`
	ExecutionElapsed time.Duration                `json:"execution_elapsed_ns"`
}

// Verify asserts receipt validity against scenario requirements.
func (r *OpenCodeEvalReceipt) Verify(expectedScenarioID string) (bool, string) {
	if r == nil {
		return false, "receipt is nil"
	}
	if r.Schema != OpenCodeEvalSchema {
		return false, fmt.Sprintf("invalid schema: expected %q, got %q", OpenCodeEvalSchema, r.Schema)
	}
	if expectedScenarioID != "" && r.ScenarioID != expectedScenarioID {
		return false, fmt.Sprintf("scenario mismatch: expected %q, got %q", expectedScenarioID, r.ScenarioID)
	}
	if r.Outcome != "PASS" {
		reason := r.FailureReason
		if reason == "" {
			reason = "unspecified failure"
		}
		return false, fmt.Sprintf("outcome not PASS: %s (%s)", r.Outcome, reason)
	}
	if r.EvaluationHash == "" {
		return false, "evaluation hash is empty"
	}
	return true, ""
}

// Regular expressions for terminal output normalization.
var (
	// Unified ANSI escape matcher: OSC (ESC ] ... BEL|ST), CSI (ESC [ ... final), or standard 2-byte escape
	reAllANSI = regexp.MustCompile(`\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]`)
	// Carriage return followed by overwrite or newline
	reCRLF = regexp.MustCompile(`\r\n`)
	// Stray lone carriage return
	reLoneCR = regexp.MustCompile(`\r`)
)

// NormalizeOpenCodeOutput strips ANSI escape sequences, collapses carriage return overwrites,
// and canonicalizes line endings to guarantee deterministic evaluation hashes.
func NormalizeOpenCodeOutput(raw string) (normalized string, ansiCount int) {
	if raw == "" {
		return "", 0
	}

	// 1. Detect and count ANSI escape sequences without overlapping matches
	matches := reAllANSI.FindAllStringIndex(raw, -1)
	ansiCount = len(matches)

	// 2. Strip all ANSI escape sequences
	s := reAllANSI.ReplaceAllString(raw, "")

	// 3. Normalize line breaks: CRLF -> LF
	s = reCRLF.ReplaceAllString(s, "\n")

	// 6. Handle carriage returns (terminal progress spinners / rewrites)
	lines := strings.Split(s, "\n")
	var cleanedLines []string
	for _, line := range lines {
		if strings.Contains(line, "\r") {
			// If line has carriage returns, emulate terminal overwrite
			segments := strings.Split(line, "\r")
			var resolved string
			for _, seg := range segments {
				if seg != "" {
					resolved = seg // last non-empty segment overwrites prior text
				}
			}
			line = resolved
		}
		// Strip C0 control characters (except tab)
		var buf strings.Builder
		for _, r := range line {
			if r == '\t' || r >= 32 && r != 127 {
				buf.WriteRune(r)
			}
		}
		trimmed := strings.TrimRight(buf.String(), " \t")
		cleanedLines = append(cleanedLines, trimmed)
	}

	res := strings.Join(cleanedLines, "\n")
	res = strings.TrimSpace(res)
	if res != "" {
		res += "\n"
	}
	return res, ansiCount
}

// ComputeEvaluationHash produces a deterministic sha256 hash of normalized output.
func ComputeEvaluationHash(normalized string) string {
	sum := sha256.Sum256([]byte(normalized))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// ComputeScenarioEvaluationHash produces a deterministic hash over all scenario adjudication records.
func ComputeScenarioEvaluationHash(records []OpenCodeAdjudicationRecord) string {
	hasher := sha256.New()
	for i, r := range records {
		fmt.Fprintf(hasher, "step=%d:tool=%s:canonical=%s:verdict=%s:reason=%s:repaired=%v:args=%s:out_digest=%s\n",
			i, r.Tool, r.CanonicalTool, r.Verdict, r.Reason, r.WasRepaired, r.RepairedArguments, r.OutputDigest)
	}
	return "sha256:" + hex.EncodeToString(hasher.Sum(nil))
}

// CanonicalOpenCodeTool maps various OpenCode desktop/CLI prefixes and aliases to canonical names.
func CanonicalOpenCodeTool(name string) string {
	name = strings.TrimSpace(name)
	// OpenCode double prefix: fak_fak_<tool>
	if strings.HasPrefix(name, "fak_fak_") {
		name = strings.TrimPrefix(name, "fak_fak_")
	}
	// OpenCode desktop prefix: opencode_<tool> or desktop_<tool>
	if strings.HasPrefix(name, "opencode_") {
		name = strings.TrimPrefix(name, "opencode_")
	}
	if strings.HasPrefix(name, "desktop_") {
		name = strings.TrimPrefix(name, "desktop_")
	}
	// OpenCode single prefix: bash_exec_command -> bash
	if name == "bash_exec_command" {
		return "bash"
	}
	if strings.HasPrefix(name, "mcp__fak__") {
		name = strings.TrimPrefix(name, "mcp__fak__")
	}

	switch strings.ToLower(name) {
	case "read_file", "view_file", "cat":
		return "read"
	case "write_file", "create_file":
		return "write"
	case "edit_file", "patch_file", "apply_patch":
		return "edit"
	case "exec_command", "shell", "terminal", "command":
		return "bash"
	case "find_files", "file_search":
		return "glob"
	case "search_contents", "content_search":
		return "grep"
	case "list_directory", "ls":
		return "list_dir"
	default:
		return strings.ToLower(name)
	}
}

// RepairOpenCodeToolCall repairs malformed or schema-deviant tool proposals frequently
// emitted by OpenCode desktop and CLI interfaces.
func RepairOpenCodeToolCall(call OpenCodeToolCall) (repaired OpenCodeToolCall, wasRepaired bool, err error) {
	repaired = call
	canonical := CanonicalOpenCodeTool(call.Name)
	if canonical != call.Name {
		repaired.Name = canonical
		wasRepaired = true
	}

	rawArgs := strings.TrimSpace(call.Arguments)
	if rawArgs == "" {
		repaired.Arguments = "{}"
		return repaired, true, nil
	}

	// 1. Unwrap double-stringified JSON: "\"{\\\"filePath\\\": ...}\""
	if strings.HasPrefix(rawArgs, `"`) && strings.HasSuffix(rawArgs, `"`) {
		var unquoted string
		if err := json.Unmarshal([]byte(rawArgs), &unquoted); err == nil {
			rawArgs = strings.TrimSpace(unquoted)
			wasRepaired = true
		}
	}

	// 2. Parse JSON arguments map
	var args map[string]any
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		// Attempt lenient repair for unquoted keys: e.g. `{command: "echo 1"}`
		repairedJSON := repairRelaxedJSON(rawArgs)
		if err2 := json.Unmarshal([]byte(repairedJSON), &args); err2 == nil {
			rawArgs = repairedJSON
			wasRepaired = true
		} else {
			return call, false, fmt.Errorf("malformed json arguments: %w", err)
		}
	}

	// 3. Normalize parameter keys according to canonical tool requirements
	updated := false
	switch canonical {
	case "read", "write", "edit":
		// Canonicalize path: file_path, path, target, filename -> filePath
		for _, key := range []string{"file_path", "path", "target", "filename", "filepath"} {
			if v, ok := args[key]; ok && v != nil {
				if _, has := args["filePath"]; !has {
					args["filePath"] = v
					delete(args, key)
					updated = true
				}
			}
		}
		if canonical == "edit" {
			// Canonicalize old_string / new_string -> oldString / newString
			if v, ok := args["old_string"]; ok {
				if _, has := args["oldString"]; !has {
					args["oldString"] = v
					delete(args, "old_string")
					updated = true
				}
			}
			if v, ok := args["new_string"]; ok {
				if _, has := args["newString"]; !has {
					args["newString"] = v
					delete(args, "new_string")
					updated = true
				}
			}
		}
	case "bash":
		// Canonicalize cmd, script -> command
		for _, key := range []string{"cmd", "script", "cli"} {
			if v, ok := args[key]; ok && v != nil {
				if _, has := args["command"]; !has {
					args["command"] = v
					delete(args, key)
					updated = true
				}
			}
		}
	case "grep":
		// Canonicalize query, search -> pattern
		for _, key := range []string{"query", "search", "search_term"} {
			if v, ok := args[key]; ok && v != nil {
				if _, has := args["pattern"]; !has {
					args["pattern"] = v
					delete(args, key)
					updated = true
				}
			}
		}
	case "glob":
		// Canonicalize dir, directory -> path
		for _, key := range []string{"dir", "directory", "folder"} {
			if v, ok := args[key]; ok && v != nil {
				if _, has := args["path"]; !has {
					args["path"] = v
					delete(args, key)
					updated = true
				}
			}
		}
	}

	if updated || wasRepaired {
		reencoded, err := json.Marshal(args)
		if err != nil {
			return call, false, fmt.Errorf("failed to marshal repaired arguments: %w", err)
		}
		repaired.Arguments = string(reencoded)
		return repaired, true, nil
	}

	return repaired, false, nil
}

var (
	// relaxed JSON unquoted key matcher: {key: "value"} -> {"key": "value"}
	reUnquotedKey = regexp.MustCompile(`([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)`)
	// trailing comma before closing brace/bracket: `,}` -> `}`
	reTrailingComma = regexp.MustCompile(`,\s*([}\]])`)
)

func repairRelaxedJSON(input string) string {
	s := reUnquotedKey.ReplaceAllString(input, `$1"$2"$3`)
	s = reTrailingComma.ReplaceAllString(s, `$1`)
	return s
}

// OpenCodeGymConfig supplies initialization configuration for the OpenCode evaluation gym.
type OpenCodeGymConfig struct {
	BaseDir         string
	Arena           *Arena
	AllowedTools    []string
	SelfModifyGlobs []string
}

// OpenCodeGymHarness coordinates hermetic sandboxed execution, repair, and adjudication
// of OpenCode desktop and CLI tool behaviors.
type OpenCodeGymHarness struct {
	cfg       OpenCodeGymConfig
	arena     *Arena
	ownsArena bool
	tempDir   string
}

// NewOpenCodeGymHarness initializes an isolated evaluation gym harness.
func NewOpenCodeGymHarness(ctx context.Context, cfg OpenCodeGymConfig) (*OpenCodeGymHarness, error) {
	if len(cfg.AllowedTools) == 0 {
		cfg.AllowedTools = []string{
			"read", "write", "edit", "bash", "glob", "grep", "list_dir",
			"read_file", "write_file", "edit_file", "exec_command",
		}
	}
	if len(cfg.SelfModifyGlobs) == 0 {
		cfg.SelfModifyGlobs = []string{
			"internal/abi/", "internal/kernel/", "internal/adjudicator/", "dos.toml", ".dos/", "internal/gym/",
		}
	}

	var arena *Arena
	var ownsArena bool
	var tempDir string

	if cfg.Arena != nil {
		arena = cfg.Arena
	} else if cfg.BaseDir != "" {
		a, err := Create(ctx, Config{
			BaseDir:       cfg.BaseDir,
			WorkspaceName: "opencode-gym-eval",
		})
		if err != nil {
			return nil, fmt.Errorf("failed creating gym arena: %w", err)
		}
		arena = a
		ownsArena = true
	} else {
		td, err := os.MkdirTemp("", "fak-opencode-gym-*")
		if err != nil {
			return nil, fmt.Errorf("failed creating temp directory: %w", err)
		}
		tempDir = td
		a, err := Create(ctx, Config{
			BaseDir:       td,
			WorkspaceName: "opencode-gym-eval-temp",
		})
		if err != nil {
			_ = os.RemoveAll(td)
			return nil, fmt.Errorf("failed creating gym arena over temp directory: %w", err)
		}
		arena = a
		ownsArena = true
	}

	return &OpenCodeGymHarness{
		cfg:       cfg,
		arena:     arena,
		ownsArena: ownsArena,
		tempDir:   tempDir,
	}, nil
}

// Close destroys the gym arena and cleans up ephemeral resources.
func (h *OpenCodeGymHarness) Close() error {
	var firstErr error
	if h.ownsArena && h.arena != nil {
		if err := h.arena.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if h.tempDir != "" {
		if err := os.RemoveAll(h.tempDir); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// WorkspacePath returns the active isolated execution root.
func (h *OpenCodeGymHarness) WorkspacePath() string {
	if h.arena != nil {
		return h.arena.Path()
	}
	return h.cfg.BaseDir
}

// AdjudicateCall evaluates an OpenCode tool proposal against capability floors,
// self-modification invariants, and argument constraints without executing side-effects.
func (h *OpenCodeGymHarness) AdjudicateCall(call OpenCodeToolCall) OpenCodeAdjudicationRecord {
	canonical := CanonicalOpenCodeTool(call.Name)
	record := OpenCodeAdjudicationRecord{
		ToolCallID:    call.ID,
		Tool:          call.Name,
		CanonicalTool: canonical,
	}

	// 1. Check capability floor
	allowed := false
	for _, a := range h.cfg.AllowedTools {
		if strings.EqualFold(a, call.Name) || strings.EqualFold(a, canonical) {
			allowed = true
			break
		}
	}
	if !allowed {
		record.Verdict = "DENY"
		record.Reason = "DEFAULT_DENY"
		record.ReasonCode = abi.ReasonDefaultDeny
		return record
	}

	// 2. Repair malformed call proposals
	repairedCall, wasRepaired, repairErr := RepairOpenCodeToolCall(call)
	if repairErr != nil {
		record.Verdict = "DENY"
		record.Reason = "MALFORMED"
		record.ReasonCode = abi.ReasonMalformed
		return record
	}
	record.WasRepaired = wasRepaired
	if wasRepaired {
		record.RepairedArguments = repairedCall.Arguments
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(repairedCall.Arguments), &args); err != nil {
		record.Verdict = "DENY"
		record.Reason = "MALFORMED"
		record.ReasonCode = abi.ReasonMalformed
		return record
	}

	// 3. Check for path-based self-modification & out-of-tree escape boundaries
	targetPath := extractTargetPath(args)
	if targetPath != "" {
		cleanTarget := filepath.ToSlash(filepath.Clean(targetPath))

		// Out of tree write / path traversal escape
		if strings.HasPrefix(cleanTarget, "../") || cleanTarget == ".." || filepath.IsAbs(targetPath) {
			// Check if absolute path escapes workspace root
			wsRoot := filepath.ToSlash(h.WorkspacePath())
			if filepath.IsAbs(targetPath) && !strings.HasPrefix(filepath.ToSlash(filepath.Clean(targetPath)), wsRoot) {
				record.Verdict = "DENY"
				record.Reason = "OUT_OF_TREE_WRITE"
				record.ReasonCode = abi.ReasonPolicyBlock
				return record
			}
			if strings.HasPrefix(cleanTarget, "../") || cleanTarget == ".." {
				record.Verdict = "DENY"
				record.Reason = "POLICY_BLOCK"
				record.ReasonCode = abi.ReasonPolicyBlock
				return record
			}
		}

		// Self-modification protection
		for _, g := range h.cfg.SelfModifyGlobs {
			if strings.Contains(cleanTarget, strings.TrimSuffix(g, "/")) {
				record.Verdict = "DENY"
				record.Reason = "SELF_MODIFY"
				record.ReasonCode = abi.ReasonSelfModify
				return record
			}
		}
	}

	// 4. Check for destructive shell command gotchas
	if canonical == "bash" {
		if cmd, ok := args["command"].(string); ok {
			lowerCmd := strings.ToLower(cmd)
			if strings.Contains(lowerCmd, "rm -rf /") ||
				strings.Contains(lowerCmd, "mkfs") ||
				strings.Contains(lowerCmd, ":(){ :|:& };:") ||
				strings.Contains(lowerCmd, "/etc/shadow") {
				record.Verdict = "DENY"
				record.Reason = "POLICY_BLOCK"
				record.ReasonCode = abi.ReasonPolicyBlock
				return record
			}
		}
	}

	if wasRepaired {
		record.Verdict = "REPAIRED"
	} else {
		record.Verdict = "ALLOW"
	}
	return record
}

func extractTargetPath(args map[string]any) string {
	for _, key := range []string{"filePath", "file_path", "path", "target", "filename"} {
		if v, ok := args[key].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// ExecuteHermetic executes an admitted OpenCode tool hermetically within the arena workspace.
func (h *OpenCodeGymHarness) ExecuteHermetic(ctx context.Context, call OpenCodeToolCall) (rawOutput string, normalizedOutput string, err error) {
	canonical := CanonicalOpenCodeTool(call.Name)
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return "", "", fmt.Errorf("invalid json arguments: %w", err)
	}

	ws := h.WorkspacePath()
	switch canonical {
	case "read":
		target, _ := args["filePath"].(string)
		if target == "" {
			return "", "", errors.New("read requires filePath")
		}
		full := filepath.Join(ws, filepath.FromSlash(target))
		data, rerr := os.ReadFile(full)
		if rerr != nil {
			return "", "", rerr
		}
		rawOutput = string(data)

	case "write":
		target, _ := args["filePath"].(string)
		content, _ := args["content"].(string)
		if target == "" {
			return "", "", errors.New("write requires filePath")
		}
		full := filepath.Join(ws, filepath.FromSlash(target))
		if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
			return "", "", err
		}
		if err := os.WriteFile(full, []byte(content), 0644); err != nil {
			return "", "", err
		}
		rawOutput = fmt.Sprintf("\x1b[32m[SUCCESS]\x1b[0m Wrote %d bytes to %s\r\n", len(content), target)

	case "edit":
		target, _ := args["filePath"].(string)
		oldStr, _ := args["oldString"].(string)
		newStr, _ := args["newString"].(string)
		if target == "" || oldStr == "" {
			return "", "", errors.New("edit requires filePath and oldString")
		}
		full := filepath.Join(ws, filepath.FromSlash(target))
		data, rerr := os.ReadFile(full)
		if rerr != nil {
			return "", "", rerr
		}
		cur := string(data)
		if !strings.Contains(cur, oldStr) {
			return "", "", fmt.Errorf("oldString %q not found in file", oldStr)
		}
		updated := strings.Replace(cur, oldStr, newStr, 1)
		if err := os.WriteFile(full, []byte(updated), 0644); err != nil {
			return "", "", err
		}
		rawOutput = fmt.Sprintf("\x1b[34m[EDITED]\x1b[0m Successfully replaced %q with %q in %s\r\n", oldStr, newStr, target)

	case "glob":
		pattern, _ := args["pattern"].(string)
		if pattern == "" {
			pattern = "*"
		}
		var matches []string
		_ = filepath.WalkDir(ws, func(path string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				rel, _ := filepath.Rel(ws, path)
				rel = filepath.ToSlash(rel)
				if matched, _ := filepath.Match(pattern, filepath.Base(rel)); matched || strings.Contains(rel, pattern) {
					matches = append(matches, rel)
				}
			}
			return nil
		})
		sort.Strings(matches)
		rawOutput = strings.Join(matches, "\r\n")

	case "grep":
		pat, _ := args["pattern"].(string)
		if pat == "" {
			return "", "", errors.New("grep requires pattern")
		}
		re, perr := regexp.Compile(pat)
		if perr != nil {
			return "", "", fmt.Errorf("invalid regex: %w", perr)
		}
		var hits []string
		_ = filepath.WalkDir(ws, func(path string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				rel, _ := filepath.Rel(ws, path)
				data, rerr := os.ReadFile(path)
				if rerr == nil {
					lines := strings.Split(string(data), "\n")
					for lineNum, line := range lines {
						if re.MatchString(line) {
							hits = append(hits, fmt.Sprintf("%s:%d: %s", filepath.ToSlash(rel), lineNum+1, strings.TrimSpace(line)))
						}
					}
				}
			}
			return nil
		})
		rawOutput = strings.Join(hits, "\r\n")

	case "bash":
		cmd, _ := args["command"].(string)
		// Safe hermetic built-in execution for tests without spawning untrusted subshells
		if strings.HasPrefix(cmd, "echo ") {
			msg := strings.TrimPrefix(cmd, "echo ")
			msg = strings.Trim(msg, `"'`)
			rawOutput = fmt.Sprintf("\x1b[1m%s\x1b[0m\r\n", msg)
		} else if cmd == "pwd" {
			rawOutput = fmt.Sprintf("%s\r\n", ws)
		} else {
			rawOutput = fmt.Sprintf("executed command: %s\r\n", cmd)
		}

	default:
		return "", "", fmt.Errorf("unsupported hermetic execution tool: %s", canonical)
	}

	norm, _ := NormalizeOpenCodeOutput(rawOutput)
	return rawOutput, norm, nil
}

// RunScenario executes an OpenCode evaluation scenario, recording adjudications,
// normalizing outputs, and generating a verified OpenCodeEvalReceipt.
func (h *OpenCodeGymHarness) RunScenario(ctx context.Context, s OpenCodeEvalScenario) (*OpenCodeEvalReceipt, error) {
	start := time.Now()
	receipt := &OpenCodeEvalReceipt{
		Schema:         OpenCodeEvalSchema,
		ScenarioID:     s.ID,
		Timestamp:      time.Now().UTC(),
		EvaluatedCalls: len(s.ToolCalls),
	}

	// 1. Reset arena to pristine baseline if CoW arena available
	if h.arena != nil {
		if err := h.arena.Reset(ctx); err != nil {
			receipt.Outcome = "FAIL"
			receipt.FailureReason = fmt.Sprintf("arena reset failed: %v", err)
			return receipt, err
		}
	}

	// 2. Seed initial files
	ws := h.WorkspacePath()
	for relPath, content := range s.InitialFiles {
		fullPath := filepath.Join(ws, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			receipt.Outcome = "FAIL"
			receipt.FailureReason = fmt.Sprintf("failed creating dir for %s: %v", relPath, err)
			return receipt, err
		}
		if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
			receipt.Outcome = "FAIL"
			receipt.FailureReason = fmt.Sprintf("failed seeding %s: %v", relPath, err)
			return receipt, err
		}
	}

	// 3. Process and adjudicate each tool call
	for _, call := range s.ToolCalls {
		adj := h.AdjudicateCall(call)
		if adj.Verdict == "ALLOW" || adj.Verdict == "REPAIRED" {
			execCall := call
			if adj.WasRepaired {
				execCall.Name = adj.CanonicalTool
				execCall.Arguments = adj.RepairedArguments
				receipt.RepairedCount++
			}
			receipt.AllowedCount++

			rawOut, normOut, err := h.ExecuteHermetic(ctx, execCall)
			if err != nil {
				adj.ExecutionError = err.Error()
			} else {
				adj.RawOutput = rawOut
				adj.NormalizedOutput = normOut
				adj.OutputDigest = ComputeEvaluationHash(normOut)
				_, adj.ANSISequencesCount = NormalizeOpenCodeOutput(rawOut)
			}
		} else {
			receipt.DeniedCount++
		}
		receipt.Adjudications = append(receipt.Adjudications, adj)
	}

	// 4. Compute deterministic evaluation hash
	receipt.EvaluationHash = ComputeScenarioEvaluationHash(receipt.Adjudications)
	receipt.ExecutionElapsed = time.Since(start)

	// 5. Verify assertions
	passed := true
	var failureReasons []string

	if s.ExpectedVerdict != "" {
		for i, adj := range receipt.Adjudications {
			if s.ExpectedVerdict == "ALLOW" && adj.Verdict != "ALLOW" && adj.Verdict != "REPAIRED" {
				passed = false
				failureReasons = append(failureReasons, fmt.Sprintf("call %d: expected ALLOW, got %s (%s)", i, adj.Verdict, adj.Reason))
			} else if s.ExpectedVerdict == "DENY" && adj.Verdict != "DENY" {
				passed = false
				failureReasons = append(failureReasons, fmt.Sprintf("call %d: expected DENY, got %s", i, adj.Verdict))
			} else if s.ExpectedVerdict == "REPAIRED" && !adj.WasRepaired {
				passed = false
				failureReasons = append(failureReasons, fmt.Sprintf("call %d: expected REPAIRED, got %s (wasRepaired=false)", i, adj.Verdict))
			}
		}
	}

	if s.ExpectedRefusalReason != "" {
		for i, adj := range receipt.Adjudications {
			if adj.Verdict == "DENY" && adj.Reason != s.ExpectedRefusalReason {
				passed = false
				failureReasons = append(failureReasons, fmt.Sprintf("call %d: expected refusal %q, got %q", i, s.ExpectedRefusalReason, adj.Reason))
			}
		}
	}

	if s.ExpectedRepairs > 0 && receipt.RepairedCount != s.ExpectedRepairs {
		passed = false
		failureReasons = append(failureReasons, fmt.Sprintf("expected %d repairs, got %d", s.ExpectedRepairs, receipt.RepairedCount))
	}

	if s.ExpectANSIStripped {
		for i, adj := range receipt.Adjudications {
			if adj.ANSISequencesCount == 0 && adj.RawOutput != "" && strings.Contains(adj.RawOutput, "\x1b") {
				passed = false
				failureReasons = append(failureReasons, fmt.Sprintf("call %d: expected ANSI escapes stripped but count was 0", i))
			}
		}
	}

	if passed {
		receipt.Outcome = "PASS"
	} else {
		receipt.Outcome = "FAIL"
		receipt.FailureReason = strings.Join(failureReasons, "; ")
	}

	return receipt, nil
}
