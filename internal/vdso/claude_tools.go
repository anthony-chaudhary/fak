package vdso

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Claude-native tool names emitted by Claude Code, Cursor, OpenCode, and codetools.
const (
	ToolClaudeRead  = "Read"
	ToolClaudeWrite = "Write"
	ToolClaudeEdit  = "Edit"
	ToolClaudeGlob  = "Glob"
	ToolClaudeGrep  = "Grep"
	ToolClaudeBash  = "Bash"
)

// IsClaudeNativeReadTool reports whether tool is a Claude/Cursor/OpenCode read-only tool.
func IsClaudeNativeReadTool(tool string) bool {
	switch tool {
	case "Read", "Grep", "Glob", "ReadFile", "read", "grep", "glob":
		return true
	default:
		return false
	}
}

// IsClaudeNativeWriteTool reports whether tool is a Claude/Cursor/OpenCode mutating tool.
func IsClaudeNativeWriteTool(tool string) bool {
	switch tool {
	case "Write", "Edit", "write", "edit", "WriteFile":
		return true
	default:
		return false
	}
}

// IsClaudeNativeTool reports whether tool is any Claude/Cursor/OpenCode tool.
func IsClaudeNativeTool(tool string) bool {
	return IsClaudeNativeReadTool(tool) || IsClaudeNativeWriteTool(tool) || tool == "Bash" || tool == "bash"
}

// isReadOnlyCall reports whether a tool call is an effect-free read operation.
// It recognizes explicit readOnlyHint+idempotentHint metadata, as well as
// Claude-native read tools (Read, Grep, Glob) and read-only Bash commands.
func isReadOnlyCall(c *abi.ToolCall) bool {
	if c == nil {
		return false
	}
	if metaTrue(c, "readOnlyHint") && metaTrue(c, "idempotentHint") {
		return true
	}
	if IsClaudeNativeReadTool(c.Tool) {
		return true
	}
	if (c.Tool == "Bash" || c.Tool == "bash") && isReadOnlyBashCall(c) {
		return true
	}
	return false
}

func isReadOnlyBashCall(c *abi.ToolCall) bool {
	if c == nil {
		return false
	}
	if metaTrue(c, "readOnlyHint") {
		return true
	}
	var args []byte
	if c.Args.Kind == abi.RefInline {
		args = c.Args.Inline
	} else if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(context.Background(), c.Args); err == nil {
			args = b
		}
	}
	cmd := ExtractToolCommand(args)
	return IsReadOnlyBashCommand(cmd)
}

// IsReadOnlyBashCommand reports whether a shell command is an effect-free read operation.
func IsReadOnlyBashCommand(cmd string) bool {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return false
	}
	// Redirection or chaining indicates potential side-effects.
	if strings.ContainsAny(cmd, ">|;&") {
		return false
	}
	fields := strings.Fields(cmd)
	if len(fields) == 0 {
		return false
	}
	switch fields[0] {
	case "cat", "head", "tail", "ls", "grep", "wc", "pwd", "find", "stat":
		return true
	case "git":
		if len(fields) >= 2 {
			switch fields[1] {
			case "status", "diff", "log", "show", "rev-parse", "branch":
				return true
			}
		}
	}
	return false
}

// ExtractToolPath extracts the target file path from tool arguments JSON.
// Supports both camelCase (filePath) and snake_case (file_path, path, etc.).
func ExtractToolPath(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	for _, k := range filePathArgKeys {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && strings.TrimSpace(s) != "" {
				if cp := fileCanonPath(s); cp != "" {
					return cp
				}
			}
		}
	}
	return ""
}

// ExtractToolPattern extracts the search pattern from Glob or Grep JSON args.
func ExtractToolPattern(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	for _, k := range []string{"pattern", "query", "regex", "search"} {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s)
			}
		}
	}
	return ""
}

// ExtractToolInclude extracts the include/glob filter from Grep JSON args.
func ExtractToolInclude(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	for _, k := range []string{"include", "glob", "filePattern"} {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s)
			}
		}
	}
	return ""
}

// ExtractToolDirectory extracts the search directory from Glob/Grep JSON args.
// Defaults to "." if omitted or empty.
func ExtractToolDirectory(args []byte) string {
	if len(args) == 0 {
		return "."
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return "."
	}
	for _, k := range []string{"path", "dir", "directory", "filePath", "file_path"} {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && strings.TrimSpace(s) != "" {
				cleaned := filepath.ToSlash(filepath.Clean(strings.TrimSpace(s)))
				if cleaned != "" {
					return cleaned
				}
			}
		}
	}
	return "."
}

// ExtractToolCommand extracts the shell command from Bash JSON args.
func ExtractToolCommand(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	for _, k := range []string{"command", "cmd", "script", "shell_command"} {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil {
				return strings.TrimSpace(s)
			}
		}
	}
	return ""
}

// deriveFileWitness computes an external world-state witness for a file/dir call.
func deriveFileWitness(tool string, args []byte) string {
	if len(args) == 0 {
		return ""
	}
	if ExtractToolPattern(args) != "" || tool == "Glob" || tool == "Grep" {
		dir := ExtractToolDirectory(args)
		st, err := os.Stat(dir)
		if err == nil {
			return "dir:" + dir + ":" + ExtractToolPattern(args) + "@" + strconv.FormatInt(st.ModTime().UnixNano(), 10)
		}
		return ""
	}
	path := ExtractToolPath(args)
	if path == "" {
		return ""
	}
	st, err := os.Stat(path)
	if err == nil && !st.IsDir() {
		mtime := st.ModTime().UnixNano()
		b, err := os.ReadFile(path)
		if err == nil {
			h := sha256.Sum256(b)
			return "file:" + path + "@" + strconv.FormatInt(mtime, 10) + ":" + hex.EncodeToString(h[:])[:24]
		}
		return "file:" + path + "@" + strconv.FormatInt(mtime, 10)
	}
	return ""
}

// extractFileValidation inspects tool call arguments and the live filesystem to populate
// disk freshness metadata for tier-2 caching.
func extractFileValidation(c *abi.ToolCall, args []byte) (filePath string, mtime int64, size int64, hashHex string, isDir bool, ok bool) {
	if c == nil || len(args) == 0 {
		return "", 0, 0, "", false, false
	}
	if ExtractToolPattern(args) != "" || c.Tool == "Glob" || c.Tool == "Grep" {
		dir := ExtractToolDirectory(args)
		st, err := os.Stat(dir)
		if err == nil && st.IsDir() {
			return dir, st.ModTime().UnixNano(), 0, "", true, true
		}
		return dir, 0, 0, "", true, true
	}
	path := ExtractToolPath(args)
	if path != "" {
		return fileStatDigest(path)
	}
	if c.Tool == "Bash" || c.Tool == "bash" {
		cmd := ExtractToolCommand(args)
		if IsReadOnlyBashCommand(cmd) {
			fields := strings.Fields(cmd)
			if len(fields) >= 2 && (fields[0] == "cat" || fields[0] == "head" || fields[0] == "tail") {
				p := fileCanonPath(fields[len(fields)-1])
				if p != "" {
					return fileStatDigest(p)
				}
			}
		}
	}
	return "", 0, 0, "", false, false
}

func fileStatDigest(p string) (string, int64, int64, string, bool, bool) {
	st, err := os.Stat(p)
	if err == nil && !st.IsDir() {
		mtime := st.ModTime().UnixNano()
		size := st.Size()
		b, readErr := os.ReadFile(p)
		var hHex string
		if readErr == nil {
			h := sha256.Sum256(b)
			hHex = hex.EncodeToString(h[:])[:24]
		}
		return p, mtime, size, hHex, false, true
	}
	return p, 0, 0, "", false, true
}
