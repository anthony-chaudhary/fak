package vdso

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ShellReadSpec describes an in-process promotable file read command (cat, head, tail, dir).
type ShellReadSpec struct {
	Op          string `json:"op"`
	FilePath    string `json:"file_path"`
	Lines       int    `json:"lines,omitempty"`
	LineNumbers bool   `json:"line_numbers,omitempty"`
}

// ShellReadResult represents the structured response of an in-process shell read execution,
// conforming to standard agent Bash tool result schemas.
type ShellReadResult struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
}

// ParseShellRead analyzes a shell command to determine if it is a safe, effect-free
// read operation (cat, head, tail, dir) that can be executed directly in-process.
// Returns the parsed ShellReadSpec and true if promotable; otherwise false.
func ParseShellRead(cmd string) (*ShellReadSpec, bool) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return nil, false
	}
	// Redirection, chaining, or command substitution forbid in-process promotion.
	if strings.ContainsAny(cmd, ">|<;`$&$\n\r") {
		return nil, false
	}

	firstWord := peekFirstWord(cmd)
	lowerFirst := strings.ToLower(firstWord)

	switch lowerFirst {
	case "get-content", "gc", "type":
		tokens := splitPowerShellTokens(cmd)
		return parsePowerShellGetContent(tokens)

	case "get-childitem", "gci", "dir":
		tokens := splitPowerShellTokens(cmd)
		return parsePowerShellGetChildItem(tokens)

	case "cat":
		if isPowerShellCat(cmd) {
			tokens := splitPowerShellTokens(cmd)
			return parsePowerShellGetContent(tokens)
		}
		args := splitShellTokens(cmd)
		if len(args) == 0 {
			return nil, false
		}
		return parseCatCommand(args)

	case "head":
		var args []string
		if strings.Contains(cmd, "\\") {
			args = splitPowerShellTokens(cmd)
		} else {
			args = splitShellTokens(cmd)
		}
		if len(args) == 0 {
			return nil, false
		}
		return parseHeadCommand(args)

	case "tail":
		var args []string
		if strings.Contains(cmd, "\\") {
			args = splitPowerShellTokens(cmd)
		} else {
			args = splitShellTokens(cmd)
		}
		if len(args) == 0 {
			return nil, false
		}
		return parseTailCommand(args)

	default:
		return nil, false
	}
}

func peekFirstWord(cmd string) string {
	cmd = strings.TrimSpace(cmd)
	idx := strings.IndexAny(cmd, " \t")
	if idx < 0 {
		return strings.Trim(cmd, "\"'")
	}
	return strings.Trim(cmd[:idx], "\"'")
}

func isPowerShellCat(cmd string) bool {
	if strings.Contains(cmd, "\\") {
		return true
	}
	lower := strings.ToLower(cmd)
	psFlags := []string{
		"-path", "-literalpath", "-totalcount", "-tail",
		"-head", "-first", "-last", "-raw",
	}
	for _, flag := range psFlags {
		if strings.Contains(lower, flag) {
			return true
		}
	}
	return false
}

func splitPowerShellTokens(cmd string) []string {
	var tokens []string
	var current strings.Builder
	inSingle := false
	inDouble := false

	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if c == '\'' && !inDouble {
			inSingle = !inSingle
			continue
		}
		if c == '"' && !inSingle {
			inDouble = !inDouble
			continue
		}
		if (c == ' ' || c == '\t') && !inSingle && !inDouble {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			continue
		}
		current.WriteByte(c)
	}
	if inSingle || inDouble {
		return nil
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

func normalizePath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	p = strings.ReplaceAll(p, "\\", "/")
	return filepath.ToSlash(filepath.Clean(p))
}

func parsePowerShellGetContent(tokens []string) (*ShellReadSpec, bool) {
	if len(tokens) == 0 {
		return nil, false
	}
	spec := &ShellReadSpec{Op: "cat"}
	var paths []string
	hasTotalCount := false
	hasTail := false

	for i := 1; i < len(tokens); i++ {
		token := tokens[i]
		lower := strings.ToLower(token)

		// Check for -Path / -LiteralPath
		if lower == "-path" || lower == "-literalpath" {
			if i+1 >= len(tokens) {
				return nil, false
			}
			i++
			paths = append(paths, tokens[i])
			continue
		}
		if strings.HasPrefix(lower, "-path:") || strings.HasPrefix(lower, "-path=") {
			val := token[6:]
			if val == "" {
				return nil, false
			}
			paths = append(paths, val)
			continue
		}
		if strings.HasPrefix(lower, "-literalpath:") || strings.HasPrefix(lower, "-literalpath=") {
			val := token[13:]
			if val == "" {
				return nil, false
			}
			paths = append(paths, val)
			continue
		}

		// Check for -TotalCount (or -Head, -First)
		if lower == "-totalcount" || lower == "-head" || lower == "-first" {
			if i+1 >= len(tokens) {
				return nil, false
			}
			i++
			n, err := strconv.Atoi(tokens[i])
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTotalCount = true
			continue
		}
		if strings.HasPrefix(lower, "-totalcount:") || strings.HasPrefix(lower, "-totalcount=") {
			val := token[12:]
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTotalCount = true
			continue
		}
		if strings.HasPrefix(lower, "-head:") || strings.HasPrefix(lower, "-head=") {
			val := token[6:]
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTotalCount = true
			continue
		}
		if strings.HasPrefix(lower, "-first:") || strings.HasPrefix(lower, "-first=") {
			val := token[7:]
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTotalCount = true
			continue
		}

		// Check for -Tail (or -Last)
		if lower == "-tail" || lower == "-last" {
			if i+1 >= len(tokens) {
				return nil, false
			}
			i++
			n, err := strconv.Atoi(tokens[i])
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTail = true
			continue
		}
		if strings.HasPrefix(lower, "-tail:") || strings.HasPrefix(lower, "-tail=") {
			val := token[6:]
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTail = true
			continue
		}
		if strings.HasPrefix(lower, "-last:") || strings.HasPrefix(lower, "-last=") {
			val := token[6:]
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
			hasTail = true
			continue
		}

		// Check for -Raw
		if lower == "-raw" {
			continue
		}

		// Unsupported flags fall back to normal shell
		if strings.HasPrefix(token, "-") {
			return nil, false
		}

		// Positional path
		paths = append(paths, token)
	}

	if hasTotalCount && hasTail {
		return nil, false
	}
	if hasTotalCount {
		spec.Op = "head"
	} else if hasTail {
		spec.Op = "tail"
	} else {
		spec.Op = "cat"
	}

	if len(paths) != 1 || paths[0] == "" || paths[0] == "-" {
		return nil, false
	}

	spec.FilePath = normalizePath(paths[0])
	return spec, true
}

func parsePowerShellGetChildItem(tokens []string) (*ShellReadSpec, bool) {
	if len(tokens) == 0 {
		return nil, false
	}
	spec := &ShellReadSpec{Op: "dir"}
	var paths []string

	for i := 1; i < len(tokens); i++ {
		token := tokens[i]
		lower := strings.ToLower(token)

		// Reject recursion explicitly
		if lower == "-recurse" || lower == "-r" || lower == "/s" {
			return nil, false
		}

		// Check for -Path / -LiteralPath
		if lower == "-path" || lower == "-literalpath" {
			if i+1 >= len(tokens) {
				return nil, false
			}
			i++
			paths = append(paths, tokens[i])
			continue
		}
		if strings.HasPrefix(lower, "-path:") || strings.HasPrefix(lower, "-path=") {
			val := token[6:]
			if val == "" {
				return nil, false
			}
			paths = append(paths, val)
			continue
		}
		if strings.HasPrefix(lower, "-literalpath:") || strings.HasPrefix(lower, "-literalpath=") {
			val := token[13:]
			if val == "" {
				return nil, false
			}
			paths = append(paths, val)
			continue
		}

		if lower == "-name" || lower == "/b" {
			continue
		}

		// Unsupported flags fall back to normal shell
		if strings.HasPrefix(token, "-") || strings.HasPrefix(token, "/") {
			return nil, false
		}

		// Positional path
		paths = append(paths, token)
	}

	if len(paths) == 0 {
		spec.FilePath = "."
	} else if len(paths) == 1 {
		if paths[0] == "" {
			spec.FilePath = "."
		} else {
			spec.FilePath = normalizePath(paths[0])
		}
	} else {
		// In-process promotion supports at most one target path
		return nil, false
	}

	return spec, true
}

func splitShellTokens(cmd string) []string {
	var tokens []string
	var current strings.Builder
	inSingle := false
	inDouble := false
	escaped := false

	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if escaped {
			current.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && !inSingle {
			escaped = true
			continue
		}
		if c == '\'' && !inDouble {
			inSingle = !inSingle
			continue
		}
		if c == '"' && !inSingle {
			inDouble = !inDouble
			continue
		}
		if (c == ' ' || c == '\t') && !inSingle && !inDouble {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			continue
		}
		current.WriteByte(c)
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

func parseCatCommand(args []string) (*ShellReadSpec, bool) {
	spec := &ShellReadSpec{Op: "cat"}
	var paths []string

	for i := 1; i < len(args); i++ {
		arg := args[i]
		if arg == "-n" || arg == "--number" {
			spec.LineNumbers = true
		} else if strings.HasPrefix(arg, "-") && arg != "-" {
			// Other cat flags (e.g. -v, -e, -t) fall back to normal shell
			return nil, false
		} else {
			paths = append(paths, arg)
		}
	}

	// In-process cat promotion supports exactly one target file
	if len(paths) != 1 || paths[0] == "-" || paths[0] == "" {
		return nil, false
	}
	spec.FilePath = normalizePath(paths[0])
	return spec, true
}

func parseHeadCommand(args []string) (*ShellReadSpec, bool) {
	spec := &ShellReadSpec{Op: "head", Lines: 10}
	var paths []string

	for i := 1; i < len(args); i++ {
		arg := args[i]
		if arg == "-n" && i+1 < len(args) {
			i++
			n, err := strconv.Atoi(strings.TrimPrefix(args[i], "+"))
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
		} else if strings.HasPrefix(arg, "-n") {
			nStr := strings.TrimPrefix(strings.TrimPrefix(arg, "-n"), "+")
			n, err := strconv.Atoi(nStr)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
		} else if strings.HasPrefix(arg, "-") && len(arg) > 1 && isAllDigits(arg[1:]) {
			n, err := strconv.Atoi(arg[1:])
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
		} else if strings.HasPrefix(arg, "-") {
			return nil, false
		} else {
			paths = append(paths, arg)
		}
	}

	if len(paths) != 1 || paths[0] == "-" || paths[0] == "" {
		return nil, false
	}
	spec.FilePath = normalizePath(paths[0])
	return spec, true
}

func parseTailCommand(args []string) (*ShellReadSpec, bool) {
	spec := &ShellReadSpec{Op: "tail", Lines: 10}
	var paths []string

	for i := 1; i < len(args); i++ {
		arg := args[i]
		if arg == "-n" && i+1 < len(args) {
			i++
			n, err := strconv.Atoi(strings.TrimPrefix(args[i], "+"))
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
		} else if strings.HasPrefix(arg, "-n") {
			nStr := strings.TrimPrefix(strings.TrimPrefix(arg, "-n"), "+")
			n, err := strconv.Atoi(nStr)
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
		} else if strings.HasPrefix(arg, "-") && len(arg) > 1 && isAllDigits(arg[1:]) {
			n, err := strconv.Atoi(arg[1:])
			if err != nil || n < 0 {
				return nil, false
			}
			spec.Lines = n
		} else if strings.HasPrefix(arg, "-") {
			return nil, false
		} else {
			paths = append(paths, arg)
		}
	}

	if len(paths) != 1 || paths[0] == "-" || paths[0] == "" {
		return nil, false
	}
	spec.FilePath = normalizePath(paths[0])
	return spec, true
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// ExecuteInProcessRead reads the target file in-process according to spec without spawning a shell process.
func ExecuteInProcessRead(spec *ShellReadSpec, workDir string) ShellReadResult {
	if spec == nil {
		return ShellReadResult{Stderr: "invalid read specification\n", ExitCode: 1}
	}

	targetPath := spec.FilePath
	if targetPath == "" {
		targetPath = "."
	}
	if !filepath.IsAbs(targetPath) && workDir != "" {
		targetPath = filepath.Join(workDir, targetPath)
	}
	targetPath = filepath.Clean(targetPath)

	fi, err := os.Stat(targetPath)
	if err != nil {
		if os.IsNotExist(err) {
			return ShellReadResult{
				Stderr:   fmt.Sprintf("%s: %s: No such file or directory\n", spec.Op, spec.FilePath),
				ExitCode: 1,
			}
		}
		return ShellReadResult{
			Stderr:   fmt.Sprintf("%s: %s: %v\n", spec.Op, spec.FilePath, err),
			ExitCode: 1,
		}
	}

	if spec.Op == "dir" {
		if fi.IsDir() {
			entries, err := os.ReadDir(targetPath)
			if err != nil {
				return ShellReadResult{
					Stderr:   fmt.Sprintf("%s: %s: %v\n", spec.Op, spec.FilePath, err),
					ExitCode: 1,
				}
			}
			var sb strings.Builder
			for _, entry := range entries {
				sb.WriteString(entry.Name())
				sb.WriteString("\n")
			}
			return ShellReadResult{
				Stdout:   sb.String(),
				ExitCode: 0,
			}
		}
		return ShellReadResult{
			Stdout:   fi.Name() + "\n",
			ExitCode: 0,
		}
	}

	if fi.IsDir() {
		return ShellReadResult{
			Stderr:   fmt.Sprintf("%s: %s: Is a directory\n", spec.Op, spec.FilePath),
			ExitCode: 1,
		}
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		return ShellReadResult{
			Stderr:   fmt.Sprintf("%s: %s: %v\n", spec.Op, spec.FilePath, err),
			ExitCode: 1,
		}
	}

	content := string(data)
	switch spec.Op {
	case "cat":
		if spec.LineNumbers {
			return ShellReadResult{
				Stdout:   formatNumberedLines(content),
				ExitCode: 0,
			}
		}
		return ShellReadResult{
			Stdout:   content,
			ExitCode: 0,
		}

	case "head":
		return ShellReadResult{
			Stdout:   takeHeadLines(content, spec.Lines),
			ExitCode: 0,
		}

	case "tail":
		return ShellReadResult{
			Stdout:   takeTailLines(content, spec.Lines),
			ExitCode: 0,
		}

	default:
		return ShellReadResult{
			Stdout:   content,
			ExitCode: 0,
		}
	}
}

func formatNumberedLines(content string) string {
	if content == "" {
		return ""
	}
	lines := strings.Split(content, "\n")
	hasTrailingNewline := strings.HasSuffix(content, "\n")
	if hasTrailingNewline && len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	var sb strings.Builder
	for i, line := range lines {
		fmt.Fprintf(&sb, "%6d  %s\n", i+1, line)
	}
	return sb.String()
}

func takeHeadLines(content string, n int) string {
	if n <= 0 || content == "" {
		return ""
	}
	lines := strings.Split(content, "\n")
	hasTrailingNewline := strings.HasSuffix(content, "\n")
	if hasTrailingNewline && len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	if n > len(lines) {
		n = len(lines)
	}
	selected := lines[:n]
	res := strings.Join(selected, "\n")
	if hasTrailingNewline || n < len(lines) {
		res += "\n"
	}
	return res
}

func takeTailLines(content string, n int) string {
	if n <= 0 || content == "" {
		return ""
	}
	lines := strings.Split(content, "\n")
	hasTrailingNewline := strings.HasSuffix(content, "\n")
	if hasTrailingNewline && len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	if n > len(lines) {
		n = len(lines)
	}
	start := len(lines) - n
	selected := lines[start:]
	res := strings.Join(selected, "\n")
	if hasTrailingNewline {
		res += "\n"
	}
	return res
}

// PromoteInProcessRead checks if call represents an in-process promotable read command.
// If promotable, it executes the read in-process and returns an *abi.Result with structured JSON
// matching standard Bash result expectations and true. Otherwise, it returns nil, false.
func PromoteInProcessRead(call *abi.ToolCall, workDir string) (*abi.Result, bool) {
	if call == nil {
		return nil, false
	}
	if !isPromotableReadTool(call.Tool) {
		return nil, false
	}

	var argsBytes []byte
	if call.Args.Kind == abi.RefInline {
		argsBytes = call.Args.Inline
	} else if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(context.Background(), call.Args); err == nil {
			argsBytes = b
		}
	}
	if len(argsBytes) == 0 {
		return nil, false
	}

	if workDir == "" {
		workDir = extractToolWorkDir(argsBytes)
	}

	cmd := ExtractToolCommand(argsBytes)
	spec, ok := ParseShellRead(cmd)
	if !ok || spec == nil {
		return nil, false
	}

	readRes := ExecuteInProcessRead(spec, workDir)
	resJSON, err := json.Marshal(readRes)
	if err != nil {
		return nil, false
	}

	status := abi.StatusOK
	if readRes.ExitCode != 0 {
		status = abi.StatusError
	}

	result := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: resJSON, Len: int64(len(resJSON)), Taint: abi.TaintTrusted},
		Status:  status,
		Meta: map[string]string{
			"served_by":     "in_process_read",
			"promoted":      "true",
			"in_process_op": spec.Op,
		},
	}
	return result, true
}

func isPromotableReadTool(tool string) bool {
	switch strings.ToLower(tool) {
	case "bash", "sh", "exec_command", "functions.exec_command",
		"powershell", "pwsh", "shell_command", "functions.shell_command", "cmd":
		return true
	default:
		return false
	}
}

func extractToolWorkDir(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		var s string
		if json.Unmarshal(args, &s) == nil {
			if json.Unmarshal([]byte(s), &m) != nil {
				return ""
			}
		} else {
			return ""
		}
	}
	for _, k := range []string{"workdir", "cwd", "working_directory", "work_dir"} {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s)
			}
		}
	}
	return ""
}
