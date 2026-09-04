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

// ShellReadSpec describes an in-process promotable file read command (cat, head, tail).
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
// read operation (cat, head, tail) that can be executed directly in-process.
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

	args := splitShellTokens(cmd)
	if len(args) == 0 {
		return nil, false
	}

	op := strings.ToLower(args[0])
	switch op {
	case "cat":
		return parseCatCommand(args)
	case "head":
		return parseHeadCommand(args)
	case "tail":
		return parseTailCommand(args)
	default:
		return nil, false
	}
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
	spec.FilePath = paths[0]
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
	spec.FilePath = paths[0]
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
	spec.FilePath = paths[0]
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
	tool := strings.ToLower(call.Tool)
	if tool != "bash" && tool != "sh" {
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
