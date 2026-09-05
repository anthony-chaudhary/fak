package vdso

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// IsPromotableShellTool returns true if tool is a recognized shell tool capable
// of being promoted to in-process read execution.
func IsPromotableShellTool(tool string) bool {
	switch strings.ToLower(tool) {
	case "bash", "sh", "exec_command", "functions.exec_command", "shell_command", "functions.shell_command", "powershell", "pwsh":
		return true
	default:
		return false
	}
}

// ExtractToolWorkdir inspects JSON arguments for workdir, work_dir, or cwd keys
// and returns the trimmed directory path if found.
func ExtractToolWorkdir(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	lowerMap := make(map[string]json.RawMessage, len(m))
	for k, v := range m {
		lowerMap[strings.ToLower(k)] = v
	}
	for _, k := range []string{"workdir", "work_dir", "cwd"} {
		if raw, ok := lowerMap[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil {
				if trimmed := strings.TrimSpace(s); trimmed != "" {
					return trimmed
				}
			}
		}
	}
	return ""
}

// ShellReadSpec describes an in-process promotable file read command (cat, head, tail, get-content, type, get-childitem).
type ShellReadSpec struct {
	Op          string `json:"op"`
	FilePath    string `json:"file_path"`
	Lines       int    `json:"lines,omitempty"`
	LineNumbers bool   `json:"line_numbers,omitempty"`
	Tail        bool   `json:"tail,omitempty"`
	NameOnly    bool   `json:"name_only,omitempty"`
	HasLines    bool   `json:"has_lines,omitempty"`
}

// ShellReadResult represents the structured response of an in-process shell read execution,
// conforming to standard agent Bash tool result schemas.
type ShellReadResult struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
}

// ParseShellRead analyzes a shell command to determine if it is a safe, effect-free
// read operation (cat, head, tail, get-content, gc, type, get-childitem, gci, dir)
// that can be executed directly in-process.
// Returns the parsed ShellReadSpec and true if promotable; otherwise false.
func ParseShellRead(cmd string) (*ShellReadSpec, bool) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return nil, false
	}
	// Redirection, chaining, or command substitution forbid in-process promotion.
	if strings.ContainsAny(cmd, ">|<;$&$\n\r") {
		return nil, false
	}

	firstWord := peekFirstCommandToken(cmd)
	op := strings.ToLower(firstWord)
	switch op {
	case "cat":
		if strings.Contains(cmd, "`") {
			return nil, false
		}
		args := splitShellTokens(cmd)
		return parseCatCommand(args)
	case "head":
		if strings.Contains(cmd, "`") {
			return nil, false
		}
		args := splitShellTokens(cmd)
		return parseHeadCommand(args)
	case "tail":
		if strings.Contains(cmd, "`") {
			return nil, false
		}
		args := splitShellTokens(cmd)
		return parseTailCommand(args)
	case "get-content", "gc":
		args := splitPowerShellTokens(cmd)
		return parseGetContentCommand(args)
	case "type":
		args := splitPowerShellTokens(cmd)
		return parseTypeCommand(args)
	case "get-childitem", "gci", "dir":
		args := splitPowerShellTokens(cmd)
		return parseGetChildItemCommand(args)
	default:
		return nil, false
	}
}

func peekFirstCommandToken(cmd string) string {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return ""
	}
	if cmd[0] == '"' || cmd[0] == '\'' {
		quote := cmd[0]
		end := strings.IndexByte(cmd[1:], quote)
		if end != -1 {
			return cmd[1 : end+1]
		}
	}
	fields := strings.Fields(cmd)
	if len(fields) > 0 {
		return fields[0]
	}
	return ""
}

func splitPowerShellTokens(cmd string) []string {
	var tokens []string
	var current strings.Builder
	inSingle := false
	inDouble := false
	tokenStarted := false

	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if inSingle {
			if c == '\'' {
				if i+1 < len(cmd) && cmd[i+1] == '\'' {
					current.WriteByte('\'')
					i++
				} else {
					inSingle = false
				}
			} else {
				current.WriteByte(c)
			}
			continue
		}

		if inDouble {
			if c == '`' {
				if i+1 < len(cmd) {
					i++
					current.WriteByte(cmd[i])
				}
				continue
			}
			if c == '"' {
				if i+1 < len(cmd) && cmd[i+1] == '"' {
					current.WriteByte('"')
					i++
				} else {
					inDouble = false
				}
				continue
			}
			current.WriteByte(c)
			continue
		}

		// Outside quotes
		if c == '`' {
			tokenStarted = true
			if i+1 < len(cmd) {
				i++
				current.WriteByte(cmd[i])
			}
			continue
		}
		if c == '\'' {
			tokenStarted = true
			inSingle = true
			continue
		}
		if c == '"' {
			tokenStarted = true
			inDouble = true
			continue
		}
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			if tokenStarted || current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
				tokenStarted = false
			}
			continue
		}
		tokenStarted = true
		current.WriteByte(c)
	}
	if tokenStarted || current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
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

func parseGetContentCommand(args []string) (*ShellReadSpec, bool) {
	spec := &ShellReadSpec{Op: "get-content"}
	var paths []string

	for i := 1; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-") {
			var flagName, flagVal string
			hasColon := false
			if idx := strings.IndexByte(arg, ':'); idx != -1 {
				flagName = strings.ToLower(arg[:idx])
				flagVal = arg[idx+1:]
				hasColon = true
			} else {
				flagName = strings.ToLower(arg)
			}

			switch flagName {
			case "-wait", "-stream", "-filter", "-include", "-exclude":
				return nil, false
			case "-raw":
				// ignore
			case "-path", "-literalpath":
				var p string
				if hasColon {
					p = flagVal
				} else if i+1 < len(args) {
					i++
					p = args[i]
				} else {
					return nil, false
				}
				paths = append(paths, p)
			case "-totalcount", "-head", "-first":
				var nStr string
				if hasColon {
					nStr = flagVal
				} else if i+1 < len(args) {
					i++
					nStr = args[i]
				} else {
					return nil, false
				}
				n, err := strconv.Atoi(strings.TrimPrefix(nStr, "+"))
				if err != nil || n < 0 {
					return nil, false
				}
				spec.Lines = n
				spec.HasLines = true
				spec.Tail = false
			case "-tail", "-last":
				var nStr string
				if hasColon {
					nStr = flagVal
				} else if i+1 < len(args) {
					i++
					nStr = args[i]
				} else {
					return nil, false
				}
				n, err := strconv.Atoi(strings.TrimPrefix(nStr, "+"))
				if err != nil || n < 0 {
					return nil, false
				}
				spec.Lines = n
				spec.HasLines = true
				spec.Tail = true
			default:
				return nil, false
			}
		} else {
			paths = append(paths, arg)
		}
	}

	if len(paths) != 1 || paths[0] == "" {
		return nil, false
	}
	if strings.ContainsAny(paths[0], "*?") {
		return nil, false
	}
	spec.FilePath = paths[0]
	return spec, true
}

func parseTypeCommand(args []string) (*ShellReadSpec, bool) {
	hasFlags := false
	for _, arg := range args[1:] {
		if strings.HasPrefix(arg, "-") {
			hasFlags = true
			break
		}
	}
	if hasFlags {
		spec, ok := parseGetContentCommand(args)
		if !ok || spec == nil {
			return nil, false
		}
		spec.Op = "type"
		return spec, true
	}

	var paths []string
	for _, arg := range args[1:] {
		paths = append(paths, arg)
	}
	if len(paths) != 1 || paths[0] == "" {
		return nil, false
	}
	if strings.ContainsAny(paths[0], "*?") {
		return nil, false
	}
	return &ShellReadSpec{
		Op:       "type",
		FilePath: paths[0],
	}, true
}

func parseGetChildItemCommand(args []string) (*ShellReadSpec, bool) {
	spec := &ShellReadSpec{Op: "get-childitem", FilePath: "."}
	var paths []string

	for i := 1; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-") {
			var flagName, flagVal string
			hasColon := false
			if idx := strings.IndexByte(arg, ':'); idx != -1 {
				flagName = strings.ToLower(arg[:idx])
				flagVal = arg[idx+1:]
				hasColon = true
			} else {
				flagName = strings.ToLower(arg)
			}

			switch flagName {
			case "-recurse", "-r", "-s", "-filter", "-include", "-exclude":
				return nil, false
			case "-name":
				spec.NameOnly = true
			case "-path", "-literalpath":
				var p string
				if hasColon {
					p = flagVal
				} else if i+1 < len(args) {
					i++
					p = args[i]
				} else {
					return nil, false
				}
				paths = append(paths, p)
			default:
				return nil, false
			}
		} else {
			paths = append(paths, arg)
		}
	}

	if len(paths) > 1 {
		return nil, false
	}
	if len(paths) == 1 {
		if paths[0] == "" || strings.ContainsAny(paths[0], "*?") {
			return nil, false
		}
		spec.FilePath = paths[0]
	}
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

	if spec.Op == "get-childitem" {
		return executeGetChildItem(spec, workDir)
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

	case "get-content", "type":
		if spec.Tail {
			return ShellReadResult{
				Stdout:   takeTailLines(content, spec.Lines),
				ExitCode: 0,
			}
		} else if spec.HasLines || spec.Lines > 0 {
			return ShellReadResult{
				Stdout:   takeHeadLines(content, spec.Lines),
				ExitCode: 0,
			}
		}
		return ShellReadResult{
			Stdout:   content,
			ExitCode: 0,
		}

	default:
		return ShellReadResult{
			Stdout:   content,
			ExitCode: 0,
		}
	}
}

func executeGetChildItem(spec *ShellReadSpec, workDir string) ShellReadResult {
	targetPath := spec.FilePath
	if !filepath.IsAbs(targetPath) && workDir != "" {
		targetPath = filepath.Join(workDir, targetPath)
	}
	targetPath = filepath.Clean(targetPath)

	fi, err := os.Stat(targetPath)
	if err != nil {
		if os.IsNotExist(err) {
			return ShellReadResult{
				Stderr:   fmt.Sprintf("Get-ChildItem: Cannot find path '%s' because it does not exist.\n", spec.FilePath),
				ExitCode: 1,
			}
		}
		return ShellReadResult{
			Stderr:   fmt.Sprintf("Get-ChildItem: %v\n", err),
			ExitCode: 1,
		}
	}

	if !fi.IsDir() {
		if spec.NameOnly {
			return ShellReadResult{
				Stdout:   fi.Name() + "\n",
				ExitCode: 0,
			}
		}
		var sb strings.Builder
		sb.WriteString(formatPowerShellHeader())
		sb.WriteString(formatPowerShellRow(fi.Name(), false, fi.ModTime(), fi.Size()))
		return ShellReadResult{
			Stdout:   sb.String(),
			ExitCode: 0,
		}
	}

	entries, err := os.ReadDir(targetPath)
	if err != nil {
		return ShellReadResult{
			Stderr:   fmt.Sprintf("Get-ChildItem: %v\n", err),
			ExitCode: 1,
		}
	}

	if spec.NameOnly {
		var sb strings.Builder
		for _, entry := range entries {
			sb.WriteString(entry.Name())
			sb.WriteByte('\n')
		}
		return ShellReadResult{
			Stdout:   sb.String(),
			ExitCode: 0,
		}
	}

	var sb strings.Builder
	sb.WriteString(formatPowerShellHeader())
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}
		sb.WriteString(formatPowerShellRow(entry.Name(), entry.IsDir(), info.ModTime(), info.Size()))
	}
	return ShellReadResult{
		Stdout:   sb.String(),
		ExitCode: 0,
	}
}

func formatPowerShellHeader() string {
	return fmt.Sprintf("%-20s%-22s%10s %s\n%-20s%-22s%10s %s\n",
		"Mode", "LastWriteTime", "Length", "Name",
		"----", "-------------", "------", "----")
}

func formatPowerShellRow(name string, isDir bool, modTime time.Time, size int64) string {
	mode := "-a---"
	lengthStr := strconv.FormatInt(size, 10)
	if isDir {
		mode = "d----"
		lengthStr = ""
	}
	timeStr := modTime.Format("1/2/2006   3:04 PM")
	return fmt.Sprintf("%-20s%-22s%10s %s\n", mode, timeStr, lengthStr, name)
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
	if !IsPromotableShellTool(call.Tool) {
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
		workDir = ExtractToolWorkdir(argsBytes)
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
