package adjudicator

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const execCommandReadOnlyFix = "use one supported read-only invocation (git status, or PowerShell Get-Content on a literal path inside the workspace); split mixed commands and gate each separately"

var execGitStatusFlags = map[string]bool{
	"--short": true, "-s": true,
	"--branch": true, "-b": true,
	"--show-stash":   true,
	"--ahead-behind": true, "--no-ahead-behind": true,
	"--column": true, "--no-column": true,
	"--renames": true, "--no-renames": true,
	"--long": true,
}

// execCommandReadOnlyVerdict gives Codex's client-side execution alias a narrow,
// argument-aware admission path. readOnlyHint requests classification; it grants
// no authority by itself. Only the positive grammars below can admit the call.
func (a *Adjudicator) execCommandReadOnlyVerdict(c *abi.ToolCall, args map[string]any) (abi.Verdict, bool) {
	if !execCommandTool(c.Tool) || c.Meta["readOnlyHint"] != "true" {
		return abi.Verdict{}, false
	}
	command, ok := commandArg(args)
	if !ok {
		return execCommandReadOnlyDeny("missing cmd"), true
	}
	workdir, err := a.execCommandWorkdir(args)
	if err != nil {
		return execCommandReadOnlyDeny(err.Error()), true
	}

	var fields []string
	if powerShellExecCommand(args) {
		fields, err = execPowerShellFields(command)
	} else {
		fields, err = execSimpleFields(command)
	}
	if err != nil {
		return execCommandReadOnlyDeny(err.Error()), true
	}
	if len(fields) == 0 {
		return execCommandReadOnlyDeny("missing executable"), true
	}

	head := strings.ToLower(baseCommand(fields[0]))
	switch head {
	case "git", "git.exe":
		if err := validateExecGitStatus(fields); err != nil {
			return execCommandReadOnlyDeny(err.Error()), true
		}
	case "get-content":
		if !powerShellExecCommand(args) {
			return execCommandReadOnlyDeny("Get-Content requires the PowerShell shell"), true
		}
		if err := validateExecGetContent(fields, workdir); err != nil {
			return execCommandReadOnlyDeny(err.Error()), true
		}
	default:
		return execCommandReadOnlyDeny("unsupported executable"), true
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor/cli-read-only"}, true
}

func execCommandTool(tool string) bool {
	return strings.EqualFold(tool, "functions.exec_command") || strings.EqualFold(tool, "exec_command")
}

func execCommandReadOnlyDeny(detail string) abi.Verdict {
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonPolicyBlock,
		By:      "monitor",
		Payload: abi.WitnessPayload{Claim: "exec_command.cmd cli_read_only " + detail},
		Meta: withDenyRule(map[string]string{
			"fix": execCommandReadOnlyFix,
		}, abi.DenyRuleCLIReadOnly),
	}
}

func (a *Adjudicator) execCommandWorkdir(args map[string]any) (string, error) {
	workdir, ok := argString(args, "workdir")
	if !ok || strings.TrimSpace(workdir) == "" {
		return "", fmt.Errorf("missing declared workdir")
	}
	root := filepath.Clean(a.receiptRoot)
	if root == "." || root == "" {
		return "", fmt.Errorf("workspace root is unavailable")
	}
	if !filepath.IsAbs(workdir) {
		workdir = filepath.Join(root, workdir)
	}
	workdir, err := filepath.Abs(workdir)
	if err != nil || !pathWithin(root, workdir) {
		return "", fmt.Errorf("workdir is outside the workspace")
	}
	return filepath.Clean(workdir), nil
}

func execPowerShellFields(command string) ([]string, error) {
	segments, ok := psSegments(command)
	if !ok || len(segments) != 1 {
		return nil, fmt.Errorf("expected one balanced invocation without shell operators")
	}
	fields := make([]string, 0, len(segments[0]))
	for _, token := range segments[0] {
		if token.text == "&" {
			return nil, fmt.Errorf("PowerShell call operators are not supported")
		}
		fields = append(fields, token.text)
	}
	return fields, nil
}

func execSimpleFields(command string) ([]string, error) {
	if strings.ContainsAny(command, ";|&\r\n") || !balancedSimpleQuotes(command) {
		return nil, fmt.Errorf("expected one balanced invocation without shell operators")
	}
	return quoteAwareFields(strings.TrimSpace(command)), nil
}

func balancedSimpleQuotes(command string) bool {
	var quote byte
	for i := 0; i < len(command); i++ {
		c := command[i]
		if quote == 0 {
			if c == '\'' || c == '"' {
				quote = c
			}
			continue
		}
		if c == '\\' && quote == '"' && i+1 < len(command) {
			i++
			continue
		}
		if c == quote {
			quote = 0
		}
	}
	return quote == 0
}

func validateExecGitStatus(fields []string) error {
	if len(fields) < 2 || !strings.EqualFold(strings.Trim(fields[1], "'\""), "status") {
		return fmt.Errorf("only git status is in the read-only grammar")
	}
	for _, raw := range fields[2:] {
		field := strings.Trim(raw, "'\"")
		lower := strings.ToLower(field)
		if execGitStatusFlags[lower] || strings.HasPrefix(lower, "--column=") || strings.HasPrefix(lower, "--find-renames=") {
			continue
		}
		if strings.HasPrefix(lower, "--porcelain=") {
			version := strings.TrimPrefix(lower, "--porcelain=")
			if version == "v1" || version == "v2" || version == "1" || version == "2" {
				continue
			}
		}
		if strings.HasPrefix(lower, "--untracked-files=") {
			mode := strings.TrimPrefix(lower, "--untracked-files=")
			if mode == "no" || mode == "normal" || mode == "all" {
				continue
			}
		}
		if strings.HasPrefix(lower, "--ignored=") {
			mode := strings.TrimPrefix(lower, "--ignored=")
			if mode == "traditional" || mode == "matching" || mode == "no" {
				continue
			}
		}
		return fmt.Errorf("git status uses an unsupported option or pathspec")
	}
	return nil
}

func powerShellExecCommand(args map[string]any) bool {
	shell, ok := argString(args, "shell")
	if !ok {
		return false
	}
	shell = strings.ToLower(filepath.Base(strings.Trim(shell, "'\"")))
	return shell == "powershell" || shell == "powershell.exe" || shell == "pwsh" || shell == "pwsh.exe"
}

func validateExecGetContent(fields []string, workdir string) error {
	if len(fields) < 2 {
		return fmt.Errorf("Get-Content is missing a path")
	}
	var paths []string
	for i := 1; i < len(fields); i++ {
		field := fields[i]
		switch strings.ToLower(field) {
		case "-path", "-literalpath":
			if i+1 >= len(fields) {
				return fmt.Errorf("Get-Content path switch is missing a value")
			}
			i++
			paths = append(paths, fields[i])
		default:
			if strings.HasPrefix(field, "-") {
				return fmt.Errorf("Get-Content uses an unsupported switch")
			}
			paths = append(paths, field)
		}
	}
	for _, operand := range paths {
		if operand == "" || strings.ContainsAny(operand, "$*?[]{}()@`:>,") || strings.Contains(operand, "::") {
			return fmt.Errorf("Get-Content uses a dynamic, provider, or redirected path")
		}
		if len(operand) >= 2 && operand[1] == ':' && !filepath.IsAbs(operand) {
			return fmt.Errorf("Get-Content uses a provider path")
		}
		target := operand
		if !filepath.IsAbs(target) {
			target = filepath.Join(workdir, target)
		}
		target, err := filepath.Abs(target)
		if err != nil || !pathWithin(workdir, target) {
			return fmt.Errorf("Get-Content path is outside workdir")
		}
	}
	return nil
}

func pathWithin(root, target string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(target))
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}
