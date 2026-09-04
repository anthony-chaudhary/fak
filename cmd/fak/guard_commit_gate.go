package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"

	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

const (
	guardCommitGateModeEnv  = "FAK_GUARD_COMMIT_GATE_MODE"
	guardCommitGateGradeEnv = "FAK_GUARD_COMMIT_GATE_MIN_GRADE"
	guardCommitGateDefault  = guardPreCompactModeEnforce
)

type guardCommitHookInput struct {
	ToolName  string `json:"tool_name"`
	ToolInput struct {
		Command string `json:"command"`
	} `json:"tool_input"`
}

type guardCommitCandidate struct {
	Subject string
	Paths   []string
}

type guardCommitGitRunner struct{ dir string }

func (g guardCommitGitRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = g.dir
	configureDispatchHelperCommand(cmd)
	return cmd.CombinedOutput()
}

func cmdGuardCommitGate(argv []string) {
	os.Exit(runGuardCommitGate(os.Stdout, os.Stderr, os.Stdin, argv))
}

// runGuardCommitGate is a PreToolUse boundary gate. It only narrows Bash-shaped git
// commit calls; every parse/infra miss fails open, while a typed lint defect blocks in
// enforce mode before git can mutate history.
func runGuardCommitGate(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("guard-commit-gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeRaw := fs.String("mode", os.Getenv(guardCommitGateModeEnv), "off|shadow|enforce")
	rootFlag := fs.String("root", "", "repo root (default: cwd/git discovery)")
	minGradeRaw := fs.String("min-grade", os.Getenv(guardCommitGateGradeEnv), "minimum accepted commit grade A-F (default A)")
	if err := fs.Parse(argv); err != nil {
		return 0
	}
	mode, err := normalizeGuardPreCompactMode(firstNonEmpty(strings.TrimSpace(*modeRaw), guardCommitGateDefault))
	if err != nil || mode == guardPreCompactModeOff {
		return 0
	}
	minScore, ok := guardCommitMinScore(*minGradeRaw)
	if !ok {
		return 0
	}
	payload, err := io.ReadAll(io.LimitReader(stdin, 1<<20))
	if err != nil {
		return 0
	}
	var input guardCommitHookInput
	if json.Unmarshal(payload, &input) != nil || !guardCommitShellTool(input.ToolName) {
		return 0
	}
	candidate, found := parseGuardCommitCandidate(input.ToolInput.Command)
	if !found {
		return 0
	}
	root := strings.TrimSpace(*rootFlag)
	if root == "" {
		root = repoRoot()
	}
	if root == "" {
		return 0
	}
	// Unreadable taxonomy must degrade to skip, not fail an agent for infrastructure.
	if _, err := os.ReadFile(filepath.Join(root, "dos.toml")); err != nil {
		return 0
	}
	if len(candidate.Paths) == 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		out, err := guardCommitGitRunner{dir: root}.Run(ctx, "git", "diff", "--cached", "--name-only", "--diff-filter=ACMR")
		if err != nil {
			return 0
		}
		candidate.Paths = strings.Fields(strings.ReplaceAll(string(out), "\\", "/"))
	}
	report := hooks.LintCommitMessage(candidate.Subject, candidate.Paths, root)
	if report.OK && report.Score >= minScore {
		fmt.Fprintf(stdout, "fak guard commit: allow grade=%s score=%d leaf=%s\n", report.Grade, report.Score, report.Leaf)
		return 0
	}
	hint := guardCommitGateHint(report)
	if mode != guardPreCompactModeEnforce {
		fmt.Fprintf(stderr, "fak guard commit: shadow CLAIM_UNWITNESSED grade=%s score=%d — %s\n", report.Grade, report.Score, hint)
		return 0
	}
	fmt.Fprintf(stderr, "fak guard commit: CLAIM_UNWITNESSED — refusing unbindable commit before history changes (grade=%s score=%d). %s\n", report.Grade, report.Score, hint)
	return 2
}

func guardCommitShellTool(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "bash", "shell_command", "functions.shell_command":
		return true
	default:
		return false
	}
}

func parseGuardCommitCandidate(command string) (guardCommitCandidate, bool) {
	tokens, ok := repoguard.ShlexSplit(command)
	if !ok {
		return guardCommitCandidate{}, false
	}
	gitAt := -1
	for i := 0; i+1 < len(tokens); i++ {
		if strings.EqualFold(filepath.Base(tokens[i]), "git") || strings.EqualFold(filepath.Base(tokens[i]), "git.exe") {
			if tokens[i+1] == "commit" {
				gitAt = i
				break
			}
		}
	}
	if gitAt < 0 {
		return guardCommitCandidate{}, false
	}
	var c guardCommitCandidate
	args := tokens[gitAt+2:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-m", "--message":
			if i+1 >= len(args) {
				return guardCommitCandidate{}, false
			}
			if c.Subject == "" {
				c.Subject = strings.Split(args[i+1], "\n")[0]
			}
			i++
		case "-F", "--file", "-C", "-c", "--reuse-message", "--reedit-message":
			return guardCommitCandidate{}, false // message source is indirect; commit-msg hook remains authoritative
		case "--":
			c.Paths = append(c.Paths, args[i+1:]...)
			i = len(args)
		}
	}
	if strings.TrimSpace(c.Subject) == "" {
		return guardCommitCandidate{}, false
	}
	return c, true
}

func guardCommitMinScore(raw string) (int, bool) {
	switch strings.ToUpper(strings.TrimSpace(firstNonEmpty(raw, "A"))) {
	case "A":
		return 90, true
	case "B":
		return 80, true
	case "C":
		return 70, true
	case "D":
		return 60, true
	case "F":
		return 0, true
	default:
		if n, err := strconv.Atoi(raw); err == nil && n >= 0 && n <= 100 {
			return n, true
		}
		return 0, false
	}
}

func guardCommitGateHint(report hooks.CommitLintReport) string {
	if len(report.Issues) > 0 {
		return strings.Join(report.Issues, "; ")
	}
	if len(report.Notes) > 0 {
		return strings.Join(report.Notes, "; ")
	}
	if report.SuggestedSubject != "" {
		return "use " + report.SuggestedSubject
	}
	return "use a witness-gradeable subject with a bindable, lane-matching (fak <leaf>) stamp"
}
