package issueorchestrator

import (
	"fmt"
	"strings"
)

// FormatOpencodePrompt formats a standard task prompt for an OpenCode worker.
func FormatOpencodePrompt(issue Issue) string {
	var b strings.Builder

	issueLabel := fmt.Sprintf("Issue #%d", issue.Number)
	if issue.Number == 0 {
		if issue.Key != "" {
			issueLabel = fmt.Sprintf("Issue %s", issue.Key)
		} else {
			issueLabel = "Issue"
		}
	}

	b.WriteString(fmt.Sprintf("%s: %s\n\n", issueLabel, issue.Title))

	lane := issue.Lane
	if lane == "" {
		lane = "unspecified"
	}
	b.WriteString(fmt.Sprintf("Lane: %s\n", lane))

	if len(issue.Paths) > 0 {
		b.WriteString(fmt.Sprintf("Boundary Paths: %s\n", strings.Join(issue.Paths, ", ")))
	} else if lane != "unspecified" {
		b.WriteString(fmt.Sprintf("Boundary Paths: internal/%s/...\n", lane))
	} else {
		b.WriteString("Boundary Paths: (unspecified)\n")
	}

	b.WriteString("\nVerification Commands:\n")
	if lane != "unspecified" {
		b.WriteString(fmt.Sprintf("- go test -v ./internal/%s/...\n", lane))
		b.WriteString(fmt.Sprintf("- go vet ./internal/%s/...\n", lane))
	} else {
		b.WriteString("- go test -v ./...\n")
		b.WriteString("- go vet ./...\n")
	}

	b.WriteString(hardwareValidationGuidance(issue))
	b.WriteString("\nInstructions:\n")
	b.WriteString("- Strictly adhere to the assigned lane and boundary paths. Do not touch root files (e.g. go.mod, go.sum, dos.toml) or files in other packages.\n")
	b.WriteString("- Execute your deliverable directly within assigned package boundaries as a leaf worker. Prohibit calling the 'task' tool or attempting nested subagent delegation (prevents subagent depth limit recursion failure, #12028).\n")
	b.WriteString("- Deliverable: Deliver a clean defect fix or feature implementation along with a deterministic reproduction/regression unit test.\n")
	b.WriteString("- Autonomous Safe Git Landing Protocol: Execute safe git landing directly within the worker process by default upon test verification:\n")
	b.WriteString("  1. Pre-flight safe sync: fak sync reconcile --apply (or fak sync check)\n")
	b.WriteString(fmt.Sprintf("  2. Stage-and-commit by explicit path: fak commit --path <changed-paths> -m \"<type>(%s): <description> (fak %s)\"\n", lane, lane))
	b.WriteString("  3. Safe unprompted push: fak sync push\n")
	b.WriteString("  Never leave finished work uncommitted or rely on external manual landing.\n")
	b.WriteString("- Provide a 3-line receipt upon completion: status/verdict, changed files & commit SHA, and test output summary, and post directly to the GitHub issue with gh issue comment.\n")
	b.WriteString("- Milestone Progress Protocol: Report milestone progress using structured comment tags in your commentary:\n")
	b.WriteString("  <!-- fak:progress milestone=\"<name>\" delta=\"+N files\" tests=\"<pass|fail>\" -->\n")
	b.WriteString("- Turn Extensions: Request budget extensions when substantive progress is ongoing:\n")
	b.WriteString("  <!-- fak:extend-turns count=\"N\" reason=\"<reason>\" -->\n")

	return b.String()
}

// BuildOpencodeChat constructs a fresh OpencodeChat session definition for an issue.
func BuildOpencodeChat(issue Issue, opts OpencodeChatOptions) OpencodeChat {
	sessionTitle := fmt.Sprintf("Issue #%d: %s", issue.Number, issue.Title)
	if issue.Number == 0 {
		if issue.Key != "" {
			sessionTitle = fmt.Sprintf("Issue %s: %s", issue.Key, issue.Title)
		} else {
			sessionTitle = fmt.Sprintf("Issue: %s", issue.Title)
		}
	}

	prompt := FormatOpencodePrompt(issue)

	var cmd []string
	if opts.Interactive {
		cmd = []string{"opencode", "run", "-i", "--title", sessionTitle}
	} else {
		cmd = []string{"opencode", "run"}
		if opts.PrintLogs {
			cmd = append(cmd, "--print-logs")
		}
		if opts.AutoApprove {
			cmd = append(cmd, "--dangerously-skip-permissions")
		}
		cmd = append(cmd, "--title", sessionTitle)
	}

	if opts.Model != "" {
		cmd = append(cmd, "-m", opts.Model)
	}
	if opts.Agent != "" {
		cmd = append(cmd, "--agent", opts.Agent)
	}
	if opts.Variant != "" {
		cmd = append(cmd, "--variant", opts.Variant)
	}
	if opts.WorktreeDir != "" {
		cmd = append(cmd, "--dir", opts.WorktreeDir)
	}
	if len(opts.ExtraArgs) > 0 {
		cmd = append(cmd, opts.ExtraArgs...)
	}

	cmd = append(cmd, prompt)

	return OpencodeChat{
		IssueNumber:  issue.Number,
		Key:          issue.Key,
		Title:        issue.Title,
		Lane:         issue.Lane,
		SessionTitle: sessionTitle,
		Worktree:     opts.WorktreeDir,
		Command:      cmd,
		Prompt:       prompt,
	}
}
