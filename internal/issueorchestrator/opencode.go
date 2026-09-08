package issueorchestrator

import (
	"fmt"
	"strconv"
	"strings"
)

// isHaloHardwareRelevant checks if the issue is relevant to AMD GPU or Strix Halo hardware validation.
func isHaloHardwareRelevant(issue Issue) bool {
	switch strings.ToLower(strings.TrimSpace(issue.Lane)) {
	case "amdgpu", "compute", "modelperfobs", "nativeperf":
		return true
	}

	for _, p := range issue.Paths {
		pLower := strings.ToLower(p)
		for _, kw := range []string{"amdgpu", "compute", "strix", "halo", "gfx115", "vulkan"} {
			if strings.Contains(pLower, kw) {
				return true
			}
		}
	}

	titleLower := strings.ToLower(issue.Title)
	for _, kw := range []string{"strix", "halo", "gfx115", "amdgpu", "rocm", "vulkan", "rdna"} {
		if strings.Contains(titleLower, kw) {
			return true
		}
	}

	return false
}

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

	b.WriteString("\nInstructions:\n")
	b.WriteString("- Strictly adhere to the assigned lane and boundary paths. Do not touch root files (e.g. go.mod, go.sum, dos.toml) or files in other packages.\n")
	b.WriteString("- Execute your deliverable directly within assigned package boundaries as a leaf worker. Prohibit calling the 'task' tool or attempting nested subagent delegation (prevents subagent depth limit recursion failure, #12028).\n")
	b.WriteString("- Follow the software-first sequence: failing deterministic contract -> smallest implementation -> focused validation -> guarded software landing -> physical qualification. Always author a deterministic software witness (reproduction test) before implementing fixes. Software validation and landing strictly precede physical qualification gates.\n")
	b.WriteString("- Distinguish coordination edges from typed pickup blocks: open dependencies and coordination edges are alignment assumptions to coordinate with adjacent workers, not stop conditions. Treat open dependencies as alignment assumptions unless typed start-blocking (non-blocking unless an explicit typed pickup block is present).\n")
	b.WriteString("- On path collision or active lease conflict, narrow and re-arbitrate a declared disjoint slice before reporting blocked. BLOCKED is permitted only when no executable disjoint slice remains.\n")
	b.WriteString("- Mandatory 4-Phase Delivery and Landing Pipeline (landing by default is required within this worker process; do NOT stop after tests):\n")
	b.WriteString("  Phase 1 [Implement]: Author reproduction test and atomic fix strictly within assigned boundary paths.\n")
	b.WriteString("  Phase 2 [Verify]: Run package verification commands and confirm all tests pass cleanly.\n")
	b.WriteString("  Phase 3 [Autonomous Safe Git Landing - MANDATORY EXIT GATE]:\n")
	b.WriteString("    You are explicitly authorized and required to commit and push right now. Leaving modified files uncommitted or unstaged is a task failure.\n")
	b.WriteString("    1. Pre-flight safe sync: fak sync check (or fak sync reconcile --apply)\n")
	b.WriteString(fmt.Sprintf("    2. Stage-and-commit by explicit path: fak commit --path <changed-paths> -m \"<type>(%s): <description> (fak %s)\"\n", lane, lane))
	b.WriteString("    3. Safe unprompted push: fak sync push\n")
	b.WriteString("    4. If push reports diverged trunk: run 'fak sync reconcile --apply' and retry 'fak sync push'.\n")
	b.WriteString("    Never leave finished work uncommitted or rely on external manual landing.\n")
	b.WriteString("- Phase 4 [Receipt]: Provide a 3-line receipt upon completion: status/verdict, changed files & LANDED COMMIT SHA on origin, and test output summary, and post directly to the GitHub issue with gh issue comment.\n")
	b.WriteString("- Hardware unavailability mandate: If hardware is unavailable/busy, the state becomes PENDING_HARDWARE, and the green software increment must still land. Only issue closure and performance claims wait for physical qualification.\n")
	b.WriteString("- Milestone Progress Protocol: Report milestone progress using structured comment tags in your commentary:\n")
	b.WriteString("  <!-- fak:progress milestone=\"<name>\" delta=\"+N files\" tests=\"<pass|fail>\" -->\n")
	b.WriteString("- Turn Extensions: Request budget extensions when substantive progress is ongoing:\n")
	b.WriteString("  <!-- fak:extend-turns count=\"N\" reason=\"<reason>\" -->\n")
	b.WriteString(hardwareValidationGuidance(issue))

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

	if opts.Agent == "" {
		opts.Agent = "worker"
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
	cmd = append(cmd, "--agent", opts.Agent)
	if opts.Variant != "" {
		cmd = append(cmd, "--variant", opts.Variant)
	}
	if opts.WorktreeDir != "" {
		cmd = append(cmd, "--dir", opts.WorktreeDir)
	}
	if opts.SubagentDepth > 0 {
		cmd = append(cmd, "--subagent-depth", strconv.Itoa(opts.SubagentDepth))
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
