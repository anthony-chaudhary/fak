package debtlane

import (
	"fmt"
	"strings"
)

// OpencodeChat defines the parameters for a dedicated OpenCode chat session to retire debt on a lane.
type OpencodeChat struct {
	Lane         string   `json:"lane"`
	UnitOfWork   string   `json:"unit_of_work"`
	Criticality  string   `json:"criticality"`
	SessionTitle string   `json:"session_title"`
	Worktree     string   `json:"worktree,omitempty"`
	Command      []string `json:"command"`
	Prompt       string   `json:"prompt"`
}

// OpencodeChatOptions configures OpenCode chat command and prompt generation.
type OpencodeChatOptions struct {
	Model       string   `json:"model,omitempty"`
	Agent       string   `json:"agent,omitempty"`
	Variant     string   `json:"variant,omitempty"`
	WorktreeDir string   `json:"worktree_dir,omitempty"`
	PrintLogs   bool     `json:"print_logs,omitempty"`
	AutoApprove bool     `json:"auto_approve,omitempty"`
	Interactive bool     `json:"interactive,omitempty"`
	PerfFocus   bool     `json:"perf_focus,omitempty"`
	ExtraArgs   []string `json:"extra_args,omitempty"`
}

// FormatDebtOpencodePrompt formats a standard task prompt for an OpenCode leaf worker retiring debt on a lane.
func FormatDebtOpencodePrompt(lane DebtLane, opts OpencodeChatOptions) string {
	var b strings.Builder

	b.WriteString(fmt.Sprintf("Maturity Debt Lane: %s (%s)\n\n", lane.Lane, lane.UnitOfWork))
	b.WriteString(fmt.Sprintf("Criticality: %s\n", lane.Criticality))
	b.WriteString(fmt.Sprintf("Current Maturity: %.1f / %.1f (%s) · Debt: %.1f pts\n",
		lane.Maturity, lane.TargetMaturity, lane.MaturityRung, lane.TotalDebt))

	unitPath := strings.ReplaceAll(lane.UnitOfWork, "\\", "/")
	b.WriteString(fmt.Sprintf("Boundary Paths: %s/...\n\n", unitPath))

	b.WriteString("Verification Commands:\n")
	b.WriteString(fmt.Sprintf("- go test -v ./%s/...\n", unitPath))
	b.WriteString(fmt.Sprintf("- go vet ./%s/...\n", unitPath))

	// Performance-critical focus & 3x harsher standards
	b.WriteString("\nRequired Maturity & Performance Deliverables (3x Harsher Continual Performance Standard):\n")

	needsBench := !lane.Evidence.Benchmarked && (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling)
	if needsBench {
		b.WriteString("- [PERFORMANCE BENCHMARK] Author substantive benchmark function: write `Benchmark<Name>(b *testing.B)` in `*_test.go` using `b.Run` or `b.N` loops. Synthetic or empty loops are prohibited. Record performance baseline in BENCHMARK-AUTHORITY.md if applicable.\n")
	}

	needsDogfood := !lane.Evidence.Dogfooded && (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling)
	if needsDogfood {
		b.WriteString("- [RUNTIME PROOF / DOGFOOD] Wire loopback integration test or real execution path. Mocks hide integration bugs; synthetic in-memory assertions do not count as dogfooding. Record proof in runtime-proofs.json.\n")
	}

	if lane.Evidence.ModularityDeficit {
		b.WriteString("- [MODULARITY DEBT] Decompose god-files (>1500 lines) and god-functions (>200 lines) into focused, single-responsibility files. Eliminate model hardcoding and high coupling to enable compiler vectorization and acceleration.\n")
	}

	if lane.Evidence.ExcessComments {
		b.WriteString("- [COMMENT HYGIENE] Prune formulaic comment gaming ('Contract:', 'Invariant:', 'Fail-closed:' comment stuffing) and comment bloat. High-quality code is self-documenting; formulaic comments are penalized as bad debt.\n")
	}

	if !lane.Evidence.HasTests {
		b.WriteString("- [UNIT TESTS] Author comprehensive unit and regression tests in `*_test.go` proving core invariants and boundary conditions.\n")
	}

	if lane.NextAction != "" {
		b.WriteString(fmt.Sprintf("- [NEXT ACTION] %s\n", lane.NextAction))
	}

	b.WriteString("\nExecution Invariants:\n")
	b.WriteString("- Strictly adhere to assigned boundary paths. Do not touch root files (e.g. go.mod, dos.toml) or files in other packages.\n")
	b.WriteString("- Leaf Worker Direct Execution: Execute deliverables directly within assigned package boundaries as a leaf worker. Prohibit calling the 'task' tool or attempting nested subagent delegation (prevents subagent depth limit recursion failure, #12028).\n")
	b.WriteString("- 4-Phase Delivery & Safe Landing Pipeline (landing by default is mandatory; do NOT stop after tests):\n")
	b.WriteString("  Phase 1 [Implement]: Author tests, benchmarks, runtime wiring, or modularity refactor strictly within assigned boundaries.\n")
	b.WriteString("  Phase 2 [Verify]: Run package verification commands (go test, go vet) and confirm all tests pass cleanly.\n")
	b.WriteString("  Phase 3 [Autonomous Safe Git Landing - MANDATORY EXIT GATE]:\n")
	b.WriteString("    1. Pre-flight safe sync: fak sync check (or fak sync reconcile --apply)\n")
	b.WriteString(fmt.Sprintf("    2. Stage-and-commit by explicit path: fak commit --path <changed-paths> -m \"<type>(%s): <description> (fak %s)\"\n", lane.Lane, lane.Lane))
	b.WriteString("    3. Safe unprompted push: fak sync push\n")
	b.WriteString("    4. If push reports diverged trunk: run 'fak sync reconcile --apply' and retry 'fak sync push'.\n")
	b.WriteString("  Phase 4 [Receipt]: Output a 3-line receipt upon completion: status, changed files & landed commit SHA, and test/benchmark output summary.\n")
	b.WriteString("- Milestone Progress Protocol: Report milestone progress using structured comment tags in your commentary:\n")
	b.WriteString("  <!-- fak:progress milestone=\"<name>\" delta=\"+N files\" tests=\"<pass|fail>\" -->\n")

	return b.String()
}

// BuildDebtOpencodeChat constructs a fresh OpencodeChat session definition for a debt lane.
func BuildDebtOpencodeChat(lane DebtLane, opts OpencodeChatOptions) OpencodeChat {
	sessionTitle := fmt.Sprintf("Debt: %s (%s)", lane.Lane, strings.ReplaceAll(lane.UnitOfWork, "\\", "/"))

	if opts.Agent == "" {
		opts.Agent = "worker"
	}
	if opts.Variant == "" {
		opts.Variant = "high"
	}

	prompt := FormatDebtOpencodePrompt(lane, opts)

	var cmd []string
	if opts.Interactive {
		cmd = []string{"opencode", "run", "-i", "--title", sessionTitle}
	} else {
		cmd = []string{"opencode", "run"}
		if opts.PrintLogs {
			cmd = append(cmd, "--print-logs")
		}
		if opts.AutoApprove {
			cmd = append(cmd, "--auto")
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
	if len(opts.ExtraArgs) > 0 {
		cmd = append(cmd, opts.ExtraArgs...)
	}

	cmd = append(cmd, prompt)

	return OpencodeChat{
		Lane:         lane.Lane,
		UnitOfWork:   lane.UnitOfWork,
		Criticality:  string(lane.Criticality),
		SessionTitle: sessionTitle,
		Worktree:     opts.WorktreeDir,
		Command:      cmd,
		Prompt:       prompt,
	}
}

// AttachOpencodeChats generates and attaches OpenCode chat sessions and commands to waves in a WavePlan.
func AttachOpencodeChats(plan *WavePlan, opts OpencodeChatOptions) {
	if plan == nil {
		return
	}
	var allCmds []string
	for i := range plan.Waves {
		w := &plan.Waves[i]
		w.OpencodeChats = make([]OpencodeChat, 0, len(w.Lanes))
		for j := range w.Lanes {
			chat := BuildDebtOpencodeChat(w.Lanes[j], opts)
			w.Lanes[j].OpencodeCommand = chat.Command
			w.OpencodeChats = append(w.OpencodeChats, chat)
			cmdStr := strings.Join(chat.Command, " ")
			allCmds = append(allCmds, cmdStr)
		}
	}
	plan.OpencodeCommands = allCmds
}
