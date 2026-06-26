package terminalbench

import (
	"fmt"
	"sort"
	"strings"
)

// FailureTaxonomySchema identifies the machine-readable failure-taxonomy
// payload emitted for a Terminal-Bench rehearsal.
const FailureTaxonomySchema = "fak.terminalbench-failure-taxonomy.v1"

// FailureCategory is a closed-vocabulary reason a Terminal-Bench task failed.
// The classifier only ever emits one of these codes, so a downstream compare
// artifact can group failures without parsing free text. The vocabulary mirrors
// the categories called for in the campaign (command error, environment setup
// miss, timeout, package/install issue, bad file edit, test misunderstanding,
// policy block) plus an explicit budget-exhaustion and an UNKNOWN escape hatch.
type FailureCategory string

const (
	// FailureNone marks a task that passed; no recovery is needed.
	FailureNone FailureCategory = "NONE"
	// FailureCommandError is a command that exited non-zero with no more
	// specific cause matched.
	FailureCommandError FailureCategory = "CMD_ERROR"
	// FailureEnvSetupMiss is a missing file, binary, permission, or unset
	// variable the task environment was expected to provide.
	FailureEnvSetupMiss FailureCategory = "ENV_SETUP_MISS"
	// FailureTimeout is a command or task that exceeded its time budget.
	FailureTimeout FailureCategory = "TIMEOUT"
	// FailurePackageInstall is a failed dependency or package install.
	FailurePackageInstall FailureCategory = "PKG_INSTALL"
	// FailureBadFileEdit is an edit that left a file syntactically broken.
	FailureBadFileEdit FailureCategory = "BAD_FILE_EDIT"
	// FailureTestMisunderstood is the agent solving the wrong thing: the trace
	// completed but the benchmark-native test oracle still failed.
	FailureTestMisunderstood FailureCategory = "TEST_MISUNDERSTOOD"
	// FailurePolicyBlock is a fak verdict that denied a command. The recovery
	// branch distinguishes a false-positive (an unnecessary block, which is
	// fak-fixable) from a correct refusal of a dangerous action (which is held,
	// never bypassed).
	FailurePolicyBlock FailureCategory = "POLICY_BLOCK"
	// FailureBudgetExhausted is a task that ran out of turn budget before
	// reaching every required milestone.
	FailureBudgetExhausted FailureCategory = "BUDGET_EXHAUSTED"
	// FailureUnknown is a failure no signal matched; it carries evidence so a
	// human can extend the taxonomy rather than hiding the gap.
	FailureUnknown FailureCategory = "UNKNOWN"
)

// RecoveryAction is a benchmark-legal, task-agnostic move the retry policy may
// take. Every action is general agent behaviour: none of them inject private
// answer knowledge or rewrite the task tests, and none of them bypass a
// correctly-refused dangerous command.
type RecoveryAction string

const (
	// RecoveryNone applies when nothing legal can improve the outcome.
	RecoveryNone RecoveryAction = "NONE"
	// RecoveryNormalizeCommand re-issues the command with normalized syntax
	// (quoting, flag spelling, path separators).
	RecoveryNormalizeCommand RecoveryAction = "NORMALIZE_COMMAND"
	// RecoveryRepairFailedCommand re-issues a non-zero command after a generic,
	// non-answer-bearing repair (e.g. create a missing parent directory).
	RecoveryRepairFailedCommand RecoveryAction = "REPAIR_FAILED_COMMAND"
	// RecoveryRetryPackageInstall retries an install with a refreshed index or
	// an alternate, still-public source.
	RecoveryRetryPackageInstall RecoveryAction = "RETRY_PACKAGE_INSTALL"
	// RecoveryRestoreCheckpointAndRetry rolls the working tree back to the last
	// good evidence checkpoint and retries the step.
	RecoveryRestoreCheckpointAndRetry RecoveryAction = "RESTORE_CHECKPOINT_AND_RETRY"
	// RecoveryEvidenceGuidedRetry resumes from captured evidence instead of
	// restarting the task from scratch.
	RecoveryEvidenceGuidedRetry RecoveryAction = "EVIDENCE_GUIDED_RETRY"
	// RecoveryRereadTestOracle re-reads the public test oracle before retrying,
	// for a task the agent solved in the wrong direction.
	RecoveryRereadTestOracle RecoveryAction = "REREAD_TEST_ORACLE"
	// RecoveryRefinePolicyFalsePositive narrows the policy so a benign command
	// that was wrongly denied is allowed. This reduces unnecessary blocks; it
	// never widens an allow for a dangerous command.
	RecoveryRefinePolicyFalsePositive RecoveryAction = "REFINE_POLICY_FALSE_POSITIVE"
	// RecoveryEscalateForReview holds a correctly-refused dangerous action for a
	// human; the dangerous command is never auto-bypassed.
	RecoveryEscalateForReview RecoveryAction = "ESCALATE_FOR_REVIEW"
)

// CommandOutcome is a per-command execution result a live Harbor/Codex
// rehearsal runner records. The local fixture replay leaves these empty; a real
// run fills the exit code, timeout flag, and a short, lower-cased stderr tail
// keyed by task command. Stderr is match material only and must never carry a
// task's private answer.
type CommandOutcome struct {
	Turn     int    `json:"turn,omitempty"`
	Command  string `json:"command,omitempty"`
	ExitCode int    `json:"exit_code,omitempty"`
	TimedOut bool   `json:"timed_out,omitempty"`
	Stderr   string `json:"stderr,omitempty"`
}

// Failed reports whether the outcome is a non-zero or timed-out command.
func (o CommandOutcome) Failed() bool { return o.TimedOut || o.ExitCode != 0 }

// FailureSignals is the evidence the classifier reads for one task. It is built
// from a TaskReport arm (always available after a replay) plus, for a live run,
// the per-command outcomes the external harness recorded.
type FailureSignals struct {
	TaskID              string
	BudgetTurns         int
	RequiredMilestones  []string
	CompletedMilestones []string
	Tests               []TestResult
	Arm                 ArmResult
	Outcomes            []CommandOutcome
}

// FailureClassification is the machine-readable verdict for one failed task.
type FailureClassification struct {
	TaskID     string          `json:"task_id"`
	Failed     bool            `json:"failed"`
	Category   FailureCategory `json:"category"`
	Reason     string          `json:"reason"`
	Detail     string          `json:"detail,omitempty"`
	Evidence   []string        `json:"evidence,omitempty"`
	Recovery   RecoveryAction  `json:"recovery"`
	Retryable  bool            `json:"retryable"`
	SafetyHold bool            `json:"safety_hold,omitempty"`
}

// installNeedles, envNeedles, and editNeedles are lower-cased stderr substrings
// that route a non-zero command to a more specific category. They match shell
// and toolchain diagnostics, not task content.
var (
	// installNeedles are terminal install-failure phrases only. Progress lines
	// (e.g. "go: downloading"), generic HTTP ("404 not found"), and bare command
	// echoes ("pip install") are deliberately excluded: they appear on
	// successful runs and on non-install failures, so matching them would
	// misroute real CMD_ERROR/edit failures to PKG_INSTALL.
	installNeedles = []string{
		"no matching distribution", "could not find a version", "unable to locate package",
		"failed to fetch", "no module named", "could not resolve dependencies",
		"npm err", "cannot find module", "missing go.sum entry",
	}
	envNeedles = []string{
		"no such file or directory", "command not found", "permission denied",
		"not found in path", "unbound variable", "is not set", "cannot access",
	}
	editNeedles = []string{
		"syntaxerror", "indentationerror", "unexpected token", "unexpected eof",
		"cannot parse", "parse error", "expected declaration", "unterminated",
	}
	// timeoutNeedles are anchored diagnostic phrases. The unanchored "timeout"
	// (matches test names and flags) and "killed" (an OOM signal, not a timeout)
	// are excluded; the reliable timeout signal is the CommandOutcome.TimedOut
	// flag the runner sets, checked first below.
	timeoutNeedles = []string{"timed out", "deadline exceeded"}
)

// Classify maps one task's failure signals to a single closed-vocabulary
// category and a benchmark-legal recovery. It is deterministic: the same
// signals always produce the same classification, and the branches are tried in
// a fixed priority order documented inline.
func Classify(s FailureSignals) FailureClassification {
	c := FailureClassification{TaskID: s.TaskID}
	if !taskFailed(s) {
		c.Category = FailureNone
		c.Reason = string(FailureNone)
		c.Recovery = RecoveryNone
		return c
	}
	c.Failed = true

	// 1. A fak verdict that blocked a command is the highest-leverage signal
	//    because it is the one cause fak itself controls. An unnecessary block
	//    is a false positive we can refine away (reducing unnecessary blocks);
	//    a dangerous block was correct and is held for review, never bypassed.
	if pb, ok := classifyPolicyBlock(s); ok {
		return pb
	}

	// 2. Live per-command outcomes give the most specific cause when present.
	if oc, ok := classifyFromOutcomes(s); ok {
		return oc
	}

	// 3. Fixture replay has no per-command exit codes: fall back to the
	//    milestone/test/budget shape of the arm.
	return classifyFromShape(s, c)
}

func classifyPolicyBlock(s FailureSignals) (FailureClassification, bool) {
	c := FailureClassification{TaskID: s.TaskID, Failed: true, Category: FailurePolicyBlock}
	// Safety takes precedence: a correctly-refused dangerous command holds the
	// whole task for review and is never auto-retried, even when a benign
	// false-positive co-occurred in the same trace. Auto-retrying a task that
	// also tried a dangerous action is exactly the move the policy must not make.
	if dn := len(s.Arm.DangerousBlocks); dn > 0 {
		c.Reason = "POLICY_BLOCK:dangerous_action_held"
		c.Detail = fmt.Sprintf("%d dangerous command(s) were correctly refused; held for review and never auto-bypassed", dn)
		c.Evidence = eventEvidence(s.Arm.DangerousBlocks)
		if un := len(s.Arm.UnnecessaryBlocks); un > 0 {
			c.Detail += fmt.Sprintf("; %d benign command(s) were also denied and should be refined, but the dangerous hold dominates this task", un)
			c.Evidence = append(c.Evidence, eventEvidence(s.Arm.UnnecessaryBlocks)...)
		}
		c.Recovery = RecoveryEscalateForReview
		c.Retryable = false
		c.SafetyHold = true
		return c, true
	}
	if n := len(s.Arm.UnnecessaryBlocks); n > 0 {
		c.Reason = "POLICY_BLOCK:unnecessary_block_false_positive"
		c.Detail = fmt.Sprintf("%d benign command(s) were denied; refining the policy removes the false positive without widening any dangerous allow", n)
		c.Evidence = eventEvidence(s.Arm.UnnecessaryBlocks)
		c.Recovery = RecoveryRefinePolicyFalsePositive
		c.Retryable = true
		return c, true
	}
	return FailureClassification{}, false
}

func classifyFromOutcomes(s FailureSignals) (FailureClassification, bool) {
	last, ok := lastFailedOutcome(s.Outcomes)
	if !ok {
		return FailureClassification{}, false
	}
	c := FailureClassification{TaskID: s.TaskID, Failed: true}
	c.Evidence = []string{outcomeEvidence(last)}
	stderr := strings.ToLower(last.Stderr)
	switch {
	case last.TimedOut:
		// The hard flag the runner sets is the only reliable timeout signal.
		c.Category = FailureTimeout
		c.Reason = "TIMEOUT:command_exceeded_budget"
		c.Recovery = RecoveryRestoreCheckpointAndRetry
	case containsAny(stderr, installNeedles):
		// Specific causes are checked before the textual timeout fallback so a
		// transient "Read timed out" line in an install log cannot shadow the
		// real install failure.
		c.Category = FailurePackageInstall
		c.Reason = "PKG_INSTALL:dependency_install_failed"
		c.Recovery = RecoveryRetryPackageInstall
	case containsAny(stderr, editNeedles):
		c.Category = FailureBadFileEdit
		c.Reason = "BAD_FILE_EDIT:broken_file_after_edit"
		c.Recovery = RecoveryRestoreCheckpointAndRetry
	case containsAny(stderr, envNeedles):
		c.Category = FailureEnvSetupMiss
		c.Reason = "ENV_SETUP_MISS:missing_path_or_permission"
		c.Recovery = RecoveryRepairFailedCommand
	case containsAny(stderr, timeoutNeedles):
		c.Category = FailureTimeout
		c.Reason = "TIMEOUT:command_exceeded_budget"
		c.Recovery = RecoveryRestoreCheckpointAndRetry
	default:
		c.Category = FailureCommandError
		c.Reason = "CMD_ERROR:nonzero_exit"
		c.Recovery = RecoveryRepairFailedCommand
	}
	c.Detail = fmt.Sprintf("turn %d command exited %d", last.Turn, last.ExitCode)
	c.Retryable = true
	return c, true
}

func classifyFromShape(s FailureSignals, c FailureClassification) FailureClassification {
	missing := missingMilestones(s.RequiredMilestones, s.CompletedMilestones)
	switch {
	case len(missing) == 0 && hasFailedTest(s.Tests):
		// Every milestone reached but the oracle still fails: the agent solved
		// the wrong thing rather than hitting an execution error.
		c.Category = FailureTestMisunderstood
		c.Reason = "TEST_MISUNDERSTOOD:oracle_failed_after_full_trace"
		c.Detail = "all milestones completed but the test oracle did not pass"
		c.Evidence = failedTestEvidence(s.Tests)
		c.Recovery = RecoveryRereadTestOracle
		c.Retryable = true
	case len(missing) > 0 && s.BudgetTurns > 0 && s.Arm.Commands >= s.BudgetTurns:
		c.Category = FailureBudgetExhausted
		c.Reason = "BUDGET_EXHAUSTED:turns_spent_before_milestones"
		c.Detail = fmt.Sprintf("%d/%d turns spent, missing milestones: %s", s.Arm.Commands, s.BudgetTurns, strings.Join(missing, ", "))
		c.Evidence = []string{"missing_milestones:" + strings.Join(missing, ",")}
		c.Recovery = RecoveryEvidenceGuidedRetry
		c.Retryable = true
	case len(missing) > 0:
		c.Category = FailureCommandError
		c.Reason = "CMD_ERROR:milestones_incomplete"
		c.Detail = "missing milestones: " + strings.Join(missing, ", ")
		c.Evidence = []string{"missing_milestones:" + strings.Join(missing, ",")}
		c.Recovery = RecoveryRepairFailedCommand
		c.Retryable = true
	default:
		c.Category = FailureUnknown
		c.Reason = "UNKNOWN:no_signal_matched"
		c.Detail = "task failed but no taxonomy branch matched; extend the taxonomy"
		c.Evidence = unknownEvidence(s)
		c.Recovery = RecoveryEvidenceGuidedRetry
		c.Retryable = false
	}
	return c
}

// ClassifyReport walks a replayed Report and classifies every failed fak-arm
// task, giving an end-to-end Report -> taxonomy path without a live harness.
func ClassifyReport(r *Report) []FailureClassification {
	if r == nil {
		return nil
	}
	out := make([]FailureClassification, 0, len(r.Tasks))
	for _, t := range r.Tasks {
		out = append(out, Classify(FailureSignals{
			TaskID:              t.ID,
			BudgetTurns:         t.BudgetTurns,
			RequiredMilestones:  t.Milestones,
			CompletedMilestones: t.Fak.MilestonesCompleted,
			Tests:               t.Tests,
			Arm:                 t.Fak,
		}))
	}
	return out
}

// RetryDirective is the retry policy's decision for one failed task on one
// attempt. The policy is conservative: it never recommends bypassing a
// correctly-refused dangerous command, so honouring it can only hold or reduce
// the unnecessary-block count, never raise it.
type RetryDirective struct {
	TaskID      string         `json:"task_id"`
	Attempt     int            `json:"attempt"`
	ShouldRetry bool           `json:"should_retry"`
	Action      RecoveryAction `json:"action"`
	Rationale   string         `json:"rationale"`
	SafetyHold  bool           `json:"safety_hold,omitempty"`
}

// RetryDirectiveFor decides whether a classified failure should be retried on
// the given attempt, capped at maxAttempts. A safety hold is never retried.
func RetryDirectiveFor(c FailureClassification, attempt, maxAttempts int) RetryDirective {
	d := RetryDirective{TaskID: c.TaskID, Attempt: attempt, Action: RecoveryNone, SafetyHold: c.SafetyHold}
	switch {
	case c.SafetyHold:
		d.Rationale = "dangerous action correctly refused; held for review, not retried"
	case !c.Failed:
		d.Rationale = "task passed; no retry"
	case !c.Retryable:
		d.Rationale = "failure is not retryable with a legal recovery"
	case attempt >= maxAttempts:
		d.Rationale = fmt.Sprintf("attempt budget exhausted (%d/%d)", attempt, maxAttempts)
	default:
		d.ShouldRetry = true
		d.Action = c.Recovery
		d.Rationale = fmt.Sprintf("retry %s with %s", string(c.Category), string(c.Recovery))
	}
	return d
}

// FailureTaxonomy is the machine-readable rollup over a set of classifications.
type FailureTaxonomy struct {
	Schema          string                  `json:"schema"`
	TaskCount       int                     `json:"task_count"`
	FailureCount    int                     `json:"failure_count"`
	CategoryCounts  map[FailureCategory]int `json:"category_counts"`
	SafetyHolds     int                     `json:"safety_holds"`
	RetryableCount  int                     `json:"retryable_count"`
	Classifications []FailureClassification `json:"classifications"`
}

// BuildFailureTaxonomy rolls a list of classifications into a counted summary.
func BuildFailureTaxonomy(cs []FailureClassification) FailureTaxonomy {
	ft := FailureTaxonomy{
		Schema:          FailureTaxonomySchema,
		TaskCount:       len(cs),
		CategoryCounts:  map[FailureCategory]int{},
		Classifications: cs,
	}
	for _, c := range cs {
		if !c.Failed {
			continue
		}
		ft.FailureCount++
		ft.CategoryCounts[c.Category]++
		if c.SafetyHold {
			ft.SafetyHolds++
		}
		if c.Retryable {
			ft.RetryableCount++
		}
	}
	return ft
}

// RenderFailureTaxonomyMarkdown renders the taxonomy as a stable, human-readable
// table for the compare artifact.
func RenderFailureTaxonomyMarkdown(ft FailureTaxonomy) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Terminal-Bench Failure Taxonomy\n\n")
	fmt.Fprintf(&b, "- Schema: `%s`\n", ft.Schema)
	fmt.Fprintf(&b, "- Tasks: `%d`\n", ft.TaskCount)
	fmt.Fprintf(&b, "- Failures: `%d`\n", ft.FailureCount)
	fmt.Fprintf(&b, "- Retryable failures: `%d`\n", ft.RetryableCount)
	fmt.Fprintf(&b, "- Safety holds (never bypassed): `%d`\n\n", ft.SafetyHolds)

	if ft.FailureCount > 0 {
		fmt.Fprintf(&b, "## Category counts\n\n| Category | Count |\n|---|---:|\n")
		for _, cat := range sortedCategories(ft.CategoryCounts) {
			fmt.Fprintf(&b, "| `%s` | %d |\n", cat, ft.CategoryCounts[cat])
		}
		fmt.Fprintf(&b, "\n")
	}

	fmt.Fprintf(&b, "## Failed tasks\n\n")
	fmt.Fprintf(&b, "| Task | Category | Reason | Recovery | Retryable | Safety hold |\n")
	fmt.Fprintf(&b, "|---|---|---|---|:---:|:---:|\n")
	for _, c := range ft.Classifications {
		if !c.Failed {
			continue
		}
		fmt.Fprintf(&b, "| `%s` | `%s` | %s | `%s` | %t | %t |\n",
			c.TaskID, c.Category, mdCell(c.Reason), c.Recovery, c.Retryable, c.SafetyHold)
	}
	return b.String()
}

func taskFailed(s FailureSignals) bool {
	if !s.Arm.TaskSuccess {
		return true
	}
	return len(missingMilestones(s.RequiredMilestones, s.CompletedMilestones)) > 0
}

func missingMilestones(required, completed []string) []string {
	if len(required) == 0 {
		return nil
	}
	done := make(map[string]bool, len(completed))
	for _, m := range completed {
		done[m] = true
	}
	var missing []string
	for _, m := range required {
		if !done[m] {
			missing = append(missing, m)
		}
	}
	sort.Strings(missing)
	return missing
}

func lastFailedOutcome(outcomes []CommandOutcome) (CommandOutcome, bool) {
	for i := len(outcomes) - 1; i >= 0; i-- {
		if outcomes[i].Failed() {
			return outcomes[i], true
		}
	}
	return CommandOutcome{}, false
}

func containsAny(haystack string, needles []string) bool {
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			return true
		}
	}
	return false
}

func eventEvidence(events []CommandEvent) []string {
	out := make([]string, 0, len(events))
	for _, e := range events {
		out = append(out, fmt.Sprintf("turn=%d verdict=%s cmd=%s", e.Turn, e.Verdict, e.Command))
	}
	return out
}

func outcomeEvidence(o CommandOutcome) string {
	return fmt.Sprintf("turn=%d exit=%d timed_out=%t cmd=%s", o.Turn, o.ExitCode, o.TimedOut, o.Command)
}

// unknownEvidence hands a human the raw arm shape for a failure that matched no
// branch, so UNKNOWN is a witness to extend the taxonomy, not a silent gap.
func unknownEvidence(s FailureSignals) []string {
	ev := []string{fmt.Sprintf(
		"arm_shape: task_success=%t test_success=%t commands=%d executed=%d denied=%d budget_turns=%d outcomes=%d",
		s.Arm.TaskSuccess, s.Arm.TestSuccess, s.Arm.Commands, s.Arm.ExecutedCommands,
		s.Arm.DeniedCommands, s.BudgetTurns, len(s.Outcomes),
	)}
	if len(s.CompletedMilestones) > 0 {
		ev = append(ev, "completed_milestones:"+strings.Join(s.CompletedMilestones, ","))
	}
	if ft := failedTestEvidence(s.Tests); len(ft) > 0 {
		ev = append(ev, ft...)
	}
	return ev
}

func hasFailedTest(tests []TestResult) bool {
	for _, t := range tests {
		if !t.Passed {
			return true
		}
	}
	return false
}

func failedTestEvidence(tests []TestResult) []string {
	var out []string
	for _, t := range tests {
		if !t.Passed {
			out = append(out, "failed_test:"+t.Name)
		}
	}
	return out
}

func sortedCategories(m map[FailureCategory]int) []FailureCategory {
	out := make([]FailureCategory, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
