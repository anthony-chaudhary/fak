package agent

import "strings"

// Effort tier constants defining model reasoning capacity allocations.
const (
	EffortTierNone     = "none"
	EffortTierLow      = "low"
	EffortTierMedium   = "medium"
	EffortTierBalanced = "balanced"
	EffortTierAdaptive = "adaptive"
	EffortTierHigh     = "high"

	BudgetTierNone            = 0
	BudgetTierLow             = 256
	BudgetTierMedium          = 1024
	BudgetTierHigh            = 2048
	BudgetBalancedRoutineTool = 0
	BudgetBalancedError       = 1536
	BudgetBalancedDefault     = 768
)

// TurnAssessment carries signals about the current turn's workload to titrate
// adaptive and balanced reasoning budgets.
type TurnAssessment struct {
	ToolName        string // tool name invoked, e.g. "read", "glob", "grep", "cat"
	IsRoutineTool   bool   // explicit indicator of routine tool inspection
	IsErrorRecovery bool   // explicit indicator of error recovery
	ErrorMessage    string // error message or failure text
}

// IsRoutine reports whether the assessment indicates routine tool inspection
// (e.g. read/glob/grep/cat/etc.) that benefits from zero thinking overhead.
func (ta TurnAssessment) IsRoutine() bool {
	if ta.IsRoutineTool {
		return true
	}
	return isRoutineToolName(ta.ToolName)
}

// IsError reports whether the assessment indicates error recovery (compiler error,
// test failure, panic, policy block) requiring higher thinking capacity.
func (ta TurnAssessment) IsError() bool {
	if ta.IsErrorRecovery {
		return true
	}
	if ta.ErrorMessage != "" {
		return true
	}
	lower := strings.ToLower(ta.ToolName)
	return strings.Contains(lower, "error") || strings.Contains(lower, "fail") || strings.Contains(lower, "panic")
}

func isRoutineToolName(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "read", "glob", "grep", "cat", "view", "read_file", "list_dir", "find", "file_search":
		return true
	}
	return false
}

// AssessTranscriptTurn inspects the trailing tool turns of a message transcript to determine
// whether the turn is routine tool inspection, error recovery, or normal work.
// It scans contiguous trailing tool messages so multi-tool batches with errors are detected.
func AssessTranscriptTurn(messages []Message) (TurnAssessment, bool) {
	if len(messages) == 0 {
		return TurnAssessment{}, false
	}
	lastIdx := len(messages) - 1
	if messages[lastIdx].Role != RoleTool {
		return TurnAssessment{}, false
	}

	ta := TurnAssessment{
		ToolName: messages[lastIdx].Name,
	}

	// Scan contiguous trailing tool messages backward.
	allRoutine := true
	for i := lastIdx; i >= 0 && messages[i].Role == RoleTool; i-- {
		msg := messages[i]
		lower := strings.ToLower(msg.Content)
		if strings.Contains(lower, "compiler error") ||
			strings.Contains(lower, "test failure") ||
			strings.Contains(lower, "panic:") ||
			strings.Contains(lower, "panic") ||
			strings.Contains(lower, "policy block") ||
			strings.Contains(lower, "policy_block") ||
			strings.Contains(lower, "exit status") {
			ta.IsErrorRecovery = true
			ta.ErrorMessage = msg.Content
			return ta, true
		}
		if !isRoutineToolName(msg.Name) {
			allRoutine = false
		}
	}

	if allRoutine {
		ta.IsRoutineTool = true
	}
	return ta, true
}

// ResolveEffortBudget resolves the reasoning token budget given an effort tier,
// an optional explicit token budget override, and optional turn context assessments.
// Explicit budget wins if set and >= 0.
func ResolveEffortBudget(effort string, explicitBudget *int, turnContext ...TurnAssessment) int {
	if explicitBudget != nil && *explicitBudget >= 0 {
		return *explicitBudget
	}

	tier := strings.ToLower(strings.TrimSpace(effort))
	switch tier {
	case EffortTierNone:
		return BudgetTierNone
	case EffortTierLow:
		return BudgetTierLow
	case EffortTierMedium:
		return BudgetTierMedium
	case EffortTierHigh:
		return BudgetTierHigh
	case EffortTierBalanced, EffortTierAdaptive:
		// Error recovery takes precedence over routine inspection.
		for _, ta := range turnContext {
			if ta.IsError() {
				return BudgetBalancedError
			}
		}
		for _, ta := range turnContext {
			if ta.IsRoutine() {
				return BudgetBalancedRoutineTool
			}
		}
		return BudgetBalancedDefault
	default:
		return 0
	}
}
