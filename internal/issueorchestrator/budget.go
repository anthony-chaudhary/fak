package issueorchestrator

import (
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
)

// Default budget and iteration thresholds.
const (
	DefaultMinTurnFloor      = 15
	DefaultExtensionTurns    = 5
	DefaultMaxBudgetCap      = 50
	DefaultDeadloopThreshold = 5
)

// BudgetVerdict describes the outcome of evaluating a worker's turn.
type BudgetVerdict string

const (
	VerdictContinue      BudgetVerdict = "VERDICT_CONTINUE"
	VerdictExtended      BudgetVerdict = "VERDICT_EXTENDED"
	VerdictDeadloopAbort BudgetVerdict = "VERDICT_DEADLOOP_ABORT"

	// Aliases matching raw uppercase specification tokens.
	VERDICT_CONTINUE       = VerdictContinue
	VERDICT_EXTENDED       = VerdictExtended
	VERDICT_DEADLOOP_ABORT = VerdictDeadloopAbort
)

// MilestoneRecord records a witnessed progress milestone reported by an agent.
type MilestoneRecord struct {
	Name       string `json:"name"`
	DeltaFiles int    `json:"delta_files"`
	TestsPass  bool   `json:"tests_pass"`
	Turn       int    `json:"turn"`
}

// ExtensionRequest records a requested extension parsed from agent commentary.
type ExtensionRequest struct {
	Count  int    `json:"count"`
	Reason string `json:"reason"`
}

// BudgetMonitor tracks per-worker turn budgets, witnessed progress milestones,
// soft advisory overrun extensions, and deadloop aborts.
type BudgetMonitor struct {
	BaselineBudget               int               `json:"baseline_budget"`
	CurrentBudget                int               `json:"current_budget"`
	CurrentTurn                  int               `json:"current_turn"`
	ConsecutiveZeroProgressTurns int               `json:"consecutive_zero_progress_turns"`
	LastProgressTurn             int               `json:"last_progress_turn"`
	Milestones                   []MilestoneRecord `json:"milestones"`
	DeadloopDetected             bool              `json:"deadloop_detected"`

	IssueNumber int      `json:"issue_number,omitempty"`
	MaxCap      int      `json:"max_cap,omitempty"`
	LastErrors  []string `json:"last_errors,omitempty"`
	Warnings    []string `json:"warnings,omitempty"`
}

// CalculateTurnBudget derives the initial dynamic turn budget B0 from points and expected steps.
// B0 = max(15, int(points * 8)). If points == 0, base on expectedSteps * 2 with floor 15.
func CalculateTurnBudget(points float64, expectedSteps int) int {
	var b0 int
	if points > 0 {
		b0 = int(points * 8)
	} else {
		b0 = expectedSteps * 2
	}
	if b0 < DefaultMinTurnFloor {
		b0 = DefaultMinTurnFloor
	}
	return b0
}

// CalculateIssueTurnBudget computes the initial budget from an Issue and optional points.
func CalculateIssueTurnBudget(issue Issue, points float64) int {
	return CalculateTurnBudget(points, issue.ExpectedSteps)
}

// NewBudgetMonitor initializes a BudgetMonitor with default caps.
func NewBudgetMonitor(issueNumber int, baselineBudget int) *BudgetMonitor {
	if baselineBudget < DefaultMinTurnFloor {
		baselineBudget = DefaultMinTurnFloor
	}
	return &BudgetMonitor{
		BaselineBudget: baselineBudget,
		CurrentBudget:  baselineBudget,
		IssueNumber:    issueNumber,
		MaxCap:         DefaultMaxBudgetCap,
	}
}

var (
	progressTagRegex   = regexp.MustCompile(`(?i)<!--\s*fak:progress\s+([^>]+?)-->`)
	milestoneAttrRegex = regexp.MustCompile(`(?i)milestone\s*=\s*["']([^"']*)["']`)
	deltaAttrRegex     = regexp.MustCompile(`(?i)delta\s*=\s*["']([^"']*)["']`)
	testsAttrRegex     = regexp.MustCompile(`(?i)tests\s*=\s*["']([^"']*)["']`)
	deltaNumRegex      = regexp.MustCompile(`[+-]?\d+`)

	extendTagRegex  = regexp.MustCompile(`(?i)<!--\s*fak:extend-turns\s+([^>]+?)-->`)
	countAttrRegex  = regexp.MustCompile(`(?i)count\s*=\s*["']([^"']*)["']`)
	reasonAttrRegex = regexp.MustCompile(`(?i)reason\s*=\s*["']([^"']*)["']`)
)

// ParseMilestones parses all milestone progress tags in outputText for the given turn.
func ParseMilestones(outputText string, turn int) []MilestoneRecord {
	matches := progressTagRegex.FindAllStringSubmatch(outputText, -1)
	if len(matches) == 0 {
		return nil
	}
	records := make([]MilestoneRecord, 0, len(matches))
	for _, m := range matches {
		attrStr := m[1]
		var name string
		if nameMatch := milestoneAttrRegex.FindStringSubmatch(attrStr); len(nameMatch) > 1 {
			name = nameMatch[1]
		}
		var deltaFiles int
		if deltaMatch := deltaAttrRegex.FindStringSubmatch(attrStr); len(deltaMatch) > 1 {
			if numMatch := deltaNumRegex.FindString(deltaMatch[1]); numMatch != "" {
				deltaFiles, _ = strconv.Atoi(numMatch)
			}
		}
		var testsPass bool
		if testsMatch := testsAttrRegex.FindStringSubmatch(attrStr); len(testsMatch) > 1 {
			val := strings.ToLower(strings.TrimSpace(testsMatch[1]))
			testsPass = val == "pass" || val == "passed" || val == "true" || val == "ok"
		}
		records = append(records, MilestoneRecord{
			Name:       name,
			DeltaFiles: deltaFiles,
			TestsPass:  testsPass,
			Turn:       turn,
		})
	}
	return records
}

// ParseExtensionRequests parses all turn extension request tags in outputText.
func ParseExtensionRequests(outputText string) []ExtensionRequest {
	matches := extendTagRegex.FindAllStringSubmatch(outputText, -1)
	if len(matches) == 0 {
		return nil
	}
	reqs := make([]ExtensionRequest, 0, len(matches))
	for _, m := range matches {
		attrStr := m[1]
		var count int
		if countMatch := countAttrRegex.FindStringSubmatch(attrStr); len(countMatch) > 1 {
			count, _ = strconv.Atoi(countMatch[1])
		}
		var reason string
		if reasonMatch := reasonAttrRegex.FindStringSubmatch(attrStr); len(reasonMatch) > 1 {
			reason = reasonMatch[1]
		}
		reqs = append(reqs, ExtensionRequest{
			Count:  count,
			Reason: reason,
		})
	}
	return reqs
}

// FormatProgressTag renders a canonical milestone progress tag.
func FormatProgressTag(milestone string, deltaFiles int, testsPass bool) string {
	status := "fail"
	if testsPass {
		status = "pass"
	}
	return fmt.Sprintf("<!-- fak:progress milestone=%q delta=\"+%d files\" tests=%q -->", milestone, deltaFiles, status)
}

// FormatExtensionTag renders a canonical turn extension request tag.
func FormatExtensionTag(count int, reason string) string {
	return fmt.Sprintf("<!-- fak:extend-turns count=%q reason=%q -->", strconv.Itoa(count), reason)
}

// RecordTurn records one agent iteration turn and returns an evaluated BudgetVerdict.
func (b *BudgetMonitor) RecordTurn(turn int, outputText string, filesModified int, errSignatures []string) BudgetVerdict {
	if b.BaselineBudget <= 0 {
		b.BaselineBudget = DefaultMinTurnFloor
	}
	if b.CurrentBudget <= 0 {
		b.CurrentBudget = b.BaselineBudget
	}
	if b.MaxCap <= 0 {
		b.MaxCap = DefaultMaxBudgetCap
	}

	b.CurrentTurn = turn

	if b.DeadloopDetected {
		return VerdictDeadloopAbort
	}

	newMilestones := ParseMilestones(outputText, turn)
	hasNewMilestone := len(newMilestones) > 0
	if hasNewMilestone {
		b.Milestones = append(b.Milestones, newMilestones...)
	}

	extensionRequests := ParseExtensionRequests(outputText)

	// Determine if test pass was witnessed in milestones or output.
	testsPassWitnessed := false
	for _, m := range newMilestones {
		if m.TestsPass {
			testsPassWitnessed = true
			break
		}
	}
	if !testsPassWitnessed && (strings.Contains(outputText, `tests="pass"`) || strings.Contains(outputText, `tests="passed"`)) {
		testsPassWitnessed = true
	}

	progressWitnessed := filesModified > 0 || testsPassWitnessed || hasNewMilestone

	if progressWitnessed {
		b.ConsecutiveZeroProgressTurns = 0
		b.LastProgressTurn = turn
	} else {
		b.ConsecutiveZeroProgressTurns++
	}

	// Check repeating error signatures.
	hasRepeatingErrors := false
	if len(errSignatures) > 0 && len(b.LastErrors) > 0 {
		prevSet := make(map[string]struct{}, len(b.LastErrors))
		for _, e := range b.LastErrors {
			prevSet[e] = struct{}{}
		}
		for _, e := range errSignatures {
			if _, ok := prevSet[e]; ok {
				hasRepeatingErrors = true
				break
			}
		}
	}

	// Deadloop check:
	// Only flag DeadloopDetected = true if >= 5 consecutive turns have zero file edits
	// and repeating error signatures or zero milestone movement.
	if filesModified == 0 && b.ConsecutiveZeroProgressTurns >= DefaultDeadloopThreshold {
		if !hasNewMilestone || hasRepeatingErrors {
			b.DeadloopDetected = true
			b.LastErrors = errSignatures
			return VerdictDeadloopAbort
		}
	}

	b.LastErrors = errSignatures

	// Overrun handling:
	// If CurrentTurn > CurrentBudget:
	// If progress was witnessed: grant dynamic extension (+5 turns, up to max cap).
	// Log soft advisory warning.
	if b.CurrentTurn > b.CurrentBudget {
		if progressWitnessed || len(extensionRequests) > 0 {
			ext := DefaultExtensionTurns
			if len(extensionRequests) > 0 && extensionRequests[0].Count > 0 {
				ext = extensionRequests[0].Count
			}
			newBudget := b.CurrentBudget + ext
			if newBudget > b.MaxCap {
				newBudget = b.MaxCap
			}
			granted := newBudget - b.CurrentBudget
			if granted > 0 {
				b.CurrentBudget = newBudget
				warn := fmt.Sprintf("WARN [budget]: worker issue #%d exceeded baseline budget; progress witnessed; granting dynamic extension (%d turns)", b.IssueNumber, granted)
				b.Warnings = append(b.Warnings, warn)
				log.Println(warn)
				return VerdictExtended
			}
		}
	}

	return VerdictContinue
}
