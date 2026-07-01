package selfquery

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

type ClarificationReason string

const (
	ClarificationMissingContext ClarificationReason = "missing_context"
	ClarificationLowConfidence  ClarificationReason = "low_confidence"
	ClarificationStaleContext   ClarificationReason = "stale_context"
)

type ClarificationChoice struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type ClarificationQuestion struct {
	Key           string                `json:"key"`
	Question      string                `json:"question"`
	Reason        ClarificationReason   `json:"reason"`
	Choices       []ClarificationChoice `json:"choices"`
	DefaultChoice string                `json:"default_choice"`
	BudgetTokens  int                   `json:"budget_tokens"`
	SourceRef     string                `json:"source_ref,omitempty"`
}

type ClarificationPlan struct {
	Questions       []ClarificationQuestion `json:"questions,omitempty"`
	Omitted         int                     `json:"omitted,omitempty"`
	BudgetTokens    int                     `json:"budget_tokens"`
	MaxQuestions    int                     `json:"max_questions"`
	MaxBudgetTokens int                     `json:"max_budget_tokens"`
	Bounded         bool                    `json:"bounded"`
}

type ClarificationOptions struct {
	MaxQuestions    int
	MaxBudgetTokens int
}

func MissingContextClarifications(keys []string) ClarificationPlan {
	assumptions := make([]ctxplan.Assumption, 0, len(keys))
	for _, key := range keys {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		assumptions = append(assumptions, ctxplan.Assumption{
			Key:    key,
			Source: ctxplan.AssumptionUnknown,
		})
	}
	report := ctxplan.AssessAssumptions(assumptions, ctxplan.DefaultAssumptionPolicy())
	return PlanClarifications(report, ClarificationOptions{})
}

func PlanClarifications(report ctxplan.AssumptionReport, opt ClarificationOptions) ClarificationPlan {
	opt = normalizeClarificationOptions(opt)
	plan := ClarificationPlan{
		MaxQuestions:    opt.MaxQuestions,
		MaxBudgetTokens: opt.MaxBudgetTokens,
		Bounded:         true,
	}
	for _, assessment := range report.Assessments {
		if assessment.Action == ctxplan.AssumptionUse {
			continue
		}
		q := clarificationQuestion(assessment)
		if len(plan.Questions) >= opt.MaxQuestions || plan.BudgetTokens+q.BudgetTokens > opt.MaxBudgetTokens {
			plan.Omitted++
			continue
		}
		plan.Questions = append(plan.Questions, q)
		plan.BudgetTokens += q.BudgetTokens
	}
	return plan
}

func normalizeClarificationOptions(opt ClarificationOptions) ClarificationOptions {
	if opt.MaxQuestions <= 0 {
		opt.MaxQuestions = 3
	}
	if opt.MaxBudgetTokens <= 0 {
		opt.MaxBudgetTokens = 120
	}
	return opt
}

func clarificationQuestion(a ctxplan.AssumptionAssessment) ClarificationQuestion {
	reason := clarificationReason(a)
	q := ClarificationQuestion{
		Key:           a.Key,
		Question:      clarificationText(a, reason),
		Reason:        reason,
		Choices:       clarificationChoices(reason),
		DefaultChoice: defaultClarificationChoice(reason),
		SourceRef:     a.SourceRef,
	}
	q.BudgetTokens = estimateQuestionBudget(q)
	return q
}

func clarificationReason(a ctxplan.AssumptionAssessment) ClarificationReason {
	if a.Action == ctxplan.AssumptionRefresh || a.Source == ctxplan.AssumptionStale {
		return ClarificationStaleContext
	}
	if strings.Contains(a.Reason, "confidence") {
		return ClarificationLowConfidence
	}
	return ClarificationMissingContext
}

func clarificationText(a ctxplan.AssumptionAssessment, reason ClarificationReason) string {
	subject := a.Key
	if strings.TrimSpace(a.Statement) != "" {
		subject = fmt.Sprintf("%s: %s", a.Key, strings.TrimSpace(a.Statement))
	}
	switch reason {
	case ClarificationStaleContext:
		return fmt.Sprintf("Refresh stale context for %s before acting.", subject)
	case ClarificationLowConfidence:
		return fmt.Sprintf("Confirm low-confidence context for %s before acting.", subject)
	default:
		return fmt.Sprintf("Provide missing context for %s before acting.", subject)
	}
}

func clarificationChoices(reason ClarificationReason) []ClarificationChoice {
	switch reason {
	case ClarificationStaleContext:
		return []ClarificationChoice{
			{Value: "refresh_source", Label: "Refresh source"},
			{Value: "provide_value", Label: "Provide value"},
			{Value: "skip_effect", Label: "Skip effect"},
		}
	default:
		return []ClarificationChoice{
			{Value: "provide_value", Label: "Provide value"},
			{Value: "page_in_context", Label: "Page in context"},
			{Value: "skip_effect", Label: "Skip effect"},
		}
	}
}

func defaultClarificationChoice(reason ClarificationReason) string {
	if reason == ClarificationStaleContext {
		return "refresh_source"
	}
	return "provide_value"
}

func estimateQuestionBudget(q ClarificationQuestion) int {
	n := len(strings.Fields(q.Question)) + 4
	for _, choice := range q.Choices {
		n += len(strings.Fields(choice.Label)) + 1
	}
	if n < 12 {
		return 12
	}
	return n
}
