package agentopt

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Family 1: Prompting & reasoning strategies.
//
// Dynamic few-shot demonstration selection prevents prompt token bloat from static
// exemplars while ensuring relevant context for specialized tool use.
// It retrieves 1-3 highly relevant demonstrations using a hybrid of semantic
// term similarity and historical/predicted tool affinity, while strictly bounding
// total demonstration token expenditure to <= 10% of the input token budget.

// FewShotExemplar represents a demonstration consisting of an input task/prompt,
// reasoning thought or intermediate steps, tool calls, and final response.
type FewShotExemplar struct {
	ID        string         `json:"id"`
	Prompt    string         `json:"prompt"`
	Thought   string         `json:"thought,omitempty"`
	ToolCalls []ToolCall     `json:"tool_calls,omitempty"`
	ToolsUsed []string       `json:"tools_used,omitempty"`
	Output    string         `json:"output"`
	Tokens    int            `json:"tokens,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// EffectiveTools returns the list of tool names associated with this exemplar,
// checking ToolsUsed first and falling back to ToolCalls.
func (e FewShotExemplar) EffectiveTools() []string {
	if len(e.ToolsUsed) > 0 {
		return e.ToolsUsed
	}
	seen := make(map[string]bool)
	var tools []string
	for _, tc := range e.ToolCalls {
		name := strings.TrimSpace(tc.Name)
		if name != "" && !seen[name] {
			seen[name] = true
			tools = append(tools, name)
		}
	}
	return tools
}

// EstimatedTokens calculates or returns the cached token cost of the exemplar.
func (e FewShotExemplar) EstimatedTokens(estimator func(string) int) int {
	if e.Tokens > 0 {
		return e.Tokens
	}
	if estimator == nil {
		estimator = EstimateTokens
	}
	content := fmt.Sprintf("%s\n%s\n%s", e.Prompt, e.Thought, e.Output)
	return estimator(content)
}

// SelectorConfig controls exemplar ranking and token budget thresholds.
type SelectorConfig struct {
	// MaxExemplars is the upper bound on selected demonstrations (1-3).
	MaxExemplars int `json:"max_exemplars"`

	// MaxTokenBudgetRatio is the maximum ratio of the input token budget (default 0.10, <= 10%).
	MaxTokenBudgetRatio float64 `json:"max_token_budget_ratio"`

	// SemanticWeight is the weight for semantic query similarity [0.0 - 1.0].
	SemanticWeight float64 `json:"semantic_weight"`

	// ToolAffinityWeight is the weight for tool affinity and overlap [0.0 - 1.0].
	ToolAffinityWeight float64 `json:"tool_affinity_weight"`

	// TokenEstimator estimates token lengths for exemplar text.
	TokenEstimator func(string) int `json:"-"`
}

// DefaultSelectorConfig returns standard production settings for Family 1 dynamic selection.
func DefaultSelectorConfig() SelectorConfig {
	return SelectorConfig{
		MaxExemplars:        3,
		MaxTokenBudgetRatio: 0.10,
		SemanticWeight:      0.60,
		ToolAffinityWeight:  0.40,
		TokenEstimator:      EstimateTokens,
	}
}

// SelectionRequest holds parameters for matching and bounding few-shot exemplars.
type SelectionRequest struct {
	Query                  string             `json:"query"`
	InputTokenBudget       int                `json:"input_token_budget"`
	PredictedTools         []string           `json:"predicted_tools,omitempty"`
	HistoricalToolAffinity map[string]float64 `json:"historical_tool_affinity,omitempty"`
}

// ExemplarScore details the relevance components for a candidate exemplar.
type ExemplarScore struct {
	ExemplarID        string  `json:"exemplar_id"`
	SemanticScore     float64 `json:"semantic_score"`
	ToolAffinityScore float64 `json:"tool_affinity_score"`
	TotalScore        float64 `json:"total_score"`
	Tokens            int     `json:"tokens"`
}

// SelectionResult represents the chosen exemplars and their budget telemetry.
type SelectionResult struct {
	Selected                []FewShotExemplar `json:"selected"`
	TotalTokens             int               `json:"total_tokens"`
	MaxAllowedTokens        int               `json:"max_allowed_tokens"`
	TokenBudgetRatio        float64           `json:"token_budget_ratio"`
	Scores                  []ExemplarScore   `json:"scores"`
	FormattedDemonstrations string            `json:"formatted_demonstrations"`
}

// DynamicFewShotSelector selects optimal demonstration exemplars dynamically.
type DynamicFewShotSelector struct {
	mu        sync.RWMutex
	config    SelectorConfig
	exemplars []FewShotExemplar
}

// NewDynamicFewShotSelector creates an initialized selector with validated bounds.
func NewDynamicFewShotSelector(cfg SelectorConfig) *DynamicFewShotSelector {
	if cfg.MaxExemplars <= 0 || cfg.MaxExemplars > 3 {
		cfg.MaxExemplars = 3
	}
	if cfg.MaxTokenBudgetRatio <= 0.0 || cfg.MaxTokenBudgetRatio > 0.10 {
		cfg.MaxTokenBudgetRatio = 0.10
	}
	if cfg.SemanticWeight <= 0 && cfg.ToolAffinityWeight <= 0 {
		cfg.SemanticWeight = 0.6
		cfg.ToolAffinityWeight = 0.4
	}
	if cfg.TokenEstimator == nil {
		cfg.TokenEstimator = EstimateTokens
	}
	return &DynamicFewShotSelector{
		config:    cfg,
		exemplars: make([]FewShotExemplar, 0),
	}
}

// AddExemplar registers a new demonstration candidate into the pool.
func (s *DynamicFewShotSelector) AddExemplar(ex FewShotExemplar) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exemplars = append(s.exemplars, ex)
}

// AddExemplars registers a slice of demonstration candidates into the pool.
func (s *DynamicFewShotSelector) AddExemplars(exs []FewShotExemplar) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exemplars = append(s.exemplars, exs...)
}

var commonStopWords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true, "but": true,
	"in": true, "on": true, "at": true, "to": true, "for": true, "with": true,
	"by": true, "from": true, "up": true, "about": true, "into": true,
	"over": true, "after": true, "is": true, "are": true, "was": true,
	"were": true, "be": true, "been": true, "being": true, "have": true,
	"has": true, "had": true, "do": true, "does": true, "did": true,
	"can": true, "could": true, "should": true, "would": true, "will": true,
	"this": true, "that": true, "these": true, "those": true, "it": true,
}

func tokenizeWordSet(text string) map[string]bool {
	tokens := make(map[string]bool)
	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}
	words := strings.FieldsFunc(strings.ToLower(text), f)
	for _, w := range words {
		if len(w) > 1 && !commonStopWords[w] {
			tokens[w] = true
		}
	}
	return tokens
}

func computeSemanticSimilarity(queryTokens map[string]bool, exText string) float64 {
	if len(queryTokens) == 0 {
		return 0.0
	}
	exTokens := tokenizeWordSet(exText)
	if len(exTokens) == 0 {
		return 0.0
	}

	intersection := 0
	for t := range queryTokens {
		if exTokens[t] {
			intersection++
		}
	}

	union := len(queryTokens) + len(exTokens) - intersection
	if union <= 0 {
		return 0.0
	}

	jaccard := float64(intersection) / float64(union)

	// Direct query term recall bonus
	recall := float64(intersection) / float64(len(queryTokens))
	return 0.5*jaccard + 0.5*recall
}

func computeToolAffinity(exTools []string, predictedTools []string, historicalAffinity map[string]float64) float64 {
	if len(exTools) == 0 {
		return 0.0
	}

	var predictedScore float64
	if len(predictedTools) > 0 {
		matchCount := 0
		predSet := make(map[string]bool)
		for _, pt := range predictedTools {
			predSet[strings.ToLower(strings.TrimSpace(pt))] = true
		}
		for _, et := range exTools {
			if predSet[strings.ToLower(strings.TrimSpace(et))] {
				matchCount++
			}
		}
		predictedScore = float64(matchCount) / float64(len(predictedTools))
	}

	var histScore float64
	if len(historicalAffinity) > 0 {
		var sum float64
		for _, et := range exTools {
			if val, ok := historicalAffinity[et]; ok {
				sum += val
			} else if val, ok := historicalAffinity[strings.ToLower(et)]; ok {
				sum += val
			}
		}
		histScore = sum / float64(len(exTools))
		if histScore > 1.0 {
			histScore = 1.0
		}
	}

	if len(predictedTools) > 0 && len(historicalAffinity) > 0 {
		return 0.6*predictedScore + 0.4*histScore
	}
	if len(predictedTools) > 0 {
		return predictedScore
	}
	if len(historicalAffinity) > 0 {
		return histScore
	}
	return 0.0
}

// Select retrieves 1-3 best matching exemplars bounded strictly by <= 10% of input token budget.
func (s *DynamicFewShotSelector) Select(ctx context.Context, req SelectionRequest) (*SelectionResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	maxAllowed := int(float64(req.InputTokenBudget) * s.config.MaxTokenBudgetRatio)
	res := &SelectionResult{
		Selected:         make([]FewShotExemplar, 0),
		TotalTokens:      0,
		MaxAllowedTokens: maxAllowed,
		TokenBudgetRatio: 0.0,
		Scores:           make([]ExemplarScore, 0),
	}

	if len(s.exemplars) == 0 || req.InputTokenBudget <= 0 || maxAllowed <= 0 {
		return res, nil
	}

	queryTokens := tokenizeWordSet(req.Query)

	type scoredItem struct {
		exemplar FewShotExemplar
		score    ExemplarScore
	}

	ranked := make([]scoredItem, 0, len(s.exemplars))
	for _, ex := range s.exemplars {
		exText := fmt.Sprintf("%s %s", ex.Prompt, ex.Thought)
		semScore := computeSemanticSimilarity(queryTokens, exText)

		tools := ex.EffectiveTools()
		toolScore := computeToolAffinity(tools, req.PredictedTools, req.HistoricalToolAffinity)

		totalScore := (s.config.SemanticWeight * semScore) + (s.config.ToolAffinityWeight * toolScore)
		tokens := ex.EstimatedTokens(s.config.TokenEstimator)

		ranked = append(ranked, scoredItem{
			exemplar: ex,
			score: ExemplarScore{
				ExemplarID:        ex.ID,
				SemanticScore:     semScore,
				ToolAffinityScore: toolScore,
				TotalScore:        totalScore,
				Tokens:            tokens,
			},
		})
	}

	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score.TotalScore != ranked[j].score.TotalScore {
			return ranked[i].score.TotalScore > ranked[j].score.TotalScore
		}
		// Tie-break: prefer exemplar with smaller token footprint
		return ranked[i].score.Tokens < ranked[j].score.Tokens
	})

	remainingBudget := maxAllowed
	maxCount := s.config.MaxExemplars
	if maxCount > 3 {
		maxCount = 3
	}

	for _, item := range ranked {
		if len(res.Selected) >= maxCount {
			break
		}
		if item.score.Tokens <= remainingBudget {
			res.Selected = append(res.Selected, item.exemplar)
			res.Scores = append(res.Scores, item.score)
			res.TotalTokens += item.score.Tokens
			remainingBudget -= item.score.Tokens
		}
	}

	if req.InputTokenBudget > 0 {
		res.TokenBudgetRatio = float64(res.TotalTokens) / float64(req.InputTokenBudget)
	}

	res.FormattedDemonstrations = formatDemonstrations(res.Selected)
	return res, nil
}

func formatDemonstrations(exemplars []FewShotExemplar) string {
	if len(exemplars) == 0 {
		return ""
	}
	var sb strings.Builder
	for i, ex := range exemplars {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(fmt.Sprintf("Example %d:\n", i+1))
		sb.WriteString(fmt.Sprintf("User: %s\n", strings.TrimSpace(ex.Prompt)))
		if ex.Thought != "" {
			sb.WriteString(fmt.Sprintf("Reasoning: %s\n", strings.TrimSpace(ex.Thought)))
		}
		sb.WriteString(fmt.Sprintf("Response: %s", strings.TrimSpace(ex.Output)))
	}
	return sb.String()
}
