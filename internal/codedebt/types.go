package codedebt

import "time"

// Category represents a structural code debt category.
type Category string

const (
	CategoryModularity          Category = "modularity"
	CategoryInternalConsistency Category = "internal_consistency"
	CategoryInternalCoherence   Category = "internal_coherence"
	CategoryUncategorized       Category = "uncategorized"
)

// CategoryDefinitions maps categories to their definitions.
var CategoryDefinitions = map[Category]string{
	CategoryModularity:          "Boundaries or units are too large or coupled to change independently.",
	CategoryInternalConsistency: "The implementation contradicts its own declared rules or conventions.",
	CategoryInternalCoherence:   "Related implementation pieces do not form a complete, intelligible whole.",
}

// KPICategories maps each scorecard KPI to its structural category.
var KPICategories = map[string][]Category{
	"architecture":       {CategoryModularity},
	"build":              {CategoryInternalCoherence},
	"vet":                {CategoryInternalCoherence},
	"format":             {CategoryInternalConsistency},
	"deps":               {CategoryInternalConsistency},
	"honesty":            {CategoryInternalConsistency},
	"tests":              {CategoryInternalCoherence},
	"assertion_strength": {CategoryInternalCoherence},
	"ship_integrity":     {CategoryInternalConsistency},
}

// Defect is a single, concrete unit of code debt.
type Defect struct {
	KPI        string     `json:"kpi"`
	Categories []Category `json:"categories"`
	Raw        string     `json:"raw"`
	Path       string     `json:"path,omitempty"`
	Package    string     `json:"package,omitempty"`
	Kind       string     `json:"kind,omitempty"`
	Line       int        `json:"line,omitempty"`
}

// Report holds the complete inventory of code quality debt.
type Report struct {
	Timestamp     time.Time             `json:"timestamp"`
	Workspace     string                `json:"workspace"`
	Deterministic bool                  `json:"deterministic"`
	TotalDebt     int                   `json:"total_debt"`
	Score         float64               `json:"score"`
	Grade         string                `json:"grade"`
	DebtByKPI     map[string]int        `json:"debt_by_kpi"`
	DebtByCat     map[Category]int      `json:"debt_by_category"`
	DebtByPkg     map[string]int        `json:"debt_by_package"`
	Defects       []Defect              `json:"defects"`
	SoftSignals   []string              `json:"soft_signals,omitempty"`
	KPISummaries  map[string]KPISummary `json:"kpi_summaries,omitempty"`
}

// KPISummary summarizes the status of a single KPI.
type KPISummary struct {
	KPI        string     `json:"kpi"`
	Score      int        `json:"score"`
	Debt       int        `json:"debt"`
	Detail     string     `json:"detail"`
	Categories []Category `json:"categories"`
}

// QueryOptions specifies filters and options for querying code debt.
type QueryOptions struct {
	KPI           string
	Category      Category
	Path          string
	Package       string
	Search        string
	Limit         int
	Deterministic bool
}

// QueryResult represents the outcome of a code debt query.
type QueryResult struct {
	TotalDebt   int              `json:"total_debt"`
	MatchedDebt int              `json:"matched_debt"`
	DebtByKPI   map[string]int   `json:"debt_by_kpi"`
	DebtByCat   map[Category]int `json:"debt_by_category"`
	DebtByPkg   map[string]int   `json:"debt_by_package"`
	Defects     []Defect         `json:"defects"`
}
