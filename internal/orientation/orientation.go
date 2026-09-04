// Package orientation exposes fak's versioned temporal product orientation.
package orientation

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Schema identifies the format version of the orientation snapshot payload.
const Schema = "fak-orientation/1"

//go:embed orientation.json
var embedded []byte

// Item represents a single product orientation dimension, defining its lifecycle
// horizon, active role, rationale, and objective transition rules.
type Item struct {
	ID               string   `json:"id"`
	Label            string   `json:"label"`
	Horizon          string   `json:"horizon"`
	CurrentRole      string   `json:"current_role"`
	WhyNow           string   `json:"why_now"`
	RetainedContract string   `json:"retained_contract"`
	EvidenceState    string   `json:"evidence_state"`
	Evidence         []string `json:"evidence"`
	IncreaseWhen     string   `json:"increase_when"`
	DecreaseWhen     string   `json:"decrease_when"`
	FutureState      string   `json:"future_state"`
}

// Snapshot represents a validated product orientation document containing
// invariant promises, the owned mediation seam, and directional items.
type Snapshot struct {
	Schema              string `json:"schema"`
	EffectiveDate       string `json:"effective_date"`
	ReviewBy            string `json:"review_by"`
	EnduringPromise     string `json:"enduring_promise"`
	OwnedSeam           string `json:"owned_seam"`
	InvestmentPrinciple string `json:"investment_principle"`
	Items               []Item `json:"items"`
}

// View wraps a Snapshot with evaluation-time freshness state and review countdowns.
type View struct {
	Snapshot
	AsOf         string `json:"as_of"`
	Freshness    string `json:"freshness"`
	DaysToReview int    `json:"days_to_review"`
}

// Current loads, decodes, and validates the embedded orientation snapshot.
func Current() (Snapshot, error) {
	var s Snapshot
	if err := json.Unmarshal(embedded, &s); err != nil {
		return Snapshot{}, fmt.Errorf("decode embedded orientation: %w", err)
	}
	if err := Validate(s); err != nil {
		return Snapshot{}, err
	}
	return s, nil
}

// Validate enforces schema constraints, date formatting, required string fields,
// unique item identifiers, and recognized horizon and evidence state values.
func Validate(s Snapshot) error {
	var problems []string
	if s.Schema != Schema {
		problems = append(problems, "schema must be "+Schema)
	}
	if _, err := time.Parse(time.DateOnly, s.EffectiveDate); err != nil {
		problems = append(problems, "effective_date must be YYYY-MM-DD")
	}
	if _, err := time.Parse(time.DateOnly, s.ReviewBy); err != nil {
		problems = append(problems, "review_by must be YYYY-MM-DD")
	}
	for field, value := range map[string]string{"enduring_promise": s.EnduringPromise, "owned_seam": s.OwnedSeam, "investment_principle": s.InvestmentPrinciple} {
		if strings.TrimSpace(value) == "" {
			problems = append(problems, field+" is required")
		}
	}
	seen := map[string]bool{}
	for i, item := range s.Items {
		prefix := fmt.Sprintf("items[%d]", i)
		if strings.TrimSpace(item.ID) == "" {
			problems = append(problems, prefix+".id is required")
		} else if seen[item.ID] {
			problems = append(problems, prefix+".id is duplicate")
		} else {
			seen[item.ID] = true
		}
		if !oneOf(item.Horizon, "constitutional", "strategic", "tactical") {
			problems = append(problems, prefix+".horizon is invalid")
		}
		if !oneOf(item.EvidenceState, "witnessed", "observed", "modeled", "not-yet") {
			problems = append(problems, prefix+".evidence_state is invalid")
		}
		for field, value := range map[string]string{"label": item.Label, "current_role": item.CurrentRole, "why_now": item.WhyNow, "retained_contract": item.RetainedContract, "increase_when": item.IncreaseWhen, "decrease_when": item.DecreaseWhen, "future_state": item.FutureState} {
			if strings.TrimSpace(value) == "" {
				problems = append(problems, prefix+"."+field+" is required")
			}
		}
		if len(item.Evidence) == 0 {
			problems = append(problems, prefix+".evidence is required")
		}
	}
	if len(s.Items) == 0 {
		problems = append(problems, "items are required")
	}
	if len(problems) > 0 {
		return fmt.Errorf("invalid orientation: %s", strings.Join(problems, "; "))
	}
	return nil
}

// Assess computes freshness classification and remaining review days for a Snapshot
// evaluated against the provided reference timestamp.
func Assess(s Snapshot, now time.Time) View {
	review, _ := time.Parse(time.DateOnly, s.ReviewBy)
	today := time.Date(now.UTC().Year(), now.UTC().Month(), now.UTC().Day(), 0, 0, 0, 0, time.UTC)
	days := int(review.Sub(today).Hours() / 24)
	freshness := "current"
	if days < 0 {
		freshness = "stale"
	} else if days <= 14 {
		freshness = "due-soon"
	}
	return View{Snapshot: s, AsOf: today.Format(time.DateOnly), Freshness: freshness, DaysToReview: days}
}

// Text renders the evaluated View into a human-readable summary.
func (v View) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "FAK ORIENTATION — %s (%d days to review)\n", strings.ToUpper(v.Freshness), v.DaysToReview)
	fmt.Fprintf(&b, "Enduring promise: %s\nOwned seam: %s\nInvestment principle: %s\n\nCurrent emphases and transition rules:\n", v.EnduringPromise, v.OwnedSeam, v.InvestmentPrinciple)
	for _, item := range v.Items {
		fmt.Fprintf(&b, "- %s [%s/%s, %s]\n  now: %s\n  retain: %s\n  decrease when: %s\n", item.Label, item.Horizon, item.CurrentRole, item.EvidenceState, item.WhyNow, item.RetainedContract, item.DecreaseWhen)
	}
	return b.String()
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}
