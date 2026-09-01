package flowmetrics

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipreadiness"
)

const AgingWIPReasonCode = "FLOW_AGING_WIP_EXCEEDS_BUDGET"

const (
	AgingActionable = "actionable"
	AgingLiveOwned  = "live-owned"
	AgingRetained   = "retained"
	AgingSuperseded = "superseded"
)

var agingSafeActions = []string{"landing", "recovery", "parking", "owned-continuation", "safety", "witnessed-supersession"}

type AgingUnit struct {
	Unit           string  `json:"unit"`
	AgeDays        float64 `json:"age_days"`
	Classification string  `json:"classification"`
}

type AgingAdmissionRequest struct {
	Intent             string
	Now                time.Time
	Budget             time.Duration
	Readiness          *wipreadiness.Receipt
	Units              []AgingUnit
	SupersessionReason string
	OverrideReason     string
}

type AgingAdmissionReceipt struct {
	Schema           string     `json:"schema"`
	Verdict          string     `json:"verdict"`
	ReasonCode       string     `json:"reason_code,omitempty"`
	Reason           string     `json:"reason"`
	Intent           string     `json:"intent"`
	BudgetDays       float64    `json:"budget_days"`
	BlockingUnit     *AgingUnit `json:"blocking_unit,omitempty"`
	SafeActions      []string   `json:"safe_actions"`
	ReadinessVerdict string     `json:"readiness_verdict,omitempty"`
	WitnessedReason  string     `json:"witnessed_reason,omitempty"`
}

func BuildAgingUnits(issues []Issue, commits []Commit, readiness *wipreadiness.Receipt, now time.Time) []AgingUnit {
	labels := make(map[int][]string, len(issues))
	for _, issue := range issues {
		labels[issue.Number] = issue.Labels
	}
	owned := map[int]bool{}
	if readiness != nil {
		for _, work := range readiness.PreservedWork() {
			if n, ok := issueNumberFromWorkID(work.ID); ok {
				owned[n] = true
			}
		}
	}
	spans := AgingWIP(BuildSpans(issues, commits), now, 0)
	units := make([]AgingUnit, 0, len(spans))
	for _, span := range spans {
		class := AgingActionable
		if owned[span.Issue] {
			class = AgingLiveOwned
		} else if retainedAgingLabels(labels[span.Issue]) {
			class = AgingRetained
		}
		units = append(units, AgingUnit{Unit: fmt.Sprintf("#%d", span.Issue), AgeDays: span.AgeHours(now) / 24, Classification: class})
	}
	return units
}

func AdmitAgingWIP(req AgingAdmissionRequest) AgingAdmissionReceipt {
	intent := strings.ToLower(strings.TrimSpace(req.Intent))
	out := AgingAdmissionReceipt{Schema: "fak-aging-wip-admission/1", Verdict: "ADMIT", Reason: "no actionable aging WIP exceeds the declared budget", Intent: intent, BudgetDays: req.Budget.Hours() / 24, SafeActions: append([]string(nil), agingSafeActions...)}
	if req.Readiness != nil {
		out.ReadinessVerdict = string(req.Readiness.Verdict)
	}
	if reason := strings.TrimSpace(req.OverrideReason); reason != "" {
		out.Reason = "explicit operator override admitted with a witnessed reason"
		out.WitnessedReason = reason
		return out
	}
	if intent == "supersession" {
		if reason := strings.TrimSpace(req.SupersessionReason); reason != "" {
			out.Reason = "supersession admitted with a witnessed reason"
			out.WitnessedReason = reason
			return out
		}
		out.Verdict, out.ReasonCode, out.Reason = "REFUSE", AgingWIPReasonCode, "supersession requires a witnessed reason"
		return out
	}
	if agingIntentExempt(intent) {
		out.Reason = "landing, recovery, parking, owned continuation, and safety work remain admitted"
		return out
	}
	if intent != "fresh" {
		out.Verdict, out.ReasonCode, out.Reason = "REFUSE", AgingWIPReasonCode, "unknown aging-WIP admission intent"
		return out
	}
	if !currentReadiness(req.Readiness, req.Now) {
		out.Verdict, out.ReasonCode, out.Reason = "REFUSE", AgingWIPReasonCode, "fresh starts require a current ready WIP receipt"
		return out
	}
	units := append([]AgingUnit(nil), req.Units...)
	sort.SliceStable(units, func(i, j int) bool {
		if units[i].AgeDays != units[j].AgeDays {
			return units[i].AgeDays > units[j].AgeDays
		}
		return units[i].Unit < units[j].Unit
	})
	budgetDays := req.Budget.Hours() / 24
	for i := range units {
		unit := units[i]
		if unit.Classification == AgingActionable && unit.AgeDays > budgetDays {
			unit.Unit = scrubAgingUnit(unit.Unit)
			out.Verdict, out.ReasonCode = "REFUSE", AgingWIPReasonCode
			out.Reason = "finish, recover, park, or witness supersession of the oldest actionable unit before unrelated fresh work"
			out.BlockingUnit = &unit
			return out
		}
	}
	return out
}

func agingIntentExempt(intent string) bool {
	switch intent {
	case "landing", "recovery", "parking", "continuation", "owned-continuation", "safety":
		return true
	default:
		return false
	}
}

func currentReadiness(receipt *wipreadiness.Receipt, now time.Time) bool {
	return receipt != nil && receipt.Verdict == wipreadiness.VerdictCurrent && !now.Before(receipt.ObservedAt) && !now.After(receipt.ExpiresAt)
}

func retainedAgingLabels(labels []string) bool {
	for _, label := range labels {
		switch strings.ToLower(strings.TrimSpace(label)) {
		case "wip/retained", "wip/parked", "status/parked", "status/retained", "superseded":
			return true
		}
	}
	return false
}

func issueNumberFromWorkID(id string) (int, bool) {
	id = strings.TrimSpace(id)
	for _, prefix := range []string{"#", "issue:", "issue/"} {
		if strings.HasPrefix(id, prefix) {
			n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(id, prefix)))
			return n, err == nil && n > 0
		}
	}
	return 0, false
}

func scrubAgingUnit(unit string) string {
	unit = strings.TrimSpace(unit)
	if n, ok := issueNumberFromWorkID(unit); ok {
		return fmt.Sprintf("#%d", n)
	}
	return "unidentified-unit"
}
