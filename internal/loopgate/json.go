package loopgate

import (
	"encoding/json"
	"fmt"
	"strings"
)

// CommitAuditResultFromJSON folds `dos commit-audit --json` into the normalized
// witness outcome. A range is witnessed only when at least one row is OK and no
// row reports an unwitnessed claim.
func CommitAuditResultFromJSON(data []byte) (WitnessResult, error) {
	var rows []struct {
		Verdict string   `json:"verdict"`
		Witness string   `json:"witness"`
		Reason  string   `json:"reason"`
		SHA     string   `json:"sha"`
		Sources []string `json:"source_files"`
		Tests   []string `json:"test_files"`
	}
	if err := json.Unmarshal(data, &rows); err != nil {
		return WitnessResult{}, fmt.Errorf("parse commit-audit json: %w", err)
	}
	if len(rows) == 0 {
		return WitnessResult{Outcome: OutcomeNotYet, Reason: "ABSTAIN", Detail: "commit-audit returned no rows"}, nil
	}
	seenOK := false
	var abstains []string
	for _, row := range rows {
		verdict := strings.ToUpper(strings.TrimSpace(row.Verdict))
		switch verdict {
		case "OK":
			seenOK = true
		case "CLAIM_UNWITNESSED":
			return WitnessResult{
				Outcome:    OutcomeNotYet,
				Reason:     verdict,
				Detail:     firstNonEmpty(row.Reason, "commit claim was not witnessed by its diff"),
				RawVerdict: verdict,
				Rung:       row.Witness,
			}, nil
		case "ABSTAIN":
			abstains = append(abstains, firstNonEmpty(row.Reason, row.SHA, "abstain"))
		default:
			return WitnessResult{
				Outcome:    OutcomeRefused,
				Reason:     ReasonSchemaUnreadable,
				Detail:     fmt.Sprintf("unknown commit-audit verdict %q", row.Verdict),
				RawVerdict: row.Verdict,
				Rung:       row.Witness,
			}, nil
		}
	}
	if seenOK {
		return WitnessResult{
			Outcome:    OutcomeWitnessed,
			Reason:     "OK",
			Detail:     "commit-audit diff witnessed the done claim",
			RawVerdict: "OK",
			Rung:       "diff-witnessed",
		}, nil
	}
	return WitnessResult{
		Outcome:    OutcomeNotYet,
		Reason:     "ABSTAIN",
		Detail:     "commit-audit abstained: " + strings.Join(abstains, "; "),
		RawVerdict: "ABSTAIN",
		Rung:       "abstain",
	}, nil
}

// VerifyResultFromJSON folds `dos verify --json`.
func VerifyResultFromJSON(data []byte) (WitnessResult, error) {
	var row struct {
		Plan    string `json:"plan"`
		Phase   string `json:"phase"`
		Shipped bool   `json:"shipped"`
		Source  string `json:"source"`
		SHA     string `json:"sha"`
		Summary string `json:"summary"`
	}
	if err := json.Unmarshal(data, &row); err != nil {
		return WitnessResult{}, fmt.Errorf("parse verify json: %w", err)
	}
	if row.Shipped {
		return WitnessResult{
			Outcome:    OutcomeWitnessed,
			Reason:     "SHIPPED",
			Detail:     firstNonEmpty(row.Summary, fmt.Sprintf("%s/%s shipped via %s", row.Plan, row.Phase, row.Source)),
			RawVerdict: "SHIPPED",
			Rung:       row.Source,
		}, nil
	}
	return WitnessResult{
		Outcome:    OutcomeNotYet,
		Reason:     "NOT_SHIPPED",
		Detail:     firstNonEmpty(row.Summary, fmt.Sprintf("%s/%s not shipped", row.Plan, row.Phase)),
		RawVerdict: "NOT_SHIPPED",
		Rung:       firstNonEmpty(row.Source, "none"),
	}, nil
}

// TestWitnessResultFromJSON folds `dos test-witness --json`.
func TestWitnessResultFromJSON(data []byte) (WitnessResult, error) {
	var row struct {
		Verdict   string `json:"verdict"`
		Witnesses bool   `json:"witnesses"`
		Reason    string `json:"reason"`
		Evidence  struct {
			Rung string `json:"rung"`
		} `json:"evidence"`
	}
	if err := json.Unmarshal(data, &row); err != nil {
		return WitnessResult{}, fmt.Errorf("parse test-witness json: %w", err)
	}
	verdict := strings.ToUpper(strings.TrimSpace(row.Verdict))
	if row.Witnesses && verdict == "DISCRIMINATES" {
		return WitnessResult{
			Outcome:    OutcomeWitnessed,
			Reason:     verdict,
			Detail:     firstNonEmpty(row.Reason, "test discriminates old and new trees"),
			RawVerdict: verdict,
			Rung:       row.Evidence.Rung,
		}, nil
	}
	return WitnessResult{
		Outcome:    OutcomeNotYet,
		Reason:     firstNonEmpty(verdict, "ABSTAIN"),
		Detail:     firstNonEmpty(row.Reason, "test witness did not discriminate"),
		RawVerdict: verdict,
		Rung:       row.Evidence.Rung,
	}, nil
}

// GenericWitnessResultFromJSON folds `dos witness --json` best-effort. The
// generic verb is plugin-shaped, so this parser accepts the stable fields the
// built-in renderer exposes while failing closed on unknown shapes.
func GenericWitnessResultFromJSON(data []byte) (WitnessResult, error) {
	var row map[string]any
	if err := json.Unmarshal(data, &row); err != nil {
		return WitnessResult{}, fmt.Errorf("parse witness json: %w", err)
	}
	belief := mapField(row, "belief")
	facts := mapField(row, "facts")
	rung := firstNonEmpty(
		stringField(row, "accountability"),
		stringField(row, "rung"),
		stringField(facts, "accountability"),
		stringField(facts, "rung"),
	)
	if boolField(row, "believe") || boolField(row, "believed") || boolField(belief, "believe") ||
		strings.EqualFold(stringField(row, "verdict"), "BELIEVED") ||
		strings.EqualFold(stringField(row, "verdict"), "RESOLVED_MATCH") {
		return WitnessResult{
			Outcome:    OutcomeWitnessed,
			Reason:     firstNonEmpty(stringField(row, "verdict"), "BELIEVED"),
			Detail:     firstNonEmpty(stringField(row, "reason"), stringField(belief, "reason"), stringField(facts, "detail"), "generic witness believed the effect"),
			RawVerdict: stringField(row, "verdict"),
			Rung:       rung,
		}, nil
	}
	if boolField(belief, "refuted") {
		return WitnessResult{
			Outcome:    OutcomeNotYet,
			Reason:     firstNonEmpty(stringField(row, "verdict"), stringField(facts, "stance"), "REFUTED"),
			Detail:     firstNonEmpty(stringField(row, "reason"), stringField(belief, "reason"), stringField(facts, "detail"), "generic witness refuted the effect"),
			RawVerdict: firstNonEmpty(stringField(row, "verdict"), stringField(facts, "stance")),
			Rung:       rung,
		}, nil
	}
	verdict := strings.ToUpper(firstNonEmpty(stringField(row, "verdict"), stringField(facts, "stance")))
	switch verdict {
	case "REFUTED", "UNRESOLVED", "RESOLVED_MISMATCH":
		return WitnessResult{
			Outcome:    OutcomeNotYet,
			Reason:     verdict,
			Detail:     firstNonEmpty(stringField(row, "reason"), stringField(belief, "reason"), stringField(facts, "detail"), "generic witness refuted the effect"),
			RawVerdict: verdict,
			Rung:       rung,
		}, nil
	case "NO_SIGNAL", "ABSTAIN", "":
		return WitnessResult{
			Outcome:    OutcomeNotYet,
			Reason:     firstNonEmpty(verdict, "ABSTAIN"),
			Detail:     firstNonEmpty(stringField(row, "reason"), stringField(belief, "reason"), stringField(facts, "detail"), "generic witness had no accountable signal"),
			RawVerdict: verdict,
			Rung:       rung,
		}, nil
	default:
		return WitnessResult{
			Outcome:    OutcomeRefused,
			Reason:     ReasonSchemaUnreadable,
			Detail:     fmt.Sprintf("unknown witness verdict %q", verdict),
			RawVerdict: verdict,
			Rung:       rung,
		}, nil
	}
}

func stringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func boolField(m map[string]any, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}

func mapField(m map[string]any, key string) map[string]any {
	if v, ok := m[key].(map[string]any); ok {
		return v
	}
	return nil
}
