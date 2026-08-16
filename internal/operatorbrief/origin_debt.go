package operatorbrief

import (
	"fmt"
	"sort"
	"strings"
)

const (
	DebtCaughtAtOrigin = "origin"
	DebtFoundLate      = "late_found"
)

// DebtWitnessRecord is the normalized task/session witness projection consumed by
// the operator brief. Debt caught at origin needs ordinary cleanup; debt first
// found later identifies a missing root control.
type DebtWitnessRecord struct {
	Source   string `json:"source"` // task or session
	ID       string `json:"id,omitempty"`
	Debt     string `json:"debt"` // origin or late_found
	Detail   string `json:"detail,omitempty"`
	Evidence string `json:"evidence,omitempty"`
}

// addDebtWitnesses folds the witness projection into the brief. The split is the
// operator-facing point: origin debt is ordinary cleanup the fleet can retire, so
// it stays delegable agent work, while debt first found late says the origin
// control that should have refused the artifact does not exist yet — a control
// gap the operator has to look at, not another cleanup ticket.
func addDebtWitnesses(r *Report, records []DebtWitnessRecord) {
	origin, late := splitDebtWitnesses(records)
	r.OriginDebt, r.LateFoundDebt = origin, late
	if len(origin) > 0 {
		r.addAgent("origin-debt", fmt.Sprintf("retire %s caught at origin", pluralDebt(len(origin))),
			debtSummary(origin), "clear the refused artifacts; the origin control already held")
	}
	if len(late) > 0 {
		r.addWatch("origin-debt", fmt.Sprintf("%s found after origin", pluralDebt(len(late))),
			debtSummary(late), "add the missing origin control so this class is refused before handoff")
	}
}

// debtSummary is the compact evidence line behind a debt bucket: the first few
// witnesses by source and id, so the operator can find them without opening the
// full brief.
func debtSummary(records []DebtWitnessRecord) string {
	const shown = 3
	parts := make([]string, 0, shown+1)
	for _, record := range records {
		if len(parts) == shown {
			parts = append(parts, fmt.Sprintf("+%d more", len(records)-shown))
			break
		}
		parts = append(parts, strings.TrimSpace(record.Source+" "+record.ID))
	}
	return strings.Join(parts, ", ")
}

func pluralDebt(n int) string {
	if n == 1 {
		return "1 debt witness"
	}
	return fmt.Sprintf("%d debt witnesses", n)
}

// debtLine renders one witness as `<source> <id>: <detail> (<evidence>)`, with
// the optional parts dropped when the witness did not carry them.
func debtLine(record DebtWitnessRecord) string {
	line := strings.TrimSpace(record.Source + " " + record.ID)
	if record.Detail != "" {
		line += ": " + record.Detail
	}
	if record.Evidence != "" {
		line += " (" + record.Evidence + ")"
	}
	return line
}

func splitDebtWitnesses(records []DebtWitnessRecord) (origin, late []DebtWitnessRecord) {
	for _, record := range records {
		switch record.Debt {
		case DebtCaughtAtOrigin:
			origin = append(origin, record)
		case DebtFoundLate:
			late = append(late, record)
		}
	}
	sortDebtWitnesses(origin)
	sortDebtWitnesses(late)
	return origin, late
}

func sortDebtWitnesses(records []DebtWitnessRecord) {
	sort.Slice(records, func(i, j int) bool {
		if records[i].Source != records[j].Source {
			return records[i].Source < records[j].Source
		}
		return records[i].ID < records[j].ID
	})
}
