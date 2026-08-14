package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// formatMicroParentReceiptRows is the read-only Info/Fleet projection of the
// authoritative execution receipt. It owns no lifecycle state and invents no
// provenance: absent fields render as "not yet".
func formatMicroParentReceiptRows(data []byte) ([]string, error) {
	var receipt microSelfcheckReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return nil, err
	}
	if receipt.Schema != "fak-micro-selfcheck/2" {
		return nil, fmt.Errorf("unsupported receipt schema %q", receipt.Schema)
	}
	parent := displayOrNotYet(receipt.ParentTaskID)
	rows := []string{fmt.Sprintf(" receipt  %s · %s · children %d", parent, displayOrNotYet(receipt.Verdict), len(receipt.Children))}
	for _, child := range receipt.Children {
		effect := "not yet"
		if child.Witnessed && strings.TrimSpace(child.EffectDigest) != "" {
			effect = child.EffectDigest
		}
		rows = append(rows, fmt.Sprintf("          %s · lease %s · session %s · %s · effect %s", displayOrNotYet(child.WorkUnitID), displayOrNotYet(child.LeaseID), displayOrNotYet(child.SessionID), displayOrNotYet(child.State), effect))
	}
	return rows, nil
}

func displayOrNotYet(value string) string {
	if strings.TrimSpace(value) == "" {
		return "not yet"
	}
	return value
}
