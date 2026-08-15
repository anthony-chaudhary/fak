package stackresolve

import (
	"fmt"
	"strings"
)

// Format renders the decisive receipt without dumping the full catalog.
func Format(receipt Receipt) string {
	var out strings.Builder
	fmt.Fprintf(&out, "%s stack for %s\n", strings.ToUpper(receipt.Status), receipt.Workload)
	if receipt.Status == "allow" {
		fmt.Fprintf(&out, "selected: %d components\n", len(receipt.Selected))
		for _, component := range receipt.Selected {
			fmt.Fprintf(&out, "  - %s (%s; %s)\n", component.ID, component.Kind, component.Evidence.Source)
		}
		for _, decision := range receipt.Decisions {
			via := ""
			if decision.Substitute {
				via = " via substitute"
			}
			fmt.Fprintf(&out, "  %s %s -> %s%s [%s]\n", decision.From, decision.Relation, decision.Chosen, via, decision.Evidence.Source)
		}
		for _, warning := range receipt.Warnings {
			fmt.Fprintf(&out, "WARN %s: %s wants %s — %s [%s]\n", warning.Code, warning.From, warning.Wanted, warning.Message, warning.Evidence.Source)
		}
		return out.String()
	}
	if receipt.Conflict == nil {
		return out.String()
	}
	fmt.Fprintf(&out, "blocker: %s (%s)\n", receipt.Conflict.Wanted, receipt.Conflict.Code)
	fmt.Fprintf(&out, "chain: %s\n", strings.Join(receipt.Conflict.Chain, " -> "))
	fmt.Fprintf(&out, "authority: %s; source: %s", receipt.Conflict.Evidence.Authority, receipt.Conflict.Evidence.Source)
	if receipt.Conflict.Evidence.Tier != "" {
		fmt.Fprintf(&out, "; proof: %s", receipt.Conflict.Evidence.Tier)
	}
	out.WriteByte('\n')
	for _, remediation := range receipt.Conflict.Remediation {
		fmt.Fprintf(&out, "remediation: %s\n", remediation)
	}
	return out.String()
}
