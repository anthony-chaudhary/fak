package disambiguation

import (
	"fmt"
	"strings"
)

// Explain renders one resolved entry for a human while QueryResult remains the
// stable machine contract.
func Explain(result QueryResponse) string {
	entry := result.Entry
	var out strings.Builder
	fmt.Fprintf(&out, "%s\n\nMeaning: %s\n", entry.Identity.CanonicalTerm, entry.Definition)
	if result.MatchedAlias != "" {
		fmt.Fprintf(&out, "Matched alias: %s\n", result.MatchedAlias)
	}
	out.WriteString("Not to confuse with:\n")
	for _, contrast := range entry.Contrasts {
		fmt.Fprintf(&out, "- %s - %s\n", contrast.CanonicalTerm, contrast.Explanation)
	}
	fmt.Fprintf(&out, "Owner: %s (lane %s)\n", entry.Owner.Leaf, entry.Owner.Lane)
	fmt.Fprintf(&out, "Freshness: %s (%s)\n", entry.Freshness.Verdict, entry.Freshness.ReasonCode)
	out.WriteString("Sources:\n")
	for _, source := range entry.Sources {
		fmt.Fprintf(&out, "- %s: %s @ %s (checked %s)\n", source.Kind, source.Locator, source.Revision, source.CheckedAt)
	}
	return out.String()
}
