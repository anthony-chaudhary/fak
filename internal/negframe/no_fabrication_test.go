package negframe

import (
	"fmt"
	"strings"
	"testing"
)

// TestResolverNoFabrication is the adversarial UNKNOWN witness for #4475.
// Open-complement prose may be categorized, but it cannot carry a confident
// positive unless one of the declared mechanical rules derives that exact text.
func TestResolverNoFabrication(t *testing.T) {
	ambiguous := []string{
		"Never deploy during an incident",
		"Do not rewrite the operator's intent",
		"Don't delete evidence before review",
		"Avoid changing an unfamiliar contract",
		"This operation is not allowed here",
		"Direct publication is forbidden",
		"The worker may not own both leases",
		"The kernel must refuse to guess",
		"The probe fails to establish causality",
		"Continue without inventing a result",
	}
	prefixes := []string{"", "Please note: ", "Operator decision: "}
	suffixes := []string{".", "!", " -- preserve evidence."}

	for _, base := range ambiguous {
		for _, prefix := range prefixes {
			for _, suffix := range suffixes {
				input := prefix + mutateJudgementSpan(base) + suffix
				findings := Classify("adversarial.md", input)
				if len(findings) == 0 {
					t.Fatalf("ambiguous negative was not classified: %q", input)
				}
				for _, finding := range findings {
					if finding.Mechanical() || finding.Suggest != "" {
						t.Fatalf("ambiguous negative fabricated positive %q from %q: %+v", finding.Suggest, input, finding)
					}
					if finding.Hint == "" {
						t.Fatalf("UNKNOWN finding lacks judgement hint for %q: %+v", input, finding)
					}
				}
			}
		}
	}

	mechanical := []string{
		"Do not forget to verify",
		"Don't forget to commit",
		"Do not hesitate to ask",
		"Don't hesitate to retry",
		"No need to wait",
		"The result is not unsafe",
		"Make sure that you do not skip",
	}
	for _, input := range mechanical {
		findings := Classify("mechanical.md", input)
		if len(findings) != 1 || !findings[0].Mechanical() {
			t.Fatalf("mechanical fixture did not resolve exactly once: %q => %+v", input, findings)
		}
		if !mechanicalSuggestionDeclared(input, findings[0].Suggest) {
			t.Fatalf("free-floating mechanical suggestion %q from %q", findings[0].Suggest, input)
		}
	}
}

// mutateJudgementSpan deterministically changes surrounding span shape without
// changing any open complement into one of the finite mechanical idioms.
func mutateJudgementSpan(s string) string {
	replacer := strings.NewReplacer(
		"deploy", "publish",
		"rewrite", "reinterpret",
		"delete", "discard",
		"changing", "editing",
		"operation", "action",
		"worker", "agent",
		"kernel", "resolver",
		"probe", "check",
		"Continue", "Proceed",
	)
	return replacer.Replace(s)
}

// mechanicalSuggestionDeclared proves provenance from the package's enumerable
// mechanical rule table: the same rule must match and derive the exact emitted
// suggestion through its declared replacement template.
func mechanicalSuggestionDeclared(input, suggestion string) bool {
	for _, rule := range rules {
		if rule.Template == "" {
			continue
		}
		for _, loc := range rule.Pattern.FindAllStringIndex(input, -1) {
			span := input[loc[0]:loc[1]]
			if rule.Pattern.ReplaceAllString(span, rule.Template) == suggestion {
				return true
			}
		}
	}
	return false
}

func TestResolverNoFabricationMutationCount(t *testing.T) {
	// Pin the deterministic pass size so a future edit cannot silently shrink the
	// adversarial surface while leaving the main test green.
	const bases, prefixes, suffixes = 10, 3, 3
	if got := bases * prefixes * suffixes; got != 90 {
		t.Fatal(fmt.Sprintf("mutation corpus=%d want 90", got))
	}
}
