// Package kimik3page validates marketing page contract and links for Kimi K3.
//
// Contract: marketing page must adhere to accessibility, responsive design, and non-hyperbolic claims.
// Invariants:
// - Guard: required structural and accessibility tokens must be present.
// - Guard: hyperbolic or unwitnessed claims are strictly forbidden (fail-closed validation).
package kimik3page

import (
	"strings"
)

// DefaultRequiredTokens returns the baseline strings that must exist in Kimi K3 marketing content.
func DefaultRequiredTokens() []string {
	return []string{
		`<!doctype html>`,
		`<meta name="viewport"`,
		`Kimi K3 for Claude Code, through fak`,
		`reasoning_effort`,
		`Native max reasoning`,
		`claude-kimi-k3.sh`,
		`claude-kimi-k3.ps1`,
		`DOGFOOD-CLAUDE.md#moonshot-kimi-k3`,
		`@media (max-width: 580px)`,
		`prefers-reduced-motion`,
		`aria-label="Request path"`,
		`social-card.svg`,
	}
}

// DefaultForbiddenClaims returns unwitnessed superlatives and marketing claims barred from content.
func DefaultForbiddenClaims() []string {
	return []string{
		"fastest",
		"cheapest",
		"best model",
		"x faster",
		"× faster",
	}
}

// PageReport details findings from page contract validation.
type PageReport struct {
	// Valid indicates whether the content satisfied all required tokens and avoided forbidden claims.
	Valid bool
	// MissingRequired lists required tokens that were not found in the content.
	MissingRequired []string
	// ForbiddenMatches lists forbidden claims detected in the content.
	ForbiddenMatches []string
}

// PageValidator encapsulates validation rules and invariants for Kimi K3 marketing pages.
type PageValidator struct {
	// Required contains the mandatory substrings that must be present.
	Required []string
	// Forbidden contains substrings that must not appear in the content.
	Forbidden []string
}

// NewPageValidator initializes a PageValidator with standard required and forbidden rules.
func NewPageValidator() *PageValidator {
	return &PageValidator{
		Required:  DefaultRequiredTokens(),
		Forbidden: DefaultForbiddenClaims(),
	}
}

// Validate executes contract validation against provided HTML content.
func (v *PageValidator) Validate(htmlContent string) PageReport {
	var report PageReport
	report.Valid = true

	for _, want := range v.Required {
		if !strings.Contains(htmlContent, want) {
			report.MissingRequired = append(report.MissingRequired, want)
			report.Valid = false
		}
	}

	lower := strings.ToLower(htmlContent)
	for _, phrase := range v.Forbidden {
		if strings.Contains(lower, strings.ToLower(phrase)) {
			report.ForbiddenMatches = append(report.ForbiddenMatches, phrase)
			report.Valid = false
		}
	}

	return report
}

// ValidatePageContent performs validation on the given HTML content using default rules.
func ValidatePageContent(htmlContent string) PageReport {
	return NewPageValidator().Validate(htmlContent)
}
