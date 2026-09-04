// Package kimik3page provides verification, contract specification,
// and link auditing for the Kimi K3 documentation and marketing assets.
package kimik3page

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Invariant: required markup elements and assets must remain present and unwitnessed claims absent.
// Guard: fail-closed on malformed pages

// Contract defines the structural, stylistic, and asset constraints
// required for the Kimi K3 documentation surface.
type Contract struct {
	// RequiredSubstrings lists mandatory HTML fragments and markup tokens.
	RequiredSubstrings []string
	// ForbiddenPhrases lists marketing claims prohibited without empirical proof.
	ForbiddenPhrases []string
	// RequiredAssets lists asset files that must exist relative to the page root.
	RequiredAssets []string
}

// Spec captures metadata and verification specifications for the Kimi K3 page.
type Spec struct {
	// PageTitle is the expected canonical page title.
	PageTitle string
	// Description summarizes the purpose of the documentation surface.
	Description string
	// Contract specifies the validation rules for the page.
	Contract Contract
}

// DefaultContract returns the canonical verification contract for the Kimi K3 page.
func DefaultContract() Contract {
	return Contract{
		RequiredSubstrings: []string{
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
		},
		ForbiddenPhrases: []string{
			"fastest",
			"cheapest",
			"best model",
			"x faster",
			"× faster",
		},
		RequiredAssets: []string{
			"social-card.svg",
		},
	}
}

// DefaultSpec returns the standard specification for the Kimi K3 documentation surface.
func DefaultSpec() Spec {
	return Spec{
		PageTitle:   "Kimi K3 for Claude Code, through fak",
		Description: "Standalone route page for fak first-class Moonshot Kimi K3 support.",
		Contract:    DefaultContract(),
	}
}

// Validate verifies that the page content and assets at root conform to the contract.
//
// Invariant: all required substrings and assets must be present.
// Guard: fail-closed on malformed pages.
func (c Contract) Validate(root string) error {
	indexPath := filepath.Join(root, "index.html")
	raw, err := os.ReadFile(indexPath)
	if err != nil {
		return fmt.Errorf("read index.html: %w", err)
	}
	page := string(raw)

	for _, want := range c.RequiredSubstrings {
		if !strings.Contains(page, want) {
			return fmt.Errorf("page missing required substring %q", want)
		}
	}

	lower := strings.ToLower(page)
	for _, phrase := range c.ForbiddenPhrases {
		if strings.Contains(lower, phrase) {
			return fmt.Errorf("unwitnessed marketing claim %q found in page", phrase)
		}
	}

	for _, asset := range c.RequiredAssets {
		assetPath := filepath.Join(root, asset)
		if _, err := os.Stat(assetPath); err != nil {
			return fmt.Errorf("required asset %q missing: %w", asset, err)
		}
	}

	return nil
}

// ValidatePage checks the Kimi K3 documentation page at root against DefaultContract.
//
// Invariant: all required structural tags, scripts, and assets must be verified.
// Guard: fail-closed on malformed pages.
func ValidatePage(root string) error {
	return DefaultContract().Validate(root)
}

// AuditLinks checks that repository-relative artifacts referenced by the
// Kimi K3 page exist and returns the list of verified paths.
//
// Invariant: every referenced repository link must exist on disk.
// Guard: fail-closed on malformed pages.
func AuditLinks(repoRoot string) ([]string, error) {
	relLinks := []string{
		"DOGFOOD-CLAUDE.md",
		filepath.Join("scripts", "claude-kimi-k3.sh"),
		filepath.Join("scripts", "claude-kimi-k3.ps1"),
		filepath.Join("internal", "agent", "kimi_k3.go"),
		filepath.Join("visuals", "brand", "fak-mark.svg"),
		filepath.Join("visuals", "brand", "fak-favicon.svg"),
	}

	verified := make([]string, 0, len(relLinks))
	for _, rel := range relLinks {
		target := filepath.Join(repoRoot, rel)
		if _, err := os.Stat(target); err != nil {
			return nil, fmt.Errorf("linked artifact %s: %w", rel, err)
		}
		verified = append(verified, rel)
	}
	return verified, nil
}
