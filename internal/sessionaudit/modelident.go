package sessionaudit

// modelident.go — the typed canonical model-identity layer for the cost
// artifact (#4635, the cost-truth gate of the #4632 production-readiness
// inventory).
//
// The audit bills by the EXACT model id the provider emitted (e.g. the dated
// `claude-haiku-4-5-20251001`), while external catalogs commonly spell the same
// family differently (`claude-haiku-4.5`, undated `claude-haiku-4-5`). The
// plano study (docs/notes/BORROW-ROUTING-SIGNALS-GATEWAY-PLANO-STUDY-2026-07-13.md)
// records this as a real matching hazard: a production cost report must not
// silently treat an unknown alias as $0 — or price it as a neighboring model.
//
// This layer resolves EXACT, test-pinned spellings only:
//
//   - a canonical fleet id resolves to itself (ProvenanceExact);
//   - a dot respelling of a canonical id resolves (ProvenanceDashDot);
//   - a pinned undated family spelling of the one dated snapshot the fleet
//     runs resolves (ProvenanceDateSuffix);
//   - EVERYTHING else is unknown and fails CLOSED: PriceForIdentity /
//     StrictModelCostUSD return ErrUnknownModelPricing (never a silent zero),
//     and the aggregate carries the raw id on an explicit UNKNOWN hold
//     (Aggregate.UnpricedModels / Aggregate.UnverifiedClaudeIDs) that the
//     compact report and its recommendations gate on.
//
// Nothing here is substring-matched, so an unknown neighbor
// (`claude-haiku-4-5-20991231`, `claude-sonnet-4-5`) cannot overmatch into a
// priced row the way the legacy tier heuristic (PriceFor/ModelTier) would.
// The rates themselves stay single-sourced in Pricing — this layer adds
// identity and provenance ON TOP of the tier book, never a second price copy.

import (
	"errors"
	"fmt"
	"strings"
)

// ModelIdentity is the typed identity of one billed model id: the RAW
// provider-emitted spelling preserved verbatim, the CANONICAL fleet id it
// resolved to (empty when unknown), and the PROVENANCE of that resolution so a
// report shows HOW a raw spelling became a priced row rather than asserting it.
type ModelIdentity struct {
	Raw        string `json:"raw"`
	Canonical  string `json:"canonical,omitempty"`
	Provenance string `json:"provenance"`
}

// Known reports whether the raw id resolved to a canonical fleet id.
func (m ModelIdentity) Known() bool { return m.Canonical != "" }

// Identity-resolution provenance values. Each names the ONE rule that resolved
// (or refused) a raw spelling; the alias tests pin every mapping per rule.
const (
	// ProvenanceExact — the raw id IS a canonical fleet id (case-insensitively).
	ProvenanceExact = "exact"
	// ProvenanceDashDot — a dot respelling of a canonical id (claude-opus-4.8).
	ProvenanceDashDot = "alias:dash-dot"
	// ProvenanceDateSuffix — the pinned undated family spelling of the one
	// dated snapshot the fleet runs (claude-haiku-4-5 → …-20251001).
	ProvenanceDateSuffix = "alias:date-suffix"
	// ProvenanceUnknown — no rule resolved the spelling; pricing fails closed.
	ProvenanceUnknown = "unknown"
)

// canonicalModelTier binds each canonical Claude fleet model id — the exact
// ids the fleet emits (#4632) — to the Pricing rate-card row it bills at.
// EXACT ids only: this is the identity layer's price provenance, not a
// matcher, and the dollar figures stay single-sourced in Pricing so the two
// can never drift.
var canonicalModelTier = map[string]string{
	"claude-opus-4-8":           "opus",
	"claude-sonnet-4-6":         "sonnet",
	"claude-sonnet-5":           "sonnet",
	"claude-haiku-4-5-20251001": "haiku",
	"claude-fable-5":            "fable",
}

// modelDateAliases is the EXACT pinned date-suffix alias table: an undated
// family spelling external catalogs commonly use, mapped to the ONE dated
// snapshot the fleet actually runs. Entries are added deliberately (with the
// alias fixture pinning each), never derived by pattern, so a DIFFERENT date
// suffix stays unknown instead of borrowing this row.
var modelDateAliases = map[string]string{
	"claude-haiku-4-5": "claude-haiku-4-5-20251001",
}

// ResolveModelID resolves a raw provider-emitted model id to its typed
// canonical identity. Resolution is exact and test-pinned — lowercase/trim,
// then the canonical set, then the dot respelling, then the pinned date-suffix
// aliases — and anything unresolved is returned UNKNOWN so pricing fails
// closed rather than substring-guessing a neighbor.
func ResolveModelID(raw string) ModelIdentity {
	id := strings.ToLower(strings.TrimSpace(raw))
	if _, ok := canonicalModelTier[id]; ok {
		return ModelIdentity{Raw: raw, Canonical: id, Provenance: ProvenanceExact}
	}
	norm := strings.ReplaceAll(id, ".", "-")
	if _, ok := canonicalModelTier[norm]; ok {
		return ModelIdentity{Raw: raw, Canonical: norm, Provenance: ProvenanceDashDot}
	}
	if c, ok := modelDateAliases[norm]; ok {
		return ModelIdentity{Raw: raw, Canonical: c, Provenance: ProvenanceDateSuffix}
	}
	return ModelIdentity{Raw: raw, Provenance: ProvenanceUnknown}
}

// ErrUnknownModelPricing fails the strict cost path CLOSED: the model id has
// no canonical identity and no published rate card, so its cost is UNKNOWN —
// it must surface as an explicit held/unpriced row, never as $0.
var ErrUnknownModelPricing = errors.New("sessionaudit: unknown model id has no pricing provenance (refusing to report $0)")

// PriceForIdentity returns the published rate card and its price-provenance
// string ("anthropic-published:<tier>") for a resolved canonical identity, and
// fails closed with ErrUnknownModelPricing for an unknown one.
func PriceForIdentity(mi ModelIdentity) (Rates, string, error) {
	if !mi.Known() {
		return Rates{}, "", fmt.Errorf("%w: %q", ErrUnknownModelPricing, mi.Raw)
	}
	tier := canonicalModelTier[mi.Canonical]
	return Pricing[tier], "anthropic-published:" + tier, nil
}

// StrictModelCostUSD is the FAIL-CLOSED companion to CostUSD (#4635):
//
//   - a canonically resolved fleet id prices from its pinned row;
//   - a non-billed harness row (<synthetic>) legitimately costs 0;
//   - a NON-Claude id with a published card (deepseek/glm/kimi, #4823) prices
//     from that card;
//   - anything else returns ErrUnknownModelPricing instead of a silent 0 —
//     and a Claude-family spelling that did NOT resolve exactly refuses the
//     neighboring-tier price the legacy substring heuristic would assign.
func StrictModelCostUSD(model string, input, cacheWrite, cacheRead, output int64) (float64, error) {
	if nonBilled(model) {
		return 0, nil
	}
	if mi := ResolveModelID(model); mi.Known() {
		r, _, err := PriceForIdentity(mi)
		if err != nil {
			return 0, err
		}
		return rawCostUSD(r, input, cacheWrite, cacheRead, output), nil
	}
	if claudeFamilySpelling(model) {
		// An unresolved Claude spelling must NOT borrow a neighboring tier via
		// the substring heuristic — that is the overmatch #4635 closes.
		return 0, fmt.Errorf("%w: unresolved Claude-family id %q", ErrUnknownModelPricing, model)
	}
	if r, ok := PriceFor(model); ok {
		return rawCostUSD(r, input, cacheWrite, cacheRead, output), nil
	}
	return 0, fmt.Errorf("%w: %q", ErrUnknownModelPricing, model)
}

// claudeFamilySpelling reports whether a model id READS as a Claude-family
// spelling — the population for which an inexact match must fail closed
// instead of neighbor-pricing. Deliberately broad (any tier word counts): the
// cost of a false positive is an explicit UNKNOWN hold, never a wrong dollar.
func claudeFamilySpelling(model string) bool {
	m := strings.ToLower(model)
	for _, sub := range []string{"claude", "opus", "sonnet", "haiku", "fable"} {
		if strings.Contains(m, sub) {
			return true
		}
	}
	return false
}
