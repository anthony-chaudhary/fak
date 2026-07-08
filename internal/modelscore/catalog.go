package modelscore

import (
	"sort"
	"strings"
)

// catalog.go — the OpenRouter-shaped discovery projection.
//
// OpenRouter exposes GET /api/v1/models: a normalized list of every model with a
// consistent id / context-length / pricing shape, so a client can DISCOVER what is
// available without hard-coding provider-specific detail. fak's `/v1/models` today
// advertises only the single SERVED model; this projection is the reusable core of a
// richer discovery surface (a `fak models` listing, or per-served-model metadata on
// `/v1/models`) built from the evidence this package already holds.
//
// It keeps THE ONE LAW: evidence, never a decision. A CatalogEntry blends nothing and
// ranks nothing — it re-shapes the raw context + cost rows into the discovery view and
// carries the Illustrative honesty flag forward, so a consumer can never mistake a
// placeholder figure for a measured one (exactly the mistake a naive catalog dump
// would invite). It is pure: same registry -> same catalog.

// CatalogPricing is a model's rough $/Mtok price in the OpenRouter-style split
// (prompt / completion). It is a cost LENS, never a bill — see the Sources and
// Illustrative fields on the entry for whether the figure is measured or a stand-in.
type CatalogPricing struct {
	PromptPerMTok     float64 `json:"prompt_per_mtok"`     // $/Mtok input
	CompletionPerMTok float64 `json:"completion_per_mtok"` // $/Mtok output
}

// CatalogEntry is one model's normalized, discovery-friendly row — the analogue of
// one element of OpenRouter's GET /api/v1/models response. ContextTokens and Pricing
// are populated only when the registry holds that evidence (a model with neither
// still appears, id-only, so the catalog lists EVERY known model, not only fully
// priced ones). Illustrative is true when ANY surfaced figure is a placeholder rather
// than a measurement; Sources carries the provenance citations behind the figures.
type CatalogEntry struct {
	ID            string          `json:"id"`
	ContextTokens int             `json:"context_tokens,omitempty"`
	Pricing       *CatalogPricing `json:"pricing,omitempty"`
	Illustrative  bool            `json:"illustrative"`
	Sources       []string        `json:"sources,omitempty"`
}

// Catalog renders the whole registry as a sorted, OpenRouter-shaped discovery list:
// one CatalogEntry per model, in model-id order (deterministic). Pure projection —
// no blend, no rank, no network.
func (r *Registry) Catalog() []CatalogEntry {
	out := make([]CatalogEntry, 0, len(r.models))
	for _, id := range r.Models() { // Models() is sorted -> deterministic
		out = append(out, r.models[id].catalogEntry())
	}
	return out
}

// CatalogEntry returns the normalized discovery row for one model, or false if the
// registry has no evidence for it. This is the single-model lookup a `/v1/models`
// handler uses to enrich the served model's advertised metadata with real,
// provenance-carrying context/pricing instead of a hard-coded constant.
func (r *Registry) CatalogEntry(model string) (CatalogEntry, bool) {
	ev, ok := r.models[model]
	if !ok {
		return CatalogEntry{}, false
	}
	return ev.catalogEntry(), true
}

// catalogEntry projects one model's raw evidence into the discovery row, folding the
// per-figure Illustrative flags into one entry-level flag and gathering the distinct
// provenance sources.
func (ev ModelEvidence) catalogEntry() CatalogEntry {
	e := CatalogEntry{ID: ev.Model}
	srcs := map[string]bool{}
	if ev.Context != nil {
		e.ContextTokens = ev.Context.Tokens
		if ev.Context.Provenance.Illustrative {
			e.Illustrative = true
		}
		if s := strings.TrimSpace(ev.Context.Provenance.Source); s != "" {
			srcs[s] = true
		}
	}
	if ev.Cost != nil {
		e.Pricing = &CatalogPricing{PromptPerMTok: ev.Cost.In, CompletionPerMTok: ev.Cost.Out}
		if ev.Cost.Provenance.Illustrative {
			e.Illustrative = true
		}
		if s := strings.TrimSpace(ev.Cost.Provenance.Source); s != "" {
			srcs[s] = true
		}
	}
	if len(srcs) > 0 {
		keys := make([]string, 0, len(srcs))
		for s := range srcs {
			keys = append(keys, s)
		}
		sort.Strings(keys)
		e.Sources = keys
	}
	return e
}
