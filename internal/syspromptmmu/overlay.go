package syspromptmmu

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// overlay.go — Rung 3 of the system-prompt MMU (#1261, epic #1258): fill the overlay
// tier by QUERY, not menu, and become the FIRST live caller of the skill-loader
// keystone (capindex.Catalog), which is built but request-path-dead today.
//
// Rung 1 emits the resident base PLAN; Rung 2 (splice.go) realizes it and swaps the
// after-breakpoint overlay cache-safely. This rung PRODUCES that overlay: from a turn's
// intent + a token budget it ranks the at-rest capability cards (cards only, no body
// paged), faults the winners up to budget, and emits each faulted body as a
// non-resident overlay segment SelectOverlay's caller hands to BuildSystemValue /
// SpliceSystemOverlay. The resident base (BaseContextPlan) is never touched, so
// resident base-context tokens stay FLAT as the catalog grows from 0 to ∞ (invariant 4:
// the base is pointers, not bodies).
//
// HIT without re-fault. capindex.CapCard.Digest is the body content hash, available at
// the CARD layer, so the rank-only digest over every positively-scored card identifies
// the exact budgeted selection WITHOUT paging a body: if every ranked card is unchanged,
// the same bodies (hence the same budget cut) apply, so a cached selection is still
// valid. OverlayCache uses that digest as its key — the tier-2 analogue of
// contextq.SkillContextRecord (contextq is tier 3, unimportable from here).
//
// Scope fence. This rung produces the overlay segments; the #1144 CapRef reconciliation
// (the duplicate copy in tier-3 contextq) is not folded here — syspromptmmu (tier 2)
// cannot import contextq, and this rung already speaks the single capindex.CapRef.
//
// Tier: mechanism (2). Imports cachemeta(1) + capindex(2) + stdlib — capindex is the
// same tier (a permitted same-tier edge, like vcachechain→vcachegov), never an upward
// import.

// OverlaySelection is the result of one queried-overlay pass.
type OverlaySelection struct {
	// Overlay is the chosen, faulted overlay segments, in rank order, to append AFTER
	// the Rung-2 breakpoint (each is SegMessage, no cache_control, Witness = card digest).
	Overlay []cachemeta.PromptSegment
	// Faulted is the CapRefs whose bodies were paged into the overlay, in rank order.
	Faulted []capindex.CapRef
	// Skipped is the positively-ranked CapRefs left out because the budget was reached
	// (the boundary card plus every lower-ranked card), in rank order.
	Skipped []capindex.CapRef
	// Tokens is the overlay's total estimated token cost (≤ the budget).
	Tokens int64
	// Digest is the cards-only rank digest over (intent, budget, every ranked card's
	// ref+body-digest). It is the HIT key: identical digest ⇒ an identical selection,
	// provable without paging a body.
	Digest string
}

// rankDigest hashes (intent, budget, every ranked card's ref+digest) — cards only, no
// body paged. Identical digest ⇒ the budgeted selection is identical, so OverlayCache
// can serve a HIT without re-faulting. NUL-separated so no field concatenation aliases
// another.
func rankDigest(intent string, budgetTokens int64, cards []capindex.CapCard) string {
	h := sha256.New()
	h.Write([]byte(intent))
	h.Write([]byte{0})
	h.Write([]byte(strconv.FormatInt(budgetTokens, 10)))
	h.Write([]byte{0})
	for _, c := range cards {
		h.Write([]byte(c.Ref.Kind))
		h.Write([]byte{0})
		h.Write([]byte(c.Ref.Name))
		h.Write([]byte{0})
		h.Write([]byte(c.Ref.Version))
		h.Write([]byte{0})
		h.Write([]byte(c.Digest))
		h.Write([]byte{0})
	}
	return witnessPrefix + hex.EncodeToString(h.Sum(nil))
}

// SelectOverlay drives the skill-loader keystone for one turn: rank the at-rest cards
// for `intent` (no body paged), then fault winners in rank order while the cumulative
// overlay tokens fit `budgetTokens`. It STOPS at the first card whose body would exceed
// the budget — that one boundary card is faulted to MEASURE it (capindex carries no
// at-rest body-size), then it and every lower-ranked card are recorded in Skipped
// without being faulted. A nil catalog or a non-positive budget yields an empty overlay.
//
// The returned overlay feeds BuildSystemValue (a fresh base) or SpliceSystemOverlay (a
// per-turn swap); both leave the resident spine+policy prefix byte-identical, so the
// Rung-2 cache invariant holds while the overlay changes.
func SelectOverlay(cat *capindex.Catalog, intent string, budgetTokens int64) OverlaySelection {
	if cat == nil || budgetTokens <= 0 {
		return OverlaySelection{Digest: rankDigest(intent, budgetTokens, nil)}
	}
	cards := cat.RankCards(intent)
	sel := OverlaySelection{Digest: rankDigest(intent, budgetTokens, cards)}

	budgetReached := false
	for _, card := range cards {
		if budgetReached {
			sel.Skipped = append(sel.Skipped, card.Ref)
			continue
		}
		cp, err := cat.Lookup(card.Ref)
		if err != nil {
			continue // a card with no live resolver is dropped, never faked
		}
		body := cp.Materialize()
		tok := estTokens(body)
		if sel.Tokens+tok > budgetTokens {
			// Boundary card: faulted only to measure. It and the rest are over budget.
			sel.Skipped = append(sel.Skipped, card.Ref)
			budgetReached = true
			continue
		}
		sel.Overlay = append(sel.Overlay, cachemeta.PromptSegment{
			Kind:    cachemeta.SegMessage, // tail content, appended after the cached prefix
			Tokens:  tok,
			Content: body,
			Witness: card.Digest, // ties the segment to its body hash (coherence breaks on revoke)
		})
		sel.Faulted = append(sel.Faulted, card.Ref)
		sel.Tokens += tok
	}
	return sel
}

// OverlayCache memoizes overlay selections by their cards-only rank digest — the tier-2
// analogue of contextq.SkillContextRecord. A re-invocation whose ranked cards are
// unchanged (same digest) is a HIT served from the cache with NO body re-faulted; a
// changed capability body (its digest moves) or a changed matching set is a MISS that
// re-selects. Safe for concurrent gateway turns.
type OverlayCache struct {
	mu      sync.Mutex
	entries map[string]OverlaySelection
}

// NewOverlayCache builds an empty overlay cache.
func NewOverlayCache() *OverlayCache {
	return &OverlayCache{entries: make(map[string]OverlaySelection)}
}

// GetOrSelect returns the overlay selection for (intent, budget), serving a HIT from the
// cache when the ranked cards are unchanged (no re-fault) and otherwise selecting fresh
// and caching the result. hit reports which path was taken. The HIT check ranks the
// cards (cards only, cheap) to derive the digest key before deciding to fault.
func (oc *OverlayCache) GetOrSelect(cat *capindex.Catalog, intent string, budgetTokens int64) (sel OverlaySelection, hit bool) {
	var key string
	if cat == nil || budgetTokens <= 0 {
		key = rankDigest(intent, budgetTokens, nil)
	} else {
		key = rankDigest(intent, budgetTokens, cat.RankCards(intent))
	}

	oc.mu.Lock()
	cached, ok := oc.entries[key]
	oc.mu.Unlock()
	if ok {
		return cached, true
	}

	sel = SelectOverlay(cat, intent, budgetTokens)
	oc.mu.Lock()
	oc.entries[sel.Digest] = sel
	oc.mu.Unlock()
	return sel, false
}
