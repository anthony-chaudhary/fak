package selfquery

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

// MultiCatalog is the cross-repo fan-out over N single-root Catalogs (#3435, epic
// #3434 — the anchor capability that makes "query my other checkouts too"
// possible). Each member Catalog is loaded and queried exactly as the single-root
// path does; MultiCatalog only concatenates their card sets BELOW the ranker,
// stamping every card with its source Root so a merged result stays attributable.
//
// This mirrors GitQL's GitQLDataProvider.provide, which loops its repos, selects
// per-repo, and appends each repo's rows into one flat result carrying a `repo`
// column — the merge happens at the data layer, so nothing downstream (ranker,
// freshness, grounding) needs multi-repo awareness. It deliberately does NOT copy
// GQL's serial .unwrap()-per-repo loop: a root that fails to load is skipped and
// reported (Skipped), never fatal — one unreadable sibling repo must not kill the
// whole query.
type MultiCatalog struct {
	members []rootCatalog
	skipped []RootLoadError
}

type rootCatalog struct {
	root string
	cat  *Catalog
}

// RootLoadError records a root that LoadMany skipped, so a bad sibling repo is
// surfaced (Skipped) rather than silently dropped or fatally aborting the fan-out.
type RootLoadError struct {
	Root string
	Err  error
}

func (e RootLoadError) Error() string { return e.Root + ": " + e.Err.Error() }
func (e RootLoadError) Unwrap() error { return e.Err }

// LoadMany loads a Catalog for each root, stamping cards with their source root so
// a cross-repo query stays attributable (#3435). Semantics:
//
//   - Roots are resolved to absolute paths and de-duplicated (querying the same
//     checkout twice would double-report every card). An empty entry resolves to
//     the the current directory.
//   - A root that fails to load is SKIPPED and recorded in Skipped(), never fatal
//     (the GQL do-not: no serial .unwrap()-per-repo panic).
//   - Only the FIRST (primary) root receives the caller's live-plane Options. The
//     live plane (process-global MCP tools, memory, context-plan) is not per-repo,
//     so fanning opt out across roots would double-report every tool card. Each
//     root still loads its OWN dev plane and its own .claude/skills cap cards.
//
// LoadMany errors only when NO root loaded at all; a partial success (some roots
// skipped) returns a usable MultiCatalog plus the recorded skips.
func LoadMany(roots []string, opt Options) (*MultiCatalog, error) {
	if len(roots) == 0 {
		roots = []string{""}
	}
	mc := &MultiCatalog{}
	seen := map[string]bool{}
	primaryTaken := false
	for _, r := range roots {
		root := strings.TrimSpace(r)
		if root == "" {
			root = "."
		}
		if abs, err := filepath.Abs(root); err == nil {
			root = abs
		}
		if seen[root] {
			continue
		}
		seen[root] = true
		perRoot := Options{DevLoader: opt.DevLoader}
		if !primaryTaken {
			perRoot = opt
		} else if opt.Dev != nil {
			perRoot.Dev = opt.Dev
		}
		cat, err := Load(root, perRoot)
		if err != nil {
			mc.skipped = append(mc.skipped, RootLoadError{Root: root, Err: err})
			continue
		}
		primaryTaken = true
		mc.members = append(mc.members, rootCatalog{root: root, cat: cat})
	}
	if len(mc.members) == 0 {
		if len(mc.skipped) > 0 {
			return mc, fmt.Errorf("LoadMany: no roots loaded, %d skipped (first: %w)", len(mc.skipped), mc.skipped[0])
		}
		return mc, errors.New("LoadMany requires at least one root")
	}
	return mc, nil
}

// Skipped returns the roots LoadMany could not load (skip-and-report). Empty when
// every requested root loaded.
func (mc *MultiCatalog) Skipped() []RootLoadError {
	return append([]RootLoadError(nil), mc.skipped...)
}

// Roots returns the source roots that loaded successfully, in fan-out order (the
// first is the primary root that carries the live plane).
func (mc *MultiCatalog) Roots() []string {
	out := make([]string, 0, len(mc.members))
	for _, m := range mc.members {
		out = append(out, m.root)
	}
	return out
}

func (mc *MultiCatalog) primaryRoot() string {
	if len(mc.members) == 0 {
		return ""
	}
	return mc.members[0].root
}

func (mc *MultiCatalog) catalogForRoot(root string) *Catalog {
	for _, m := range mc.members {
		if m.root == root {
			return m.cat
		}
	}
	return nil
}

// Cards returns the Root-stamped union of every member catalog's cards for the
// plane, re-sorted so the merge is deterministic (cardLess tiebreaks on Root).
// Like Catalog.Cards it leaves Freshness empty — that advisory rung is computed
// only on the Query path — so a MultiCatalog digest surface stays stable.
func (mc *MultiCatalog) Cards(plane Plane) []FeatureCard {
	var out []FeatureCard
	for _, m := range mc.members {
		cards := m.cat.Cards(plane)
		for i := range cards {
			cards[i].Root = m.root
		}
		out = append(out, cards...)
	}
	sortCards(out)
	return out
}

// Query ranks the cross-repo union against req.Query, mirroring Catalog.Query so
// the single-root and multi-root paths behave identically apart from the fan-out.
// Freshness is computed PER ROOT (each card's cited artifact resolves against its
// own checkout) and applied with a root-qualified key so two repos carrying a
// same-named card never cross-contaminate rungs.
func (mc *MultiCatalog) Query(req Request) (Response, error) {
	return runQuery(req, queryPaths{
		root:    mc.primaryRoot(),
		cards:   mc.Cards,
		rungs:   mc.freshnessRungs,
		apply:   applyFreshnessMulti,
		rungKey: func(c FeatureCard) string { return c.Root + "\x00" + cardKey(c) },
		detail: func(card FeatureCard, q string) (Detail, error) {
			// The detail must be built by the card's OWN checkout, since its DetailRef
			// resolves there. A card whose Root names no member (it cannot, but the
			// fallback keeps the fan-out total) falls back to the primary.
			owner := mc.catalogForRoot(card.Root)
			if owner == nil {
				owner = mc.members[0].cat
			}
			return owner.detail(card, q)
		},
	})
}

// freshnessRungs computes the advisory currency rungs per source root, keyed by
// root+"\x00"+cardKey so cross-root same-named cards stay independent. Supersession
// is scoped within a root (a note supersedes only its own repo's siblings) and
// staleness resolves each DetailRef against that root.
func (mc *MultiCatalog) freshnessRungs(all []FeatureCard) map[string]string {
	byRoot := map[string][]FeatureCard{}
	for _, c := range all {
		byRoot[c.Root] = append(byRoot[c.Root], c)
	}
	out := map[string]string{}
	for root, cs := range byRoot {
		rungs := freshnessByKey(cs)
		for k, v := range stalenessByKey(cs, root) {
			rungs[k] = v
		}
		for k, v := range rungs {
			out[root+"\x00"+k] = v
		}
	}
	return out
}

// applyFreshnessMulti stamps the root-qualified rungs onto the ranked cards.
func applyFreshnessMulti(cards []FeatureCard, rungs map[string]string) {
	if len(rungs) == 0 {
		return
	}
	for i := range cards {
		if r, ok := rungs[cards[i].Root+"\x00"+cardKey(cards[i])]; ok {
			cards[i].Freshness = r
		}
	}
}
