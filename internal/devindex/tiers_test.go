package devindex

// The C1 (#2230) coverage witness for the epic-#2228 verb-tier classification.
// The contract these tests hold, forever:
//
//   1. TOTAL: every verb the cmd/fak/main.go dispatch switch routes has a tier —
//      a new dispatch case lands classified or this file reds the build. The
//      tier decision is made consciously at authoring time, never by silent
//      accretion (the exact ambiguity the epic exists to kill).
//   2. LIVE: every verbTiers key names a verb that actually dispatches — a
//      renamed/deleted verb cannot leave a ghost classification behind.
//   3. SMALL FRONT DOOR: the frontdoor tier stays at or under its ratified
//      ceiling. Growing the product surface is a deliberate, reviewed act
//      (bump the ceiling in the same commit), not a drive-by.
//
// "No verb in two tiers" needs no test: verbTiers is one map literal, and a
// duplicate key is a compile error.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// liveTierTokens loads the real repo's dispatch tokens, skipping (like
// TestRealRepoDogfood) when the tree is not readable — outside a repo there is
// no live switch to reconcile against.
func liveTierTokens(t *testing.T) []string {
	t.Helper()
	root := FindRoot(".")
	b, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skip("cmd/fak/main.go unreadable (no repo root); tier coverage is only checkable in-repo")
	}
	return mainDispatchVerbs(b)
}

// TestVerbTierCoverageIsTotal reds when any live dispatch token (canonical or
// alias spelling) resolves to no tier. The failure message names every gap so
// the fix is mechanical: add the verb to ONE tier block in tiers.go.
func TestVerbTierCoverageIsTotal(t *testing.T) {
	var missing []string
	for _, tok := range liveTierTokens(t) {
		if _, ok := TierOf(tok); !ok {
			missing = append(missing, tok)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("dispatched verbs with NO tier (classify each in internal/devindex/tiers.go, one tier block each): %s",
			strings.Join(missing, ", "))
	}
}

// TestVerbTiersNameOnlyLiveVerbs is the converse: every classification key must
// still be a live verb, resolved the same way TierOf resolves a token — via the
// manifest's spellings when curated, else the raw token. A verb renamed or
// removed from the dispatch switch must take its tier row with it.
func TestVerbTiersNameOnlyLiveVerbs(t *testing.T) {
	live := map[string]bool{}
	for _, tok := range liveTierTokens(t) {
		live[tok] = true
		if v, ok := manifestVerbByName(tok); ok {
			live[strings.ToLower(v.Name)] = true // canonical name of an alias spelling
		}
	}
	var dead []string
	for key := range verbTiers {
		if !live[key] {
			dead = append(dead, key)
		}
	}
	if len(dead) > 0 {
		t.Fatalf("verbTiers classifies verbs that no longer dispatch (remove the rows): %s",
			strings.Join(dead, ", "))
	}
}

// TestFrontdoorTierStaysSmall is the product-surface ratchet: the ratified
// frontdoor set is 24 named verbs (epic #2228 / #2230 — ~20 concepts once the
// replay/top/pull/ls companions fold into run/ps/model). Promoting a verb to
// the front door means bumping this ceiling IN THE SAME COMMIT, with the
// reasoning in the commit message — the review is the point of the gate.
func TestFrontdoorTierStaysSmall(t *testing.T) {
	const ceiling = 24
	var front []string
	for key, tier := range verbTiers {
		if tier == TierFrontdoor {
			front = append(front, key)
		}
	}
	if len(front) > ceiling {
		t.Fatalf("frontdoor tier has %d verbs (ceiling %d): %s — most verbs are dev; promote deliberately or classify as dev",
			len(front), ceiling, strings.Join(front, ", "))
	}
}

// TestTierOfCanonicalizesAliases pins the alias path: a flag-shaped or alternate
// spelling answers with its canonical verb's tier, compiled-in (no repo needed).
func TestTierOfCanonicalizesAliases(t *testing.T) {
	cases := []struct {
		tok  string
		want VerbTier
	}{
		{"guard", TierFrontdoor},
		{"-h", TierFrontdoor},        // alias of help
		{"--version", TierFrontdoor}, // alias of version
		{"benchloop", TierDev},       // alias of bench-loop
		{"SWEEP", TierDev},           // case-insensitive
		{"guard-stophook", TierHidden},
	}
	for _, c := range cases {
		got, ok := TierOf(c.tok)
		if !ok || got != c.want {
			t.Errorf("TierOf(%q) = (%q, %v), want (%q, true)", c.tok, got, ok, c.want)
		}
	}
	if _, ok := TierOf("no-such-verb-ever"); ok {
		t.Error("TierOf invented a tier for an unknown token")
	}
}
