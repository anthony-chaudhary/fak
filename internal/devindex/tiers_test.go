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
//
// Liveness spans BOTH artifacts. The runtime/dev split (#6022) moves dev-owned
// verbs out of `cmd/fak` and into `cmd/fak-dev`; such a verb is still live and
// still needs its tier row, it just dispatches from the other binary. Reading
// only the runtime switch would read every migration as a removal and demand
// the row be deleted, which would silently drop the classification the ratchet
// exists to keep. `cmd/fak-dev/main.go` is optional so this stays checkable in
// a tree that predates the split.
func liveTierTokens(t *testing.T) []string {
	t.Helper()
	root := FindRoot(".")
	b, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skip("cmd/fak/main.go unreadable (no repo root); tier coverage is only checkable in-repo")
	}
	tokens := mainDispatchVerbs(b)
	if devB, devErr := os.ReadFile(filepath.Join(root, "cmd", "fak-dev", "main.go")); devErr == nil {
		tokens = append(tokens, devDispatchVerbs(devB)...)
	}
	return tokens
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
//
// The C1 (#2230) bootstrap sweep once tolerated a dated set of rows whose
// dispatch arms were peer work still in flight on the shared trunk. All those
// arms have since landed, so the exception is gone and the liveness gate now
// holds with no escape hatch — deleting that set was the #2230 close-out
// witness. Steady-state, a new verb's tier row rides the same commit as its
// case arm; there is no longer any home for a row without a live arm.
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
// frontdoor set is 25 named verbs (epic #2228 / #2230 — ~20 concepts once the
// replay/top/pull/ls companions fold into run/ps/model). Promoting a verb to
// the front door means bumping this ceiling IN THE SAME COMMIT, with the
// reasoning in the commit message — the review is the point of the gate.
//
// 25 (was 24): `ablate` promoted — it now renders the live savings dashboard by
// default (one command, no `fak console ablate`) and earns its own front-door
// category (help.go overviewGroups), so it is a product surface, not dev tooling.
//
// 26 (was 25): `agent` promoted — README.md leads with `fak agent --offline` as
// "the shortest public proof" and it is the first command a new evaluator is told
// to run, but it was classified dev, so `fak help` did not list it and
// `fak help agent` printed "fak dev agent — …". A newcomer could not find the one
// command the front page told them to run. It is a product surface (#5464).
// 27 (was 26): `capabilities` promoted - the installed product can now answer
// outcome-language queries such as "token savings" and "turn control" without
// requiring the separately built maintainer executable. This is the discovery
// front door for the other 26 verbs, not another specialist operation.
// 28 (was 27): `study` is the product-owned local content-addressed receipt store;
// its separate study-* forge/classify/link operators remain development commands.
func TestCurrentDispatchedVerbTiersStayClassified(t *testing.T) {
	for _, name := range []string{"architecture", "catchup", "codex-resume", "component", "config", "fanout", "harness", "progress", "quantbench", "stale-work", "test-quality", "windows-setup", "work-delivery"} {
		if got, ok := TierOf(name); !ok || got != TierDev {
			t.Errorf("TierOf(%q)=(%q,%v), want dev", name, got, ok)
		}
	}
	if got, ok := TierOf("resume"); !ok || got != TierDev {
		t.Errorf("TierOf(resume)=(%q,%v), want dev recovery control", got, ok)
	}
}

func TestFrontdoorTierStaysSmall(t *testing.T) {
	const ceiling = 28
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
