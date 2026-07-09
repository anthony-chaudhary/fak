package ablate

import (
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// TestFeatureCatalogCoversEveryCacheLever proves the catalog carries a card for each cache
// lever cacheEffectForFeature classifies, so `fak ablate --list` never omits a lever a run
// would attribute. It walks the KnownFeatures set and asserts every token that produces a
// CacheEffect has a matching card (and vice versa) — the two stay in lockstep.
func TestFeatureCatalogCoversEveryCacheLever(t *testing.T) {
	carded := map[string]bool{}
	for _, c := range FeatureCatalog() {
		carded[c.Token] = true
	}
	for _, f := range KnownFeatures {
		_, produces := cacheEffectForFeature(f, true, metrics.Arm{}, FeatureConfig{}, "inkernel")
		if produces != carded[f] {
			t.Fatalf("feature %q: produces cache effect=%v but carded=%v (catalog and cacheEffectForFeature disagree)", f, produces, carded[f])
		}
	}
}

// TestFeatureCatalogSeedsCacheEffect proves the per-arm CacheEffect's STATIC fields come from
// the card, so the live-run classification and the --list menu are one source of truth.
func TestFeatureCatalogSeedsCacheEffect(t *testing.T) {
	for _, card := range FeatureCatalog() {
		e, ok := cacheEffectForFeature(card.Token, true, metrics.Arm{}, FeatureConfig{}, "inkernel")
		if !ok {
			t.Fatalf("carded feature %q produced no cache effect", card.Token)
		}
		if e.Owner != card.Owner || e.Plane != card.Plane || e.Component != card.Component || e.Dependency != card.Dependency {
			t.Fatalf("feature %q effect static fields diverge from card:\n effect=%+v\n card  =%+v", card.Token, e, card)
		}
	}
}

// TestFeatureCatalogSortedAndPopulated proves the menu is stably ordered and every field a
// user reads is filled — no blank plane/fidelity/summary can reach the CLI.
func TestFeatureCatalogSortedAndPopulated(t *testing.T) {
	cards := FeatureCatalog()
	if len(cards) == 0 {
		t.Fatal("empty catalog")
	}
	if !sort.SliceIsSorted(cards, func(i, j int) bool { return cards[i].Token < cards[j].Token }) {
		t.Fatal("catalog not sorted by token")
	}
	for _, c := range cards {
		if c.Token == "" || c.Plane == "" || c.Fidelity == "" || c.Summary == "" || c.Owner == "" {
			t.Fatalf("catalog card has a blank user-facing field: %+v", c)
		}
		// vdso is the only runtime-settable (in-process) lever; every other is env-gated.
		if c.RuntimeSettable != (c.Token == FeatureVDSO) {
			t.Fatalf("feature %q RuntimeSettable=%v, want %v", c.Token, c.RuntimeSettable, c.Token == FeatureVDSO)
		}
		if c.Token != FeatureVDSO && c.EnvVar == "" {
			t.Fatalf("env-gated feature %q has no EnvVar in its card", c.Token)
		}
	}
}

// TestExpandPresetsWireCache proves @wire-cache expands to exactly the provider-prompt-cache
// levers, sorted and deduped, so one flag sweeps a whole cache dimension.
func TestExpandPresetsWireCache(t *testing.T) {
	got, err := ExpandPresets([]string{"@wire-cache"})
	if err != nil {
		t.Fatalf("expand: %v", err)
	}
	want := []string{FeatureBreakpointPlan, FeaturePrefixGuard, FeatureTTL1H, FeatureUncachedTrim}
	sort.Strings(want)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("@wire-cache expanded to %v, want %v", got, want)
	}
	// Every expanded token must actually be a provider-prompt-cache lever.
	for _, tok := range got {
		if c, _ := cardFor(tok); c.Plane != "provider_prompt_cache_control" {
			t.Fatalf("@wire-cache yielded %q on plane %q, not provider_prompt_cache_control", tok, c.Plane)
		}
	}
}

// TestExpandPresetsMixedDedups proves a raw list mixing bare tokens and a preset flattens,
// preserves order, and collapses a token that also appears inside the preset.
func TestExpandPresetsMixedDedups(t *testing.T) {
	got, err := ExpandPresets([]string{"vdso", "@wire-cache", "bp_plan"})
	if err != nil {
		t.Fatalf("expand: %v", err)
	}
	// vdso leads, then the four wire levers once each; the trailing duplicate bp_plan collapses.
	if got[0] != FeatureVDSO {
		t.Fatalf("expected vdso first, got %v", got)
	}
	seen := map[string]int{}
	for _, tok := range got {
		seen[tok]++
	}
	if seen[FeatureBreakpointPlan] != 1 {
		t.Fatalf("bp_plan appeared %d times, want 1 (dedup failed): %v", seen[FeatureBreakpointPlan], got)
	}
	// vdso (a bare lead token, off the wire plane) + the wire-cache levers, each once;
	// the trailing bp_plan is already a wire lever, so the deduped total is the preset's
	// lever count plus the one bare lead token — assert that relation, not a frozen count.
	wantLen := len(PresetExpansion("wire-cache")) + 1
	if len(got) != wantLen {
		t.Fatalf("expanded to %d tokens, want %d (vdso + wire levers, bp_plan deduped): %v", len(got), wantLen, got)
	}
}

// TestExpandPresetsUnknownFailsLoud proves a mistyped preset is a hard error, never a silent
// zero-arm sweep.
func TestExpandPresetsUnknownFailsLoud(t *testing.T) {
	if _, err := ExpandPresets([]string{"@nope"}); err == nil {
		t.Fatal("unknown preset @nope should fail, got nil error")
	}
}

// TestBuildSweepExpandsPresets proves preset expansion is wired into BuildSweep, so the whole
// arm matrix (and both rungs) see the flattened token list from one `--sweep @wire-cache`.
func TestBuildSweepExpandsPresets(t *testing.T) {
	configs, err := BuildSweep([]string{"@wire-cache"})
	if err != nil {
		t.Fatalf("BuildSweep: %v", err)
	}
	// The matrix is a fail-closed all-off baseline, one arm isolating each preset lever,
	// and (since the preset carries >1 lever) an all-on arm — so the arm count tracks the
	// preset's expansion (levers + 2), not a frozen total.
	wantArms := len(PresetExpansion("wire-cache")) + 2
	if len(configs) != wantArms {
		t.Fatalf("@wire-cache built %d arms, want %d (all-off + one per lever + all-on): %v", len(configs), wantArms, armNames(configs))
	}
}

func armNames(cs []FeatureConfig) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.Name
	}
	return out
}
