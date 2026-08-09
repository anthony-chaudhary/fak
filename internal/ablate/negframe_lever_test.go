package ablate

// negframe_lever_test.go — the #3568 DoD's own named witnesses for the `negframe_reframe`
// lever, living where the DoD names them: `go test ./internal/ablate -run TestEnvGated` and
// `-run TestCatalog`.
//
// The lever itself already shipped (#3568, and its SessionStart wiring in #5365) and
// cmd/fak's TestNegframeAblationFeatureRegistered covers it from the consumer side. What was
// missing was the gate: BOTH `-run` patterns the DoD names matched ZERO tests here, so each
// command printed "no tests to run" and exited 0 — a green-looking no-op that would keep
// reporting PASS after the registration was deleted. These tests make those two commands
// mean something.
//
// Every assertion is paired with a NEGATIVE control on an unregistered token, so a helper
// that started answering true for everything reds instead of silently ratifying the lever.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// negframeUnregisteredToken is the negative control: a token no init ever registers, used to
// prove each helper below actually discriminates rather than answering true for any string.
const negframeUnregisteredToken = "negframe_reframe_not_a_registered_feature"

// TestEnvGatedNegframeReframeRung is the DoD's `-run TestEnvGated` witness: the reframe lever
// rides the rung-2 subprocess env, so `fak ablate --sweep negframe_reframe` re-execs a child
// that genuinely reads the environment instead of flipping an in-process knob the parent
// already resolved.
func TestEnvGatedNegframeReframeRung(t *testing.T) {
	if !EnvGated(FeatureNegframeReframe) {
		t.Fatalf("EnvGated(%q) = false, want true — the sweep would take the in-process rung and #3546's arms would share one process", FeatureNegframeReframe)
	}
	if EnvGated(negframeUnregisteredToken) {
		t.Fatalf("EnvGated(%q) = true for an unregistered token; the assertion above proves nothing", negframeUnregisteredToken)
	}

	// The child carries this exact env, and cmd/fak's guardNegframeEnvVar reads it back. A
	// rename on either side silently strands the sweep on the default-on treatment arm.
	c, ok := registeredConcept(FeatureNegframeReframe)
	if !ok {
		t.Fatalf("%q is not registered as a concept", FeatureNegframeReframe)
	}
	if c.EnvVar != "FAK_ABLATE_NEGFRAME_REFRAME" {
		t.Fatalf("concept EnvVar = %q, want %q (the env the sweep child carries)", c.EnvVar, "FAK_ABLATE_NEGFRAME_REFRAME")
	}
}

// TestCatalogCardsNegframeReframeLever is the DoD's `-run TestCatalog` witness: the lever is
// carded, so `fak ablate --list` advertises it, and the card agrees with both the concept
// registry and the per-arm CacheEffect a live run attributes.
func TestCatalogCardsNegframeReframeLever(t *testing.T) {
	card, ok := cardFor(FeatureNegframeReframe)
	if !ok {
		t.Fatalf("%q has no FeatureCard; `fak ablate --list` would omit the #3546 lever", FeatureNegframeReframe)
	}
	if _, bogus := cardFor(negframeUnregisteredToken); bogus {
		t.Fatalf("cardFor(%q) returned a card for an unregistered token; the lookup above proves nothing", negframeUnregisteredToken)
	}
	if !knownFeature(card.Token) {
		t.Fatalf("carded token %q is not sweepable", card.Token)
	}
	if card.EnvVar != "FAK_ABLATE_NEGFRAME_REFRAME" {
		t.Fatalf("card EnvVar = %q, want %q (--list would print an env the child never reads)", card.EnvVar, "FAK_ABLATE_NEGFRAME_REFRAME")
	}

	// The lever rewrites model-visible prefix BYTES rather than provider cache retention, so
	// it is carded on the context_view plane and stays lossless (the reframe is
	// token-superset-safe). Both arms must classify, else the sweep reports one-sided.
	if card.Plane != "context_view" || card.Fidelity != "lossless" {
		t.Fatalf("card plane/fidelity = %q/%q, want context_view/lossless", card.Plane, card.Fidelity)
	}
	for _, on := range []bool{true, false} {
		e, produces := cacheEffectForFeature(FeatureNegframeReframe, on, metrics.Arm{}, FeatureConfig{}, "inkernel")
		if !produces {
			t.Fatalf("on=%v arm produced no CacheEffect; the AblationReport would carry no negframe attribution", on)
		}
		if e.Component != card.Component {
			t.Fatalf("on=%v effect component = %q, want the card's %q", on, e.Component, card.Component)
		}
	}
}
