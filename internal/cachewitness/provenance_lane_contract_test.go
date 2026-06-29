package cachewitness

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

type visualLane string

const (
	laneWitnessed visualLane = "WITNESSED"
	laneObserved  visualLane = "OBSERVED"
	laneModeled   visualLane = "MODELED"
)

type laneStyle struct {
	Color  string
	Hatch  string
	Legend string
}

type renderPrimitive struct {
	Name       string
	Source     string
	Provenance Provenance
	Lane       visualLane
}

var provenanceLaneStyles = map[visualLane]laneStyle{
	laneWitnessed: {
		Color:  "#1a7f37",
		Hatch:  "solid",
		Legend: "WITNESSED - fak-authored witness",
	},
	laneObserved: {
		Color:  "#0969da",
		Hatch:  "diagonal",
		Legend: "OBSERVED - provider or external telemetry",
	},
	laneModeled: {
		Color:  "#bf8700",
		Hatch:  "crosshatch",
		Legend: "MODELED - projection until measured",
	},
}

var provenanceSourceLane = map[string]Provenance{
	"max_delta_zero":           Witnessed,
	"reuse_bit":                Witnessed,
	"provider_cache_read":      Observed,
	"page_down_tier_projected": Modeled,
}

func validateRenderPrimitive(p renderPrimitive) error {
	want, ok := provenanceSourceLane[p.Source]
	if !ok {
		return fmt.Errorf("unknown provenance source %q", p.Source)
	}
	if p.Provenance != want {
		return fmt.Errorf("%s provenance = %s, want %s from %s", p.Name, p.Provenance, want, p.Source)
	}
	if p.Lane != visualLane(want) {
		return fmt.Errorf("%s rendered on %s lane with %s provenance", p.Name, p.Lane, p.Provenance)
	}
	if st := provenanceLaneStyles[p.Lane]; st.Color == "" || st.Hatch == "" || st.Legend == "" {
		return fmt.Errorf("%s lane %s has incomplete visible encoding", p.Name, p.Lane)
	}
	return nil
}

func TestModeledPrimitiveCannotRenderOnWitnessedLane(t *testing.T) {
	err := validateRenderPrimitive(renderPrimitive{
		Name:       "projected page-down-tier value",
		Source:     "page_down_tier_projected",
		Provenance: Modeled,
		Lane:       laneWitnessed,
	})
	if err == nil {
		t.Fatal("MODELED primitive rendered on WITNESSED lane without an honesty error")
	}
	if !strings.Contains(err.Error(), "WITNESSED lane with MODELED provenance") {
		t.Fatalf("wrong error for modeled-on-witnessed lane: %v", err)
	}
}

func TestProvenanceLaneStylesAreVisibleAndDistinct(t *testing.T) {
	seen := map[string]visualLane{}
	for lane, style := range provenanceLaneStyles {
		if !strings.Contains(style.Legend, string(lane)) {
			t.Fatalf("%s legend %q does not name the lane", lane, style.Legend)
		}
		signature := style.Color + "|" + style.Hatch
		if prior, ok := seen[signature]; ok {
			t.Fatalf("%s and %s render identically as %s", lane, prior, signature)
		}
		seen[signature] = lane
	}
}

func TestKnownC8SourcesMapToSeparateProvenanceLanes(t *testing.T) {
	cases := []renderPrimitive{
		{Name: "bit-exact comparison", Source: "max_delta_zero", Provenance: Witnessed, Lane: laneWitnessed},
		{Name: "reuse bit", Source: "reuse_bit", Provenance: Witnessed, Lane: laneWitnessed},
		{Name: "provider cache_read", Source: "provider_cache_read", Provenance: Observed, Lane: laneObserved},
		{Name: "projected page-down-tier value", Source: "page_down_tier_projected", Provenance: Modeled, Lane: laneModeled},
	}
	for _, tc := range cases {
		if err := validateRenderPrimitive(tc); err != nil {
			t.Fatalf("%s: %v", tc.Name, err)
		}
	}
}

func TestC8ProvenanceLaneContractDocIsPresent(t *testing.T) {
	b, err := os.ReadFile("../../docs/proofs/ctxsafety-provenance-lane.md")
	if err != nil {
		t.Fatalf("read render contract doc: %v", err)
	}
	doc := string(b)
	for _, needle := range []string{
		"WITNESSED",
		"OBSERVED",
		"MODELED",
		"max|Δ|=0",
		"cache_read",
		"projected page-down-tier",
		"No primitive may render a MODELED point identically to a WITNESSED point",
	} {
		if !strings.Contains(doc, needle) {
			t.Fatalf("render contract doc missing %q", needle)
		}
	}
}
