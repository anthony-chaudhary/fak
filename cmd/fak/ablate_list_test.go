package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ablate"
)

// TestAblateListHuman pins the `fak ablate --list` catalog visual: every cache lever in
// the catalog must appear with its plane/fidelity/env, the presets must be listed, and a
// non-cache sweepable knob (normgate) must NOT leak in — the whole point of the catalog is
// that it is the cache subset, not all of KnownFeatures.
func TestAblateListHuman(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runAblate(&out, &errb, []string{"--list"}); rc != 0 {
		t.Fatalf("runAblate --list rc=%d stderr=%q", rc, errb.String())
	}
	got := out.String()

	// Every catalog lever, its plane, fidelity, and env gate must be visible.
	for _, c := range ablate.FeatureCatalog() {
		if !strings.Contains(got, c.Token) {
			t.Errorf("--list output missing lever %q", c.Token)
		}
		if !strings.Contains(got, c.Plane) {
			t.Errorf("--list output missing plane %q (lever %q)", c.Plane, c.Token)
		}
		if c.EnvVar != "" && !strings.Contains(got, c.EnvVar) {
			t.Errorf("--list output missing env gate %q (lever %q)", c.EnvVar, c.Token)
		}
	}

	// The presets are part of the visual.
	for _, name := range ablate.PresetNames() {
		if !strings.Contains(got, ablate.PresetPrefix+name) {
			t.Errorf("--list output missing preset %q", ablate.PresetPrefix+name)
		}
	}

	// A non-cache sweepable feature has no FeatureCard, so it must not appear as a lever
	// row. normgate is sweepable (KnownFeatures) but is a guard knob, not a cache lever.
	if strings.Contains(got, "normgate") {
		t.Errorf("--list leaked a non-cache sweepable feature (normgate) into the cache catalog")
	}

	// Header sanity: the count in the banner must match the catalog size.
	if !strings.Contains(got, "the cache-lever catalog (9 levers") {
		t.Errorf("--list banner does not report 9 levers; catalog size drifted or banner stale:\n%s", got)
	}
}

// TestAblateListJSON proves the machine form emits exactly the catalog cards plus the
// preset expansions, so a consumer (or a doc-regen) reads the same source of truth the
// human table renders from.
func TestAblateListJSON(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runAblate(&out, &errb, []string{"--list", "--json"}); rc != 0 {
		t.Fatalf("runAblate --list --json rc=%d stderr=%q", rc, errb.String())
	}

	var payload struct {
		Catalog []ablate.FeatureCard `json:"catalog"`
		Presets map[string][]string  `json:"presets"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("--list --json did not emit valid JSON: %v\n%s", err, out.String())
	}

	want := ablate.FeatureCatalog()
	if len(payload.Catalog) != len(want) {
		t.Fatalf("--list --json catalog size = %d, want %d", len(payload.Catalog), len(want))
	}
	for i, c := range want {
		if payload.Catalog[i].Token != c.Token {
			t.Errorf("catalog[%d] token = %q, want %q (order must be stable/sorted)", i, payload.Catalog[i].Token, c.Token)
		}
		if payload.Catalog[i].EnvVar != c.EnvVar {
			t.Errorf("catalog[%d] env_var = %q, want %q", i, payload.Catalog[i].EnvVar, c.EnvVar)
		}
	}

	// Presets must round-trip the same expansion ExpandPresets uses.
	for _, name := range ablate.PresetNames() {
		got := strings.Join(payload.Presets[name], ",")
		wantExp := strings.Join(ablate.PresetExpansion(name), ",")
		if got != wantExp {
			t.Errorf("preset %q expansion = %q, want %q", name, got, wantExp)
		}
	}
}
