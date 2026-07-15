package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ablate"
)

// printAblateCatalog renders `fak ablate --list`: the static cache-lever catalog behind
// FeatureCatalog(). It is the "see what I can try WITHOUT running a replay" visual — every
// row is one FeatureCard from internal/ablate/catalog.go, the same classification a live
// arm reports, so the printed menu can never drift from what a sweep actually measures.
// The human table stays narrow (the long per-lever summaries print as a second block); the
// JSON form emits the raw cards plus the preset expansions for a machine reader.
func printAblateCatalog(w io.Writer, asJSON bool) {
	cards := ablate.FeatureCatalog()
	presets := ablateCatalogPresets()

	rungs := ablateRungCatalog()

	if asJSON {
		payload := struct {
			Catalog []ablate.FeatureCard `json:"catalog"`
			Presets map[string][]string  `json:"presets"`
			Rungs   []string             `json:"rungs"`
		}{Catalog: cards, Presets: presets, Rungs: rungs}
		b, _ := json.MarshalIndent(payload, "", "  ")
		_, _ = w.Write(b)
		_, _ = io.WriteString(w, "\n")
		return
	}

	tokenPreset := ablateTokenPresets(presets)

	fmt.Fprintf(w, "== fak ablate --list: the cache-lever catalog (%d levers, all fak-owned) ==\n", len(cards))
	fmt.Fprintln(w, "source: internal/ablate/catalog.go — the single source of truth --list and a live arm share")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "%-14s %-12s %-30s %-20s %-12s %-5s %s\n",
		"lever", "preset", "plane", "component", "fidelity", "rung", "env gate")
	for _, c := range cards {
		fmt.Fprintf(w, "%-14s %-12s %-30s %-20s %-12s %-5s %s\n",
			c.Token, ablateEmDash(tokenPreset[c.Token]), c.Plane, c.Component,
			c.Fidelity, ablateRung(c), ablateEnvCell(c))
	}

	fmt.Fprintln(w)
	fmt.Fprintln(w, "presets (sweep a whole cache plane in one flag):")
	for _, name := range ablate.PresetNames() {
		fmt.Fprintf(w, "  @%-11s %s\n", name, strings.Join(presets[name], ","))
	}

	fmt.Fprintln(w)
	fmt.Fprintln(w, "what each lever does to the cache:")
	for _, c := range cards {
		fmt.Fprintf(w, "  %-14s %s\n", c.Token, c.Summary)
	}

	fmt.Fprintln(w)
	fmt.Fprintln(w, "rung 1 = flipped in-process (runtime-settable); rung 2 = re-exec'd in a child with FAK_* set.")
	fmt.Fprintln(w, "sweep e.g.: fak ablate --sweep vdso   |   --sweep @wire-cache   |   --sweep radix,compressor")

	// The adjudicator-rung axis is a DIFFERENT sweep (over a turnbench trace, not a cache
	// arm): `--rungs` masks one PDP/PEP link at a time. List those names here so an operator
	// discovers them without reading Go — distinct from the cache "rung 1/2" column above.
	fmt.Fprintln(w)
	fmt.Fprintf(w, "adjudicator rungs (a DIFFERENT axis — flip one PDP/PEP link over a turnbench trace, %d levers):\n", len(rungs))
	fmt.Fprintf(w, "  %s\n", strings.Join(rungs, ", "))
	fmt.Fprintln(w, "rung sweep e.g.: fak ablate --rungs --trace FILE   |   --rungs=grammar,ifc-sink --trace FILE")
}

// ablateCatalogPresets returns the preset name -> sorted token expansion map, the same
// grouping ExpandPresets substitutes at sweep time.
func ablateCatalogPresets() map[string][]string {
	out := map[string][]string{}
	for _, name := range ablate.PresetNames() {
		out[name] = ablate.PresetExpansion(name)
	}
	return out
}

// ablateTokenPresets inverts the preset map to token -> "@preset", so each catalog row can
// name the one preset it belongs to (a lever on a plane with no preset gets "").
func ablateTokenPresets(presets map[string][]string) map[string]string {
	out := map[string]string{}
	names := make([]string, 0, len(presets))
	for name := range presets {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		for _, tok := range presets[name] {
			out[tok] = ablate.PresetPrefix + name
		}
	}
	return out
}

// ablateRung reports which sweep rung a lever runs under: rung 1 is the one in-process
// runtime-settable knob (vdso); every env-gated lever is rung 2 (subprocess re-exec).
func ablateRung(c ablate.FeatureCard) string {
	if c.RuntimeSettable {
		return "1"
	}
	return "2"
}

// ablateEnvCell renders the env-gate column: the FAK_* the rung-2 child carries, or a
// note that the one runtime knob needs no env gate.
func ablateEnvCell(c ablate.FeatureCard) string {
	if c.EnvVar == "" {
		return "— (in-process runtime knob)"
	}
	return c.EnvVar
}

// ablateEmDash renders an empty cell as an em-dash so a blank never reads as missing data.
func ablateEmDash(s string) string {
	if s == "" {
		return "—"
	}
	return s
}
