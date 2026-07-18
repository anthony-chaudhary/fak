package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guard_policy_explain.go implements `fak guard policy explain` (#5172, epic #5170
// Track A) — the read-only operator view of the guard capability floor's MUTABILITY
// model. `--dump-policy` shows the floor's contents; this verb shows which parts of
// that floor can change at runtime, through which authorized channel, and where each
// knob's current value came from. Every knob is bucketed into one of four amendment
// classes:
//
//   - FROZEN         — never amendable at runtime; changes ship with the binary.
//   - RATCHET        — tighten-only at runtime (the deny overlay: fak guard deny).
//   - GATED-WIDEN    — widens only via an authorized OUT-OF-BAND operator channel
//     (the allow overlay: fak guard allow in the operator's own shell, then a
//     policy reload) — a wrapped agent can never widen its own floor.
//   - SELF-AMENDABLE — session-local presentation knobs; no verdict depends on them.
//
// Provenance names which layer supplied the value: embedded (the shipped
// guard-default-policy.json), user-overlay (~/.fak/guard), repo-overlay
// (.fak/guard), env-overlay (FAK_GUARD_*_OVERLAY override), or reload (a live
// gateway hot reload — only observable from the running gateway, so this static
// verb renders the launch-time layers and names reload as a channel, not a
// provenance it can witness). Read-only by construction: it changes no verdict
// and writes nothing, mirroring `fak guard allow --list`.
const (
	policyClassFrozen        = "FROZEN"
	policyClassRatchet       = "RATCHET"
	policyClassGatedWiden    = "GATED-WIDEN"
	policyClassSelfAmendable = "SELF-AMENDABLE"
)

// policyExplainKnob is one row of the explain report: a named part of the effective
// floor, its amendment class, the channels authorized to amend it, its current
// value, and the layer that supplied that value.
//
// TODO(#5171): consume policy.PolicyKnobRegistry once landed — sibling A1 owns the
// authoritative knob→class registry in internal/policy/amendment.go; until it lands
// this file carries a small representative fallback (policyExplainKnobs) so the
// verb stands alone. The fallback mirrors the semantics documented on Manifest and
// loadGuardCapabilityFloor rather than inventing new ones.
type policyExplainKnob struct {
	Name       string
	Class      string
	Channels   string
	Value      string
	Provenance string
}

const (
	policyChannelsFrozen = "none — ships with the binary (release only)"
	policyChannelsDeny   = "fak guard deny; POST /v1/fak/policy/reload"
	policyChannelsAllow  = "fak guard allow (operator shell); POST /v1/fak/policy/reload"
	policyChannelsLaunch = "guard launch flags"
)

// policyExplainAllowProvenance maps an allow-overlay layer name (guard_allow.go's
// guardAllowOverlayPaths: "user" / "repo" / "env") to the explain provenance vocab.
func policyExplainAllowProvenance(layerName string) string {
	switch layerName {
	case "user":
		return "user-overlay"
	case "repo":
		return "repo-overlay"
	case "env":
		return "env-overlay"
	}
	return layerName
}

// policyExplainKnobs builds the fallback knob rows for one amendment class from the
// embedded floor manifest plus the launch-time overlay layers. Embedded rows always
// render (so every group is non-empty and the grouping is visible even on a bare
// checkout); overlay rows render one per operator-added entry, carrying the layer
// that supplied them as provenance.
func policyExplainKnobs(class string, m policy.Manifest, allowLayers []guardAllowOverlayLayer, allowByLayer []guardAllowOverlay, deny guardDenyOverlay, denyProvenance string) []policyExplainKnob {
	countOf := func(n int, singular, plural string) string {
		return fmt.Sprintf("%d %s", n, pluralWord(n, singular, plural))
	}
	var rows []policyExplainKnob
	switch class {
	case policyClassFrozen:
		rows = append(rows,
			policyExplainKnob{"danger.arg_rules", class, policyChannelsFrozen, countOf(len(m.ArgRules), "rule", "rules"), "embedded"},
			policyExplainKnob{"self_modify.globs", class, policyChannelsFrozen, countOf(len(m.SelfModifyGlobs), "glob", "globs"), "embedded"},
			policyExplainKnob{"egress.metadata_block", class, policyChannelsFrozen, "always-on (cloud metadata + link-local)", "embedded"},
		)
	case policyClassRatchet:
		// Only the v1 name-level deny list is rendered here; the deny overlay's
		// richer tighten-only fields belong to the #5171 registry rows once landed.
		rows = append(rows, policyExplainKnob{"deny.tools", class, policyChannelsDeny, countOf(len(m.Deny), "reason", "reasons"), "embedded"})
		for _, tool := range deny.Deny {
			rows = append(rows, policyExplainKnob{fmt.Sprintf("deny.tools[%s]", tool), class, policyChannelsDeny, "denied", denyProvenance})
		}
	case policyClassGatedWiden:
		rows = append(rows,
			policyExplainKnob{"allow.tools", class, policyChannelsAllow, countOf(len(m.Allow), "tool", "tools"), "embedded"},
			policyExplainKnob{"allow.prefixes", class, policyChannelsAllow, countOf(len(m.AllowPrefix), "prefix", "prefixes"), "embedded"},
		)
		for i, layer := range allowLayers {
			prov := policyExplainAllowProvenance(layer.Name)
			for _, tool := range allowByLayer[i].Allow {
				rows = append(rows, policyExplainKnob{fmt.Sprintf("allow.tools[%s]", tool), class, policyChannelsAllow, "allow", prov})
			}
			for _, prefix := range allowByLayer[i].AllowPrefix {
				rows = append(rows, policyExplainKnob{fmt.Sprintf("allow.prefixes[%s]", prefix), class, policyChannelsAllow, "allow", prov})
			}
		}
	case policyClassSelfAmendable:
		rows = append(rows,
			policyExplainKnob{"report.debug_stats", class, policyChannelsLaunch + " (--debug-stats/--quiet)", "on", "embedded"},
			policyExplainKnob{"report.banner", class, policyChannelsLaunch + " (--banner)", "auto", "embedded"},
		)
	}
	return rows
}

// runGuardPolicyExplain renders the effective floor grouped by amendment class.
// allowLayers/denyPath are injected (production passes guardAllowOverlayPaths() /
// guardDenyOverlayPath()) so tests can point the report at a constructed overlay.
func runGuardPolicyExplain(stdout, stderr io.Writer, allowLayers []guardAllowOverlayLayer, denyPath string) int {
	m, err := policy.ParseManifest(guardDefaultPolicyJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard policy explain: embedded floor: %v\n", err)
		return 2
	}
	allowByLayer := make([]guardAllowOverlay, len(allowLayers))
	for i, layer := range allowLayers {
		ov, err := loadGuardAllowOverlay(layer.Path)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard policy explain: %v\n", err)
			return 2
		}
		allowByLayer[i] = ov
	}
	deny, err := loadGuardDenyOverlay(denyPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard policy explain: %v\n", err)
		return 2
	}
	denyProvenance := "repo-overlay"
	if strings.TrimSpace(os.Getenv(guardDenyOverlayEnv)) != "" {
		denyProvenance = "env-overlay"
	}

	fmt.Fprintln(stdout, "fak guard policy explain — effective guard floor by amendment class")
	fmt.Fprintln(stdout, "floor: built-in guard floor (embedded guard-default-policy.json) + launch-time overlays")
	groups := []struct {
		class, legend string
	}{
		{policyClassFrozen, "never amendable at runtime (release only)"},
		{policyClassRatchet, "tighten-only at runtime"},
		{policyClassGatedWiden, "widens only via an authorized out-of-band operator channel"},
		{policyClassSelfAmendable, "session-local presentation; no verdict depends on them"},
	}
	for _, g := range groups {
		fmt.Fprintf(stdout, "\n== %s — %s ==\n", g.class, g.legend)
		for _, k := range policyExplainKnobs(g.class, m, allowLayers, allowByLayer, deny, denyProvenance) {
			fmt.Fprintf(stdout, "  %-36s  class=%-14s  value=%-40s  provenance=%-12s  channels=%s\n",
				k.Name, k.Class, k.Value, k.Provenance, k.Channels)
		}
	}
	return 0
}

// cmdGuardPolicy dispatches the `fak guard policy <subverb>` family. Only `explain`
// exists today (`policy diff` is sibling A3, #5170); anything else is a usage error.
func cmdGuardPolicy(argv []string) {
	if len(argv) == 1 && argv[0] == "explain" {
		os.Exit(runGuardPolicyExplain(os.Stdout, os.Stderr, guardAllowOverlayPaths(), guardDenyOverlayPath()))
	}
	fmt.Fprintln(os.Stderr, "usage: fak guard policy explain")
	os.Exit(2)
}
