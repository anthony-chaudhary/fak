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
	// Scope is the guard_allow_scope.go SCOPE that owns an overlay-supplied entry
	// (#5180): the narrowest layer naming it. Empty for rows that are not overlay
	// entries (the embedded floor, whose value ships with the binary and has no scope).
	Scope string
}

const (
	policyChannelsFrozen = "none — ships with the binary (release only)"
	policyChannelsDeny   = "fak guard deny; POST /v1/fak/policy/reload"
	policyChannelsAllow  = "fak guard allow (operator shell); POST /v1/fak/policy/reload"
	policyChannelsLaunch = "guard launch flags"
)

// policyExplainAllowProvenance maps an allow-overlay layer name (guard_allow.go's
// guardAllowOverlayPaths: "user" / "repo" / "env", plus guard_allow_scope.go's
// "session") to the explain provenance vocab.
func policyExplainAllowProvenance(layerName string) string {
	switch layerName {
	case "user":
		return "user-overlay"
	case "repo":
		return "repo-overlay"
	case "env":
		return "env-overlay"
	case guardAllowScopeSession:
		return "session-overlay"
	}
	return layerName
}

// policyExplainScopeOrder lists the widening scopes broadest-first, matching the
// guard_allow_scope.go precedence table (guardAllowScopeRank). Rendered once under the
// GATED-WIDEN header so an operator reading a `scope=` cell can see where that scope
// sits in the precedence order and whether a widening recorded there survives a relaunch.
var policyExplainScopeOrder = []string{"repo", "user", "env", guardAllowScopeSession}

// policyExplainAllowEntry is one operator-added overlay entry attributed to the scope
// that OWNS it.
type policyExplainAllowEntry struct {
	Name  string
	Scope string
}

// policyExplainResolvedAllowEntries folds the allow layers into ONE row per distinct
// entry, attributed to the RESOLVED scope — the narrowest layer that names it — instead
// of one row per layer that happens to repeat it (#5180). This is the same rule
// guardAllowWinningScope enforces (higher rank wins, later layer wins a tie), so the
// explain report and the overlay's own provenance seam can never disagree about which
// scope owns a widening. First-seen order is preserved so the report is deterministic.
func policyExplainResolvedAllowEntries(layers []guardAllowOverlayLayer, byLayer []guardAllowOverlay, prefix bool) []policyExplainAllowEntry {
	var out []policyExplainAllowEntry
	at := make(map[string]int)
	for i, layer := range layers {
		if i >= len(byLayer) {
			break
		}
		list := byLayer[i].Allow
		if prefix {
			list = byLayer[i].AllowPrefix
		}
		for _, entry := range list {
			j, seen := at[entry]
			if !seen {
				at[entry] = len(out)
				out = append(out, policyExplainAllowEntry{Name: entry, Scope: layer.Name})
				continue
			}
			if guardAllowScopeRank(layer.Name) >= guardAllowScopeRank(out[j].Scope) {
				out[j].Scope = layer.Name
			}
		}
	}
	return out
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
			policyExplainKnob{"danger.arg_rules", class, policyChannelsFrozen, countOf(len(m.ArgRules), "rule", "rules"), "embedded", ""},
			policyExplainKnob{"self_modify.globs", class, policyChannelsFrozen, countOf(len(m.SelfModifyGlobs), "glob", "globs"), "embedded", ""},
			policyExplainKnob{"egress.metadata_block", class, policyChannelsFrozen, "always-on (cloud metadata + link-local)", "embedded", ""},
		)
	case policyClassRatchet:
		// Only the v1 name-level deny list is rendered here; the deny overlay's
		// richer tighten-only fields belong to the #5171 registry rows once landed.
		rows = append(rows, policyExplainKnob{"deny.tools", class, policyChannelsDeny, countOf(len(m.Deny), "reason", "reasons"), "embedded", ""})
		for _, tool := range deny.Deny {
			rows = append(rows, policyExplainKnob{fmt.Sprintf("deny.tools[%s]", tool), class, policyChannelsDeny, "denied", denyProvenance, ""})
		}
	case policyClassGatedWiden:
		rows = append(rows,
			policyExplainKnob{"allow.tools", class, policyChannelsAllow, countOf(len(m.Allow), "tool", "tools"), "embedded", ""},
			policyExplainKnob{"allow.prefixes", class, policyChannelsAllow, countOf(len(m.AllowPrefix), "prefix", "prefixes"), "embedded", ""},
		)
		// One row per distinct entry, carrying the scope that OWNS it — not one row per
		// layer repeating it, which would show the same widening two or three times and
		// leave the operator to guess which layer actually governs.
		for _, e := range policyExplainResolvedAllowEntries(allowLayers, allowByLayer, false) {
			rows = append(rows, policyExplainKnob{fmt.Sprintf("allow.tools[%s]", e.Name), class, policyChannelsAllow, "allow", policyExplainAllowProvenance(e.Scope), e.Scope})
		}
		for _, e := range policyExplainResolvedAllowEntries(allowLayers, allowByLayer, true) {
			rows = append(rows, policyExplainKnob{fmt.Sprintf("allow.prefixes[%s]", e.Name), class, policyChannelsAllow, "allow", policyExplainAllowProvenance(e.Scope), e.Scope})
		}
	case policyClassSelfAmendable:
		rows = append(rows,
			policyExplainKnob{"report.debug_stats", class, policyChannelsLaunch + " (--debug-stats/--quiet)", "on", "embedded", ""},
			policyExplainKnob{"report.banner", class, policyChannelsLaunch + " (--banner)", "auto", "embedded", ""},
		)
	}
	return rows
}

// guardPolicyExplainAllowLayers names the allow-layer set the explain report must
// render: the same one loadGuardAllowOverlayLayers folds into the enforced floor,
// session scope included. The CLI adapter (runGuardPolicyExplainVerb, guard_policy.go)
// spells that expression out at its own call site, so this exists to give the tests ONE
// name for it — a witness that re-derived the layer set by hand could agree with itself
// while both drifted from the floor, which is exactly the session-scope blind spot
// described below.
func guardPolicyExplainAllowLayers() []guardAllowOverlayLayer {
	return guardAllowLayersWithSessionScope(guardAllowOverlayPaths())
}

// runGuardPolicyExplain renders the effective floor grouped by amendment class.
// allowLayers/denyPath are injected so tests can point the report at a constructed
// overlay; production passes the session-scoped layer set (the value
// guardPolicyExplainAllowLayers names) and guardDenyOverlayPath().
//
// The allowLayers set is load-bearing, not cosmetic: explain claims to show the floor the
// guard enforces, so it must be handed the SAME layer set the floor loader reads
// (loadGuardAllowOverlayLayers). Passing the bare guardAllowOverlayPaths() would silently
// omit the session scope and render a session-scoped widening as absent — the report
// telling the operator the floor is narrower than it is. Hence the named helper above.
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
		// The widening scopes are only meaningful for the gated-widen channel, so the
		// precedence legend prints once here rather than trailing every row: it answers
		// "where does this scope sit, and does a widening recorded there survive the
		// next launch" in one place, and the per-row scope= cell then reads as a key
		// into it (#5180).
		if g.class == policyClassGatedWiden {
			fmt.Fprintln(stdout, "  scope precedence — broadest first; the NARROWEST scope naming an entry owns it:")
			for _, s := range policyExplainScopeOrder {
				fmt.Fprintf(stdout, "    scope=%-8s%s\n", s, guardAllowScopeDurabilityNote(s))
			}
		}
		for _, k := range policyExplainKnobs(g.class, m, allowLayers, allowByLayer, deny, denyProvenance) {
			scope := k.Scope
			if scope == "" {
				scope = "-" // not an overlay entry; the embedded floor has no widening scope
			}
			fmt.Fprintf(stdout, "  %-36s  class=%-14s  value=%-40s  provenance=%-12s  scope=%-8s  channels=%s\n",
				k.Name, k.Class, k.Value, k.Provenance, scope, k.Channels)
		}
	}
	return 0
}
