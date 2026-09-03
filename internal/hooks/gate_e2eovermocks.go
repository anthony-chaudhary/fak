package hooks

import (
	"sort"
	"strings"
)

// gate_e2eovermocks.go — the E2E_OVER_MOCKS advisory gate (issue #2901, Hermes-inspiration
// epic #2871). Hermes' rubric: for anything touching a security boundary, resolution chain,
// config propagation, remote backend, or file/network I/O, exercise the REAL path against a
// temp home — "mocks hide integration bugs." fak's answer is the `/verify` skill (drive the
// real flow, not just a green mock). This gate turns that rule into an enforced commit-boundary
// check: a staged diff touching a SECURITY-CRITICAL floor/quarantine/adjudicator package must
// carry a witnessed end-to-end run — the `/verify` output — or attest one, before it can land.
//
// ADVISORY BY DEFAULT (DefaultMode "warn"), exactly like PRIOR_ART: it never reds a shared
// trunk out of the box — it prints which security surface changed and how to satisfy the rule.
// Set FLEET_E2E_GUARD=block to hard-enforce it ("failing the merge otherwise", per the issue),
// or ALLOW_NO_E2E=1 to skip it once. A pre-commit gate sees the STAGED DIFF, not the commit
// message, so the witness is keyed on an added line (the `/verify` artifact staged alongside
// the change, or an "E2E-verified:" trailer in a touched file), the same seam PRIOR_ART uses
// for its "Prior-art:" suppression.

// securityCriticalPrefixes is the initial set of security-boundary and runtime-critical package
// trees the E2E rule guards — the adjudicator (capability decisions), the egress floor (blocks the
// cloud-metadata SSRF class), the deployable capability floor (policy manifest), the write-time
// result quarantine/normalize gate, the gateway (wire protocol and streaming flow), and repoguard
// (tool interception hooks). These are the "adjudicator/floor/quarantine/runtime" surfaces the issue
// names: a change here that ships only against mocks is exactly the failure Hermes warns about.
// The set is intentionally explicit and minimal for the first slice; widen it as more security
// surfaces earn the rule.
var securityCriticalPrefixes = []string{
	"internal/adjudicator/", // the capability adjudicator (the floor decision engine)
	"internal/egressfloor/", // the network-egress floor (cloud-metadata SSRF block)
	"internal/policy/",      // the deployable capability floor (policy manifest)
	"internal/normgate/",    // the write-time result quarantine / normalize gate
	"internal/gateway/",     // the wire gateway proxy, SSE streaming, and route adapters
	"internal/repoguard/",   // tool-use interception and security hooks
}

// e2eWitnessTrailer is the attestation token an author stages to satisfy the rule: it certifies
// that a real end-to-end run (the `/verify` skill, integration test, or dogfood probe) drove the change,
// not a green mock. Case-insensitive; matched anywhere in an added line so staging the `/verify` output
// (which carries this header) or adding an "E2E-verified:" or "Shift-left-verified:" trailer in a
// touched test/doc both silence the gate.
const (
	e2eWitnessTrailer       = "e2e-verified:"
	reasonShiftLeftUnproven = "SHIFT_LEFT_UNPROVEN"
)

// matchSecurityPrefix reports whether p (a repo-relative staged path) lives under one of the
// security-critical package trees, and returns the matched prefix. Backslashes are normalized to
// forward slashes so a Windows-shaped path matches the same as a POSIX one.
func matchSecurityPrefix(p string) (string, bool) {
	p = strings.ReplaceAll(p, "\\", "/")
	for _, prefix := range securityCriticalPrefixes {
		if strings.HasPrefix(p, prefix) {
			return prefix, true
		}
	}
	return "", false
}

// gateE2EOverMocks emits ONE E2E_OVER_MOCKS finding per distinct security-critical package the
// staged diff touches, UNLESS the diff itself adds a line carrying the "E2E-verified:" token —
// that lets an author who staged the `/verify` witness (or attested the run) silence it.
// Findings are deduped by matched prefix (touching three adjudicator files gives one finding)
// and sorted by prefix for determinism, mirroring gatePriorArt's shape.
func gateE2EOverMocks(d *StagedDiff) ([]Finding, error) {
	// matched: prefix -> first touched path under it, deduped by prefix.
	matched := map[string]string{}
	for _, raw := range d.StagedPaths {
		if prefix, ok := matchSecurityPrefix(raw); ok {
			if _, seen := matched[prefix]; !seen {
				matched[prefix] = strings.ReplaceAll(raw, "\\", "/")
			}
		}
	}

	// The denominator is the security-critical surface this commit actually touched, deduped by
	// prefix — the set the gate judges, not the staged set it was handed (#5602).
	//
	// Computed BEFORE the witness check on purpose. Suppressing first and counting later would
	// report zero candidates for an attested commit that touched forty security prefixes, which
	// reads as "no security surface here" — a denominator that misdescribes the run is worse
	// than none, and this gate's whole job is to make that surface visible.
	d.NoteCandidates("E2E_OVER_MOCKS", len(matched), "touched security-surface prefix(es)")

	// Witness: any added line carrying a recognized witness trailer quiets the whole gate — the
	// author has staged/attested the real end-to-end run the rule asks for.
	for _, al := range d.AddedLines() {
		lower := strings.ToLower(al.Text)
		if strings.Contains(lower, e2eWitnessTrailer) || strings.Contains(lower, "shift-left-verified:") {
			return nil, nil
		}
	}

	prefixes := make([]string, 0, len(matched))
	for p := range matched {
		prefixes = append(prefixes, p)
	}
	sort.Strings(prefixes)

	var findings []Finding
	for _, prefix := range prefixes {
		findings = append(findings, Finding{
			Gate:   "E2E_OVER_MOCKS",
			File:   matched[prefix],
			Line:   0,
			Detail: e2eDetail(prefix),
		})
	}
	return findings, nil
}

// e2eDetail renders the one-line advisory for a touched security surface: which floor/quarantine
// tree changed, the Hermes rule, and how to satisfy it (drive the real path, stage the `/verify`
// output, or add an "E2E-verified:" or "Shift-left-verified:" trailer).
func e2eDetail(prefix string) string {
	return `security-critical surface "` + prefix + `" changed without a witnessed end-to-end run — ` +
		`Hermes' rule: "mocks hide integration bugs". Drive the REAL path (the /verify skill, integration test, or dogfood probe) against a ` +
		`temp home and stage its output, or add an "E2E-verified:" or "Shift-left-verified:" trailer citing the run, before landing. ` +
		`(advisory; FLEET_E2E_GUARD=block enforces, ALLOW_NO_E2E=1 skips once)`
}
