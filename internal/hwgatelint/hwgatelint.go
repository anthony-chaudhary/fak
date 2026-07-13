// Package hwgatelint is the sensor for the "local machine is the compute
// boundary" regression — the hardware-gate anti-pattern.
//
// An agent working a device/GPU/heavy-compute task on the control host (a
// laptop with no CUDA device, no Node.js, not enough RAM) keeps mis-scoping the
// work and stopping: "not yet — no GPU on this host", "can't run the device
// witness without CUDA", "this laptop can't test that". Every one of those
// treats the local machine as the boundary when it is only the CONTROL POINT.
// The fleet's whole purpose is to dispatch that work to the machine that CAN
// witness it: a sanctioned compute node (GCP `fak-realmodel`, the DGX, `da33`,
// the nightrun pipeline).
//
// This package is the dual of internal/headlesslint. headlesslint catches a
// turn that ended by asking a HUMAN; hwgatelint catches a turn that ended by
// declaring a LOCAL-HARDWARE blocker as terminal. The doctrine is the same
// shape — an autonomous worker must ACT, not stop — but the remediation is
// fixed: redirect to the sanctioned compute node (or, if the credential/bridge
// session is missing, produce the exact ready-to-run command sequence and hand
// it to the operator; that still counts as using the lab). The canonical route
// is the SSOT in the local-machine-not-a-constraint fleet memory and
// docs/sanctioned-compute-nodes.md.
//
// Detection is co-occurrence on a line: a HARDWARE token (names GPU/CUDA/device,
// a missing runtime like Node.js, or "this machine/laptop/host" as the limit)
// AND a BLOCKER framing (can't / no / unable / requires / without / not
// available) on the SAME line. A whole-text SUPPRESSION clears the turn when it
// already names the sanctioned route — an agent that said "no local GPU, so I
// dispatched to the DGX" did the right thing and must never be flagged. Favoring
// precision (no false positives) suits a soak-first rollout.
//
// Pure and stdlib-only: text in, typed findings out, no I/O and no clock. Any
// layer — a Stop hook, a loop gate, an operator brief — can fold an agent's
// final turn through the same taxonomy.
package hwgatelint

import (
	"regexp"
	"strings"
)

// Schema is the versioned envelope tag for a Report.
const Schema = "fak-hwgate-lint/1"

// Verdict is the closed top-level judgment over a scanned text.
const (
	// Clean: no hardware-gate note found (or the turn already named the
	// sanctioned route) — the output is fleet-safe.
	Clean = "clean"
	// HardwareGated: at least one note declares a local-hardware blocker as a
	// terminal state instead of dispatching to a sanctioned compute node.
	HardwareGated = "hardware_gated"
)

// Redirect is the fixed remediation for every hardware-gate finding: the
// sanctioned compute-node route. It is the operative one-paragraph form of the
// SSOT (local-machine-not-a-constraint memory / docs/fleet-compute-nodes.md).
// Node identities stay generic here — the private hostnames live only in the
// operator's local fleet memory, never in the repo.
const Redirect = "Do not stop for lack of LOCAL hardware — this host is the CONTROL POINT, not the compute boundary. Dispatch the work to a sanctioned compute node: GCP `fak-realmodel` (L4 sm_89, or a spot A100/H100) for a live CUDA/GPU serve or device witness; the on-prem DGX (8× A100 sm_80) via `dgxbridge` for a device-GEMM/kernel witness; `da33` for DC-network or CPU-bench work; or the nightrun pipeline for a nightly device-witness ledger. If the credential/bridge session for the right node is missing, produce the exact ready-to-run command sequence and hand it to the operator — that still counts as using the lab, never \"blocked by local hardware.\""

// Class is the closed vocabulary of hardware-gate anti-patterns. Each member
// names WHICH local resource the turn declared missing; the remediation is the
// same fixed Redirect for all of them.
type Class string

const (
	// NoLocalGPU declares a GPU/CUDA/accelerator device missing as the blocker:
	// "no GPU on this host", "can't run without CUDA", "requires an accelerator".
	NoLocalGPU Class = "NO_LOCAL_GPU"

	// NoLocalRuntime declares a non-Go runtime missing as the blocker — most
	// often Node.js, which the fleet has no in-repo project for: "Node.js isn't
	// installed", "no npm here".
	NoLocalRuntime Class = "NO_LOCAL_RUNTIME"

	// LocalBoundary declares THIS machine the limit without naming a specific
	// device: "can't test that on this laptop", "not reproducible on this box",
	// "no way to benchmark locally".
	LocalBoundary Class = "LOCAL_BOUNDARY"
)

// Finding is one detected hardware-gate note.
type Finding struct {
	Class   Class  `json:"class"`
	Line    int    `json:"line"`
	Match   string `json:"match"`
	Excerpt string `json:"excerpt"`
	Reason  string `json:"reason"`
	Node    string `json:"node"` // the sanctioned node this class most naturally routes to
}

// Report is the fold over one scanned text.
type Report struct {
	Schema     string        `json:"schema"`
	Verdict    string        `json:"verdict"`
	Count      int           `json:"count"`
	Suppressed bool          `json:"suppressed,omitempty"` // a sanctioned-route mention cleared an otherwise-gated turn
	Redirect   string        `json:"redirect,omitempty"`
	Classes    map[Class]int `json:"classes"`
	Findings   []Finding     `json:"findings"`
}

func re(p string) *regexp.Regexp { return regexp.MustCompile(`(?i)` + p) }

// blockerRE is the shared "this is terminal, I'm stuck" framing. A hardware
// token only becomes a finding when one of these co-occurs on the same line —
// so "the GPU node ran the witness" (no blocker) never trips, but "can't, no
// GPU here" does.
var blockerRE = []*regexp.Regexp{
	re(`\bcan'?t\b`),
	re(`\bcannot\b`),
	re(`\bcan not\b`),
	re(`\bcould ?n'?t\b`),
	re(`\bunable to\b`),
	re(`\bnot able to\b`),
	re(`\bno way to\b`),
	re(`\bthere'?s no way\b`),
	re(`\bblocked?\b`),
	re(`\bblocker\b`),
	re(`\bnot possible\b`),
	re(`\bimpossible\b`),
	re(`\bnot enough\b`),
	re(`\binsufficient\b`),
	re(`\bwon'?t (be able to|run|work)\b`),
	re(`\brequires?\b`),
	re(`\brequired\b`),
	re(`\bneeds?\b`),
	re(`\bneeded\b`),
	re(`\bwithout\b`),
	re(`\bnot (available|installed|present|supported)\b`),
	re(`\bunavailable\b`),
	re(`\bisn'?t (available|installed|present|here)\b`),
	re(`\bnot yet\b`),
	re(`\bno\b`), // "no GPU", "no CUDA device", "no local accelerator"
}

// sanctionedRouteRE recognises a turn that ALREADY redirected to the lab. Its
// presence anywhere in the text suppresses the whole scan: the agent named the
// right node (or the act of dispatching/handing the operator a command), so it
// did not stop at the local boundary. Favors precision over recall.
var sanctionedRouteRE = []*regexp.Regexp{
	re(`\bfak-realmodel\b`),
	re(`\bdgxbridge\b`),
	re(`\b(the )?dgx\b`),
	re(`\bda33\b`),
	re(`\bnightrun\b`),
	re(`\bgcp\b`),
	re(`\bgcloud\b`),
	re(`\bspot (a100|h100|instance)\b`),
	re(`\bdispatch(ed|ing)? (it |the |this )?(to|onto) (the |a )?(fleet|lab|remote|gpu|dgx|node|gcp)\b`),
	re(`\brun(ning)? (it |the witness |this )?on (the |a )?(fleet|lab|remote|gpu|dgx|da33|gcp|node)\b`),
	re(`\bhand(ed|ing)? (it |the operator |off )?(to )?the operator\b`),
	re(`\bready-to-run\b`),
	re(`\bremote (node|gpu|host)\b`),
	re(`\bsanctioned (compute )?node\b`),
}

// classSpec binds a Class to its hardware-token patterns and its natural
// sanctioned node. Specs are evaluated in slice order; the first whose pattern
// matches a (blocker-bearing) line wins that line, so the specific device
// classes precede the generic LocalBoundary catch-all.
type classSpec struct {
	class  Class
	node   string
	reason string
	res    []*regexp.Regexp
}

var specs = []classSpec{
	{
		class:  NoLocalGPU,
		node:   "GCP `fak-realmodel` (L4 sm_89) or a spot A100/H100; the DGX via `dgxbridge` for device-GEMM",
		reason: "declares a GPU/CUDA/accelerator device missing as a terminal blocker; the laptop is the control point, not the compute boundary — dispatch the device work to a GPU node",
		res: []*regexp.Regexp{
			re(`\bgpus?\b`),
			re(`\bcuda\b`),
			re(`\bnvidia\b`),
			re(`\baccelerators?\b`),
			re(`\bvram\b`),
			re(`\bsm_\d{2}\b`),
			re(`\bdevice[- ]gemm\b`),
			re(`\bcompute capability\b`),
		},
	},
	{
		class:  NoLocalRuntime,
		node:   "a fleet node that has the runtime (dispatch or hand the operator the command)",
		reason: "declares a non-Go runtime (Node.js/npm) missing as a terminal blocker; run it on a fleet node that has the runtime rather than treating the local host as the limit",
		res: []*regexp.Regexp{
			re(`\bnode\.?js\b`),
			re(`\bnpm\b`),
			re(`\bnode runtime\b`),
			re(`\byarn\b`),
			re(`\bpnpm\b`),
		},
	},
	{
		class:  LocalBoundary,
		node:   "the sanctioned compute node for the task (GCP/DGX/da33/nightrun)",
		reason: "treats THIS machine as the compute boundary; the fleet's point is to dispatch the work to the node that can run it — never report \"blocked for lack of local hardware\"",
		res: []*regexp.Regexp{
			re(`\bon this (windows )?(machine|laptop|host|box|pc)\b`),
			re(`\bthis (windows )?(machine|laptop|host|box) (can'?t|cannot|lacks|is not|isn'?t)\b`),
			re(`\blocal(ly)? (test|run|verify|benchmark|reproduce|witness|serve)\b`),
			re(`\bheavy compute\b`),
			re(`\benough (ram|memory|compute)\b`),
			re(`\b\d+\s?(gb|tb) (of )?(ram|vram|memory)\b`),
		},
	},
}

// lineHasBlocker reports whether the terminal-blocker framing is present.
func lineHasBlocker(low string) bool {
	for _, r := range blockerRE {
		if r.MatchString(low) {
			return true
		}
	}
	return false
}

// hasSanctionedRoute reports whether the text already names the lab route.
func hasSanctionedRoute(low string) bool {
	for _, r := range sanctionedRouteRE {
		if r.MatchString(low) {
			return true
		}
	}
	return false
}

// Scan folds a text into a Report. A whole-text sanctioned-route mention
// suppresses everything (the agent already redirected). Otherwise each line
// bearing both a blocker framing and a hardware token yields one Finding (the
// first matching Class wins), all sharing the fixed Redirect.
func Scan(text string) Report {
	rep := Report{Schema: Schema, Verdict: Clean, Classes: map[Class]int{}}
	low := strings.ToLower(text)
	if hasSanctionedRoute(low) {
		rep.Suppressed = true
		return rep
	}
	for i, raw := range splitLines(text) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		lowLine := strings.ToLower(line)
		if !lineHasBlocker(lowLine) {
			continue
		}
		for _, sp := range specs {
			m := firstMatch(sp.res, lowLine, line)
			if m == "" {
				continue
			}
			rep.Findings = append(rep.Findings, Finding{
				Class:   sp.class,
				Line:    i + 1,
				Match:   clip(m, 120),
				Excerpt: clip(line, 200),
				Reason:  sp.reason,
				Node:    sp.node,
			})
			rep.Classes[sp.class]++
			break
		}
	}
	rep.Count = len(rep.Findings)
	if rep.Count > 0 {
		rep.Verdict = HardwareGated
		rep.Redirect = Redirect
	}
	return rep
}

// firstMatch returns the original-case substring of the first pattern that
// matches the lowercased line, or "" if none match.
func firstMatch(res []*regexp.Regexp, low, orig string) string {
	for _, r := range res {
		if loc := r.FindStringIndex(low); loc != nil {
			return strings.TrimSpace(orig[loc[0]:loc[1]])
		}
	}
	return ""
}

func splitLines(text string) []string {
	return strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
}

func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
