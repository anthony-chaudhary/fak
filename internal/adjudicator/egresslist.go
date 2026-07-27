// egresslist.go — the OPERATOR-CONFIGURABLE egress band, sitting between the hardwired
// cloud-metadata floor and the research WebFetch allowlist.
//
// The floor above it (egressfloor.Classify, decide.go's rungEgress) is HARDWIRED and
// mandatory: the metadata/link-local class is never a legitimate destination and no policy
// field can re-open it. This layer is the part a deployment gets to configure — operator
// block_hosts, operator allow_hosts (exceptions), subscribed community block lists, and
// the strict-allowlist `restrict` posture. Because it runs AFTER the floor, an allow rule
// naming the metadata host cannot un-block the SSRF class: the floor has already returned.
//
// Rule compilation and adblock precedence live in internal/egresslist; this file is only
// the wiring — policy fields in, one Decision per destination host out.
package adjudicator

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// compileEgressList compiles a policy's egress rules ONCE, at policy install time (New /
// SetPolicy), so the decide path pays only map probes per call and never re-parses a
// community list. Returns nil when the policy configures no rules at all, which is the
// common case: a nil *egresslist.List Decides None for every host, so an unconfigured
// policy costs nothing and the layer is provably inert.
//
// An unknown bundled-list name is skipped here rather than reported: the policy loader
// already rejects it LOUDLY at load (internal/policy validates names against
// egresslist.BundledListNames before ever building a Policy), so by the time a Policy
// reaches this function the names are known-good. Skipping keeps compilation total —
// a name that somehow slipped through cannot panic the reference monitor.
func compileEgressList(p Policy) *egresslist.List {
	if len(p.EgressBlockHosts) == 0 && len(p.EgressAllowHosts) == 0 && len(p.EgressBlockLists) == 0 {
		return nil
	}
	b := egresslist.NewBuilder()
	if len(p.EgressBlockHosts) > 0 {
		b.AddRules("policy.block_hosts", p.EgressBlockHosts, egresslist.Block)
	}
	if len(p.EgressAllowHosts) > 0 {
		b.AddRules("policy.allow_hosts", p.EgressAllowHosts, egresslist.Allow)
	}
	for _, name := range p.EgressBlockLists {
		if text, ok := egresslist.BundledList(name); ok {
			b.AddFilterText(name, text)
		}
	}
	l := b.Build()
	if l.Empty() {
		return nil
	}
	return l
}

// egressListVerdict resolves one tool call against the compiled list and the restrict
// posture. It reports ok=false to mean "this layer is silent, fall through to the next
// egress rung" — never an implicit allow.
//
// Destinations come from egressfloor.Destinations, deliberately NOT re-derived here: that
// helper exists so this layer and the hardwired floor agree on what "the destination" of a
// call is, covering both a WebFetch url and a host buried in a Bash command line.
//
// Precedence, in order:
//
//   - BLOCK on any destination wins outright. Within a single host, allow beats block
//     (adblock `@@` semantics, resolved inside List.Decide). ACROSS hosts we take the
//     fail-safe reading instead: a call reaching one blocked host is refused even if it
//     also reaches a sanctioned one, so a beacon cannot ride along with a legitimate
//     fetch. That makes the outcome independent of destination order.
//   - ALLOW is a positive admit for WebFetch ONLY, mirroring researchEgressVerdict. This
//     scoping is load-bearing, not cosmetic: returning Allow short-circuits the remaining
//     rungs, so admitting any tool on the strength of a hostname would let
//     `curl https://docs.example.com && rm -rf /` past the write/self-modify rungs on the
//     strength of its URL. For every other tool an allow rule means only "not blocked" and
//     the call continues down the normal chain.
//   - RESTRICT closes the tail: with EgressRestrict set, a destination no allow rule names
//     is POLICY_BLOCK even though no block rule matched it. This inverts the default
//     posture from "reachable unless listed" to "unreachable unless listed", and it applies
//     to every tool with a destination — restrict only ever tightens, so widening its
//     reach cannot open anything.
func egressListVerdict(l *egresslist.List, p Policy, tool string, args map[string]any) (abi.Verdict, bool) {
	if l == nil && !p.EgressRestrict {
		return abi.Verdict{}, false // layer not configured: provably inert
	}
	dests := egressfloor.Destinations(tool, args)
	if len(dests) == 0 {
		return abi.Verdict{}, false // no network destination: nothing for this layer to say
	}
	sanctioned := ""
	for _, host := range dests {
		switch d := l.Decide(host); d.Kind {
		case egresslist.Block:
			// Bounded disclosure: the witness names the host and WHICH list spoke, never
			// the full URL or the rest of the policy.
			return abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  egressfloor.ReasonEgressBlock,
				By:      "monitor/egress-list",
				Payload: abi.WitnessPayload{Claim: "egress blocked by " + d.Source + ": " + host},
				Meta: map[string]string{
					"egress_list": d.Source,
					"egress_rule": d.Rule,
					"host":        host,
				},
			}, true
		case egresslist.Allow:
			if sanctioned == "" {
				sanctioned = host
			}
		}
	}
	if sanctioned != "" && strings.EqualFold(tool, "WebFetch") {
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor/egress-list",
			Meta: map[string]string{
				"egress_list": "allowlisted",
				"host":        sanctioned,
			},
		}, true
	}
	if p.EgressRestrict && sanctioned == "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonPolicyBlock,
			By:      "monitor/egress-list",
			Payload: abi.WitnessPayload{Claim: "egress restricted, host not allowlisted: " + dests[0]},
			Meta: map[string]string{
				"egress_posture": "restrict",
				"host":           dests[0],
			},
		}, true
	}
	return abi.Verdict{}, false
}
