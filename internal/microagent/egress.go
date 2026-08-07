// Per-agent network egress control (#2017, epic #2000 M17): the missing half of
// sandbox isolation. The ToolExec seam (#2018) already proves a tool CALL was
// adjudicated before it ran, but a compute-isolated agent with open egress can
// still post a secret to an attacker host or be steered by content it fetched.
// This file adds the network dimension to the same seam, on the same terms: a
// property of the SEAM, not a convention each backend re-implements — so
// whatever the isolation level (goroutine, subprocess, container, microVM), the
// egress floor is identical and a new backend cannot forget it.
//
// Three rules, all fail-closed, each able only to ADD a denial:
//
//  1. DEFAULT-DENY. A governed action's declared destination must match the
//     allowlist. There are no implicit exemptions — not loopback, not
//     link-local — because the classic exfiltration and SSRF targets (a cloud
//     instance-metadata endpoint, a sidecar on 127.0.0.1) are exactly the hosts
//     an "obviously local, obviously fine" carve-out would hand over.
//  2. UNTRUSTED TAKES EXACT HOSTS ONLY. At the untrusted pole a suffix entry
//     (".example.com") is refused at CONSTRUCTION: a subdomain wildcard is a
//     published exfil channel — the attacker picks `secret-in-the-name.example.com`
//     and the allowlist waves it through. Trusted levels may still use suffixes.
//  3. RESIDENCY WINS OVER THE ALLOWLIST. The destination is re-adjudicated
//     through the SAME kernel floor as an engine route, so the residency /
//     data-leak gate already in the tree (internal/engine residencyGate, rank 12)
//     decides it. A tenant-scoped or sensitivity-tagged payload is therefore
//     denied even to an ALLOWLISTED host: the operator allowlisted a
//     destination, not a data-residency exception.
//
// Generation intent: gen/future (#2017 is labelled gen/future) — this is an
// OPTION on the sandbox-network surface, not a current-product commitment.
// Nothing in the default serve/guard/dispatch path constructs an EgressPolicy;
// it is reachable only through ToolExec.WithEgress, and an executor without one
// behaves exactly as it did before this file existed. Closing evidence for the
// generation frame:
//
//   - Promotion evidence: egress_test.go witnesses the acceptance of #2017 —
//     an untrusted agent is DENIED a non-allowlisted host, an allowlisted host
//     PASSES, and the denial is recorded with its destination. Promote to a
//     default-on floor once the microagent Host drives agent loops whose actions
//     actually open sockets (the #2001 RunArm extraction), at which point rule 1
//     should move from "the action DECLARED a destination" to an enforced
//     stripped network namespace / proxy the backend cannot route around.
//   - Demotion / retirement criteria: retire this decision layer if the exec
//     backends grow real per-action network namespaces (container/microVM tiers
//     with an enforced deny-by-default netns), because the kernel then enforces
//     what this layer can only decide — keep the allowlist as the netns's
//     configuration input and delete the seam check.
//   - INVALIDATING ASSUMPTION (the load-bearing one): this layer gates the
//     destination an action DECLARES (ToolAction.Dest). It is an honest floor
//     only for a backend that cannot reach the network by any other path — a
//     subprocess on a host with open networking can simply dial a socket and
//     never declare it. If measurement shows real actions reaching hosts they
//     did not declare, this assumption is invalid and the decision must move
//     below the process boundary (netns/firewall/proxy), where declaration is
//     not optional. The subprocess tier is NOT that boundary today; the
//     container and microVM tiers are where rule 1 becomes enforcement.
package microagent

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// EgressTrustUntrusted is the reserved fail-closed trust pole. It mirrors
// internal/policy.TrustUntrusted (the isolation dial's reserved level) by VALUE
// rather than by import: policy is the manifest layer and this leaf decides on a
// level string the caller already resolved. An empty level is treated as this
// pole too — unknown provenance is untrusted provenance.
const EgressTrustUntrusted = "untrusted"

// egressBy names this adjudication site in a verdict's By field.
const egressBy = "microagent-egress"

// Structured refusals for the egress floor.
var (
	ErrEgressDenied      = errors.New("microagent: egress policy refused the action's destination")
	ErrEgressNoAllowlist = errors.New("microagent: NewEgressPolicy requires at least one allowlist entry (an empty allowlist reaches nothing; pass one explicitly to say so)")
	ErrEgressWildcard    = errors.New("microagent: egress allowlist entry \"*\" is not a legal entry (no open-egress escape hatch)")
	ErrEgressSuffix      = errors.New("microagent: an untrusted egress allowlist takes EXACT hosts only (a subdomain suffix is an exfil channel)")
)

// EgressPolicy is one agent's network-egress floor: the trust level it runs at
// plus the destinations it may reach. Build it with NewEgressPolicy — the
// validation lives there, so a policy value that exists has already been proven
// well-formed, and attach it with ToolExec.WithEgress.
//
// A nil *EgressPolicy is the UNGOVERNED legacy posture (Decide defers), which is
// what every ToolExec built before this file had. That is deliberate: an
// executor is governed because a caller attached a policy, never by accident.
type EgressPolicy struct {
	// TrustLevel is the isolation-dial trust level this agent runs at.
	// EgressTrustUntrusted (or empty) selects the fail-closed pole.
	trustLevel string
	// exact holds lowercase exact-match hosts; suffix holds ".example.com"
	// entries (never populated at the untrusted pole — refused at construction).
	exact  map[string]bool
	suffix []string
	// Audit receives every non-allow egress verdict before Run returns, so a
	// denial is RECORDED and not merely returned. Nil is legal: the verdict on
	// ToolResult.Verdict is still the record, carrying destination and reason.
	Audit func(abi.Verdict)
}

// NewEgressPolicy compiles and validates one agent's allowlist. Everything that
// can be refused is refused HERE, at construction, rather than at decision time:
// a malformed allowlist is an operator error that should fail loud when the
// policy is built, never a surprise deny (or worse, a surprise allow) on the
// hot path.
//
// Entries are hosts, not URLs: "api.example.com" (exact) or ".example.com"
// (that domain and any subdomain). A scheme, a path, a port, an empty entry, or
// the wildcard "*" is refused; at the untrusted pole a suffix entry is refused
// too (rule 2 — see the file header).
func NewEgressPolicy(trustLevel string, allow ...string) (*EgressPolicy, error) {
	lv := strings.ToLower(strings.TrimSpace(trustLevel))
	if lv == "" {
		lv = EgressTrustUntrusted
	}
	if len(allow) == 0 {
		return nil, ErrEgressNoAllowlist
	}
	p := &EgressPolicy{trustLevel: lv, exact: map[string]bool{}}
	for i, raw := range allow {
		e := strings.ToLower(strings.TrimSpace(raw))
		switch {
		case e == "":
			return nil, fmt.Errorf("microagent: egress allowlist[%d]: entry is empty", i)
		case e == "*":
			return nil, fmt.Errorf("%w (allowlist[%d])", ErrEgressWildcard, i)
		case strings.Contains(e, "://"), strings.Contains(e, "/"):
			return nil, fmt.Errorf("microagent: egress allowlist[%d]: %q is a URL, not a host (drop the scheme and path)", i, raw)
		case strings.Contains(e, ":"):
			// A port would imply the policy gates ports; it does not — the
			// destination HOST is the unit. Refuse rather than silently ignore.
			return nil, fmt.Errorf("microagent: egress allowlist[%d]: %q carries a port (the allowlist is host-only)", i, raw)
		case strings.HasPrefix(e, "."):
			if len(e) == 1 {
				return nil, fmt.Errorf("microagent: egress allowlist[%d]: %q is not a domain suffix", i, raw)
			}
			if lv == EgressTrustUntrusted {
				return nil, fmt.Errorf("%w: allowlist[%d] = %q (name the exact hosts instead)", ErrEgressSuffix, i, raw)
			}
			p.suffix = append(p.suffix, e)
		default:
			p.exact[strings.TrimSuffix(e, ".")] = true
		}
	}
	return p, nil
}

// TrustLevel reports the resolved trust level (empty normalizes to the
// untrusted pole), so a caller and an audit row can confirm which posture
// actually decided rather than trusting the level they passed in.
func (p *EgressPolicy) TrustLevel() string {
	if p == nil {
		return ""
	}
	return p.trustLevel
}

// Decide adjudicates ONE destination for this agent and returns the verdict the
// seam acts on:
//
//   - VerdictDefer  — nothing to decide (no policy attached, or no declared
//     destination). The seam proceeds; this is the ungoverned legacy path.
//   - VerdictAllow  — the destination matched the allowlist AND the residency
//     floor did not object to routing this payload there.
//   - anything else — an egress REFUSAL. The seam must not dispatch.
//
// The floor argument is the same KernelFloor the seam adjudicates calls
// through. Decide re-submits the ORIGINAL call with its Engine route set to the
// destination host, which is how the residency / data-leak gate already in the
// tree (internal/engine residencyGate) gets to see an egress destination at all:
// that gate denies a sensitivity-tagged payload bound for a route it does not
// recognize as on-box, and an arbitrary hostname is exactly that. Composing this
// way — rather than re-deriving "is this payload sensitive" here — means the
// egress floor inherits the residency rules instead of drifting from them, and
// any future route-aware adjudicator governs egress the day it is registered.
//
// The re-adjudication is consulted BEFORE the allowlist and is decisive: an
// allowlisted host does not buy a residency exception. A nil floor skips the
// re-adjudication (the allowlist still applies) — the decision is then weaker,
// never wider.
func (p *EgressPolicy) Decide(ctx context.Context, dest string, call *abi.ToolCall, floor KernelFloor) abi.Verdict {
	if p == nil || strings.TrimSpace(dest) == "" {
		return abi.Verdict{Kind: abi.VerdictDefer, By: egressBy}
	}
	host, err := EgressHost(dest)
	if err != nil {
		// An undecodable destination is a refusal, not a pass: the floor cannot
		// allow what it cannot name.
		return p.record(abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonMalformed,
			By:     egressBy,
			Meta:   p.meta(dest, "", "undecodable destination"),
		})
	}

	// Rule 3 — residency first, and it wins. Re-adjudicate the SAME call as if
	// routed to this destination so the registered residency floor decides it.
	if floor != nil && call != nil {
		routed := *call
		routed.Engine = host
		if v := floor.Decide(ctx, &routed); v.Kind != abi.VerdictAllow && v.Kind != abi.VerdictDefer {
			// Report the residency floor's own verdict — its By/Reason are the
			// forensics — with the egress destination attached.
			out := v
			out.Meta = mergeMeta(v.Meta, p.meta(dest, host, "residency floor refused the destination"))
			return p.record(out)
		}
	}

	// Rules 1 and 2 — the allowlist is the only opt-in.
	if p.allows(host) {
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   egressBy,
			Meta: p.meta(dest, host, "allowlisted"),
		}
	}
	return p.record(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonDefaultDeny,
		By:     egressBy,
		Meta:   p.meta(dest, host, "host is not on the egress allowlist"),
	})
}

// allows reports whether host matches the compiled allowlist. Exact entries
// match exactly; a ".example.com" suffix entry matches that domain and any
// subdomain of it (and is unreachable at the untrusted pole, where construction
// refused suffixes).
func (p *EgressPolicy) allows(host string) bool {
	if p.exact[host] {
		return true
	}
	for _, s := range p.suffix {
		if host == strings.TrimPrefix(s, ".") || strings.HasSuffix(host, s) {
			return true
		}
	}
	return false
}

// record hands a non-allow verdict to the Audit sink (if any) and returns it, so
// every refusal this floor issues is written down exactly once, at the single
// point every refusal passes through.
func (p *EgressPolicy) record(v abi.Verdict) abi.Verdict {
	if p != nil && p.Audit != nil {
		p.Audit(v)
	}
	return v
}

// meta builds the forensic row for an egress decision: what was asked for, what
// it resolved to, which posture decided, and why.
func (p *EgressPolicy) meta(dest, host, rule string) map[string]string {
	m := map[string]string{
		"egress_dest": dest,
		"trust_level": p.trustLevel,
		"egress_rule": rule,
	}
	if host != "" {
		m["egress_host"] = host
	}
	return m
}

// mergeMeta overlays add onto base without mutating either (base wins on a key
// collision: another adjudicator's own forensics are not overwritten by ours).
func mergeMeta(base, add map[string]string) map[string]string {
	out := make(map[string]string, len(base)+len(add))
	for k, v := range add {
		out[k] = v
	}
	for k, v := range base {
		out[k] = v
	}
	return out
}

// EgressHost normalizes a declared destination to the bare host the allowlist is
// matched against. It accepts the three shapes an action realistically carries —
// a URL ("https://api.example.com/v1/chat"), a host:port ("api.example.com:443"),
// and a bare host ("api.example.com") — and lowercases, strips the trailing root
// dot, and unwraps an IPv6 literal's brackets so "API.Example.com." and
// "api.example.com" cannot resolve to two different allowlist decisions.
func EgressHost(dest string) (string, error) {
	s := strings.TrimSpace(dest)
	if s == "" {
		return "", errors.New("microagent: empty egress destination")
	}
	if strings.Contains(s, "://") {
		u, err := url.Parse(s)
		if err != nil {
			return "", fmt.Errorf("microagent: parse egress destination %q: %w", dest, err)
		}
		if u.Host == "" {
			return "", fmt.Errorf("microagent: egress destination %q has no host", dest)
		}
		s = u.Host
	}
	if h, _, err := net.SplitHostPort(s); err == nil {
		s = h
	}
	s = strings.Trim(strings.TrimSpace(s), "[]")
	s = strings.ToLower(strings.TrimSuffix(s, "."))
	if s == "" {
		return "", fmt.Errorf("microagent: egress destination %q has no host", dest)
	}
	if strings.ContainsAny(s, "/ \t") {
		return "", fmt.Errorf("microagent: egress destination %q is not a host", dest)
	}
	return s, nil
}
