package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// TestEgressListBlockHostAndSubdomain proves an operator block_hosts entry refuses the
// host AND its subdomains with EGRESS_BLOCK while an unrelated host is unaffected.
func TestEgressListBlockHostAndSubdomain(t *testing.T) {
	a := New(Policy{
		Allow:            map[string]bool{"WebFetch": true},
		EgressBlockHosts: []string{"tracker.example"},
	})
	for _, host := range []string{"tracker.example", "cdn.tracker.example"} {
		v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://`+host+`/x"}`))
		if v.Kind != abi.VerdictDeny || v.Reason != egressfloor.ReasonEgressBlock {
			t.Fatalf("WebFetch %s = %v/%s, want Deny/EGRESS_BLOCK", host, v.Kind, abi.ReasonName(v.Reason))
		}
	}
	// A host not on any rule falls through to the (allowed) posture, not blocked.
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://safe.example/x"}`))
	if v.Kind == abi.VerdictDeny && v.Reason == egressfloor.ReasonEgressBlock {
		t.Fatalf("WebFetch safe.example was EGRESS_BLOCKED, want not blocked")
	}
}

// TestEgressListAllowWinsOverBlock proves an allow (exception) host carves back open a
// host a block rule would otherwise refuse — the adblock '@@' precedence end to end — and
// that for WebFetch it is a positive admit even when nothing else allows the tool.
func TestEgressListAllowWinsOverBlock(t *testing.T) {
	a := New(Policy{
		EgressBlockHosts: []string{"example.com"},
		EgressAllowHosts: []string{"docs.example.com"},
	})
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://docs.example.com/guide"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("WebFetch docs.example.com = %v/%s, want Allow (exception carves it open)", v.Kind, abi.ReasonName(v.Reason))
	}
	// A sibling under the block rule with no allow carve-out stays blocked.
	v = a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://ads.example.com/x"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != egressfloor.ReasonEgressBlock {
		t.Fatalf("WebFetch ads.example.com = %v/%s, want Deny/EGRESS_BLOCK (still blocked)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressListBundledList proves a policy subscribing to a bundled community list
// refuses a host that list carries — the full name->BundledList->compile->decide path.
func TestEgressListBundledList(t *testing.T) {
	a := New(Policy{
		Allow:            map[string]bool{"WebFetch": true},
		EgressBlockLists: []string{"sample-malware"},
	})
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://malware.example/payload"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != egressfloor.ReasonEgressBlock {
		t.Fatalf("WebFetch malware.example = %v/%s, want Deny/EGRESS_BLOCK (on sample-malware list)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressListDoesNotUnblockMetadataFloor is the load-bearing invariant: the hardwired
// cloud-metadata floor runs FIRST, so even an explicit allow of the metadata host cannot
// un-block the SSRF class.
func TestEgressListDoesNotUnblockMetadataFloor(t *testing.T) {
	a := New(Policy{
		Allow:            map[string]bool{"WebFetch": true},
		EgressAllowHosts: []string{"169.254.169.254"},
	})
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"http://169.254.169.254/latest/meta-data/"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != egressfloor.ReasonEgressBlock {
		t.Fatalf("WebFetch metadata host = %v/%s, want Deny/EGRESS_BLOCK (floor wins over allow rule)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressListBlocksShellCommandHost proves the list layer reaches the SHELL path too:
// a curl to a block-listed host buried in a Bash command line is refused, mirroring the
// hardwired floor's shell coverage.
func TestEgressListBlocksShellCommandHost(t *testing.T) {
	a := New(Policy{
		Allow:            map[string]bool{"Bash": true},
		EgressBlockHosts: []string{"tracker.example"},
	})
	v := a.Adjudicate(context.Background(), inlineCall("Bash", `{"command":"curl -s https://cdn.tracker.example/beacon"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != egressfloor.ReasonEgressBlock {
		t.Fatalf("Bash curl tracker = %v/%s, want Deny/EGRESS_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressRestrictStrictAllowlist proves the restrict knob turns EgressAllowHosts into a
// strict allowlist: a listed host is admitted, an unlisted host is POLICY_BLOCK even
// though it is on no block rule — the "restrict" posture.
func TestEgressRestrictStrictAllowlist(t *testing.T) {
	a := New(Policy{
		EgressRestrict:   true,
		EgressAllowHosts: []string{"docs.internal"},
	})
	// Listed host: admitted (positive allow via the block layer's exception path).
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://docs.internal/guide"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("WebFetch docs.internal under restrict = %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	// Unlisted host on no block rule: refused because restrict makes the allowlist total.
	v = a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://random.example/x"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("WebFetch random.example under restrict = %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressRestrictEmptyAllowlistBlocksAllFetch proves restrict with no allowlist is the
// fully-closed egress: every WebFetch host is POLICY_BLOCK.
func TestEgressRestrictEmptyAllowlistBlocksAllFetch(t *testing.T) {
	a := New(Policy{EgressRestrict: true})
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://anything.example/x"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("WebFetch under empty restrict = %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressDefaultAllowedFetchesFreely proves the DEFAULT posture (restrict off, no
// research allowlist): a block list denies its hosts, but any other host is reachable —
// the opposite of a strict allowlist.
func TestEgressDefaultAllowedFetchesFreely(t *testing.T) {
	a := New(Policy{
		Allow:            map[string]bool{"WebFetch": true},
		EgressBlockLists: []string{"sample-malware"},
	})
	// On the block list: denied.
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://malware.example/x"}`))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("WebFetch malware.example = %v, want Deny", v.Kind)
	}
	// Everything else: reachable (default-allowed), NOT a POLICY_BLOCK allowlist miss.
	v = a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://news.example/x"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("WebFetch news.example = %v/%s, want Allow (default-allowed)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestEgressListAbsentIsZeroCost proves a policy with no list configured leaves egressList
// nil and behaves exactly as before (the block path is never entered).
func TestEgressListAbsentIsZeroCost(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"WebFetch": true}})
	if got := a.state.Load().egressList; got != nil {
		t.Fatalf("egressList = %v, want nil when no list configured", got)
	}
	v := a.Adjudicate(context.Background(), inlineCall("WebFetch", `{"url":"https://anything.example/x"}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("WebFetch with no list = %v/%s, want Allow (unchanged path)", v.Kind, abi.ReasonName(v.Reason))
	}
}
