package engine

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fleetRoute is the route string a KindFleet target actually stamps into
// abi.ToolCall.Engine, built through modelroute rather than hand-spelled, so these
// tests cannot drift from the shape the switch emits.
func fleetRoute(t *testing.T, account, model string) string {
	t.Helper()
	return modelroute.Target{Kind: modelroute.KindFleet, Account: account, UpstreamModel: model}.EngineRoute()
}

// sensitiveFleetCall is a tenant-scoped payload routed to an org-operated host —
// exactly the call the residency gate exists to deny by default.
func sensitiveFleetCall(route string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool:   "summarize",
		Engine: route,
		Args:   abi.Ref{Kind: abi.RefInline, Scope: abi.ScopeTenant},
	}
}

// declareTrustedHost declares one verified host for the duration of a test and
// restores the fail-closed empty boundary afterwards, so the process-wide policy
// never leaks into another test in this package.
func declareTrustedHost(t *testing.T, account, baseURL string) {
	t.Helper()
	t.Setenv("FAK_TEST_FLEET_TOKEN", "s3cr3t")
	t.Cleanup(func() {
		if err := DeclareFleetTrustBoundary(); err != nil {
			t.Fatalf("clearing the fleet trust boundary: %v", err)
		}
	})
	if err := DeclareFleetTrustBoundary(FleetHost{
		Account: account, BaseURL: baseURL, CredEnv: "FAK_TEST_FLEET_TOKEN",
	}); err != nil {
		t.Fatalf("declaring fleet host %q: %v", account, err)
	}
}

// TestFleetTrustBoundaryDefaultIsClosed pins the pre-#5421 floor as the DEFAULT.
// Declaring `kind: fleet` is an ownership assertion, not a boundary, so with no
// operator declaration a tenant-scoped payload routed to an org-operated host is
// still denied — the behavior TestFleetZoneStaysRemoteAtTheFloor describes, now
// asserted at the gate that actually refuses the call.
func TestFleetTrustBoundaryDefaultIsClosed(t *testing.T) {
	if hosts := FleetTrustBoundary(); len(hosts) != 0 {
		t.Fatalf("the default fleet trust boundary must be EMPTY, got %+v", hosts)
	}
	route := fleetRoute(t, "gpu07", "glm-5.2")
	v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonScopeCrossing {
		t.Fatalf("undeclared fleet route %q: got %v/%s, want Deny/SCOPE_CROSSING",
			route, v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestDeclaredFleetHostAdmitsSensitivePayload is the issue's positive half: an
// operator CAN declare a specific authenticated org host as inside the trust
// boundary, and the floor then admits a sensitive payload to it.
func TestDeclaredFleetHostAdmitsSensitivePayload(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	route := fleetRoute(t, "gpu07", "glm-5.2")
	if v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route)); v.Kind != abi.VerdictDefer {
		t.Fatalf("declared fleet route %q: got %v/%s, want Defer (admitted)",
			route, v.Kind, abi.ReasonName(v.Reason))
	}
	// The declaration reads back as policy, carrying an env-var NAME and no secret.
	hosts := FleetTrustBoundary()
	if len(hosts) != 1 || hosts[0].Account != "gpu07" || hosts[0].CredEnv != "FAK_TEST_FLEET_TOKEN" {
		t.Fatalf("declared boundary read-back = %+v", hosts)
	}
}

// TestUndeclaredFleetHostStaysDenied is the issue's negative half, and the reason
// the widening is safe: declaring ONE host admits exactly that host. A different
// org-operated account — same kind, same zone, same operator — is still denied.
func TestUndeclaredFleetHostStaysDenied(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	for _, account := range []string{"gpu99", "gpu0", "gpu077", "GPU07-staging"} {
		route := fleetRoute(t, account, "glm-5.2")
		v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonScopeCrossing {
			t.Fatalf("undeclared fleet account %q (route %q): got %v/%s, want Deny/SCOPE_CROSSING",
				account, route, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestFleetTrustBoundaryLeavesTheResidencyClassifierAlone is requirement 3 and 4 of
// the issue held together: widening the GATE must not move the FLOOR. Even with the
// host declared, the route still reads REMOTE at the residency classifier, still
// reads off-box at the tier-1 mirror, and is still attributable as self-hosted. A
// future change that widened localRoute instead of the gate would fail here (and
// would silently desync modelroute.IsRemoteRoute, which cannot see this policy).
func TestFleetTrustBoundaryLeavesTheResidencyClassifierAlone(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	route := fleetRoute(t, "gpu07", "glm-5.2")
	if !insideFleetTrustBoundary(route) {
		t.Fatalf("route %q should be inside the declared boundary", route)
	}
	if !remoteRoute(route) {
		t.Fatalf("a DECLARED fleet route %q must still read REMOTE at the floor — trust is not residency", route)
	}
	if localRoute(route) {
		t.Fatalf("a declared fleet route %q must never join the on-box family list", route)
	}
	if !modelroute.IsRemoteRoute(route) {
		t.Fatalf("the tier-1 mirror must still read %q as remote — it cannot see this policy and must not need to", route)
	}
	zone := modelroute.ZoneOfRoute(route)
	if zone.OnBox() {
		t.Fatalf("zone %q of a declared fleet route must stay off-box", zone)
	}
	if !zone.SelfHosted() {
		t.Fatalf("zone %q of a declared fleet route must stay attributable as self-hosted", zone)
	}
}

// TestFleetTrustBoundaryNeedsAnAccountNamingRoute pins the fail-closed parse: only
// "fleet:<account>/…" names an account the allowlist can match. The other route
// shapes the family matcher accepts carry no account segment, so no declaration can
// admit them, and a vendor route that merely reuses the declared id is untouched.
func TestFleetTrustBoundaryNeedsAnAccountNamingRoute(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	for _, route := range []string{
		"fleet", "fleet-gpu07/glm-5.2", "fleet/gpu07", "fleet:", "fleet:/glm-5.2",
		"openai:gpu07/gpt-5.5", "anthropic:gpu07/claude-opus-5", "fleetwood:gpu07/m",
		"notfleet:gpu07/m", "gpu07",
	} {
		if insideFleetTrustBoundary(route) {
			t.Fatalf("route %q must not resolve into the trust boundary — it names no declared fleet account", route)
		}
		v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonScopeCrossing {
			t.Fatalf("route %q: got %v/%s, want Deny/SCOPE_CROSSING",
				route, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestFleetTrustBoundaryMatchesTheRouteCaseInsensitively pins that the match keys on
// the same normalization the floor uses, and that a namespaced upstream model id
// (qwen/qwen3.6-27b) does not truncate the account segment.
func TestFleetTrustBoundaryMatchesTheRouteCaseInsensitively(t *testing.T) {
	declareTrustedHost(t, "GPU07", "https://gpu-07.corp.internal:8000/v1")
	for _, route := range []string{
		"fleet:gpu07/glm-5.2", "FLEET:GPU07/GLM-5.2", "  fleet:GPU07/glm-5.2  ",
		"fleet:gpu07/qwen/qwen3.6-27b", "fleet:gpu07",
	} {
		if !insideFleetTrustBoundary(route) {
			t.Fatalf("route %q names the declared account and must be admitted", route)
		}
	}
}

// TestFleetTrustBoundaryRefusesUnverifiableDeclarations is the "not sufficient" half
// of the issue made executable: ownership alone does not declare a boundary. Each
// case is a host an operator might plausibly write and that this floor must refuse —
// and a refusal must leave the PREVIOUS policy untouched rather than half-apply.
func TestFleetTrustBoundaryRefusesUnverifiableDeclarations(t *testing.T) {
	t.Setenv("FAK_TEST_FLEET_TOKEN", "s3cr3t")
	t.Cleanup(func() { _ = DeclareFleetTrustBoundary() })

	cases := []struct {
		name string
		host FleetHost
	}{
		{"plain http is not a boundary", FleetHost{Account: "gpu07", BaseURL: "http://gpu-07.corp.internal:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
		{"no transport scheme", FleetHost{Account: "gpu07", BaseURL: "gpu-07.corp.internal:8000", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
		{"missing base_url", FleetHost{Account: "gpu07", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
		{"loopback is a device, not a fleet", FleetHost{Account: "gpu07", BaseURL: "https://127.0.0.1:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
		{"localhost is a device, not a fleet", FleetHost{Account: "gpu07", BaseURL: "https://localhost:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
		{"unauthenticated host", FleetHost{Account: "gpu07", BaseURL: "https://gpu-07.corp.internal:8000/v1"}},
		{"cred_env unset at declaration", FleetHost{Account: "gpu07", BaseURL: "https://gpu-07.corp.internal:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN_ABSENT"}},
		{"pasted secret instead of an env name", FleetHost{Account: "gpu07", BaseURL: "https://gpu-07.corp.internal:8000/v1", CredEnv: "sk-live-abc123"}},
		{"no account id", FleetHost{BaseURL: "https://gpu-07.corp.internal:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
		{"account id carries route delimiters", FleetHost{Account: "gpu07/glm", BaseURL: "https://gpu-07.corp.internal:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}},
	}
	for _, c := range cases {
		if err := DeclareFleetTrustBoundary(c.host); err == nil {
			t.Fatalf("%s: declaration was ACCEPTED (%+v) — the boundary must refuse it", c.name, c.host)
		}
		if hosts := FleetTrustBoundary(); len(hosts) != 0 {
			t.Fatalf("%s: a refused declaration installed policy %+v", c.name, hosts)
		}
	}
}

// TestFleetTrustBoundaryDeclarationIsWholeOrNothing pins that one bad host in a
// batch installs NOTHING. A half-applied allowlist is the failure mode the issue
// warns about — it reads as protection while covering only the entries that parsed.
func TestFleetTrustBoundaryDeclarationIsWholeOrNothing(t *testing.T) {
	t.Setenv("FAK_TEST_FLEET_TOKEN", "s3cr3t")
	t.Cleanup(func() { _ = DeclareFleetTrustBoundary() })

	good := FleetHost{Account: "gpu07", BaseURL: "https://gpu-07.corp.internal:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}
	bad := FleetHost{Account: "gpu08", BaseURL: "http://gpu-08.corp.internal:8000/v1", CredEnv: "FAK_TEST_FLEET_TOKEN"}
	if err := DeclareFleetTrustBoundary(good, bad); err == nil {
		t.Fatal("a batch containing an unverifiable host must be refused whole")
	}
	if hosts := FleetTrustBoundary(); len(hosts) != 0 {
		t.Fatalf("a refused batch left policy behind: %+v", hosts)
	}
	if insideFleetTrustBoundary("fleet:gpu07/glm-5.2") {
		t.Fatal("the good half of a refused batch must NOT be live")
	}
	// A duplicate account id is likewise refused: two records for one id would make
	// which credential is enforced depend on declaration order.
	if err := DeclareFleetTrustBoundary(good, good); err == nil {
		t.Fatal("a duplicate account id must be refused")
	}
}

// TestFleetTrustBoundaryClosesWhenTheCredentialIsWithdrawn pins the adjudication-time
// re-read: the boundary is only as live as the authenticated transport behind it. An
// operator who unsets the secret closes the boundary without editing the policy.
func TestFleetTrustBoundaryClosesWhenTheCredentialIsWithdrawn(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	route := fleetRoute(t, "gpu07", "glm-5.2")
	if v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route)); v.Kind != abi.VerdictDefer {
		t.Fatalf("declared fleet route %q should be admitted while the credential is present, got %v", route, v.Kind)
	}
	t.Setenv("FAK_TEST_FLEET_TOKEN", "")
	if insideFleetTrustBoundary(route) {
		t.Fatalf("route %q must fall outside the boundary once its credential is withdrawn", route)
	}
	v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonScopeCrossing {
		t.Fatalf("withdrawn credential for %q: got %v/%s, want Deny/SCOPE_CROSSING",
			route, v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestFleetTrustBoundaryClearsBackToTheClosedDefault pins that the widening is
// REVERSIBLE by the same verb that applied it — an empty declaration restores the
// pre-#5421 floor exactly, so a bad policy is one call away from being retired.
func TestFleetTrustBoundaryClearsBackToTheClosedDefault(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	route := fleetRoute(t, "gpu07", "glm-5.2")
	if !insideFleetTrustBoundary(route) {
		t.Fatalf("route %q should start inside the declared boundary", route)
	}
	if err := DeclareFleetTrustBoundary(); err != nil {
		t.Fatalf("clearing the boundary: %v", err)
	}
	if hosts := FleetTrustBoundary(); len(hosts) != 0 {
		t.Fatalf("cleared boundary still reads %+v", hosts)
	}
	v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonScopeCrossing {
		t.Fatalf("cleared boundary for %q: got %v/%s, want Deny/SCOPE_CROSSING",
			route, v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestFleetTrustBoundaryNeverWidensANonFleetRemoteRoute is the containment check: a
// declared boundary must not become a general residency escape. Every non-fleet
// remote route the gate denied before is still denied while a host is declared.
func TestFleetTrustBoundaryNeverWidensANonFleetRemoteRoute(t *testing.T) {
	declareTrustedHost(t, "gpu07", "https://gpu-07.corp.internal:8000/v1")
	for _, route := range []string{
		"remote", "openai:acct/gpt-5.5", "anthropic:work/claude-opus-5",
		"litellm/gpt-4o", "openrouter/anthropic/claude-3.5", "my-proxy", LLMDEngineID,
	} {
		v := (residencyGate{}).Adjudicate(context.Background(), sensitiveFleetCall(route))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonScopeCrossing {
			t.Fatalf("remote route %q under a declared boundary: got %v/%s, want Deny/SCOPE_CROSSING",
				route, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}
