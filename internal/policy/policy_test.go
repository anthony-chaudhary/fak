package policy

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// TestRoundTrip is the load-bearing invariant for --dump: the built-in
// DefaultPolicy, rendered to a manifest and parsed back, must reconstruct the
// SAME runtime Policy. If this drifts, an adopter who dumps-edits-loads silently
// loses a baked-in protection.
func TestRoundTrip(t *testing.T) {
	want := adjudicator.DefaultPolicy()
	got, err := FromPolicy(want).ToPolicy()
	if err != nil {
		t.Fatalf("round-trip ToPolicy: %v", err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("round-trip mismatch:\n want=%+v\n got =%+v", want, got)
	}
}

// TestParseFromDumpBytes exercises the full byte path (JSON -> Parse), not just
// the in-memory struct round-trip.
func TestParseFromDumpBytes(t *testing.T) {
	want := adjudicator.DefaultPolicy()
	b := FromPolicy(want).JSON()
	got, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse(dump): %v", err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("byte round-trip mismatch:\n want=%+v\n got =%+v", want, got)
	}
}

func TestUnknownDenyReasonRejected(t *testing.T) {
	_, err := Parse([]byte(`{"deny":{"rm_rf":"NUKE_EVERYTHING"}}`))
	if err == nil {
		t.Fatal("expected error for unknown deny reason, got nil")
	}
	if !strings.Contains(err.Error(), "NUKE_EVERYTHING") {
		t.Fatalf("error should name the offending reason: %v", err)
	}
	// and it should list the valid vocabulary to be actionable
	if !strings.Contains(err.Error(), "DEFAULT_DENY") {
		t.Fatalf("error should list the valid vocabulary: %v", err)
	}
}

func TestUnknownFieldRejected(t *testing.T) {
	// "allows" is a typo for "allow" — must fail loudly, not silently no-op.
	_, err := Parse([]byte(`{"allows":["read_file"]}`))
	if err == nil {
		t.Fatal("expected error for unknown field, got nil")
	}
}

func TestEmptyManifestIsFailClosed(t *testing.T) {
	p, err := Parse([]byte(`{}`))
	if err != nil {
		t.Fatalf("empty manifest should be valid: %v", err)
	}
	if len(p.Allow) != 0 || len(p.AllowPrefix) != 0 {
		t.Fatalf("empty manifest should allow nothing: %+v", p)
	}
	if !strings.Contains(Summary(p), "fail-closed") {
		t.Fatalf("Summary should flag the empty floor as fail-closed:\n%s", Summary(p))
	}
}

func TestVersionGating(t *testing.T) {
	cases := []struct {
		name    string
		ver     string
		wantErr bool
	}{
		{"omitted", "", false},
		{"current", "fak-policy/v1", false},
		{"future minor", "fak-policy/v1.3", false},
		{"future major", "fak-policy/v2", true},
		{"garbage", "not-a-version", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := Manifest{Version: tc.ver, Allow: []string{"read_file"}}
			_, err := m.ToPolicy()
			if (err != nil) != tc.wantErr {
				t.Fatalf("version %q: err=%v wantErr=%v", tc.ver, err, tc.wantErr)
			}
		})
	}
}

func TestAdmitAndLogPostureLoadsAndRoundTrips(t *testing.T) {
	p, err := Parse([]byte(`{"posture":"admit_and_log"}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if p.Posture != adjudicator.PostureAdmitAndLog {
		t.Fatalf("posture = %v, want PostureAdmitAndLog", p.Posture)
	}
	a := adjudicator.New(p)
	v := a.Adjudicate(context.Background(), &abi.ToolCall{
		Tool: "read_batch_fixture",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	})
	if v.Kind != abi.VerdictAllow || v.Meta["would_deny"] != "DEFAULT_DENY" {
		t.Fatalf("admit-and-log read verdict = %+v, want Allow with would_deny", v)
	}
	if got, err := FromPolicy(p).ToPolicy(); err != nil || !reflect.DeepEqual(got, p) {
		t.Fatalf("posture round-trip mismatch err=%v got=%+v want=%+v", err, got, p)
	}
	if !strings.Contains(Summary(p), "admit_and_log") {
		t.Fatalf("Summary should surface posture:\n%s", Summary(p))
	}
}

func TestUnknownPostureRejected(t *testing.T) {
	_, err := Parse([]byte(`{"posture":"audit_only"}`))
	if err == nil {
		t.Fatal("expected error for unknown posture, got nil")
	}
	if !strings.Contains(err.Error(), "audit_only") || !strings.Contains(err.Error(), "admit_and_log") {
		t.Fatalf("error should name the bad and valid postures: %v", err)
	}
}

// TestLoadedPolicyIsLoadBearing proves the manifest actually drives the
// adjudicator: a tool the manifest allows resolves ALLOW; one it denies resolves
// DENY with the cited reason; an unlisted tool hits the fail-closed default.
func TestLoadedPolicyIsLoadBearing(t *testing.T) {
	p, err := Parse([]byte(`{
		"allow": ["search_flights"],
		"allow_prefix": ["read_"],
		"deny": {"exfiltrate": "SECRET_EXFIL"}
	}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	a := adjudicator.New(p)
	ctx := context.Background()
	// Inline Ref args: the adjudicator reads RefInline bytes directly, so this
	// test is hermetic — no registered Ref resolver / driver set needed.
	call := func(tool string) abi.Verdict {
		ref := abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}
		return a.Adjudicate(ctx, &abi.ToolCall{Tool: tool, Args: ref})
	}

	if v := call("search_flights"); v.Kind != abi.VerdictAllow {
		t.Errorf("allowed tool: got %v, want ALLOW", v.Kind)
	}
	if v := call("read_refund_policy"); v.Kind != abi.VerdictAllow {
		t.Errorf("allow_prefix tool: got %v, want ALLOW", v.Kind)
	}
	if v := call("exfiltrate"); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSecretExfil {
		t.Errorf("denied tool: got kind=%v reason=%v, want DENY/SECRET_EXFIL", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("delete_account"); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Errorf("unlisted tool: got kind=%v reason=%v, want DENY/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestArgRulesAreLoadBearing(t *testing.T) {
	p, err := Parse([]byte(`{
		"allow": ["write_file", "run_shell"],
		"arg_rules": [
			{"tool":"write_file", "arg":"path", "allow_glob":"./out/**"},
			{"tool":"run_shell", "arg":"cmd", "deny_regex":"rm|push --force", "reason":"POLICY_BLOCK"}
		]
	}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	a := adjudicator.New(p)
	ctx := context.Background()
	call := func(tool, args string) abi.Verdict {
		ref := abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}
		return a.Adjudicate(ctx, &abi.ToolCall{Tool: tool, Args: ref})
	}

	if v := call("write_file", `{"path":"./out/report.txt"}`); v.Kind != abi.VerdictAllow {
		t.Fatalf("write under ./out: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("write_file", `{"path":"./out/../secret.txt"}`); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("write path escape: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("run_shell", `{"cmd":"git status --short"}`); v.Kind != abi.VerdictAllow {
		t.Fatalf("benign shell command: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("run_shell", `{"cmd":"git push --force origin main"}`); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("dangerous shell command: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
	if !strings.Contains(Summary(p), "write_file.path allow_glob ./out/**") {
		t.Fatalf("Summary should include arg rules:\n%s", Summary(p))
	}
}

func TestArgRuleValidation(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"missing matcher", `{"arg_rules":[{"tool":"write_file","arg":"path"}]}`},
		{"two matchers", `{"arg_rules":[{"tool":"write_file","arg":"path","allow_glob":"./out/**","deny_regex":"rm"}]}`},
		{"bad regex", `{"arg_rules":[{"tool":"run_shell","arg":"cmd","deny_regex":"["}]}`},
		{"unknown reason", `{"arg_rules":[{"tool":"run_shell","arg":"cmd","deny_regex":"rm","reason":"NOPE"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse([]byte(tc.json)); err == nil {
				t.Fatal("expected validation error, got nil")
			}
		})
	}
}

func TestIFCRuntimeConfigIsLoadBearing(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"allow": ["send_email", "Bash", "handoff_to_security"],
		"safe_sinks": ["handoff_to_security"],
		"authorize": [{"tool":"send_email", "sink":"EGRESS"}],
		"sources": {"read_corp_kb_policytest": "trusted_local"}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	ctx := context.Background()
	led := ifc.NewLedger()
	ApplySources(rt)
	ifcPolicy := testIFCPolicy(rt)
	stamp := ifc.NewStampGate(led, ifcPolicy)
	trustedRead := &abi.ToolCall{Tool: "read_corp_kb_policytest", TraceID: "trusted"}
	stamp.Admit(ctx, trustedRead, &abi.Result{Status: abi.StatusOK})
	if got := led.Level("trusted"); got != abi.TaintTrusted {
		t.Fatalf("custom trusted source left trace %v, want trusted", got)
	}

	stamp.Admit(ctx, &abi.ToolCall{Tool: "read_webpage", TraceID: "tainted"}, &abi.Result{Status: abi.StatusOK})
	if got := led.Level("tainted"); got != abi.TaintTainted {
		t.Fatalf("external read left trace %v, want tainted", got)
	}

	chain := []abi.Adjudicator{ifc.NewSinkGate(led, ifcPolicy), adjudicator.New(rt.Adjudicator)}
	call := func(tool, args string) abi.Verdict {
		return kernel.Fold(ctx, chain, &abi.ToolCall{
			Tool:    tool,
			TraceID: "tainted",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		})
	}
	if v := call("send_email", `{"to":"ok@partner.example.com","body":"approved update"}`); v.Kind != abi.VerdictAllow {
		t.Fatalf("authorized egress: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("Bash", `{"cmd":"echo sensitive"}`); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("unauthorized exec sink: got %v/%s, want Deny/TRUST_VIOLATION", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("handoff_to_security", `{"reason":"needs human review"}`); v.Kind != abi.VerdictAllow {
		t.Fatalf("configured safe sink: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if s := SummaryRuntime(rt); !strings.Contains(s, "read_corp_kb_policytest -> trusted_local") || !strings.Contains(s, "send_email -> EGRESS") {
		t.Fatalf("SummaryRuntime omitted IFC config:\n%s", s)
	}
}

func testIFCPolicy(rt Runtime) ifc.Policy {
	p := ifc.Policy{}
	if len(rt.SafeSinks) > 0 {
		p.SafeSinks = make(map[string]bool, len(rt.SafeSinks))
		for _, tool := range rt.SafeSinks {
			p.SafeSinks[tool] = true
		}
	}
	if len(rt.AuthorizeRules) > 0 {
		p.Authorize = func(c *abi.ToolCall, into ifc.SinkClass) bool {
			if c == nil {
				return false
			}
			for _, r := range rt.AuthorizeRules {
				if c.Tool == r.Tool && strings.EqualFold(r.Sink, into.String()) {
					return true
				}
			}
			return false
		}
	}
	return p
}

func TestIFCRuntimeValidation(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"unknown sink", `{"authorize":[{"tool":"send_email","sink":"NETWORK"}]}`},
		{"missing authorize tool", `{"authorize":[{"sink":"EGRESS"}]}`},
		{"unknown source", `{"sources":{"read_x":"trusted_remote"}}`},
		{"empty safe sink", `{"safe_sinks":[""]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseRuntime([]byte(tc.json)); err == nil {
				t.Fatal("expected validation error, got nil")
			}
		})
	}
}

func TestReasonByNameInverse(t *testing.T) {
	for _, name := range abi.ReasonNames() {
		c, ok := abi.ReasonByName(name)
		if !ok {
			t.Errorf("ReasonByName(%q) not found", name)
			continue
		}
		if got := abi.ReasonName(c); got != name {
			t.Errorf("inverse mismatch: %q -> %d -> %q", name, c, got)
		}
	}
	if _, ok := abi.ReasonByName("DEFINITELY_NOT_A_REASON"); ok {
		t.Error("ReasonByName should reject an unknown name")
	}
}
