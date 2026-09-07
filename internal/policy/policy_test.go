package policy

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func TestAllowExactPreservesCommandBytes(t *testing.T) {
	const command = `powershell.exe -NoProfile -Command "Get-Process"`
	literal := command
	p, err := (Manifest{Allow: []string{"Bash"}, ArgRules: []ArgRule{{Tool: "Bash", Arg: "command", AllowExact: &literal}}}).ToPolicy()
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := Parse(FromPolicy(p).JSON())
	if err != nil || !reflect.DeepEqual(p, roundTrip) {
		t.Fatalf("round trip dropped exact predicate: %v", err)
	}
	if !strings.Contains(Summary(p), "allow_exact") {
		t.Fatal("summary omitted exact predicate")
	}
	k := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{adjudicator.New(roundTrip)}))
	k.SetVDSO(false)
	for _, raw := range []string{
		`{"command":` + string(mustExactJSON(t, command)) + `}`,
		`{"command":` + string(mustExactJSON(t, command+" & echo changed")) + `}`,
		`{"command":` + string(mustExactJSON(t, "ignored/../"+command)) + `}`,
		`{"command":` + string(mustExactJSON(t, " "+command)) + `}`,
		`{"command":` + string(mustExactJSON(t, strings.ToUpper(command))) + `}`,
		`{"command":null}`, `{"command":123}`, `{}`,
	} {
		_, v := k.Syscall(context.Background(), &abi.ToolCall{
			Tool: "Bash", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(raw)},
		})
		want := abi.VerdictDeny
		if raw == `{"command":`+string(mustExactJSON(t, command))+`}` {
			want = abi.VerdictAllow
		}
		if v.Kind != want {
			t.Fatalf("args %s: verdict=%v want=%v", raw, v.Kind, want)
		}
	}
	changed := literal + " changed"
	next, err := (Manifest{Allow: []string{"Bash"}, ArgRules: []ArgRule{{Tool: "Bash", Arg: "command", AllowExact: &changed}}}).ToPolicy()
	if err != nil || DiffAmendment(p, next).Empty() {
		t.Fatalf("exact-value amendment was lost: %v", err)
	}
	for _, raw := range []string{
		`{"arg_rules":[{"tool":"Bash","arg":"command","allow_exact":"","allow_glob":"**"}]}`,
		`{"arg_rules":[{"tool":"Bash","arg":"command","allow_exact":null}]}`,
	} {
		if _, err := Parse([]byte(raw)); err == nil {
			t.Fatalf("invalid matcher accepted: %s", raw)
		}
	}
	empty, err := Parse([]byte(`{"allow":["Bash"],"arg_rules":[{"tool":"Bash","arg":"command","allow_exact":""}]}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{`{"command":""}`, `{}`, `{"command":null}`} {
		v := adjudicator.New(empty).Adjudicate(context.Background(), &abi.ToolCall{Tool: "Bash", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(raw)}})
		if (v.Kind == abi.VerdictAllow) != (raw == `{"command":""}`) {
			t.Fatalf("empty literal presence not enforced: %s => %v", raw, v.Kind)
		}
	}
}

func mustExactJSON(t *testing.T, value string) []byte {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

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

func TestDefaultOpenPostureLoadsAndRoundTrips(t *testing.T) {
	p, err := Parse([]byte(`{"posture":"default_open"}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if p.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("posture = %v, want PostureDefaultOpen", p.Posture)
	}
	a := adjudicator.New(p)
	v := a.Adjudicate(context.Background(), &abi.ToolCall{
		Tool: "custom_random_tool",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	})
	if v.Kind != abi.VerdictAllow || v.Meta["posture"] != "default_open" {
		t.Fatalf("default_open verdict = %+v, want Allow with posture=default_open", v)
	}
	if got, err := FromPolicy(p).ToPolicy(); err != nil || !reflect.DeepEqual(got, p) {
		t.Fatalf("posture round-trip mismatch err=%v got=%+v want=%+v", err, got, p)
	}
	if !strings.Contains(Summary(p), "default_open") {
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

// TestDeniesToolUnconditionally proves the manifest-level "is this a blanket block?"
// predicate (#752/#773): true only when no argument value could ever flip the tool to
// ALLOW, so promptmmu may safely drop its definition from the advertised surface.
func TestDeniesToolUnconditionally(t *testing.T) {
	// A real floor: two tools affirmatively allowed, one of them narrowed by an
	// arg-conditional rule, plus an explicit blanket deny.
	m, err := ParseManifest([]byte(`{
		"allow": ["read_file", "write_file"],
		"deny": {"exfiltrate": "SECRET_EXFIL"},
		"arg_rules": [
			{"tool":"write_file", "arg":"path", "allow_glob":"./out/**"}
		]
	}`))
	if err != nil {
		t.Fatalf("ParseManifest: %v", err)
	}

	// (a) a blanket name-level deny is unconditional — no arg can make it allow.
	if !m.DeniesToolUnconditionally("exfiltrate") {
		t.Error("blanket-denied tool should report unconditionally denied")
	}
	// (b) an arg-conditional tool is NOT unconditional — some path values are allowed.
	if m.DeniesToolUnconditionally("write_file") {
		t.Error("arg-conditional tool (some args allowed) must NOT report unconditionally denied")
	}
	// (c) an affirmatively allowed tool with no restriction is not denied at all.
	if m.DeniesToolUnconditionally("read_file") {
		t.Error("affirmatively allowed tool must NOT report unconditionally denied")
	}
	// (d) under a real fail-closed floor, a tool absent from every allow case is never
	// reachable for ANY arg — droppable.
	if !m.DeniesToolUnconditionally("unlisted_tool") {
		t.Error("unlisted tool under a real fail-closed floor is never admitted → droppable")
	}

	// A manifest with NO affirmative-allow surface (the unconfigured / default-allow
	// case) must report false for everything — never prune against a zero floor.
	open, err := ParseManifest([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseManifest empty: %v", err)
	}
	if open.DeniesToolUnconditionally("anything") {
		t.Error("empty/unconfigured manifest must NOT prune any tool (zero-floor guard)")
	}

	// A manifest that does not resolve (unknown deny reason) fails safe → false.
	bad := Manifest{Allow: []string{"read_file"}, Deny: map[string]string{"x": "NOPE_NOT_A_REASON"}}
	if bad.DeniesToolUnconditionally("x") {
		t.Error("an unresolvable manifest must report false (never prune against a floor that did not load)")
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
	if v := call("send_email", `{"to":"ok@partner.example.com","body":"approved update"}`); v.Kind != abi.VerdictRequireWitness || v.Meta["reversibility_class"] != string(adjudicator.ReversibilityOutwardFacing) {
		t.Fatalf("authorized egress preview: got %v/%s meta=%v, want RequireWitness/outward-facing",
			v.Kind, abi.ReasonName(v.Reason), v.Meta)
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
	if len(rt.SafeSinks) > 0 || len(rt.AuthorizeRules) > 0 || len(rt.Sources) > 0 {
		p.GatedSinks = ifc.StrictGatedSinks()
	}
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

// TestLintWritesLoadsAndRoundTrips: the opt-in write-lint rung (#536) is a real
// operator surface — `lint_writes: true` in a manifest turns on
// adjudicator.Policy.LintWrites, omitted means off, and FromPolicy/ToPolicy
// round-trip it exactly so --dump/--check stay honest.
func TestLintWritesLoadsAndRoundTrips(t *testing.T) {
	on, err := Parse([]byte(`{"lint_writes":true,"allow":["write_file"]}`))
	if err != nil {
		t.Fatalf("Parse lint_writes:true: %v", err)
	}
	if !on.LintWrites {
		t.Fatal("lint_writes:true must set Policy.LintWrites")
	}

	off, err := Parse([]byte(`{"allow":["write_file"]}`))
	if err != nil {
		t.Fatalf("Parse omitted lint_writes: %v", err)
	}
	if off.LintWrites {
		t.Fatal("omitted lint_writes must default to false (off by default)")
	}

	for _, p := range []adjudicator.Policy{on, off} {
		got, err := FromPolicy(p).ToPolicy()
		if err != nil || !reflect.DeepEqual(got, p) {
			t.Fatalf("lint_writes round-trip mismatch err=%v got=%+v want=%+v", err, got, p)
		}
	}
}

// TestManifestRateLimitRoundTrip is the issue-#699 acceptance witness (criterion
// 1): a fak-policy/v1 manifest with a rate_limit block parses, round-trips through
// the manifest level (it is NOT an adjudicator.Policy field, so FromPolicy/ToPolicy
// is the wrong vehicle — its round-trip is JSON -> Manifest -> Runtime -> JSON),
// and surfaces in SummaryRuntime.
func TestManifestRateLimitRoundTrip(t *testing.T) {
	const js = `{"allow":["search_kb"],"rate_limit":{"max_calls":5,"max_cost":1000,"key":"tool","retry_after_ms":250}}`
	m, err := ParseManifest([]byte(js))
	if err != nil {
		t.Fatalf("ParseManifest: %v", err)
	}
	if m.RateLimit == nil || m.RateLimit.MaxCalls != 5 || m.RateLimit.MaxCost != 1000 ||
		m.RateLimit.Key != "tool" || m.RateLimit.RetryAfterMS != 250 {
		t.Fatalf("parsed rule = %+v, want the declared cap", m.RateLimit)
	}
	rt, err := m.ToRuntime()
	if err != nil {
		t.Fatalf("ToRuntime: %v", err)
	}
	if got := rt.RateLimit; got == nil || *got != *m.RateLimit {
		t.Fatalf("runtime rule = %+v, want %+v", got, m.RateLimit)
	}
	// JSON -> Manifest -> JSON round-trips the rule byte-for-byte.
	m2, err := ParseManifest(m.JSON())
	if err != nil {
		t.Fatalf("re-parse marshal: %v", err)
	}
	if m2.RateLimit == nil || *m2.RateLimit != *m.RateLimit {
		t.Fatalf("JSON round-trip lost the rule: got %+v want %+v", m2.RateLimit, m.RateLimit)
	}
	if !strings.Contains(SummaryRuntime(rt), "rate limit         : 5 call(s) / 1000 cost per tool") {
		t.Fatalf("SummaryRuntime should surface the declared rate limit:\n%s", SummaryRuntime(rt))
	}
}

// TestManifestRateLimitRejectsBadBlocks is the issue-#699 fail-loud witness
// (criterion 1): an unknown key-mode, a negative cap, or an all-zero (no-cap)
// block is rejected at load — never silently installed.
func TestManifestRateLimitRejectsBadBlocks(t *testing.T) {
	cases := []struct {
		name string
		js   string
	}{
		{"unknown key-mode", `{"rate_limit":{"max_calls":1,"key":"tenant"}}`},
		{"negative calls", `{"rate_limit":{"max_calls":-1}}`},
		{"negative cost", `{"rate_limit":{"max_cost":-5}}`},
		{"negative retry_after", `{"rate_limit":{"max_calls":1,"retry_after_ms":-10}}`},
		{"no cap declared", `{"rate_limit":{"key":"tool"}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse([]byte(tc.js)); err == nil {
				t.Fatal("expected validation error, got nil")
			}
		})
	}
}

// TestManifestWithoutRateLimitIsAbsent proves criterion 2 at the manifest level: a
// manifest with NO rate_limit block resolves to a nil (inert) rule, and the
// existing name-level FromPolicy/ToPolicy round-trip is unaffected (rate_limit is
// not an adjudicator.Policy field).
func TestManifestWithoutRateLimitIsAbsent(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"allow":["search_kb"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.RateLimit != nil {
		t.Fatalf("absent rate_limit must resolve to nil (inert), got %+v", rt.RateLimit)
	}
	if !strings.Contains(SummaryRuntime(rt), "rate limit         : (none") {
		t.Fatalf("SummaryRuntime should flag the inert rate limit:\n%s", SummaryRuntime(rt))
	}
	p := adjudicator.DefaultPolicy()
	if got, err := FromPolicy(p).ToPolicy(); err != nil || !reflect.DeepEqual(got, p) {
		t.Fatalf("name-level round-trip drifted: err=%v got=%+v", err, got)
	}
}

func TestCLIReadOnlyArgRuleCompiles(t *testing.T) {
	m := Manifest{Version: Version, Posture: "fail_closed", AllowPrefix: []string{"Bash"}, ArgRules: []ArgRule{{Tool: "Bash", Arg: "command", CLIReadOnly: true}}}
	runtime, err := m.ToRuntime()
	if err != nil {
		t.Fatal(err)
	}
	if len(runtime.Adjudicator.ArgPredicates) != 1 || runtime.Adjudicator.ArgPredicates[0].Kind != adjudicator.ArgCLIReadOnly {
		t.Fatalf("predicates=%+v", runtime.Adjudicator.ArgPredicates)
	}
	m.ArgRules[0].DenyRegex = "push"
	if _, err := m.ToRuntime(); err == nil {
		t.Fatal("cli_read_only plus deny_regex must fail exactly-one matcher validation")
	}
}

func TestAllowExactArgumentPredicate(t *testing.T) {
	exactCmd := "systemctl restart nginx"
	rule := ArgRule{
		Tool:       "run_shell",
		Arg:        "cmd",
		AllowExact: &exactCmd,
		Reason:     "POLICY_BLOCK",
	}

	// 1. Exact command matches (returns allow)
	t.Run("exact matches allow", func(t *testing.T) {
		if !rule.Matches(exactCmd) {
			t.Errorf("Matches(%q) should be true", exactCmd)
		}
		if !rule.Match(exactCmd) {
			t.Errorf("Match(%q) should be true", exactCmd)
		}
		if !rule.Evaluate(exactCmd) {
			t.Errorf("Evaluate(%q) should be true", exactCmd)
		}
		if v := rule.EvaluateVerdict(exactCmd); v.Kind != abi.VerdictAllow {
			t.Errorf("EvaluateVerdict(%q) got %v, want VerdictAllow", exactCmd, v)
		}
		// Via map of arguments
		argsMap := map[string]any{"cmd": exactCmd}
		if !rule.Matches(argsMap) {
			t.Errorf("Matches(map) should be true for exact command")
		}
		if v := rule.EvaluateVerdict(argsMap); v.Kind != abi.VerdictAllow {
			t.Errorf("EvaluateVerdict(map) got %v, want VerdictAllow", v)
		}
		// Via map[string]string
		argsStrMap := map[string]string{"cmd": exactCmd}
		if !rule.Matches(argsStrMap) {
			t.Errorf("Matches(map[string]string) should be true for exact command")
		}
		// Via JSON string
		argsJSON := `{"cmd":"systemctl restart nginx"}`
		if !rule.Matches(argsJSON) {
			t.Errorf("Matches(json) should be true for exact command")
		}
		if v := rule.EvaluateVerdict(argsJSON); v.Kind != abi.VerdictAllow {
			t.Errorf("EvaluateVerdict(json) got %v, want VerdictAllow", v)
		}
	})

	// 2. Near-matches return deny
	t.Run("near matches deny", func(t *testing.T) {
		nearMatches := []struct {
			name string
			val  string
		}{
			{"trailing space", "systemctl restart nginx "},
			{"leading space", " systemctl restart nginx"},
			{"different casing upper", "Systemctl restart nginx"},
			{"different casing allcaps", "SYSTEMCTL RESTART NGINX"},
			{"appended flag", "systemctl restart nginx --force"},
			{"prepended command", "sudo systemctl restart nginx"},
			{"missing token", "systemctl restart"},
			{"inner double space", "systemctl  restart nginx"},
			{"newline appended", "systemctl restart nginx\n"},
		}
		for _, tc := range nearMatches {
			t.Run(tc.name, func(t *testing.T) {
				if rule.Matches(tc.val) {
					t.Errorf("near-match %q should return false", tc.val)
				}
				if v := rule.EvaluateVerdict(tc.val); v.Kind != abi.VerdictDeny {
					t.Errorf("near-match %q got verdict %v, want VerdictDeny", tc.val, v)
				}
				// Also via args map
				m := map[string]any{"cmd": tc.val}
				if rule.Matches(m) {
					t.Errorf("near-match map %q should return false", tc.val)
				}
				if v := rule.EvaluateVerdict(m); v.Kind != abi.VerdictDeny {
					t.Errorf("near-match map %q got verdict %v, want VerdictDeny", tc.val, v)
				}
			})
		}
	})

	// 3. Missing argument or non-string argument fails closed
	t.Run("fails closed on missing or non-string", func(t *testing.T) {
		badInputs := []struct {
			name string
			val  any
		}{
			{"missing key in map", map[string]any{"other_arg": exactCmd}},
			{"empty map", map[string]any{}},
			{"nil map value", map[string]any{"cmd": nil}},
			{"nil input", nil},
			{"integer argument", map[string]any{"cmd": 123}},
			{"boolean argument", map[string]any{"cmd": true}},
			{"array argument", map[string]any{"cmd": []string{"systemctl", "restart", "nginx"}}},
			{"raw integer", 123},
			{"raw bool", true},
		}
		for _, tc := range badInputs {
			t.Run(tc.name, func(t *testing.T) {
				if rule.Matches(tc.val) {
					t.Errorf("input %s (%v) should fail closed (return false)", tc.name, tc.val)
				}
				if v := rule.EvaluateVerdict(tc.val); v.Kind != abi.VerdictDeny {
					t.Errorf("input %s (%v) got verdict %v, want VerdictDeny", tc.name, tc.val, v)
				}
			})
		}

		// Explicit empty string AllowExact
		emptyStr := ""
		emptyRule := ArgRule{Tool: "run_shell", Arg: "cmd", AllowExact: &emptyStr}
		if !emptyRule.Matches("") {
			t.Errorf("explicit empty string should match empty string")
		}
		if emptyRule.Matches(" ") {
			t.Errorf("explicit empty string should not match space")
		}
		if emptyRule.Matches(nil) {
			t.Errorf("explicit empty string should fail closed on nil")
		}
		if emptyRule.Matches(map[string]any{"cmd": nil}) {
			t.Errorf("explicit empty string should fail closed on nil map value")
		}
		if emptyRule.Matches(map[string]any{}) {
			t.Errorf("explicit empty string should fail closed on missing map key")
		}
	})

	// 4. JSON manifest round-trip serialization/deserialization
	t.Run("JSON manifest round-trip", func(t *testing.T) {
		manifestJSON := `{
  "version": "fak-policy/v1",
  "allow": [
    "run_shell"
  ],
  "arg_rules": [
    {
      "tool": "run_shell",
      "arg": "cmd",
      "allow_exact": "systemctl restart nginx",
      "reason": "POLICY_BLOCK"
    }
  ]
}`
		m, err := ParseManifest([]byte(manifestJSON))
		if err != nil {
			t.Fatalf("ParseManifest failed: %v", err)
		}
		if len(m.ArgRules) != 1 {
			t.Fatalf("expected 1 arg_rule, got %d", len(m.ArgRules))
		}
		r := m.ArgRules[0]
		if r.AllowExact == nil || *r.AllowExact != exactCmd {
			t.Fatalf("expected AllowExact %q, got %v", exactCmd, r.AllowExact)
		}

		// Verify ToRuntime / ToPolicy compiles
		rt, err := m.ToRuntime()
		if err != nil {
			t.Fatalf("ToRuntime failed: %v", err)
		}
		if len(rt.Adjudicator.ArgPredicates) != 1 {
			t.Fatalf("expected 1 ArgPredicate in Adjudicator, got %d", len(rt.Adjudicator.ArgPredicates))
		}
		pred := rt.Adjudicator.ArgPredicates[0]
		if pred.Kind != ArgAllowExact || pred.Glob != exactCmd {
			t.Fatalf("expected ArgAllowExact predicate with glob %q, got kind=%v glob=%q", exactCmd, pred.Kind, pred.Glob)
		}

		// Re-serialize to JSON and re-parse
		serialized := m.JSON()
		m2, err := ParseManifest(serialized)
		if err != nil {
			t.Fatalf("ParseManifest(m.JSON()) failed: %v", err)
		}
		if len(m2.ArgRules) != 1 || m2.ArgRules[0].AllowExact == nil || *m2.ArgRules[0].AllowExact != exactCmd {
			t.Fatalf("JSON round-trip mismatch: got %+v", m2.ArgRules)
		}

		// Policy round-trip FromPolicy
		fromPol := FromPolicy(rt.Adjudicator)
		if len(fromPol.ArgRules) != 1 || fromPol.ArgRules[0].AllowExact == nil || *fromPol.ArgRules[0].AllowExact != exactCmd {
			t.Fatalf("FromPolicy round-trip mismatch: got %+v", fromPol.ArgRules)
		}

		// Summary includes allow_exact
		summary := Summary(rt.Adjudicator)
		if !strings.Contains(summary, "allow_exact systemctl restart nginx") {
			t.Fatalf("Summary missing allow_exact: %s", summary)
		}
	})

	// 5. Exclusivity validation
	t.Run("exclusivity validation", func(t *testing.T) {
		// allow_exact + allow_glob
		bad1 := `{
  "allow": ["run_shell"],
  "arg_rules": [
    {
      "tool": "run_shell",
      "arg": "cmd",
      "allow_exact": "ls",
      "allow_glob": "./*"
    }
  ]
}`
		if _, err := Parse([]byte(bad1)); err == nil {
			t.Fatal("expected error when setting both allow_exact and allow_glob")
		}

		// allow_exact + deny_regex
		bad2 := `{
  "allow": ["run_shell"],
  "arg_rules": [
    {
      "tool": "run_shell",
      "arg": "cmd",
      "allow_exact": "ls",
      "deny_regex": "rm"
    }
  ]
}`
		if _, err := Parse([]byte(bad2)); err == nil {
			t.Fatal("expected error when setting both allow_exact and deny_regex")
		}
	})
}
