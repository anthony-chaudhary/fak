package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

func TestApplyRuntimeInstallsIFCManifestPolicy(t *testing.T) {
	rt, err := policy.ParseRuntime([]byte(`{
		"allow": ["send_email", "Bash", "send_to_security"],
		"safe_sinks": ["send_to_security"],
		"authorize": [{"tool":"send_email", "sink":"EGRESS"}],
		"sources": {"read_cmd_policy_fixture": "trusted_local"}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}

	const trustedTrace = "cmd-policy-trusted-source"
	const taintedTrace = "cmd-policy-tainted-flow"
	ctx := context.Background()
	ifc.Default.Reset(trustedTrace)
	ifc.Default.Reset(taintedTrace)
	applyRuntime(rt)
	defer func() {
		ifc.ConfigureDefaultPolicy(ifc.Policy{})
		ifc.Default.Reset(trustedTrace)
		ifc.Default.Reset(taintedTrace)
	}()

	ifc.DefaultStampGate.Admit(ctx,
		&abi.ToolCall{Tool: "read_cmd_policy_fixture", TraceID: trustedTrace},
		&abi.Result{Status: abi.StatusOK},
	)
	if got := ifc.Default.Level(trustedTrace); got != abi.TaintTrusted {
		t.Fatalf("configured trusted source left trace %v, want trusted", got)
	}

	ifc.Default.Raise(taintedTrace, abi.TaintTainted)
	chain := []abi.Adjudicator{ifc.DefaultSinkGate, adjudicator.New(rt.Adjudicator)}
	call := func(tool, args string) abi.Verdict {
		return kernel.Fold(ctx, chain, &abi.ToolCall{
			Tool:    tool,
			TraceID: taintedTrace,
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		})
	}

	if v := call("send_email", `{"to":"ok@partner.example.com","body":"approved update"}`); v.Kind != abi.VerdictAllow {
		t.Fatalf("authorized egress: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("Bash", `{"cmd":"echo sensitive"}`); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("unauthorized exec sink: got %v/%s, want Deny/TRUST_VIOLATION", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := call("send_to_security", `{"reason":"needs human review"}`); v.Kind != abi.VerdictAllow {
		t.Fatalf("configured safe sink: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestPolicyReloaderSwapsAdjudicatorPolicy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "floor.json")
	if err := os.WriteFile(path, []byte(`{
		"allow": ["after_reload"],
		"deny": {"exfiltrate": "SECRET_EXFIL"}
	}`), 0o600); err != nil {
		t.Fatalf("write policy: %v", err)
	}
	defer func() {
		adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy())
		ifc.ConfigureDefaultPolicy(ifc.Policy{})
	}()

	reload := policyReloader(path)
	if reload == nil {
		t.Fatal("policyReloader returned nil for a policy path")
	}
	resp, err := reload(context.Background())
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !resp.Reloaded || resp.Source != path || !strings.Contains(resp.Summary, "allow (exact)      : 1") {
		t.Fatalf("reload response = %+v, want source + summary", resp)
	}

	v := adjudicator.Default.Adjudicate(context.Background(), &abi.ToolCall{
		Tool: "after_reload",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	})
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("reloaded policy did not allow after_reload: %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := adjudicator.Default.Adjudicate(context.Background(), &abi.ToolCall{Tool: "exfiltrate"}); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("reloaded deny rule = %v/%s, want Deny/SECRET_EXFIL", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestResetTraceClearsIFCLedger(t *testing.T) {
	const trace = "cmd-reset-trace"
	ifc.Default.Reset(trace)
	ifc.Default.Raise(trace, abi.TaintTainted)
	t.Cleanup(func() { ifc.Default.Reset(trace) })

	if got := ifc.Default.Level(trace); got != abi.TaintTainted {
		t.Fatalf("precondition level = %v, want Tainted", got)
	}
	if err := resetTrace(context.Background(), trace); err != nil {
		t.Fatalf("resetTrace: %v", err)
	}
	if got := ifc.Default.Level(trace); got != abi.TaintTrusted {
		t.Fatalf("after reset level = %v, want Trusted", got)
	}
	if err := resetTrace(context.Background(), " "); err == nil {
		t.Fatal("blank trace reset should fail")
	}
}

func TestCmdDebugTombstonePersists(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	rec, _, err := cdb.IngestSession(ctx, "../../testdata/cdb/session.jsonl", "cmd-tombstone")
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}

	cmdDebug([]string{
		"--dir", dir,
		"--cmd", "tombstone",
		"--step", "0",
		"--reason", "agent marked stale context",
		"--requested-by", "agent:self-audit",
	})

	s, err := recall.Load(dir)
	if err != nil {
		t.Fatalf("load after tombstone: %v", err)
	}
	if !s.Tombstoned(0) {
		t.Fatal("debug tombstone command did not persist the tombstone")
	}
	if _, err := s.Resolve(ctx, 0); !errors.Is(err, recall.ErrTombstoned) {
		t.Fatalf("tombstoned page resolve: want ErrTombstoned, got %v", err)
	}
}

// TestResolveRequiredKeyFailsClosed locks in the #255 fix: a --…-key-env flag
// that names an unset/empty var must report ok=false so cmdServe fails closed,
// never warn-and-serve-unauthenticated. The lookup is injected (no process env).
func TestResolveRequiredKeyFailsClosed(t *testing.T) {
	env := map[string]string{
		"FAK_GATEWAY_KEY": "s3cret",
		"FAK_EMPTY":       "",
	}
	lookup := func(k string) string { return env[k] }

	cases := []struct {
		name    string
		envName string
		wantKey string
		wantOK  bool
	}{
		{"flag-unset-auth-not-requested", "", "", true},
		{"flag-set-secret-present", "FAK_GATEWAY_KEY", "s3cret", true},
		{"flag-set-var-empty-fail-closed", "FAK_EMPTY", "", false},
		{"flag-set-var-missing-fail-closed", "FAK_DOES_NOT_EXIST", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			key, ok := resolveRequiredKey(tc.envName, lookup)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v (envName=%q)", ok, tc.wantOK, tc.envName)
			}
			if key != tc.wantKey {
				t.Fatalf("key = %q, want %q", key, tc.wantKey)
			}
		})
	}
}
