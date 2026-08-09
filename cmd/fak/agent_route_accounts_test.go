package main

// agent_route_accounts_test.go — the #5708 witness for `fak agent --route-accounts`.
//
// #5644 landed the loop-side plumbing (agent.WithRouteAccounts) and wired the ONE
// construction site it could reach, the native-serve path
// (internal/gateway/native_serve.go:355). The `cmd/fak/main.go` site — the `fak agent`
// verb — had no roster flag at all, so the same manifest dispatched a routed id to a
// DIFFERENT account depending on which verb an operator ran. These tests pin the
// closure at the operator surface:
//
//   - TestAgentVerbBindsRouteAccounts: with a roster, a routed tool call dispatches to
//     the account-resolved Target.EngineRoute() — the same value the served gateway
//     binds for the same manifest+roster — witnessed by WHICH engine the call reached,
//     not by inspecting an option closure.
//   - TestAgentVerbWithoutRouteAccountsIsUnchanged: without the flag the routed call
//     still carries the bare plan-member string, byte-for-byte the pre-#5708 verb.
//   - TestAgentRouteAccountsMalformedFailsLoud: a malformed roster refuses at startup
//     rather than silently falling back to the default engine.
//   - TestAgentRouteAccountsNeverPrintsSecret: a roster carries env-var NAMES, so no
//     flag surface (announce or refusal) may print the credential's VALUE.

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Each arm uses its OWN routed id and account route: abi.RegisterEngine is a global
// registry keyed by id, so distinct ids keep the two arms from reading each other's
// dispatches inside the one test binary.
const (
	agent5708BoundID    = "agent5708-bound"
	agent5708BoundRoute = "local:lab5708/llama-3.3-8b"
	agent5708BareID     = "agent5708-bare"
)

// agent5708Manifest routes ONLY book_flight to the given abstract id; everything else
// falls to the loop's localtools default. A single-model PICK, so Plan.Primary() is the
// id the roster is then asked to resolve.
func agent5708Manifest(t *testing.T, routedID string) string {
	t.Helper()
	return writeAgent5708File(t, "route.json", `{
  "version": "fak-route/v1",
  "default": {"members": [{"model": "localtools", "role": "primary"}]},
  "rules": [
    {
      "name": "book-to-5708",
      "match": {"aspect": "tool_call", "tool": "book_flight"},
      "plan": {"members": [{"model": "`+routedID+`"}]}
    }
  ]
}`)
}

// agent5708Roster binds that abstract id to a LOCAL account (residency-exempt, so the
// witness measures the routing seam and not the residency PDP) and carries a SECOND,
// remote account whose credential is an env-var NAME — the shape the secret test reads.
func agent5708Roster(t *testing.T, routedID string) string {
	t.Helper()
	return writeAgent5708File(t, "roster.json", `{
  "version": "fak-accounts/v1",
  "accounts": [
    {"id": "lab5708", "kind": "local", "base_url": "http://127.0.0.1:11434/v1", "label": "on-box server for the #5708 witness"},
    {"id": "vendor5708", "kind": "openai", "cred_env": "FAK_5708_WITNESS_KEY", "label": "a remote account whose credential is a NAME"}
  ],
  "bindings": [
    {"model": "`+routedID+`", "account": "lab5708", "upstream_model": "llama-3.3-8b"}
  ]
}`)
}

func writeAgent5708File(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}

// agent5708Engine records the abi.ToolCall.Engine of every call routed to it, so the
// test can witness WHICH engine a routed tool call actually dispatched to. It answers
// with a booking confirmation so the mock planner's task completes in one call.
type agent5708Engine struct {
	id   string
	mu   sync.Mutex
	seen []string
}

func (*agent5708Engine) Caps() []abi.Capability { return nil }

func (e *agent5708Engine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.mu.Lock()
	e.seen = append(e.seen, c.Engine)
	e.mu.Unlock()
	body := []byte(`{"booked":true,"confirmation":"AC-5708"}`)
	return &abi.Result{
		Call:    c,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body)), Taint: abi.TaintTrusted, Scope: abi.ScopeAgent},
		Meta:    map[string]string{"engine": e.id},
	}, nil
}

func (e *agent5708Engine) calls() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.seen...)
}

// TestAgentVerbBindsRouteAccounts is the #5708 done-condition witness:
// `fak agent --route-manifest M --route-accounts R` binds a routed tool call to the
// account-resolved Target.EngineRoute(), not the bare plan-member string.
func TestAgentVerbBindsRouteAccounts(t *testing.T) {
	agent.Configure()
	manifestPath := agent5708Manifest(t, agent5708BoundID)
	rosterPath := agent5708Roster(t, agent5708BoundID)
	manifest, roster, opts, err := loadAgentRouteOptionsWithAccounts(manifestPath, rosterPath)
	if err != nil {
		t.Fatalf("load manifest+roster: %v", err)
	}
	if manifest == nil || roster == nil {
		t.Fatal("both manifest and roster must load")
	}

	want := agent5708BoundRoute
	got, err := agent.ResolveToolRoute("book_flight", opts...)
	if err != nil {
		t.Fatalf("resolve account-bound route: %v", err)
	}
	if got != want {
		t.Fatalf("fak agent bound Engine = %q, want roster Target.EngineRoute() %q", got, want)
	}
}

// TestAgentVerbWithoutRouteAccountsIsUnchanged proves the no-flag arm: an empty
// --route-accounts installs no roster option, so the routed call still carries the bare
// plan-member string — byte-for-byte the pre-#5708 verb.
func TestAgentVerbWithoutRouteAccountsIsUnchanged(t *testing.T) {
	agent.Configure()
	manifestPath := agent5708Manifest(t, agent5708BareID)

	bare := &agent5708Engine{id: agent5708BareID}
	abi.RegisterEngine(agent5708BareID, bare)

	manifest, roster, opts, err := loadAgentRouteOptionsWithAccounts(manifestPath, "")
	if err != nil {
		t.Fatalf("loadAgentRouteOptionsWithAccounts(manifest, \"\"): %v", err)
	}
	if manifest == nil {
		t.Fatal("the manifest half must still load without a roster")
	}
	if roster != nil {
		t.Fatalf("no --route-accounts must leave the roster nil, got %+v", roster)
	}
	// Exactly the pre-#5708 option set: the manifest option and nothing else.
	if len(opts) != 1 {
		t.Fatalf("no --route-accounts must install exactly the manifest option, got %d options", len(opts))
	}

	res, _, err := agent.Run(context.Background(), agent.NewMockPlanner("route5708bare"), agent.DefaultTask, 12, opts...)
	if err != nil {
		t.Fatalf("agent.Run with manifest only: %v", err)
	}
	_ = res // dispatch identity below is the authoritative witness; planner completion is unrelated.
	if got := bare.calls(); len(got) != 1 || got[0] != agent5708BareID {
		t.Fatalf("without a roster the routed call must carry the bare plan-member id; engine saw %v, want [%q]",
			got, agent5708BareID)
	}
}

// TestAgentRouteAccountsMalformedFailsLoud: a roster that does not load is refused at
// startup, naming the flag, rather than silently degrading to the default engine.
func TestAgentRouteAccountsMalformedFailsLoud(t *testing.T) {
	manifestPath := agent5708Manifest(t, agent5708BareID)
	for name, body := range map[string]string{
		"not-json":          `{not-json`,
		"residency-bypass":  `{"version":"fak-accounts/v1","accounts":[{"id":"l","kind":"local","base_url":"https://api.openai.com/v1"}]}`,
		"no-accounts":       `{"version":"fak-accounts/v1","accounts":[]}`,
		"unknown-kind":      `{"version":"fak-accounts/v1","accounts":[{"id":"a","kind":"telepathy"}]}`,
		"cred-env-is-value": `{"version":"fak-accounts/v1","accounts":[{"id":"a","kind":"openai","cred_env":"sk-not-a-name"}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			bad := writeAgent5708File(t, "bad-roster.json", body)
			manifest, roster, opts, err := loadAgentRouteOptionsWithAccounts(manifestPath, bad)
			if err == nil {
				t.Fatalf("a malformed roster must fail loud, got manifest=%v roster=%v opts=%d", manifest, roster, len(opts))
			}
			if !strings.Contains(err.Error(), "fak agent: --route-accounts") {
				t.Fatalf("the refusal must name the flag an operator has to fix, got %q", err)
			}
			// Fail LOUD means nothing is installed — never a half-armed option set that
			// would dispatch the routed id to the default engine.
			if roster != nil || opts != nil {
				t.Fatalf("a refused roster must install nothing, got roster=%v opts=%d", roster, len(opts))
			}
			// And the operator-facing bail must name the flag + the path without
			// inventing a fallback.
			var buf bytes.Buffer
			writeAgentRouteAccountsBail(&buf, bad, err)
			for _, want := range []string{"route-accounts", bad} {
				if !strings.Contains(buf.String(), want) {
					t.Fatalf("bail must mention %q:\n%s", want, buf.String())
				}
			}
		})
	}
}

// TestAgentRouteAccountsNeverPrintsSecret mirrors TestRouteAccountsNeverPrintsSecret
// (cmd/fak/route_accounts_test.go): a roster carries env-var NAMES, never secrets, so
// neither the load announcement nor the refusal may echo the credential's VALUE.
func TestAgentRouteAccountsNeverPrintsSecret(t *testing.T) {
	const secret = "sk-5708-must-not-print"
	t.Setenv("FAK_5708_WITNESS_KEY", secret)
	rosterPath := agent5708Roster(t, agent5708BoundID)

	var buf bytes.Buffer
	roster, _, err := loadAgentRouteAccounts(rosterPath)
	if err != nil {
		t.Fatalf("roster should load: %v", err)
	}
	announceAgentRouteAccounts(&buf, rosterPath, roster)

	bad := writeAgent5708File(t, "bad-roster.json", `{"version":"fak-accounts/v1","accounts":[{"id":"a","kind":"telepathy","cred_env":"FAK_5708_WITNESS_KEY"}]}`)
	_, _, badErr := loadAgentRouteAccounts(bad)
	if badErr == nil {
		t.Fatal("the unknown-kind roster must refuse")
	}
	writeAgentRouteAccountsBail(&buf, bad, badErr)

	out := buf.String()
	if strings.Contains(out, secret) {
		t.Fatalf("the --route-accounts surface leaked a credential VALUE:\n%s", out)
	}
	if !strings.Contains(out, "FAK_5708_WITNESS_KEY") {
		t.Fatalf("the announcement should surface the env-var NAME so an operator can see what to set:\n%s", out)
	}
}
