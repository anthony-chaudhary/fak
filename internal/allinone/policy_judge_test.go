package allinone

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const policyTestServer = "policy-mcp-server"

func writePolicyLock(t *testing.T) string {
	t.Helper()
	return writeContractLock(t, `[
    {
      "id": "`+policyTestServer+`",
      "version": "1.0.0",
      "digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "source": "definitely-missing-binary-policy",
      "provider": "mcp",
      "provides": ["safe", "danger"]
    }
  ]`, `[]`)
}

func writePolicyManifest(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(p, []byte(body), 0600); err != nil {
		t.Fatalf("write policy: %v", err)
	}
	return p
}

// postAgentTool drives one agent-session tool call and returns the decoded
// NDJSON events.
func postAgentTool(t *testing.T, addr, tool string) []map[string]any {
	t.Helper()
	body := `{"goal":"policy probe","tool":` + jsonQuote(tool) + `,"args":{"q":1}}`
	resp, err := http.Post("http://"+addr+"/v1/fak/agent/sessions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST session: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("session status = %d, want 200", resp.StatusCode)
	}
	var events []map[string]any
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("decode event %q: %v", line, err)
		}
		events = append(events, ev)
	}
	return events
}

func hasEvent(events []map[string]any, kind, tool string) bool {
	for _, ev := range events {
		if ev["event"] == kind && ev["tool"] == tool {
			return true
		}
	}
	return false
}

// TestUpLockPolicyEnforcesManifest proves --policy reaches locked
// adjudication: the manifest-denied tool never dispatches, the allowed one does.
func TestUpLockPolicyEnforcesManifest(t *testing.T) {
	safe := mcpbroker.NamespaceTool(policyTestServer, "safe")
	danger := mcpbroker.NamespaceTool(policyTestServer, "danger")
	policyPath := writePolicyManifest(t, `{"version":"fak-policy/v1","allow":[`+jsonQuote(safe)+`],"deny":{`+jsonQuote(danger)+`:"POLICY_BLOCK"}}`)

	sup, err := NewSupervisor(Config{
		LockPath:   writePolicyLock(t),
		Addr:       "127.0.0.1:0",
		PolicyPath: policyPath,
		Engine:     "mock",
		Mock:       true,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer func() { _ = sup.Shutdown(context.Background()) }()

	denied := postAgentTool(t, sup.Addr(), danger)
	if hasEvent(denied, "call", danger) {
		t.Fatalf("manifest-denied tool %q dispatched under --lock --policy; events=%v", danger, denied)
	}
	if !hasEvent(denied, "denied", danger) && !hasEvent(denied, "deny", danger) {
		t.Fatalf("no denied event for %q; events=%v", danger, denied)
	}
	for _, ev := range denied {
		if (ev["event"] == "denied" || ev["event"] == "deny") && ev["reason"] != "POLICY_BLOCK" {
			t.Fatalf("denied reason = %v, want POLICY_BLOCK", ev["reason"])
		}
	}

	allowed := postAgentTool(t, sup.Addr(), safe)
	if !hasEvent(allowed, "call", safe) {
		t.Fatalf("manifest-allowed tool %q not dispatched under --lock --policy; events=%v", safe, allowed)
	}
}

// TestUpLockPolicyLoadFailureFailsClosed proves an unreadable or malformed
// --policy refuses startup instead of running with no policy.
func TestUpLockPolicyLoadFailureFailsClosed(t *testing.T) {
	cases := map[string]string{
		"missing":   filepath.Join(t.TempDir(), "absent-policy.json"),
		"malformed": writePolicyManifest(t, `{"version":"fak-policy/v1","deny":`),
		"bad_reason": writePolicyManifest(t,
			`{"version":"fak-policy/v1","deny":{"x":"NOT_A_REGISTERED_REASON"}}`),
	}
	for name, policyPath := range cases {
		t.Run(name, func(t *testing.T) {
			sup, err := NewSupervisor(Config{
				LockPath:   writePolicyLock(t),
				Addr:       "127.0.0.1:0",
				PolicyPath: policyPath,
				Engine:     "mock",
				Mock:       true,
			})
			if err != nil {
				t.Fatalf("NewSupervisor: %v", err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			err = sup.Start(ctx)
			defer func() { _ = sup.Shutdown(context.Background()) }()
			var loadErr *policy.LoadError
			if !errors.As(err, &loadErr) {
				t.Fatalf("Start err = %v, want a *policy.LoadError", err)
			}
			if addr := sup.Addr(); addr != "" {
				t.Fatalf("supervisor bound %q despite an unloadable policy", addr)
			}
			if _, err := sup.DryRunTopology(); !errors.As(err, &loadErr) {
				t.Fatalf("DryRunTopology err = %v, want a *policy.LoadError", err)
			}
		})
	}
}

func startPolicySupervisor(t *testing.T, manifest string) *Supervisor {
	t.Helper()
	sup, err := NewSupervisor(Config{
		LockPath:   writePolicyLock(t),
		Addr:       "127.0.0.1:0",
		PolicyPath: writePolicyManifest(t, manifest),
		Engine:     "mock",
		Mock:       true,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	if err := sup.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = sup.Shutdown(context.Background()) })
	return sup
}

// TestUpLockPolicyBareAllowDoesNotWidenToNamespaced: allowing a bare tool name
// must not admit a same-named tool on an MCP server the manifest never named.
func TestUpLockPolicyBareAllowDoesNotWidenToNamespaced(t *testing.T) {
	danger := mcpbroker.NamespaceTool(policyTestServer, "danger")
	sup := startPolicySupervisor(t, `{"version":"fak-policy/v1","posture":"fail_closed","allow":["danger"]}`)
	events := postAgentTool(t, sup.Addr(), danger)
	if hasEvent(events, "call", danger) {
		t.Fatalf("bare allow %q admitted namespaced %q; events=%v", "danger", danger, events)
	}
}
