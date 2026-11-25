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
	"time"

	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// TestUpLockPolicyEnforcement verifies that PolicyPath (--policy) in fak up --lock
// is loaded fail-closed at startup and enforced on MCP/tool calls rather than
// silently ignored (Fixes #14511).
func TestUpLockPolicyEnforcement(t *testing.T) {
	lockFile := writeContractLock(t, `[
    {
      "id": "support-mcp",
      "version": "1.0.0",
      "digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "source": "support-mcp-mock",
      "provider": "mcp",
      "provides": ["search_kb", "refund_payment", "export_data"]
    }
  ]`, `[]`)

	t.Run("missing_policy_fails_closed", func(t *testing.T) {
		missingPath := filepath.Join(t.TempDir(), "does-not-exist-policy.json")
		sup, err := NewSupervisor(Config{
			LockPath:   lockFile,
			PolicyPath: missingPath,
			Addr:       "127.0.0.1:0",
			Mock:       true,
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}

		if _, err := sup.DryRunTopology(); err == nil {
			t.Fatal("DryRunTopology succeeded with missing policy file; want fail-closed error")
		} else {
			var loadErr *policy.LoadError
			if !errors.As(err, &loadErr) || loadErr.Op != policy.LoadOpRead {
				t.Fatalf("DryRunTopology err = %v, want *policy.LoadError(read)", err)
			}
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err == nil {
			_ = sup.Shutdown(context.Background())
			t.Fatal("Start succeeded with missing policy file; want fail-closed error")
		} else {
			var loadErr *policy.LoadError
			if !errors.As(err, &loadErr) || loadErr.Op != policy.LoadOpRead {
				t.Fatalf("Start err = %v, want *policy.LoadError(read)", err)
			}
		}
		if addr := sup.Addr(); addr != "" {
			t.Fatalf("supervisor bound %q after missing policy error; want no listener", addr)
		}
		if sup.Adjudicator() != nil {
			t.Fatal("Adjudicator() non-nil after failed policy load")
		}
	})

	t.Run("invalid_policy_json_fails_closed", func(t *testing.T) {
		badPolicyPath := filepath.Join(t.TempDir(), "bad-policy.json")
		if err := os.WriteFile(badPolicyPath, []byte(`{not valid json`), 0600); err != nil {
			t.Fatalf("write bad policy: %v", err)
		}
		sup, err := NewSupervisor(Config{
			LockPath:   lockFile,
			PolicyPath: badPolicyPath,
			Addr:       "127.0.0.1:0",
			Mock:       true,
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}

		if _, err := sup.DryRunTopology(); err == nil {
			t.Fatal("DryRunTopology succeeded with invalid policy JSON; want fail-closed error")
		} else {
			var loadErr *policy.LoadError
			if !errors.As(err, &loadErr) || loadErr.Op != policy.LoadOpParse {
				t.Fatalf("DryRunTopology err = %v, want *policy.LoadError(parse)", err)
			}
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err == nil {
			_ = sup.Shutdown(context.Background())
			t.Fatal("Start succeeded with invalid policy JSON; want fail-closed error")
		} else {
			var loadErr *policy.LoadError
			if !errors.As(err, &loadErr) || loadErr.Op != policy.LoadOpParse {
				t.Fatalf("Start err = %v, want *policy.LoadError(parse)", err)
			}
		}
		if addr := sup.Addr(); addr != "" {
			t.Fatalf("supervisor bound %q after invalid policy error; want no listener", addr)
		}
	})

	t.Run("enforces_explicit_deny_and_permits_allowed_tool", func(t *testing.T) {
		policyPath := filepath.Join(t.TempDir(), "policy.json")
		policyJSON := `{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["search_kb"],
  "deny": {
    "refund_payment": "POLICY_BLOCK"
  }
}`
		if err := os.WriteFile(policyPath, []byte(policyJSON), 0600); err != nil {
			t.Fatalf("write policy: %v", err)
		}

		sup, err := NewSupervisor(Config{
			LockPath:   lockFile,
			PolicyPath: policyPath,
			Addr:       "127.0.0.1:0",
			Mock:       true,
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}

		spec, err := sup.DryRunTopology()
		if err != nil {
			t.Fatalf("DryRunTopology: %v", err)
		}
		if spec.PolicyPath != policyPath {
			t.Fatalf("TopologySpec.PolicyPath = %q, want %q", spec.PolicyPath, policyPath)
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err != nil {
			t.Fatalf("Start: %v", err)
		}
		t.Cleanup(func() {
			shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer shutdownCancel()
			_ = sup.Shutdown(shutdownCtx)
		})

		if sup.Adjudicator() == nil {
			t.Fatal("expected non-nil Adjudicator when PolicyPath is configured")
		}

		baseURL := "http://" + sup.Addr()
		refundTool := mcpbroker.NamespaceTool("support-mcp", "refund_payment")
		searchTool := mcpbroker.NamespaceTool("support-mcp", "search_kb")
		exportTool := mcpbroker.NamespaceTool("support-mcp", "export_data")

		type wireEvent struct {
			Event  string          `json:"event"`
			Tool   string          `json:"tool"`
			Reason string          `json:"reason"`
			Status string          `json:"status"`
			Error  string          `json:"error"`
			Result json.RawMessage `json:"result"`
		}
		postSession := func(t *testing.T, tool string) []wireEvent {
			t.Helper()
			payload, _ := json.Marshal(map[string]any{
				"goal": "policy check",
				"tool": tool,
				"args": map[string]any{"order_id": "ord-1"},
			})
			resp, err := http.Post(baseURL+"/v1/fak/agent/sessions", "application/json", strings.NewReader(string(payload)))
			if err != nil {
				t.Fatalf("POST /v1/fak/agent/sessions: %v", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want 200", resp.StatusCode)
			}
			var events []wireEvent
			scan := bufio.NewScanner(resp.Body)
			for scan.Scan() {
				var ev wireEvent
				if err := json.Unmarshal(scan.Bytes(), &ev); err != nil {
					t.Fatalf("decode NDJSON %q: %v", scan.Text(), err)
				}
				events = append(events, ev)
			}
			if err := scan.Err(); err != nil {
				t.Fatalf("scan NDJSON: %v", err)
			}
			return events
		}

		// 1. Explicitly denied tool must be refused and never executed.
		denyEvents := postSession(t, refundTool)
		seenDeny := false
		seenEndDenied := false
		for _, ev := range denyEvents {
			if ev.Event == "call" {
				t.Fatalf("denied tool %q was executed (emitted call event: %+v)", refundTool, ev)
			}
			if ev.Event == "deny" && ev.Tool == refundTool && ev.Reason == "POLICY_BLOCK" {
				seenDeny = true
			}
			if ev.Event == "session.end" && ev.Status == "denied" {
				seenEndDenied = true
			}
		}
		if !seenDeny || !seenEndDenied {
			t.Fatalf("expected deny(POLICY_BLOCK) and session.end(denied) for %q; got %+v", refundTool, denyEvents)
		}

		// 2. Direct broker RouteCall must also enforce the same policy filter.
		brokerResp, err := sup.Broker().RouteCall(context.Background(), mcpbroker.CallRequest{
			Tool:      refundTool,
			Arguments: json.RawMessage(`{"order_id":"ord-2"}`),
		})
		if err != nil {
			t.Fatalf("RouteCall(%q) unexpected err: %v", refundTool, err)
		}
		if !brokerResp.Filtered || brokerResp.FilterReason != "POLICY_BLOCK" {
			t.Fatalf("RouteCall(%q) = %+v, want Filtered=true FilterReason=POLICY_BLOCK", refundTool, brokerResp)
		}

		// 3. Unlisted tool under fail_closed posture must be denied by default.
		defaultDenyEvents := postSession(t, exportTool)
		seenDefaultDeny := false
		for _, ev := range defaultDenyEvents {
			if ev.Event == "call" {
				t.Fatalf("unlisted tool %q was executed under fail_closed posture", exportTool)
			}
			if ev.Event == "deny" && ev.Tool == exportTool {
				seenDefaultDeny = true
			}
		}
		if !seenDefaultDeny {
			t.Fatalf("expected deny event for unlisted tool %q; got %+v", exportTool, defaultDenyEvents)
		}

		// 4. Explicitly allowed tool must succeed.
		allowEvents := postSession(t, searchTool)
		seenCall := false
		seenEndOK := false
		for _, ev := range allowEvents {
			if ev.Event == "deny" {
				t.Fatalf("allowed tool %q was unexpectedly denied: %+v", searchTool, ev)
			}
			if ev.Event == "call" && ev.Tool == searchTool {
				seenCall = true
			}
			if ev.Event == "session.end" && ev.Status == "ok" {
				seenEndOK = true
			}
		}
		if !seenCall || !seenEndOK {
			t.Fatalf("expected call and session.end(ok) for allowed tool %q; got %+v", searchTool, allowEvents)
		}

		// 5. Memory journal must record the policy denial audit entry.
		recordedDeny := false
		for _, entry := range sup.Memory().Entries() {
			if entry.Type == "call.denied" && entry.Tool == refundTool {
				recordedDeny = true
				break
			}
		}
		if !recordedDeny {
			t.Fatalf("expected call.denied memory journal entry for %q", refundTool)
		}
	})
}
