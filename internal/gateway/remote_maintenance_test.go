package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

type remoteMaintenanceAdapterRequest struct {
	Schema          string             `json:"schema"`
	Phase           string             `json:"phase"`
	ProfileID       string             `json:"profile_id"`
	OperationDigest string             `json:"operation_digest"`
	Grant           *remotemaint.Grant `json:"grant,omitempty"`
}

type remoteMaintenanceAdapterResponse struct {
	Schema  string               `json:"schema"`
	Grant   *remotemaint.Grant   `json:"grant,omitempty"`
	Receipt *remotemaint.Receipt `json:"receipt,omitempty"`
}

type remoteMaintenanceAdjudicator struct {
	allow bool
	call  atomic.Pointer[abi.ToolCall]
}

type countingRemoteMaintenanceEngine struct{ calls atomic.Int32 }

func (*countingRemoteMaintenanceEngine) Caps() []abi.Capability { return nil }
func (e *countingRemoteMaintenanceEngine) Complete(ctx context.Context, call *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	return echoEngine{}.Complete(ctx, call)
}

func (*remoteMaintenanceAdjudicator) Caps() []abi.Capability { return nil }
func (a *remoteMaintenanceAdjudicator) Adjudicate(_ context.Context, call *abi.ToolCall) abi.Verdict {
	copyCall := *call
	copyCall.Meta = make(map[string]string, len(call.Meta))
	for key, value := range call.Meta {
		copyCall.Meta[key] = value
	}
	a.call.Store(&copyCall)
	if a.allow && call.Tool == remoteMaintenanceTool {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "remote-maintenance-test"}
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "remote-maintenance-test"}
}

type fakeRemoteMaintenanceAuthority struct {
	grant                      remotemaint.Grant
	receipt                    remotemaint.Receipt
	resolveErr, validateErr    error
	executeErr                 error
	resolve, validate, execute atomic.Int32
}

type remoteMaintenanceResultDeny struct{}

func (remoteMaintenanceResultDeny) Caps() []abi.Capability { return nil }
func (remoteMaintenanceResultDeny) Admit(_ context.Context, _ *abi.ToolCall, result *abi.Result) abi.Verdict {
	result.Payload = abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"attacker":"result-admitter-secret"}`)}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "remote-result-test"}
}

func (a *fakeRemoteMaintenanceAuthority) Resolve(context.Context, remotemaint.Request) (remotemaint.Grant, error) {
	a.resolve.Add(1)
	return a.grant, a.resolveErr
}
func (a *fakeRemoteMaintenanceAuthority) Validate(context.Context, remotemaint.Grant) error {
	a.validate.Add(1)
	return a.validateErr
}
func (a *fakeRemoteMaintenanceAuthority) Execute(context.Context, remotemaint.Grant) (remotemaint.Receipt, error) {
	a.execute.Add(1)
	return a.receipt, a.executeErr
}

func validRemoteMaintenanceGrant(now time.Time) remotemaint.Grant {
	return remotemaint.Grant{
		ProfileID: "ops-profile", OperationDigest: strings.Repeat("1", 64), ContextID: strings.Repeat("2", 64),
		Target: "maintenance.example.invalid", HostKey: "SHA256:" + base64.RawStdEncoding.EncodeToString(make([]byte, 32)),
		Operation: remotemaint.OperationSystemdProfileInstallReload, PayloadSHA256: strings.Repeat("3", 64),
		Holder: "operator", BootID: "boot-1", PriorStateSHA256: strings.Repeat("4", 64),
		Paths: []string{"/etc/systemd/system/fak.service"}, PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "maintenance-stage"}},
		Units: []string{"fak.service"}, LeasePins: []remotemaint.LeasePin{{NodeID: "node-1", ID: "lease-1", Epoch: 1, ExpiresAt: now.Add(10 * time.Minute).UTC()}},
		ExpiresAt: now.Add(5 * time.Minute).UTC(),
	}
}

func newRemoteMaintenanceTestServer(t *testing.T, allow bool) (*Server, *fakeRemoteMaintenanceAuthority, *remoteMaintenanceAdjudicator) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	adj := &remoteMaintenanceAdjudicator{allow: allow}
	abi.RegisterAdjudicator(0, adj)
	srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-test", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	grant := validRemoteMaintenanceGrant(time.Now().UTC())
	receipt := remotemaint.Receipt{
		Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(grant), ProfileID: grant.ProfileID,
		OperationDigest: grant.OperationDigest, ContextID: grant.ContextID, Target: grant.Target, HostKey: grant.HostKey,
		Operation: grant.Operation, BootID: grant.BootID, StartedAt: time.Now().Add(-time.Second).UTC(),
		EndedAt: time.Now().UTC(), Outcome: "applied", Verified: true,
	}
	authority := &fakeRemoteMaintenanceAuthority{grant: grant, receipt: receipt}
	srv.SetRemoteMaintenanceAuthority(authority)
	return srv, authority, adj
}

func remoteMaintenanceArgs() string {
	return `{"profile_id":"ops-profile","operation_digest":"` + strings.Repeat("1", 64) + `"}`
}

func remoteMaintenanceMCPResponse(t *testing.T, result any) SyscallResponse {
	response, _ := remoteMaintenanceMCPWire(t, result)
	return response
}

func remoteMaintenanceMCPWire(t *testing.T, result any) (SyscallResponse, bool) {
	t.Helper()
	wrapped, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("MCP result=%#v", result)
	}
	blocks, ok := wrapped["content"].([]map[string]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("MCP result=%#v", result)
	}
	var response SyscallResponse
	if err := json.Unmarshal([]byte(blocks[0]["text"].(string)), &response); err != nil {
		t.Fatal(err)
	}
	isError, _ := wrapped["isError"].(bool)
	return response, isError
}

func TestRemoteMaintenanceRegisteredSearchableAndDirectlyCallable(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	found := false
	for _, descriptor := range srv.exposedToolDescriptors() {
		if descriptor["name"] == remoteMaintenanceTool {
			found = true
			schema := string(descriptor["inputSchema"].(json.RawMessage))
			for _, forbidden := range []string{"target", "path", "command", "read_only", "meta"} {
				if strings.Contains(schema, `"`+forbidden+`"`) {
					t.Fatalf("descriptor exposes caller authority %q: %s", forbidden, schema)
				}
			}
		}
	}
	if !found {
		t.Fatal("tools/list registry missing fak_remote_maintenance")
	}
	search, err := srv.toolsSearch(ToolsSearchRequest{Tool: remoteMaintenanceTool, DetailLevel: "full"})
	if err != nil || len(search.Tools) != 1 || search.Tools[0]["name"] != remoteMaintenanceTool {
		t.Fatalf("tool search=%+v err=%v", search, err)
	}
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_remote_maintenance","arguments":`+remoteMaintenanceArgs()+`}`))
	if rpcErr != nil {
		t.Fatalf("direct MCP call: %+v", rpcErr)
	}
	response := remoteMaintenanceMCPResponse(t, result)
	if response.Verdict.Kind != "ALLOW" || response.Result == nil {
		t.Fatalf("direct response=%+v", response)
	}
	if authority.execute.Load() != 1 {
		t.Fatalf("direct call executions=%d", authority.execute.Load())
	}
}

func TestRemoteMaintenanceFakAdjudicateAndSyscallEntrypoints(t *testing.T) {
	t.Run("adjudicate", func(t *testing.T) {
		srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
		params := `{"name":"fak_adjudicate","arguments":{"tool":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `,"read_only":true}}`
		result, rpcErr := srv.callTool(context.Background(), json.RawMessage(params))
		if rpcErr != nil {
			t.Fatalf("fak_adjudicate: %+v", rpcErr)
		}
		response := remoteMaintenanceMCPResponse(t, result)
		if response.Verdict.Kind != "ALLOW" || response.Receipt == nil || response.Receipt.Execution != "not_executed" {
			t.Fatalf("adjudicate response=%+v", response)
		}
		if authority.execute.Load() != 0 {
			t.Fatalf("adjudicate executions=%d", authority.execute.Load())
		}
	})
	t.Run("syscall", func(t *testing.T) {
		srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
		params := `{"name":"fak_syscall","arguments":{"tool":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `,"read_only":true}}`
		result, rpcErr := srv.callTool(context.Background(), json.RawMessage(params))
		if rpcErr != nil {
			t.Fatalf("fak_syscall: %+v", rpcErr)
		}
		response := remoteMaintenanceMCPResponse(t, result)
		if response.Verdict.Kind != "ALLOW" || response.Result == nil {
			t.Fatalf("syscall response=%+v", response)
		}
		if authority.execute.Load() != 1 {
			t.Fatalf("syscall executions=%d", authority.execute.Load())
		}
	})
}

func TestRemoteMaintenanceAdjudicateValidatesWithoutExecuting(t *testing.T) {
	srv, authority, adj := newRemoteMaintenanceTestServer(t, true)
	verdict, _, err := srv.adjudicate(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), true, "", "remote-adjudicate")
	if err != nil || verdict.Kind != "ALLOW" {
		t.Fatalf("adjudicate verdict=%+v err=%v", verdict, err)
	}
	if authority.resolve.Load() != 1 || authority.validate.Load() != 1 || authority.execute.Load() != 0 {
		t.Fatalf("authority calls resolve=%d validate=%d execute=%d", authority.resolve.Load(), authority.validate.Load(), authority.execute.Load())
	}
	call := adj.call.Load()
	if call == nil || call.Meta["readOnlyHint"] != "false" || call.Meta["destructive"] != "true" {
		t.Fatalf("caller read_only widened maintenance call: %+v", call)
	}
}

func TestRemoteMaintenanceSyscallExecutesPreparedGrantOnceAndAdmitsResult(t *testing.T) {
	srv, authority, adj := newRemoteMaintenanceTestServer(t, true)
	verdict, result, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), true, "", "remote-syscall")
	if err != nil || verdict.Kind != "ALLOW" || result == nil || result.Status != "OK" {
		t.Fatalf("syscall verdict=%+v result=%+v err=%v", verdict, result, err)
	}
	if authority.resolve.Load() != 1 || authority.validate.Load() != 2 || authority.execute.Load() != 1 {
		t.Fatalf("authority calls resolve=%d validate=%d execute=%d", authority.resolve.Load(), authority.validate.Load(), authority.execute.Load())
	}
	var receipt remotemaint.Receipt
	if err := json.Unmarshal([]byte(result.Content), &receipt); err != nil || receipt.GrantDigest == "" {
		t.Fatalf("result receipt=%+v err=%v", receipt, err)
	}
	policyCall := adj.call.Load()
	if policyCall == nil || policyCall.Meta["remote_maintenance_wire_args_sha256"] == "" || policyCall.Meta["remote_maintenance_grant_sha256"] == "" || policyCall.Meta["remote_maintenance_policy_facts_sha256"] == "" || policyCall.Meta["remote_maintenance_resource_index"] == "" {
		t.Fatalf("trusted-scope decision missing audit hashes: %+v", policyCall)
	}
	if args := string(resolveBytes(context.Background(), policyCall.Args)); !strings.Contains(args, `"remote_paths"`) || !strings.Contains(args, `/etc/systemd/system/fak.service`) {
		t.Fatalf("trusted-scope decision args=%s", args)
	}
}

func TestRemoteMaintenanceKernelDenyExecutesZero(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, false)
	verdict, result, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), false, "", "remote-deny")
	if err != nil || verdict.Kind != "DENY" || result != nil {
		t.Fatalf("syscall verdict=%+v result=%+v err=%v", verdict, result, err)
	}
	if authority.execute.Load() != 0 {
		t.Fatalf("kernel DENY executed authority %d times", authority.execute.Load())
	}
	if authority.resolve.Load() != 0 {
		t.Fatalf("initial kernel DENY resolved authority %d times", authority.resolve.Load())
	}
}

func TestRemoteMaintenancePolicyRecognizedAliasesStillCrossCanonicalScopeFloors(t *testing.T) {
	aliases := []string{
		"functions." + remoteMaintenanceTool,
		"mcp__fak__" + remoteMaintenanceTool,
		"mcp__fak_guard__" + remoteMaintenanceTool,
	}
	for _, alias := range aliases {
		for _, scope := range []struct {
			name       string
			path       string
			wantReason string
		}{
			{name: "credential", path: "/fixture/.aws/credentials", wantReason: "POLICY_BLOCK"},
			{name: "self-config", path: "/root/.fak/guard/remote-maintenance.json", wantReason: "SELF_MODIFY"},
		} {
			t.Run(alias+"/"+scope.name, func(t *testing.T) {
				abi.ResetForTest()
				abi.RegisterRegionBackend(inlineBackend{})
				abi.RegisterEngine("test", echoEngine{})
				abi.RegisterEngine("mock", engine.MockEngine)
				floor := adjudicator.Policy{
					Posture:          adjudicator.PostureFailClosed,
					Allow:            map[string]bool{alias: true, remoteMaintenanceTool: true},
					SelfModifyGlobs:  []string{".fak/guard/"},
					BlockedPathGlobs: []string{".aws/"},
				}
				runtime := policy.Runtime{Adjudicator: floor}
				abi.RegisterAdjudicator(0, adjudicator.New(floor))
				srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-alias-floor", VDSO: true, PolicyRuntime: &runtime})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(srv.Close)
				grant := validRemoteMaintenanceGrant(time.Now().UTC())
				grant.Paths = []string{scope.path}
				authority := &fakeRemoteMaintenanceAuthority{grant: grant}
				srv.SetRemoteMaintenanceAuthority(authority)
				verdict, result, err := srv.syscall(context.Background(), alias, remoteMaintenanceArgs(), false, "", "remote-alias-floor")
				if err != nil || verdict.Kind != "DENY" || verdict.Reason != scope.wantReason || result != nil {
					t.Fatalf("alias scope verdict=%+v result=%+v err=%v", verdict, result, err)
				}
				if authority.resolve.Load() != 1 || authority.execute.Load() != 0 {
					t.Fatalf("authority resolve=%d execute=%d", authority.resolve.Load(), authority.execute.Load())
				}
			})
		}

		t.Run(alias+"/initial-deny", func(t *testing.T) {
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterEngine("mock", engine.MockEngine)
			floor := adjudicator.Policy{
				Posture: adjudicator.PostureFailClosed,
				Allow:   map[string]bool{remoteMaintenanceTool: true},
				Deny:    map[string]abi.ReasonCode{alias: abi.ReasonPolicyBlock},
			}
			runtime := policy.Runtime{Adjudicator: floor}
			abi.RegisterAdjudicator(0, adjudicator.New(floor))
			srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-alias-deny", VDSO: true, PolicyRuntime: &runtime})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			grant := validRemoteMaintenanceGrant(time.Now().UTC())
			authority := &fakeRemoteMaintenanceAuthority{grant: grant}
			srv.SetRemoteMaintenanceAuthority(authority)
			verdict, result, err := srv.syscall(context.Background(), alias, remoteMaintenanceArgs(), false, "", "remote-alias-deny")
			if err != nil || verdict.Kind != "DENY" || result != nil {
				t.Fatalf("initial alias verdict=%+v result=%+v err=%v", verdict, result, err)
			}
			if authority.resolve.Load() != 0 || authority.execute.Load() != 0 {
				t.Fatalf("initial alias DENY reached authority resolve=%d execute=%d", authority.resolve.Load(), authority.execute.Load())
			}
		})
	}
}

func TestRemoteMaintenanceExemptionDoesNotAdmitOrdinaryBash(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	engine := &countingRemoteMaintenanceEngine{}
	abi.RegisterEngine("test", engine)
	abi.RegisterEngine("mock", engine)
	abi.RegisterAdjudicator(0, adjudicator.New(adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   map[string]bool{remoteMaintenanceTool: true, "Bash": true},
	}))
	srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-bash-floor", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	grant := validRemoteMaintenanceGrant(time.Now().UTC())
	authority := &fakeRemoteMaintenanceAuthority{grant: grant}
	srv.SetRemoteMaintenanceAuthority(authority)
	for _, tc := range []struct {
		name    string
		command string
	}{
		{name: "computed-program-and-flag", command: `p=$(printf fak); f=--remote-$(printf maintenance); "$p" guard allow "$f" reviewed.json --ttl 5m`},
		{name: "computed-flag-with-session-strip", command: `m=--remote; m+=-maintenance; env -u FAK_GUARD_SESSION_ID fak guard allow "$m" reviewed.json --ttl 15m`},
		{name: "opaque-ssh", command: `ssh maintenance.example.invalid true`},
	} {
		verdict, result, err := srv.syscall(context.Background(), "Bash", `{"command":`+string(mustJSON(t, tc.command))+`}`, true, "", "ordinary-bash-"+tc.name)
		if err != nil || verdict.Kind != "DENY" || verdict.Reason != "DEFAULT_DENY" {
			t.Fatalf("%s verdict=%+v result=%+v err=%v", tc.name, verdict, result, err)
		}
		if engine.calls.Load() != 0 {
			t.Fatalf("%s executed engine %d times", tc.name, engine.calls.Load())
		}
	}
	if authority.resolve.Load() != 0 || authority.execute.Load() != 0 {
		t.Fatalf("ordinary Bash reached maintenance authority resolve=%d execute=%d", authority.resolve.Load(), authority.execute.Load())
	}
	verdict, result, err := srv.syscall(context.Background(), "Bash", `{"command":"git status --short"}`, true, "", "known-readonly-bash")
	if err != nil || verdict.Kind != "ALLOW" || result == nil || result.Status != "OK" {
		t.Fatalf("known read-only Bash verdict=%+v result=%+v err=%v", verdict, result, err)
	}
	if engine.calls.Load() != 1 {
		t.Fatalf("known read-only Bash executed engine %d times, want 1", engine.calls.Load())
	}
}

func TestRemoteMaintenanceTrustedScopeCrossesApplicablePathFloors(t *testing.T) {
	for _, tc := range []struct {
		name       string
		paths      []string
		target     string
		entry      string
		wantKind   string
		wantReason string
		wantExec   int32
	}{
		{name: "host-credential-direct-mcp", paths: []string{"/etc/passwd"}, entry: remoteMaintenanceTool, wantKind: "DENY", wantReason: "POLICY_BLOCK"},
		{name: "any-user-credential-direct-mcp", paths: []string{"/fixture/.aws/credentials"}, entry: remoteMaintenanceTool, wantKind: "DENY", wantReason: "POLICY_BLOCK"},
		{name: "guard-self-config-fak-syscall", paths: []string{"/root/.fak/guard/remote-maintenance.json"}, entry: "fak_syscall", wantKind: "DENY", wantReason: "SELF_MODIFY"},
		{name: "mixed-benign-then-credential", paths: []string{"/etc/systemd/system/fak.service", "/fixture/.aws/credentials"}, entry: remoteMaintenanceTool, wantKind: "DENY", wantReason: "POLICY_BLOCK"},
		{name: "metadata-target-userinfo-port", paths: []string{"/etc/systemd/system/fak.service"}, target: "operator@169.254.169.254:22", entry: remoteMaintenanceTool, wantKind: "DENY", wantReason: "EGRESS_BLOCK"},
		{name: "systemd-credential-basename", paths: []string{"/etc/systemd/system/fak.service.d/.env.conf"}, entry: remoteMaintenanceTool, wantKind: "DENY", wantReason: "POLICY_BLOCK"},
		{name: "legitimate-systemd", paths: []string{"/etc/systemd/system/fak.service"}, entry: remoteMaintenanceTool, wantKind: "ALLOW", wantExec: 1},
		{name: "legitimate-systemd-dropin", paths: []string{"/etc/systemd/system/fak.service.d/90-maintenance.conf"}, entry: remoteMaintenanceTool, wantKind: "ALLOW", wantExec: 1},
		{name: "legitimate-systemd-dropin-parent", paths: []string{"/etc/systemd/system/fak.service.d"}, entry: remoteMaintenanceTool, wantKind: "ALLOW", wantExec: 1},
		{name: "legitimate-systemd-recovery-sibling", paths: []string{"/etc/systemd/system/fak.service.d/90-maintenance.conf.fak-recovery-tmp"}, entry: remoteMaintenanceTool, wantKind: "ALLOW", wantExec: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			abi.ResetForTest()
			abi.RegisterReason(egressfloor.ReasonEgressBlock, egressfloor.ReasonEgressBlockName)
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterEngine("mock", engine.MockEngine)
			policy := adjudicator.Policy{
				Posture:          adjudicator.PostureFailClosed,
				Allow:            map[string]bool{remoteMaintenanceTool: true},
				SelfModifyGlobs:  []string{".fak/guard/"},
				BlockedPathGlobs: []string{"/etc/", ".aws/", ".env", ".npmrc", ".kube/config"},
			}
			abi.RegisterAdjudicator(0, adjudicator.New(policy))
			srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-floor-test", VDSO: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			grant := validRemoteMaintenanceGrant(time.Now().UTC())
			grant.Paths = tc.paths
			if tc.target != "" {
				grant.Target = tc.target
			}
			receipt := remotemaint.Receipt{
				Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(grant), ProfileID: grant.ProfileID,
				OperationDigest: grant.OperationDigest, ContextID: grant.ContextID, Target: grant.Target, HostKey: grant.HostKey,
				Operation: grant.Operation, BootID: grant.BootID, StartedAt: time.Now().Add(-time.Second).UTC(),
				EndedAt: time.Now().UTC(), Outcome: remotemaint.OutcomeApplied, Verified: true,
			}
			authority := &fakeRemoteMaintenanceAuthority{grant: grant, receipt: receipt}
			srv.SetRemoteMaintenanceAuthority(authority)
			params := `{"name":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `}`
			if tc.entry == "fak_syscall" {
				params = `{"name":"fak_syscall","arguments":{"tool":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `}}`
			}
			result, rpcErr := srv.callTool(context.Background(), json.RawMessage(params))
			if rpcErr != nil {
				t.Fatalf("entry=%s rpcErr=%+v", tc.entry, rpcErr)
			}
			response := remoteMaintenanceMCPResponse(t, result)
			if response.Verdict.Kind != tc.wantKind || response.Verdict.Reason != tc.wantReason {
				t.Fatalf("entry=%s response=%+v", tc.entry, response)
			}
			if authority.execute.Load() != tc.wantExec {
				t.Fatalf("Execute calls=%d want=%d", authority.execute.Load(), tc.wantExec)
			}
			if tc.wantExec == 0 && authority.resolve.Load() != 1 {
				t.Fatalf("scope floor must run after exactly one trusted Resolve; got %d", authority.resolve.Load())
			}
		})
	}
}

func TestRemoteMaintenancePolicyResourceBoundExecutesZero(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	grant := validRemoteMaintenanceGrant(time.Now().UTC())
	grant.Paths = make([]string, 0, adjudicator.MaxRemoteMaintenancePolicyResources+1)
	for i := 1; i <= adjudicator.MaxRemoteMaintenancePolicyResources+1; i++ {
		grant.Paths = append(grant.Paths, "/var/tmp/"+strings.Repeat("a", i))
	}
	authority.grant = grant
	_, result, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), false, "", "remote-resource-bound")
	if err == nil || result != nil {
		t.Fatalf("resource overflow result=%+v err=%v", result, err)
	}
	if authority.execute.Load() != 0 {
		t.Fatalf("resource overflow executed authority %d times", authority.execute.Load())
	}
}

func TestRemoteMaintenanceUnrepresentableTargetExecutesZero(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	authority.grant.Target = "maintenance.example.invalid/unrepresentable"
	_, result, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), false, "", "remote-target-shape")
	if err == nil || result != nil {
		t.Fatalf("unrepresentable target result=%+v err=%v", result, err)
	}
	if authority.execute.Load() != 0 {
		t.Fatalf("unrepresentable target executed authority %d times", authority.execute.Load())
	}
}

func TestRemoteMaintenanceLazyDefaultInvalidContextDeniesInvocationWithoutBrickingStartup(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content func(time.Time) []byte
		want    error
	}{
		{name: "invalid", content: func(time.Time) []byte { return []byte(`{"schema":`) }, want: remotemaint.ErrAuthority},
		{name: "expired", content: func(now time.Time) []byte {
			body, err := json.Marshal(remotemaint.Config{Schema: remotemaint.ConfigSchema, Profiles: []remotemaint.Profile{{
				ProfileID: "ops-profile", Target: "maintenance.example.invalid",
				HostKey: "SHA256:" + base64.RawStdEncoding.EncodeToString(make([]byte, 32)), Operation: remotemaint.OperationSystemdProfileInstallReload,
				PayloadSHA256: strings.Repeat("3", 64), Holder: "operator", ExpiresAt: now.Add(-time.Minute).UTC(),
			}}})
			if err != nil {
				panic(err)
			}
			return body
		}, want: remotemaint.ErrExpired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			t.Setenv("HOME", home)
			t.Setenv("USERPROFILE", home)
			configPath := filepath.Join(home, ".fak", "guard", "remote-maintenance.json")
			if err := os.MkdirAll(filepath.Dir(configPath), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(configPath, tc.content(time.Now().UTC()), 0o600); err != nil {
				t.Fatal(err)
			}

			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			engine := &countingRemoteMaintenanceEngine{}
			abi.RegisterEngine("test", engine)
			abi.RegisterEngine("mock", engine)
			abi.RegisterAdjudicator(0, &remoteMaintenanceAdjudicator{allow: true})
			srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-default-context", VDSO: true})
			if err != nil {
				t.Fatalf("base startup was bricked by optional context: %v", err)
			}
			t.Cleanup(srv.Close)
			_, result, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), false, "", "remote-default-context")
			if !errors.Is(err, tc.want) || result != nil {
				t.Fatalf("invocation result=%+v err=%v, want %v", result, err, tc.want)
			}
			if engine.calls.Load() != 0 {
				t.Fatalf("invalid default context executed engine %d times", engine.calls.Load())
			}
		})
	}
}

// fak-test:runtime medium est=4s
func TestRemoteMaintenanceLazyDefaultNativeAuthorityDirectMCP(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	executableBytes, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	executableSum := sha256.Sum256(executableBytes)
	now := time.Now().UTC()
	grant := validRemoteMaintenanceGrant(now)
	config := remotemaint.Config{Schema: remotemaint.ConfigSchema, Profiles: []remotemaint.Profile{{
		ProfileID: grant.ProfileID, Target: grant.Target, HostKey: grant.HostKey,
		Operation: grant.Operation, PayloadSHA256: grant.PayloadSHA256, Holder: grant.Holder,
		Paths: grant.Paths, PathPrefixes: grant.PathPrefixes, Units: grant.Units,
		LeaseNodes: []string{"node-1"}, ExpiresAt: now.Add(time.Hour).UTC(),
		Authority: remotemaint.Command{Executable: executable, SHA256: hex.EncodeToString(executableSum[:]), Args: []string{"-test.run=^TestRemoteMaintenanceNativeAuthorityProcess$", "--", "remote-maintenance-native-adapter"}},
	}}}
	configPath := filepath.Join(home, ".fak", "guard", "remote-maintenance.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o700); err != nil {
		t.Fatal(err)
	}
	configBytes, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, configBytes, 0o600); err != nil {
		t.Fatal(err)
	}

	shippedBytes, err := os.ReadFile(filepath.Join("..", "..", "cmd", "fak", "guard-default-policy.json"))
	if err != nil {
		t.Fatal(err)
	}
	shippedPolicy, err := policy.Parse(shippedBytes)
	if err != nil {
		t.Fatal(err)
	}
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, adjudicator.New(shippedPolicy))
	srv, err := New(Config{EngineID: "test", Model: "remote-maintenance-native-authority", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_remote_maintenance","arguments":`+remoteMaintenanceArgs()+`}`))
	if rpcErr != nil {
		t.Fatalf("direct MCP native authority: %+v", rpcErr)
	}
	response, isError := remoteMaintenanceMCPWire(t, result)
	if isError || response.Verdict.Kind != "ALLOW" || response.Result == nil || response.Result.Status != "OK" {
		t.Fatalf("native authority response=%+v isError=%v", response, isError)
	}
	var receipt remotemaint.Receipt
	if err := json.Unmarshal([]byte(response.Result.Content), &receipt); err != nil || receipt.Outcome != remotemaint.OutcomeApplied || !receipt.Verified || receipt.GrantDigest == "" || receipt.ProfileID != grant.ProfileID || receipt.OperationDigest != grant.OperationDigest || receipt.Target != grant.Target {
		t.Fatalf("native authority receipt=%+v err=%v", receipt, err)
	}
}

func TestRemoteMaintenanceNativeAuthorityProcess(t *testing.T) {
	marker := false
	for _, arg := range os.Args {
		if arg == "remote-maintenance-native-adapter" {
			marker = true
			break
		}
	}
	if !marker {
		return
	}
	raw, err := io.ReadAll(os.Stdin)
	if err != nil {
		os.Exit(2)
	}
	var request remoteMaintenanceAdapterRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		os.Exit(2)
	}
	response := remoteMaintenanceAdapterResponse{Schema: remotemaint.ResponseEnvelopeSchema}
	switch request.Phase {
	case "resolve":
		grant := validRemoteMaintenanceGrant(time.Now().UTC())
		grant.ProfileID = request.ProfileID
		grant.OperationDigest = request.OperationDigest
		response.Grant = &grant
	case "validate":
		response.Grant = request.Grant
	case "execute":
		if request.Grant == nil {
			os.Exit(2)
		}
		now := time.Now().UTC()
		receipt := remotemaint.Receipt{
			Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(*request.Grant), ProfileID: request.Grant.ProfileID,
			OperationDigest: request.Grant.OperationDigest, ContextID: request.Grant.ContextID, Target: request.Grant.Target,
			HostKey: request.Grant.HostKey, Operation: request.Grant.Operation, BootID: request.Grant.BootID,
			StartedAt: now.Add(-time.Millisecond), EndedAt: now, Outcome: remotemaint.OutcomeApplied, Verified: true,
		}
		response.Receipt = &receipt
	default:
		os.Exit(2)
	}
	if err := json.NewEncoder(os.Stdout).Encode(response); err != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

func TestRemoteMaintenanceNeverReplaysEffectFromVDSO(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	for i := 0; i < 2; i++ {
		verdict, result, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), true, "", "remote-repeat")
		if err != nil || verdict.Kind != "ALLOW" || result == nil {
			t.Fatalf("call %d verdict=%+v result=%+v err=%v", i, verdict, result, err)
		}
	}
	if authority.execute.Load() != 2 {
		t.Fatalf("two admitted maintenance calls executed %d times; effect was replayed", authority.execute.Load())
	}
}

func TestRemoteMaintenanceResultFloorCannotLeakOriginalReceipt(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	abi.RegisterResultAdmitter(0, remoteMaintenanceResultDeny{})
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_remote_maintenance","arguments":`+remoteMaintenanceArgs()+`}`))
	if rpcErr != nil {
		t.Fatalf("result-floor denial dropped into RPC error: %+v", rpcErr)
	}
	response, isError := remoteMaintenanceMCPWire(t, result)
	if !isError || response.Verdict.Kind != "ALLOW" || response.Result == nil || response.Result.Status != "ERROR" {
		t.Fatalf("response=%+v isError=%v", response, isError)
	}
	if authority.execute.Load() != 1 {
		t.Fatalf("pre-admitted effect executions=%d", authority.execute.Load())
	}
	if strings.Contains(response.Result.Content, "result-admitter-secret") {
		t.Fatalf("result admission denial leaked rejected payload: %q", response.Result.Content)
	}
	var receipt remotemaint.Receipt
	if err := json.Unmarshal([]byte(response.Result.Content), &receipt); err != nil || receipt.Outcome != remotemaint.OutcomeUnknownEffect || receipt.GrantDigest != remotemaint.GrantDigest(authority.grant) || receipt.ContextID != authority.grant.ContextID {
		t.Fatalf("safe result-admission receipt=%+v err=%v", receipt, err)
	}
	if response.Result.Meta["remote_maintenance_authorization"] != "ALLOW" || response.Result.Meta["remote_maintenance_result_admission"] != "DENY" || response.Result.Meta["remote_maintenance_effect"] != "attempted_unknown" {
		t.Fatalf("stage audit meta=%+v", response.Result.Meta)
	}
}

func TestRemoteMaintenanceBoundUnknownEffectSurvivesMCPErrorWire(t *testing.T) {
	for _, tool := range []string{remoteMaintenanceTool, "fak_syscall"} {
		t.Run(tool, func(t *testing.T) {
			srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
			authority.receipt.Outcome = remotemaint.OutcomeUnknownEffect
			authority.receipt.Verified = false
			authority.receipt.BackupDisposition = remotemaint.BackupRetained
			authority.executeErr = errors.New("private adapter stderr must not cross the wire")
			params := `{"name":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `}`
			if tool == "fak_syscall" {
				params = `{"name":"fak_syscall","arguments":{"tool":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `}}`
			}
			result, rpcErr := srv.callTool(context.Background(), json.RawMessage(params))
			if rpcErr != nil {
				t.Fatalf("structured failure was dropped into RPC error: %+v", rpcErr)
			}
			response, isError := remoteMaintenanceMCPWire(t, result)
			if !isError || response.Result == nil || response.Result.Status != "ERROR" || response.Verdict.Kind != "ALLOW" {
				t.Fatalf("response=%+v isError=%v", response, isError)
			}
			if strings.Contains(response.Result.Content, "private adapter stderr") {
				t.Fatalf("raw authority error leaked: %s", response.Result.Content)
			}
			var receipt remotemaint.Receipt
			if err := json.Unmarshal([]byte(response.Result.Content), &receipt); err != nil || receipt.Outcome != remotemaint.OutcomeUnknownEffect || receipt.BackupDisposition != remotemaint.BackupRetained {
				t.Fatalf("failure receipt=%+v err=%v", receipt, err)
			}
			if authority.execute.Load() != 1 {
				t.Fatalf("Execute calls=%d, want 1", authority.execute.Load())
			}
		})
	}
}

func TestRemoteMaintenanceVerifiedRollbackIsMCPErrorWithAuthorizationAllow(t *testing.T) {
	for _, tool := range []string{remoteMaintenanceTool, "fak_syscall"} {
		t.Run(tool, func(t *testing.T) {
			srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
			authority.receipt.Outcome = remotemaint.OutcomeRolledBack
			authority.receipt.Verified = true
			authority.receipt.BackupDisposition = remotemaint.BackupRestored
			params := `{"name":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `}`
			if tool == "fak_syscall" {
				params = `{"name":"fak_syscall","arguments":{"tool":"fak_remote_maintenance","arguments":` + remoteMaintenanceArgs() + `}}`
			}
			result, rpcErr := srv.callTool(context.Background(), json.RawMessage(params))
			if rpcErr != nil {
				t.Fatalf("verified rollback was dropped into RPC error: %+v", rpcErr)
			}
			response, isError := remoteMaintenanceMCPWire(t, result)
			if !isError || response.Verdict.Kind != "ALLOW" || response.Result == nil || response.Result.Status != "ERROR" {
				t.Fatalf("rollback response=%+v isError=%v", response, isError)
			}
			var receipt remotemaint.Receipt
			if err := json.Unmarshal([]byte(response.Result.Content), &receipt); err != nil || receipt.Outcome != remotemaint.OutcomeRolledBack || !receipt.Verified || receipt.BackupDisposition != remotemaint.BackupRestored {
				t.Fatalf("rollback receipt=%+v err=%v", receipt, err)
			}
			if authority.execute.Load() != 1 {
				t.Fatalf("Execute calls=%d, want 1", authority.execute.Load())
			}
		})
	}
}

func TestRemoteMaintenanceInvalidFailureIdentityIsReplacedByBoundUnknownEffect(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	authority.receipt.Outcome = remotemaint.OutcomeRollbackFailed
	authority.receipt.Verified = false
	authority.receipt.BackupDisposition = remotemaint.BackupRetained
	authority.receipt.BootID = "attacker-selected-boot"
	authority.executeErr = errors.New("untrusted failure")
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_remote_maintenance","arguments":`+remoteMaintenanceArgs()+`}`))
	if rpcErr != nil {
		t.Fatalf("invalid failure dropped synthesized receipt: %+v", rpcErr)
	}
	response, isError := remoteMaintenanceMCPWire(t, result)
	if !isError || response.Result == nil || response.Result.Status != "ERROR" || response.Verdict.Kind != "ALLOW" {
		t.Fatalf("invalid failure response=%+v isError=%v", response, isError)
	}
	if strings.Contains(response.Result.Content, "attacker-selected-boot") || strings.Contains(response.Result.Content, "untrusted failure") {
		t.Fatalf("invalid failure identity leaked: %s", response.Result.Content)
	}
	var receipt remotemaint.Receipt
	if err := json.Unmarshal([]byte(response.Result.Content), &receipt); err != nil || receipt.Outcome != remotemaint.OutcomeUnknownEffect || receipt.GrantDigest != remotemaint.GrantDigest(authority.grant) {
		t.Fatalf("synthesized failure receipt=%+v err=%v", receipt, err)
	}
	if authority.execute.Load() != 1 {
		t.Fatalf("Execute calls=%d, want 1", authority.execute.Load())
	}
}

func TestRemoteMaintenanceRejectsCallerSelectedScopeAndChangedAuthority(t *testing.T) {
	srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
	bad := `{"profile_id":"ops-profile","operation_digest":"` + strings.Repeat("1", 64) + `","target":"attacker.invalid"}`
	if _, _, err := srv.syscall(context.Background(), remoteMaintenanceTool, bad, false, "", "remote-extra"); !errors.Is(err, remotemaint.ErrInvalidRequest) {
		t.Fatalf("extra caller scope err=%v", err)
	}
	if authority.resolve.Load() != 0 || authority.execute.Load() != 0 {
		t.Fatalf("invalid request reached authority resolve=%d execute=%d", authority.resolve.Load(), authority.execute.Load())
	}
	authority.grant.OperationDigest = strings.Repeat("a", 64)
	if _, _, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), false, "", "remote-changed"); !errors.Is(err, remotemaint.ErrChanged) {
		t.Fatalf("changed grant err=%v", err)
	}
	if authority.execute.Load() != 0 {
		t.Fatalf("changed authority executed %d times", authority.execute.Load())
	}
}

func TestRemoteMaintenanceRejectsInvalidResolvedScopeAndExpiry(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*remotemaint.Grant)
		want error
	}{
		{name: "scope", edit: func(grant *remotemaint.Grant) { grant.Paths = []string{"relative/path"} }, want: remotemaint.ErrScope},
		{name: "expired", edit: func(grant *remotemaint.Grant) { grant.ExpiresAt = time.Now().Add(-time.Minute).UTC() }, want: remotemaint.ErrExpired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, authority, _ := newRemoteMaintenanceTestServer(t, true)
			tc.edit(&authority.grant)
			if _, _, err := srv.syscall(context.Background(), remoteMaintenanceTool, remoteMaintenanceArgs(), false, "", "remote-invalid"); !errors.Is(err, tc.want) {
				t.Fatalf("err=%v want errors.Is %v", err, tc.want)
			}
			if authority.execute.Load() != 0 {
				t.Fatalf("invalid grant executed %d times", authority.execute.Load())
			}
		})
	}
}
