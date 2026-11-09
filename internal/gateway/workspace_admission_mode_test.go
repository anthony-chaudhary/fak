package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// newWorkspaceAdmissionModeTestServer builds an isolated server pinned to the
// named admission mode for the mode-matrix tests.
func newWorkspaceAdmissionModeTestServer(t *testing.T, mode WorkspaceAdmissionMode) *Server {
	t.Helper()
	newTestServer(t)
	abi.RegisterAdjudicator(-1, permissiveAdmissionTestAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true, WorkspaceAdmission: string(mode)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// fak-test:runtime fast est=1s
func TestResolveWorkspaceAdmissionModePrecedence(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
		want WorkspaceAdmissionMode
	}{
		{name: "empty config resolves open", want: WorkspaceAdmissionOpen},
		{name: "permissive bool alone resolves permissive", cfg: Config{WorkspaceAdmissionPermissive: true}, want: WorkspaceAdmissionPermissive},
		{name: "explicit strict wins over permissive bool", cfg: Config{WorkspaceAdmission: "strict", WorkspaceAdmissionPermissive: true}, want: WorkspaceAdmissionStrict},
		{name: "explicit open beats permissive bool", cfg: Config{WorkspaceAdmission: "open", WorkspaceAdmissionPermissive: true}, want: WorkspaceAdmissionOpen},
		{name: "trimmed upper-case token parses", cfg: Config{WorkspaceAdmission: "  STRICT "}, want: WorkspaceAdmissionStrict},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveWorkspaceAdmissionMode(tc.cfg); got != tc.want {
				t.Fatalf("resolved %q, want %q", got, tc.want)
			}
		})
	}
	if got, err := ParseWorkspaceAdmissionMode(""); err != nil || got != "" {
		t.Fatalf("empty mode must parse clean: got %q err %v", got, err)
	}
	if _, err := ParseWorkspaceAdmissionMode("bogus"); err == nil {
		t.Fatal("unknown token must fail loud")
	}
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionDowngradeMatrix(t *testing.T) {
	for _, tc := range []struct {
		mode                   WorkspaceAdmissionMode
		defaultDeny, leaseHeld bool
		defaultDenyBy          string
	}{
		{mode: "", defaultDeny: true, leaseHeld: true, defaultDenyBy: "lease-admission-open"},
		{mode: WorkspaceAdmissionOpen, defaultDeny: true, leaseHeld: true, defaultDenyBy: "lease-admission-open"},
		{mode: WorkspaceAdmissionPermissive, defaultDeny: true, leaseHeld: false, defaultDenyBy: "lease-admission-permissive"},
		{mode: WorkspaceAdmissionStrict, defaultDeny: false, leaseHeld: false},
	} {
		s := &Server{workspaceAdmissionMode: tc.mode}
		if admit, by := s.workspaceAdmissionDowngrade(abi.ReasonName(abi.ReasonDefaultDeny)); admit != tc.defaultDeny || (admit && by != tc.defaultDenyBy) {
			t.Errorf("mode %q DEFAULT_DENY downgrade=(%t,%q), want (%t,%q)", tc.mode, admit, by, tc.defaultDeny, tc.defaultDenyBy)
		}
		if admit, _ := s.workspaceAdmissionDowngrade(abi.ReasonName(abi.ReasonLeaseHeld)); admit != tc.leaseHeld {
			t.Errorf("mode %q LEASE_HELD downgrade=%t, want %t", tc.mode, admit, tc.leaseHeld)
		}
	}
}

// installWorkspaceAdmissionModeAuthority installs a workspace authority that either
// fails closed (authorityErr) or reports a peer lease holding protected/**.
func installWorkspaceAdmissionModeAuthority(t *testing.T, root string, authorityErr error, peer bool) {
	t.Helper()
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		view := WorkspaceLeaseAdmissionView{WorkspaceRoot: root}
		if peer {
			view.Leases = []WorkspaceAdmissionLease{{SessionID: "peer", TreeGlobs: []string{"protected/**"}}}
		}
		return view, authorityErr
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionModeAdjudicatePosture(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeArgs := `{"tool":"allow_write","arguments":{"file_path":` + strconv.Quote(filepath.Join(root, "probe.txt")) + `}}`
	peerArgs := `{"tool":"allow_write","arguments":{"file_path":` + strconv.Quote(filepath.Join(root, "protected", "probe.txt")) + `}}`

	for _, tc := range []struct {
		mode             WorkspaceAdmissionMode
		authorityErr     error
		peer             bool
		args             string
		kind, reason, by string
		wouldDeny        string
		detailKind       string
	}{
		// DEFAULT_DENY-class refusal (unavailable authority): open admits-and-logs,
		// permissive admits-and-logs, strict refuses.
		{mode: WorkspaceAdmissionOpen, authorityErr: errWorkspaceAuthorityUnavailable, args: writeArgs, kind: "ALLOW", by: "lease-admission-open", wouldDeny: "DEFAULT_DENY", detailKind: "open"},
		{mode: WorkspaceAdmissionPermissive, authorityErr: errWorkspaceAuthorityUnavailable, args: writeArgs, kind: "ALLOW", by: "lease-admission-permissive", wouldDeny: "DEFAULT_DENY", detailKind: "permissive"},
		{mode: WorkspaceAdmissionStrict, authorityErr: errWorkspaceAuthorityUnavailable, args: writeArgs, kind: "DENY", reason: "DEFAULT_DENY", by: "lease-admission"},
		// Peer LEASE_HELD conflict: open downgrades it too; permissive and strict stand.
		{mode: WorkspaceAdmissionOpen, peer: true, args: peerArgs, kind: "ALLOW", by: "lease-admission-open", wouldDeny: "LEASE_HELD", detailKind: "open"},
		{mode: WorkspaceAdmissionPermissive, peer: true, args: peerArgs, kind: "DENY", reason: "LEASE_HELD", by: "lease-admission"},
		{mode: WorkspaceAdmissionStrict, peer: true, args: peerArgs, kind: "DENY", reason: "LEASE_HELD", by: "lease-admission"},
		// Kernel policy denial stands in every mode: lease admission never runs after a DENY.
		{mode: WorkspaceAdmissionOpen, args: `{"tool":"deny_write","arguments":{"file_path":"x"}}`, kind: "DENY", reason: abi.ReasonName(abi.ReasonPolicyBlock), by: "test"},
		{mode: WorkspaceAdmissionStrict, args: `{"tool":"deny_write","arguments":{"file_path":"x"}}`, kind: "DENY", reason: abi.ReasonName(abi.ReasonPolicyBlock), by: "test"},
	} {
		name := tc.kind + "/" + string(tc.mode)
		if tc.reason != "" {
			name += "/" + tc.reason
		}
		t.Run(name, func(t *testing.T) {
			installWorkspaceAdmissionModeAuthority(t, root, tc.authorityErr, tc.peer)
			response, rpcErr := callFakAdjudicate(t, newWorkspaceAdmissionModeTestServer(t, tc.mode), tc.args)
			if rpcErr != nil || response.Verdict.Kind != tc.kind || (tc.reason != "" && response.Verdict.Reason != tc.reason) {
				t.Fatalf("response=%+v error=%+v, want %s/%s", response, rpcErr, tc.kind, tc.reason)
			}
			if tc.by != "" && response.Verdict.By != tc.by {
				t.Fatalf("deciding authority=%q, want %q", response.Verdict.By, tc.by)
			}
			if tc.wouldDeny != "" {
				if response.Verdict.Detail["would_deny"] != tc.wouldDeny || response.Verdict.Detail["workspace_admission"] != tc.detailKind || response.Verdict.Detail["would_deny_by"] == "" || response.Verdict.Detail["kernel_by"] == "" {
					t.Fatalf("admitted verdict lost forensic evidence: %+v", response.Verdict)
				}
			}
		})
	}
}

var errWorkspaceAuthorityUnavailable = contextError("workspace lease authority unavailable")

type contextError string

func (e contextError) Error() string { return string(e) }

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionModeOperatorApprovalOverride(t *testing.T) {
	const approved = "rm -rf build"
	for _, mode := range []WorkspaceAdmissionMode{WorkspaceAdmissionOpen, WorkspaceAdmissionPermissive, WorkspaceAdmissionStrict} {
		t.Run(string(mode), func(t *testing.T) {
			root := t.TempDir()
			// Install ONLY the workspace admission authority, leaving the lease-plane
			// presence provider nil: an unprovable shell footprint then fails closed
			// (DEFAULT_DENY) at the presence stage, which is the refusal the
			// operator-approval override admits. A working empty presence view would
			// admit the call through the kernel verdict alone and never exercise the
			// override.
			SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
				return WorkspaceLeaseAdmissionView{WorkspaceRoot: root}, nil
			})
			t.Cleanup(func() { SetWorkspaceLeaseAdmissionProvider(nil) })
			// The fixture adjudicator allows bash; swap in the operator-approved floor
			// record for exactly one command so the override path is exercised in every mode.
			newTestServer(t)
			abi.RegisterAdjudicator(-1, approvedBashAdj{approvedCmd: approved})
			srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true, WorkspaceAdmission: string(mode)})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"bash","arguments":{"command":"`+approved+`"}}`)
			if rpcErr != nil || response.Verdict.Kind != "ALLOW" || response.Verdict.By != "lease-admission-operator-approved" {
				t.Fatalf("approved call under %s: response=%+v error=%+v, want ALLOW by lease-admission-operator-approved", mode, response, rpcErr)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionModeSyscallExecution(t *testing.T) {
	for _, tc := range []struct {
		mode         WorkspaceAdmissionMode
		kind, reason string
		calls        int64
		detailKind   string
	}{
		{mode: WorkspaceAdmissionOpen, kind: "ALLOW", calls: 1, detailKind: "open"},
		{mode: WorkspaceAdmissionPermissive, kind: "ALLOW", calls: 1, detailKind: "permissive"},
		{mode: WorkspaceAdmissionStrict, kind: "DENY", reason: "DEFAULT_DENY"},
	} {
		t.Run(string(tc.mode), func(t *testing.T) {
			root := t.TempDir()
			setPermissiveAdmissionTestCWD(t, root)
			installWorkspaceAdmissionModeAuthority(t, root, nil, false)
			engine := &pluginWriteFenceEngine{}
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("mode-execution", engine)
			abi.RegisterAdjudicator(0, toolAdj{})
			srv, err := New(Config{EngineID: "mode-execution", Model: "test-model", VDSO: true, WorkspaceAdmission: string(tc.mode)})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			response := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
				"tool":      "allow_write",
				"arguments": map[string]any{"file_path": filepath.Join(root, "probe.txt"), "content": "fake-engine-only"},
				"trace_id":  "mode-execution-test",
			})
			if response.Verdict.Kind != tc.kind || (tc.reason != "" && response.Verdict.Reason != tc.reason) {
				t.Fatalf("verdict=%+v, want %s/%s", response.Verdict, tc.kind, tc.reason)
			}
			if got := engine.calls.Load(); got != tc.calls {
				t.Fatalf("engine calls=%d, want %d", got, tc.calls)
			}
			if tc.calls == 1 {
				if response.Verdict.By != "lease-admission-"+tc.detailKind ||
					response.Verdict.Detail["workspace_admission"] != tc.detailKind ||
					response.Verdict.Detail["would_deny"] != "DEFAULT_DENY" ||
					response.Verdict.Detail["atomic_workspace_guard"] != "unavailable" {
					t.Fatalf("execution lost admission evidence: %+v", response.Verdict)
				}
			}
		})
	}
}
