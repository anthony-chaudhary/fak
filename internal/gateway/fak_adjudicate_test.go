package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

type protectedWriteTransformAdj struct{ target string }

func (a protectedWriteTransformAdj) Caps() []abi.Capability { return nil }
func (a protectedWriteTransformAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	ref, _ := abi.ActiveResolver().Put(ctx, []byte(`{"file_path":`+strconv.Quote(a.target)+`,"content":"must-not-run"}`))
	return abi.Verdict{Kind: abi.VerdictTransform, By: "test", Payload: abi.TransformPayload{NewArgs: ref}}
}

func callFakAdjudicate(t *testing.T, srv *Server, arguments string) (SyscallResponse, *rpcError) {
	t.Helper()
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_adjudicate","arguments":`+arguments+`}`))
	if rpcErr != nil {
		return SyscallResponse{}, rpcErr
	}
	blocks, ok := result.(map[string]any)["content"].([]map[string]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("MCP content = %#v", result)
	}
	var response SyscallResponse
	if err := json.Unmarshal([]byte(blocks[0]["text"].(string)), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return response, nil
}

func assertAdjudicateReceipt(t *testing.T, receipt *AdjudicateReceipt, outcome string) {
	t.Helper()
	if receipt == nil {
		t.Fatal("receipt is nil")
	}
	if receipt.Schema != "fak-adjudicate-receipt/1" || receipt.Outcome != outcome {
		t.Fatalf("receipt = %+v", receipt)
	}
	if receipt.DurationNS < 0 || receipt.Execution != "not_executed" || receipt.Provenance != "kernel_decide" {
		t.Fatalf("receipt evidence = %+v", receipt)
	}
}

func TestFakAdjudicateReceiptOutcomesAndTrace(t *testing.T) {
	srv := newTestServer(t)
	cases := []struct {
		name, tool, outcome, kind string
		repaired                  bool
	}{
		{"allow", "allow_read", "allowed", "ALLOW", false},
		{"deny", "deny_thing", "denied", "DENY", false},
		{"transform", "transform_x", "transformed", "TRANSFORM", true},
		{"witness", "witness_ship", "witness_required", "REQUIRE_WITNESS", false},
		{"default-deny", "unknown_tool", "denied", "DENY", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"`+tc.tool+`","arguments":{"secret":"must-not-leak"},"trace_id":"kept-trace"}`)
			if rpcErr != nil {
				t.Fatalf("call: %+v", rpcErr)
			}
			if response.Verdict.Kind != tc.kind || response.TraceID != "kept-trace" || response.Result != nil {
				t.Fatalf("legacy response = %+v", response)
			}
			assertAdjudicateReceipt(t, response.Receipt, tc.outcome)
			if tc.repaired != (len(response.RepairedArguments) != 0) {
				t.Fatalf("repaired_arguments = %q, want present=%v", response.RepairedArguments, tc.repaired)
			}
			encoded, _ := json.Marshal(response)
			if strings.Contains(string(encoded), "must-not-leak") {
				t.Fatalf("response leaked raw arguments: %s", encoded)
			}
		})
	}

	minted, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_read","arguments":{}}`)
	if rpcErr != nil || !strings.HasPrefix(minted.TraceID, "gw-") {
		t.Fatalf("minted trace response = %+v, error = %+v", minted, rpcErr)
	}
}

func TestFakAdjudicateFailureIsSanitized(t *testing.T) {
	srv := newTestServer(t)
	_, rpcErr := callFakAdjudicate(t, srv, `{"tool":"","arguments":{"path":"C:\\\\private\\\\secret","token":"sk-do-not-leak"}}`)
	if rpcErr == nil {
		t.Fatal("missing tool should fail")
	}
	if rpcErr.Message != "fak_adjudicate failed" {
		t.Fatalf("message = %q", rpcErr.Message)
	}
	encoded, _ := json.Marshal(rpcErr)
	for _, leak := range []string{"private", "secret", "sk-do-not-leak", "missing tool name"} {
		if strings.Contains(string(encoded), leak) {
			t.Fatalf("error leaked %q: %s", leak, encoded)
		}
	}
	data, ok := rpcErr.Data.(AdjudicateReceipt)
	if !ok {
		t.Fatalf("error data type = %T", rpcErr.Data)
	}
	assertAdjudicateReceipt(t, &data, "failed")
	if data.Error == nil || data.Error.Code != "invalid_arguments" || data.Error.Source != "gateway" {
		t.Fatalf("sanitized error = %+v", data.Error)
	}
}

func TestFakAdjudicateDiscoveryDocumentsReceiptAndNeverExecute(t *testing.T) {
	var found map[string]any
	for _, descriptor := range toolDescriptors() {
		if descriptor["name"] == "fak_adjudicate" {
			found = descriptor
			break
		}
	}
	if found == nil {
		t.Fatal("fak_adjudicate descriptor missing")
	}
	description, _ := found["description"].(string)
	for _, want := range []string{"WITHOUT executing", "fak-adjudicate-receipt/1", "not_executed", "kernel_decide", "Repaired arguments appear only for TRANSFORM"} {
		if !strings.Contains(description, want) {
			t.Fatalf("description missing %q: %q", want, description)
		}
	}
}

func TestFakAdjudicateEnforcesCanonicalLeaseAdmissionBeforeProposedWrite(t *testing.T) {
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Clean(filepath.Join(packageDir, "..", ".."))
	if err := os.Chdir(repoRoot); err != nil {
		t.Fatalf("chdir repo root: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(packageDir) })

	peerPath := "internal/gateway/.fak-adjudicate-11834-peer-probe"
	if _, err := os.Stat(peerPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("peer probe must start absent, stat error=%v", err)
	}
	refs := map[string]string{
		"refs/fak/locks/gateway-peer":         `{"id":"gateway-peer","tree_globs":["internal/gateway/**"],"holder":"node-peer/peer-session","acquired_unix":1000,"ttl_seconds":0,"generation":1,"session_id":"peer-session"}`,
		"refs/fak/locks/session-peer-session": `{"id":"peer-session","host":"node-peer","pcb_state":"RUNNING","updated_at":1000,"ttl_seconds":0}`,
	}
	installLeasePlane(t, leaseref.NewWithRunner(fakeLockRunner(refs), repoRoot))
	workspaceLeases := []WorkspaceAdmissionLease{{TreeGlobs: []string{"internal/gateway/**"}, SessionID: "peer-session"}}
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: repoRoot, Leases: workspaceLeases}, nil
	})
	t.Cleanup(func() { SetWorkspaceLeaseAdmissionProvider(nil) })
	srv := newTestServer(t)

	conflict, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"`+peerPath+`","content":"must-not-run"},"trace_id":"requesting-session"}`)
	if rpcErr != nil {
		t.Fatalf("conflicting proposed write: %+v", rpcErr)
	}
	if conflict.Verdict.Kind != "DENY" || conflict.Verdict.Reason != "LEASE_HELD" || conflict.Verdict.By != "lease-admission" {
		t.Fatalf("conflicting verdict = %+v, want DENY/LEASE_HELD by lease-admission", conflict.Verdict)
	}
	assertAdjudicateReceipt(t, conflict.Receipt, "denied")
	if _, err := os.Stat(peerPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("adjudication executed or created peer probe, stat error=%v", err)
	}
	spoofed, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"`+peerPath+`"},"trace_id":"peer-session"}`)
	if rpcErr != nil || spoofed.Verdict.Kind != "DENY" {
		t.Fatalf("wire trace spoof claimed peer lease: response=%+v error=%+v", spoofed, rpcErr)
	}

	disjoint, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"docs/.fak-adjudicate-11834-disjoint"},"trace_id":"requesting-session"}`)
	if rpcErr != nil || disjoint.Verdict.Kind != "ALLOW" {
		t.Fatalf("disjoint proposed write = %+v, error=%+v, want ALLOW", disjoint, rpcErr)
	}
	readOnly, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_read","arguments":{"file_path":"`+peerPath+`"},"read_only":true,"trace_id":"requesting-session"}`)
	if rpcErr != nil || readOnly.Verdict.Kind != "ALLOW" {
		t.Fatalf("read-only overlap = %+v, error=%+v, want ALLOW", readOnly, rpcErr)
	}

	delete(refs, "refs/fak/locks/gateway-peer")
	workspaceLeases = nil
	released, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"`+peerPath+`"},"trace_id":"requesting-session"}`)
	if rpcErr != nil || released.Verdict.Kind != "ALLOW" {
		t.Fatalf("released proposed write = %+v, error=%+v, want ALLOW", released, rpcErr)
	}

	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{}, errors.New("authority unavailable")
	})
	authorityError, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"`+peerPath+`"},"trace_id":"requesting-session"}`)
	if rpcErr != nil {
		t.Fatalf("authority-error proposed write: %+v", rpcErr)
	}
	if authorityError.Verdict.Kind != "DENY" || authorityError.Verdict.Reason != "DEFAULT_DENY" || authorityError.Verdict.By != "lease-admission" {
		t.Fatalf("authority-error verdict = %+v, want fail-closed DEFAULT_DENY", authorityError.Verdict)
	}
	readDuringError, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_read","arguments":{"file_path":"`+peerPath+`"},"read_only":true}`)
	if rpcErr != nil || readDuringError.Verdict.Kind != "ALLOW" {
		t.Fatalf("read during authority error = %+v, error=%+v, want ALLOW", readDuringError, rpcErr)
	}

	selfRefs := map[string]string{
		"refs/fak/locks/gateway-self":        `{"id":"gateway-self","tree_globs":["internal/gateway/**"],"holder":"node-self/own-session","acquired_unix":1000,"ttl_seconds":0,"generation":1,"session_id":"own-session"}`,
		"refs/fak/locks/session-own-session": `{"id":"own-session","host":"node-self","pcb_state":"RUNNING","updated_at":1000,"ttl_seconds":0}`,
	}
	installLeasePlane(t, leaseref.NewWithRunner(fakeLockRunner(selfRefs), repoRoot))
	workspaceLeases = []WorkspaceAdmissionLease{{TreeGlobs: []string{"internal/gateway/**"}, SessionID: "own-session"}}
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: repoRoot, OwnSession: "own-session", Leases: workspaceLeases}, nil
	})
	srv.SetDefaultTraceID("own-session")
	spoofedSelf, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"`+peerPath+`"}}`)
	if rpcErr != nil || spoofedSelf.Verdict.Kind != "DENY" || spoofedSelf.Verdict.Reason != "LEASE_HELD" || spoofedSelf.TraceID != "own-session" {
		t.Fatalf("caller-selected own-session text claimed lease: response=%+v error=%+v", spoofedSelf, rpcErr)
	}
	if _, err := os.Stat(peerPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("proposed-write path executed unexpectedly, stat error=%v", err)
	}
}

func TestFakAdjudicateMalformedWorkspaceAuthorityAndSyscallFailClosed(t *testing.T) {
	root := t.TempDir()
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: []WorkspaceAdmissionLease{{SessionID: "peer"}}}, nil
	})
	t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })
	srv := newTestServer(t)
	path := filepath.Join(root, "protected.txt")

	malformed, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"`+filepath.ToSlash(path)+`"}}`)
	if rpcErr != nil || malformed.Verdict.Kind != "DENY" || malformed.Verdict.Reason != "DEFAULT_DENY" {
		t.Fatalf("malformed authority response=%+v error=%+v", malformed, rpcErr)
	}
	result, rpcErr := srv.callTool(context.Background(), json.RawMessage(`{"name":"fak_syscall","arguments":{"tool":"allow_write","arguments":{"file_path":"`+filepath.ToSlash(path)+`","content":"must-not-run"}}}`))
	if rpcErr != nil {
		t.Fatalf("fak_syscall: %+v", rpcErr)
	}
	encoded, _ := json.Marshal(result)
	if !strings.Contains(string(encoded), `\"kind\":\"DENY\"`) || !strings.Contains(string(encoded), "atomic workspace lease guard is unavailable") {
		t.Fatalf("protected syscall was not denied: %s", encoded)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("protected syscall executed: %v", err)
	}
}

func TestFakAdjudicateCanonicalizesSymlinkAliasIntoPeerLease(t *testing.T) {
	root := t.TempDir()
	protected := filepath.Join(root, "internal", "gateway")
	if err := os.MkdirAll(protected, 0o755); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(root, "alias")
	if err := os.Symlink(protected, alias); err != nil {
		if runtime.GOOS != "windows" {
			t.Fatalf("create directory symlink: %v", err)
		}
		if out, junctionErr := exec.Command("cmd", "/c", "mklink", "/J", alias, protected).CombinedOutput(); junctionErr != nil {
			t.Fatalf("create directory junction after symlink refusal %v: %v: %s", err, junctionErr, out)
		}
	}
	canonical, ok := adjudicateWorkspacePath(root, "", "alias/probe.txt")
	if !ok || canonical != "internal/gateway/probe.txt" {
		t.Fatalf("canonical alias target = %q, ok=%t", canonical, ok)
	}
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: []WorkspaceAdmissionLease{{TreeGlobs: []string{"internal/gateway/**"}}}}, nil
	})
	t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })

	response, rpcErr := callFakAdjudicate(t, newTestServer(t), `{"tool":"allow_write","arguments":{"file_path":"alias/probe.txt"}}`)
	if rpcErr != nil || response.Verdict.Kind != "DENY" || response.Verdict.Reason != "LEASE_HELD" {
		t.Fatalf("symlink alias response=%+v error=%+v", response, rpcErr)
	}
}

func TestFakAdjudicateDOSHolderTextMatchingProcessSessionDoesNotClaimLease(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{
			WorkspaceRoot: root,
			OwnSession:    "spoofed-owner-text",
			Leases:        []WorkspaceAdmissionLease{{TreeGlobs: []string{"internal/gateway/**"}}},
		}, nil
	})
	t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })

	response, rpcErr := callFakAdjudicate(t, newTestServer(t), `{"tool":"allow_write","arguments":{"file_path":"internal/gateway/probe.txt"}}`)
	if rpcErr != nil || response.Verdict.Kind != "DENY" || response.Verdict.Reason != "LEASE_HELD" {
		t.Fatalf("spoofed holder response=%+v error=%+v", response, rpcErr)
	}
}

func TestFakAdjudicateDeleteAndMoveTargetsRespectPeerLease(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "outside"), 0o755); err != nil {
		t.Fatal(err)
	}
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: []WorkspaceAdmissionLease{{TreeGlobs: []string{"protected/**"}}}}, nil
	})
	t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })
	srv := newTestServer(t)
	for _, tc := range []struct{ name, tool, args string }{
		{"delete", "allow_delete", `{"file_path":"protected/delete.txt"}`},
		{"move destination", "allow_move", `{"file_path":"outside/source.txt","destination":"protected/moved.txt"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"`+tc.tool+`","arguments":`+tc.args+`}`)
			if rpcErr != nil || response.Verdict.Kind != "DENY" || response.Verdict.Reason != "LEASE_HELD" {
				t.Fatalf("response=%+v error=%+v", response, rpcErr)
			}
		})
	}
}

func TestFakAdjudicateChecksTransformedWriteTarget(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: []WorkspaceAdmissionLease{{TreeGlobs: []string{"protected/**"}}}}, nil
	})
	t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })

	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, protectedWriteTransformAdj{target: "protected/transformed.txt"})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"disjoint/original.txt"}}`)
	if rpcErr != nil || response.Verdict.Kind != "DENY" || response.Verdict.Reason != "LEASE_HELD" {
		t.Fatalf("transformed target response=%+v error=%+v", response, rpcErr)
	}
	if !strings.Contains(string(response.RepairedArguments), "protected/transformed.txt") {
		t.Fatalf("repaired arguments = %q", response.RepairedArguments)
	}
}
