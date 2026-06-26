package fakclient_test

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

// The fak gateway's wire DTOs live in internal/gateway/wire.go, which Go's
// internal/ rule seals from any out-of-tree consumer — so the public SDK in this
// package must carry its OWN copy of those types. These tests gate that copy
// against drift: a server DTO field that is renamed, retyped, or added without a
// matching SDK change makes the round-trip JSON differ and fails the build. It is
// the in-code analogue of the route-drift gate (internal/gateway/openapi_spec_test.go)
// that issue #205 box 1 added for the OpenAPI spec.

// assertServerToClient marshals a fully-populated SERVER value, decodes it into
// the SDK type, re-marshals, and asserts the JSON is identical. A field the
// server emits that the SDK drops (a rename, a missing field) changes the JSON
// and fails — proving the SDK reads every field the server sends.
func assertServerToClient(t *testing.T, serverVal, sdkPtr any) {
	t.Helper()
	serverJSON, err := json.Marshal(serverVal)
	if err != nil {
		t.Fatalf("marshal server value: %v", err)
	}
	if err := json.Unmarshal(serverJSON, sdkPtr); err != nil {
		t.Fatalf("decode server JSON into SDK type: %v", err)
	}
	sdkJSON, err := json.Marshal(sdkPtr)
	if err != nil {
		t.Fatalf("re-marshal SDK value: %v", err)
	}
	if string(serverJSON) != string(sdkJSON) {
		t.Fatalf("SDK type drifted from server wire shape:\n server: %s\n sdk:    %s", serverJSON, sdkJSON)
	}
}

// assertClientToServer is the request-direction dual: a field the SDK sends that
// the server cannot read (a rename, a missing field) changes the JSON and fails.
func assertClientToServer(t *testing.T, sdkVal, serverPtr any) {
	t.Helper()
	sdkJSON, err := json.Marshal(sdkVal)
	if err != nil {
		t.Fatalf("marshal SDK value: %v", err)
	}
	if err := json.Unmarshal(sdkJSON, serverPtr); err != nil {
		t.Fatalf("decode SDK JSON into server type: %v", err)
	}
	serverJSON, err := json.Marshal(serverPtr)
	if err != nil {
		t.Fatalf("re-marshal server value: %v", err)
	}
	if string(sdkJSON) != string(serverJSON) {
		t.Fatalf("SDK request drifted from server wire shape:\n sdk:    %s\n server: %s", sdkJSON, serverJSON)
	}
}

func TestVerdictAndResponseParity(t *testing.T) {
	// Every field set non-zero so an omitempty cannot hide a mismatch.
	server := gateway.SyscallResponse{
		Verdict: gateway.WireVerdict{
			Kind:        "DENY",
			Reason:      "SELF_MODIFY",
			By:          "selfmod",
			Disposition: "ESCALATE",
			Detail:      map[string]string{"claim": "kernel.go"},
		},
		Result: &gateway.ResultEnvelope{
			Status:  "OK",
			Content: `{"rows":3}`,
			Meta:    map[string]string{"admit": "quarantined"},
		},
		RepairedArguments: json.RawMessage(`{"path":"/srv/data.csv"}`),
		TraceID:           "t-7f3a9c",
	}
	assertServerToClient(t, server, &fakclient.SyscallResponse{})
}

func TestChangesResponseParity(t *testing.T) {
	server := gateway.ChangesResponse{
		Events: []gateway.CoherenceEvent{{
			Kind:       "mutation",
			Seq:        43,
			Tool:       "write_file",
			Tags:       []string{"fs:/srv"},
			Witness:    "sha256:abc",
			Evicted:    2,
			WorldVer:   7,
			TrustEpoch: 1,
		}},
		Cursor: 43,
	}
	assertServerToClient(t, server, &fakclient.ChangesResponse{})
}

func TestRevokeResponseParity(t *testing.T) {
	server := gateway.RevokeResponse{Witness: "sha256:abc123", Evicted: 3, TrustEpoch: 17}
	assertServerToClient(t, server, &fakclient.RevokeResponse{})
}

func TestSyscallRequestParity(t *testing.T) {
	sdk := fakclient.SyscallRequest{
		Tool:      "read_file",
		Arguments: json.RawMessage(`{"path":"test.txt"}`),
		ReadOnly:  true,
		Witness:   "sha256:w",
		TraceID:   "t-1",
		Principal: "tenant-7",
	}
	assertClientToServer(t, sdk, &gateway.SyscallRequest{})
}

func TestAdmitRequestParity(t *testing.T) {
	sdk := fakclient.AdmitRequest{
		Tool:    "fetch_url",
		Result:  json.RawMessage(`{"rows":3}`),
		Witness: "sha256:w",
		TraceID: "t-1",
	}
	assertClientToServer(t, sdk, &gateway.AdmitRequest{})
}
