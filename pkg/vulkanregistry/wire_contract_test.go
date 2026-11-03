package vulkanregistry

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
// fak-test:usefulness protects=vulkan-identity-json-wire-contract
func TestIdentityJSONWireContract(t *testing.T) {
	in := Identity{
		ReceiptSchema: "fak.vulkan-build-receipt.v1",
		ID:            "vulkan-spirv.current42",
		SHA256:        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
		ModuleCount:   42,
	}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("json.Marshal(%+v) error: %v", in, err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal(%s) error: %v", raw, err)
	}
	if len(got) != 4 {
		t.Fatalf("marshaled Identity has %d keys (%s); want exactly 4: receipt_schema, id, sha256, module_count", len(got), raw)
	}
	wantStrings := map[string]string{
		"receipt_schema": in.ReceiptSchema,
		"id":             in.ID,
		"sha256":         in.SHA256,
	}
	for k, want := range wantStrings {
		if s, ok := got[k].(string); !ok || s != want {
			t.Fatalf("key %q = %v; want %q", k, got[k], want)
		}
	}
	if n, ok := got["module_count"].(float64); !ok || int(n) != in.ModuleCount {
		t.Fatalf("module_count = %v; want %d", got["module_count"], in.ModuleCount)
	}
	var out Identity
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("json.Unmarshal(%s) into Identity error: %v", raw, err)
	}
	if out != in {
		t.Fatalf("round trip = %+v; want %+v", out, in)
	}
}
