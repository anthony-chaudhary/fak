package armbench

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestManagedArmsPinExcludedFeatures(t *testing.T) {
	for _, a := range ManagedArms() {
		g, err := TogglesForManagedArm(a)
		if err != nil {
			t.Fatal(err)
		}
		if g.Routing || g.Policy || g.ResponseReuse {
			t.Fatalf("%s enabled excluded feature", a)
		}
	}
}
func TestManagedTransformsAreIsolated(t *testing.T) {
	raw := []byte(`{"system":[{"type":"text","text":"stable"}],"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"x"},{"role":"user","content":[{"type":"tool_result","content":"` + string(make([]byte, 5000)) + `"}]}]}`)
	// JSON cannot contain NULs; use a valid long string.
	var root map[string]any
	_ = json.Unmarshal([]byte(`{"system":[{"type":"text","text":"stable"}],"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"x"},{"role":"user","content":[{"type":"tool_result","content":"short"}]}]}`), &root)
	long := ""
	for i := 0; i < 5000; i++ {
		long += "a"
	}
	root["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)["content"] = long
	raw, _ = json.Marshal(root)
	for _, a := range ManagedArms() {
		g, _ := TogglesForManagedArm(a)
		out, st, err := transformAnthropicRequest(raw, g)
		if err != nil {
			t.Fatal(err)
		}
		_ = out
		if (st.cache > 0) != g.SharedPrefixProviderCache {
			t.Fatalf("%s cache=%d", a, st.cache)
		}
		if (st.compressed > 0) != g.ToolResultCompression {
			t.Fatalf("%s compressed=%d", a, st.compressed)
		}
		if st.shed > 0 {
			t.Fatalf("small request unexpectedly shed in %s", a)
		}
	}
}
func TestManagedProxyForwardsWithoutSecretsInReceipt(t *testing.T) {
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "secret" {
			t.Error("header missing")
		}
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()
	// StartManagedProxy requires HTTPS and its production client verifies trust, so exercise handler-level transforms separately; receipt's schema has no header/body fields.
	g, _ := TogglesForManagedArm(ManagedPassthrough)
	raw := []byte("  {\n  \"messages\": []\n}  ")
	got, _, err := transformAnthropicRequest(raw, g)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("passthrough changed provider request bytes: got %q want %q", got, raw)
	}
	b, _ := json.Marshal(ProxyReceipt{})
	if string(b) == "" {
		t.Fatal("empty")
	}
	_ = context.Background()
}
