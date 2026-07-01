package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// vllmBodyCapture starts a minimal streaming vLLM stand-in that records each
// request body and returns one finished SSE frame so Complete drains cleanly.
// Handler assertions use Errorf, never Fatalf, since it runs off the test
// goroutine.
func vllmBodyCapture(t *testing.T, bodies chan<- map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body JSON: %v", err)
		}
		bodies <- body
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
}

// TestVLLMCacheSaltIsolatesTenants proves two tenants presenting a byte-identical
// prefix are lowered with DIFFERENT cache_salt values, so vLLM's prefix cache can
// never alias them across the trust boundary. (Acceptance 1.)
func TestVLLMCacheSaltIsolatesTenants(t *testing.T) {
	ctx := context.Background()
	bodies := make(chan map[string]any, 2)
	srv := vllmBodyCapture(t, bodies)
	defer srv.Close()
	e := NewVLLMEngine(VLLMConfig{BaseURL: srv.URL + "/v1", Model: "served", WorkerID: "w"})

	prefix := `{"messages":[{"role":"user","content":"identical prefix"}]}`
	call := func(tenant string) map[string]any {
		if _, err := e.Complete(ctx, &abi.ToolCall{
			Tool: "chat",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(prefix)},
			Meta: map[string]string{MetaCacheTenant: tenant},
		}); err != nil {
			t.Fatalf("Complete(%s): %v", tenant, err)
		}
		return <-bodies
	}
	a := call("tenant-a")
	b := call("tenant-b")
	saltA, _ := a["cache_salt"].(string)
	saltB, _ := b["cache_salt"].(string)
	if saltA == "" || saltB == "" {
		t.Fatalf("both requests must carry a cache_salt: a=%q b=%q", saltA, saltB)
	}
	if saltA == saltB {
		t.Fatalf("two tenants with a byte-identical prefix must get different cache_salt, both = %q", saltA)
	}
	if strings.Contains(saltA, "tenant-a") || strings.Contains(saltB, "tenant-b") {
		t.Fatalf("cache_salt must not embed the raw tenant: a=%q b=%q", saltA, saltB)
	}
}

// TestVLLMPriorityFromTurnIntentGatedByAdvertisement proves a fak TurnIntent
// priority is lowered to the vLLM priority field ONLY when the served engine
// advertises priority scheduling, and degrades to no field otherwise.
// (Acceptance 2.)
func TestVLLMPriorityFromTurnIntentGatedByAdvertisement(t *testing.T) {
	ctx := context.Background()
	prefix := `{"messages":[{"role":"user","content":"hi"}]}`

	bodies := make(chan map[string]any, 1)
	srv := vllmBodyCapture(t, bodies)
	defer srv.Close()
	on := NewVLLMEngine(VLLMConfig{BaseURL: srv.URL + "/v1", Model: "served", PriorityScheduling: true})
	if _, err := on.Complete(ctx, &abi.ToolCall{Tool: "chat", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(prefix)}, Meta: map[string]string{MetaTurnPriority: "2"}}); err != nil {
		t.Fatalf("Complete on: %v", err)
	}
	got := <-bodies
	p, ok := got["priority"]
	if !ok {
		t.Fatalf("priority-scheduling engine must emit a priority field: %#v", got)
	}
	if pf, ok := p.(float64); !ok || int(pf) != 2 {
		t.Fatalf("priority = %#v, want 2", p)
	}

	bodies2 := make(chan map[string]any, 1)
	srv2 := vllmBodyCapture(t, bodies2)
	defer srv2.Close()
	off := NewVLLMEngine(VLLMConfig{BaseURL: srv2.URL + "/v1", Model: "served"})
	if _, err := off.Complete(ctx, &abi.ToolCall{Tool: "chat", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(prefix)}, Meta: map[string]string{MetaTurnPriority: "2"}}); err != nil {
		t.Fatalf("Complete off: %v", err)
	}
	got2 := <-bodies2
	if _, ok := got2["priority"]; ok {
		t.Fatalf("engine without priority scheduling must omit the priority field: %#v", got2)
	}
}

// TestVLLMCacheAttributionRecordsSaltFamilyNoSecret proves the trace records the
// salt family label and the priority, that the recorded salt matches the one put
// on the wire, and that NO raw identity secret leaks into the result meta.
// (Acceptance 3 and 4.)
func TestVLLMCacheAttributionRecordsSaltFamilyNoSecret(t *testing.T) {
	ctx := context.Background()
	bodies := make(chan map[string]any, 1)
	srv := vllmBodyCapture(t, bodies)
	defer srv.Close()
	e := NewVLLMEngine(VLLMConfig{BaseURL: srv.URL + "/v1", Model: "served", PriorityScheduling: true})
	res, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
		Meta: map[string]string{MetaCacheTenant: "secret-tenant", MetaCacheAuthority: "authZ", MetaTurnPriority: "5"},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	body := <-bodies

	wantSalt := deriveCacheSalt("secret-tenant", "authZ", "")
	if wantSalt == "" || res.Meta["cache_salt"] != wantSalt {
		t.Fatalf("meta cache_salt = %q, want %q", res.Meta["cache_salt"], wantSalt)
	}
	if !strings.HasPrefix(res.Meta["cache_salt"], "fak-") {
		t.Fatalf("cache_salt family must be a fak- digest label: %q", res.Meta["cache_salt"])
	}
	if res.Meta["priority"] != "5" {
		t.Fatalf("meta priority = %q, want 5", res.Meta["priority"])
	}
	for k, v := range res.Meta {
		if strings.Contains(v, "secret-tenant") || strings.Contains(v, "authZ") {
			t.Fatalf("meta[%q]=%q leaks raw identity", k, v)
		}
	}
	if s, _ := body["cache_salt"].(string); s != wantSalt {
		t.Fatalf("wire cache_salt = %q, want %q (must equal the recorded family)", s, wantSalt)
	}
}

// TestDeriveCacheSaltAndPriorityPureProperties pins the derivation invariants:
// determinism, per-axis sensitivity, null-separated boundaries, empty-degrades,
// and integer-only priority parsing.
func TestDeriveCacheSaltAndPriorityPureProperties(t *testing.T) {
	if deriveCacheSalt("t", "a", "f") != deriveCacheSalt("t", "a", "f") {
		t.Fatal("salt must be deterministic for a fixed identity")
	}
	base := deriveCacheSalt("t", "a", "f")
	if deriveCacheSalt("t2", "a", "f") == base || deriveCacheSalt("t", "a2", "f") == base || deriveCacheSalt("t", "a", "f2") == base {
		t.Fatal("changing any identity axis must change the salt")
	}
	if deriveCacheSalt("ab", "c", "") == deriveCacheSalt("a", "bc", "") {
		t.Fatal("axis boundaries must not collide by concatenation")
	}
	if deriveCacheSalt("", "", "") != "" {
		t.Fatal("empty identity must yield empty salt")
	}
	if _, ok := derivePriority(""); ok {
		t.Fatal("unset priority must not be ok")
	}
	if _, ok := derivePriority("high"); ok {
		t.Fatal("non-integer priority must not be ok")
	}
	if p, ok := derivePriority(" 3 "); !ok || p != 3 {
		t.Fatalf("derivePriority = %d,%v want 3,true", p, ok)
	}
}
