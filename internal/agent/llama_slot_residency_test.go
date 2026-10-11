package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime fast est=100ms lane=default
func TestLlamaSingleSlotRequiresOptInAndDiscoveredCapability(t *testing.T) {
	for _, tc := range []struct {
		name       string
		optIn      bool
		provider   Provider
		discovered int64
		wantRead   bool
	}{
		{name: "opt_out_known_single", provider: ProviderOpenAI, discovered: 1},
		{name: "non_openai_known_single", optIn: true, provider: ProviderAnthropic, discovered: 1},
		{name: "unknown", optIn: true, provider: ProviderOpenAI},
		{name: "known_multi", optIn: true, provider: ProviderOpenAI, discovered: 4},
		{name: "known_single", optIn: true, provider: ProviderOpenAI, discovered: 1, wantRead: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var reads atomic.Int32
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				reads.Add(1)
				if r.Method != http.MethodGet || r.URL.Path != "/slots" {
					t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				}
				_, _ = w.Write([]byte(`[{"id_task":42,"is_processing":false,"n_prompt_tokens":123}]`))
			}))
			t.Cleanup(up.Close)
			p := &HTTPPlanner{Provider: tc.provider, BaseURL: up.URL + "/v1", LlamaSlotAffinity: tc.optIn}
			st := llamaSlotStateFor(p.BaseURL)
			st.slots.Store(tc.discovered)
			t.Cleanup(func() { llamaSlotRegistry.Delete(p.BaseURL) })
			got, ok := p.LlamaSingleSlot(context.Background(), time.Second)
			if ok != tc.wantRead {
				t.Fatalf("ok=%v, want %v", ok, tc.wantRead)
			}
			wantReads := int32(0)
			if tc.wantRead {
				wantReads = 1
				if got.IDTask != 42 || got.Processing || got.PromptTokens != 123 {
					t.Fatalf("slot=%+v", got)
				}
			}
			if reads.Load() != wantReads {
				t.Fatalf("HTTP reads=%d, want %d", reads.Load(), wantReads)
			}
			if st.probing.Load() {
				t.Fatal("residency read started capability discovery")
			}
		})
	}
}
