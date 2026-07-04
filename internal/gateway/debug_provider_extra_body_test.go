package gateway

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestDebugVarsExposeProviderExtraBodyKeysOnly(t *testing.T) {
	srv := newTestServer(t)
	hp := agent.NewHTTPPlanner("http://upstream.example/v1", "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M", "")
	if err := hp.SetExtraBodyJSON(`{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}`); err != nil {
		t.Fatalf("SetExtraBodyJSON: %v", err)
	}
	srv.planner = hp

	vars := srv.debugVars(time.Now())
	if !vars.Upstream.ProviderExtraBodySet {
		t.Fatal("/debug/vars did not report provider extra body set")
	}
	wantKeys := []string{"chat_template_kwargs", "top_k"}
	if !reflect.DeepEqual(vars.Upstream.ProviderExtraBodyKeys, wantKeys) {
		t.Fatalf("provider extra body keys = %v, want %v", vars.Upstream.ProviderExtraBodyKeys, wantKeys)
	}
	raw, err := json.Marshal(vars.Upstream)
	if err != nil {
		t.Fatalf("marshal upstream vars: %v", err)
	}
	text := string(raw)
	for _, forbidden := range []string{"preserve_thinking", "20"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("/debug/vars leaked provider extra body value %q in %s", forbidden, text)
		}
	}
}

func TestDebugVarsProviderExtraBodyUnset(t *testing.T) {
	srv := newTestServer(t)
	vars := srv.debugVars(time.Now())
	if vars.Upstream.ProviderExtraBodySet || len(vars.Upstream.ProviderExtraBodyKeys) != 0 {
		t.Fatalf("unset provider extra body reported as set: %+v", vars.Upstream)
	}
}
